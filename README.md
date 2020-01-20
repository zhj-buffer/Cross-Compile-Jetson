## Mxnet cross compile

Monday, 08. April 2019 01:13PM. The latest version of [jetpack](https://developer.nvidia.com/embedded/downloads) is 28.3 for now, feel free to update the link I attached as default.

#### Host:
	Ubuntu 18.04
#### Target: 
	Jetson Nano


#### 1. Prepare the build enviroment
##### 1. Download the Roofs Image from jetpack
[jetson-nano-sd-r32.1-2019-03-18.zip](http://developer.nvidia.com/embedded/dlc/jetson-nano-dev-kit-sd-card-image)
##### 2. change to root user
>		alan@C:~$ sudo su - root
>		root@C:~# mkdir -p /home/alan/rootfs-nano
>		root@C:~# cd /home/alan/rootfs-nano && cp /home/alan/Downloads/jetson-nano-sd-r32.1-2019-03-18.zip .
>		root@C:~# unzip jetson-nano-sd-r32.1-2019-03-18.zip 
>		root@C:~# ls
>		root@C:~# fdisk jetson-nano-sd-r32.1-2019-03-18.img

		Welcome to fdisk (util-linux 2.31.1).
		Changes will remain in memory only, until you decide to write them.
		Be careful before using the write command.
		
		
		Command (m for help): p
		Disk jetson-nano-sd-r32.1-2019-03-18.img: 12 GiB, 12884901888 bytes, 25165824 sectors
		Units: sectors of 1 * 512 = 512 bytes
		Sector size (logical/physical): 512 bytes / 512 bytes
		I/O size (minimum/optimal): 512 bytes / 512 bytes
		Disklabel type: gpt
		Disk identifier: D048AD43-24FD-4DED-B06E-7BB8ED98158C
		
		Device                                Start      End  Sectors  Size Type
		jetson-nano-sd-r32.1-2019-03-18.img1  24576 25165790 25141215   12G Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img2   2048     2303      256  128K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img3   4096     4991      896  448K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img4   6144     7295     1152  576K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img5   8192     8319      128   64K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img6  10240    10623      384  192K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img7  12288    13439     1152  576K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img8  14336    14463      128   64K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img9  16384    17663     1280  640K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img10 18432    19327      896  448K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img11 20480    20735      256  128K Linux filesystem
		jetson-nano-sd-r32.1-2019-03-18.img12 22528    22687      160   80K Linux filesystem
		
		Partition table entries are not in disk order.
		
		Command (m for help): 

From here you can see the rootfs start from block 24576 with block size 512bytes. So We need to mount the image from offset 24576 * 512 = 12582912

>		root@C:~# mount -o loop,offset=12582912 jetson-nano-sd-r32.1-2019-03-18.img /mnt
>		root@C:~# ls /mnt/
		bin  boot  dev  etc  home  lib  lost+found  media  mnt  mxnet  opt  proc  README.txt  root  run  sbin  snap  srv  sys  tmp  usr  var
>		root@C:~#

I recommand to cp all the files from image to the local folder. because all the changes under /mnt will change the origin image, and also there is a size limitation from Image. Here its max size is 12GB. Maybe not enough for the later use.
>		root@C:~# cp /mnt/*  /home/alan/rootfs-nano -rapvf
>		root@C:~#  apt install qemu-user-static
>		root@C:~# cp /usr/bin/qemu-arm-static usr/bin/
>		root@C:~# cp /usr/bin/qemu-aarch64-static usr/bin/
>		root@C:~# 

Here is a mount shell script (ch-mount.sh) to help chroot to the target file system. Here we use qemu to setup a simulation to install and get some packages conveniently. Of course you can copy from  a real device.

		#!/bin/bash
		
		function mnt() {
		    echo "MOUNTING"
		    sudo mount -t proc /proc ${2}proc
		    sudo mount -t sysfs /sys ${2}sys
		    sudo mount -o bind /dev ${2}dev
		    sudo mount -o bind /run ${2}run
		    #sudo mount -vt tmpfs shm ${2}dev/shm
		    #sudo mount -t /dev/shm ${2}dev/shm
		    sudo chroot ${2}
		}
		
		function umnt() {
		    echo "UNMOUNTING"
		    sudo umount ${2}proc
		    sudo umount ${2}sys
		    #sudo umount ${2}dev/shm
		    sudo umount ${2}dev
		    sudo umount ${2}run
		    sudo umount ${2}host
		}
		
		
		if [ "$1" == "-m" ] && [ -n "$2" ] ;
		then
		    mnt $1 $2
		elif [ "$1" == "-u" ] && [ -n "$2" ];
		then
		    umnt $1 $2
		else
		    echo ""
		    echo "Either 1'st, 2'nd or both parameters were missing"
		    echo ""
		    echo "1'st parameter can be one of these: -m(mount) OR -u(umount)"
		    echo "2'nd parameter is the full path of rootfs directory(with trailing '/')"
		    echo ""
		    echo "For example: ch-mount -m /media/sdcard/"
		    echo ""
		    echo 1st parameter : ${1}
		    echo 2nd parameter : ${2}
		fi

	root@C:~# cp /etc/resolv.conf etc/
	root@C:~# bash ch-mount.sh -m rootfs-nano 

Now you entered a simulation system and experience just like you are on target board.  It's usefull to get some packages and dependency libs without a dev kit board on hand.

	root@C:/# apt update
	root@C:/# 

Keep the session for the target, open another session for the Host.
#### 2. Get the source code.
	git clone --recursive https://github.com/dmlc/mxnet
	
#### 3. Install the dependency

Install the dependency on the Host and Target in the same time.

	sudo apt-get update
	sudo apt-get install -y \
	    apt-transport-https \
	    build-essential \
	    ca-certificates \
	    cmake \
	    curl \
	    git \
	    libatlas-base-dev \
	    libcurl4-openssl-dev \
	    libjemalloc-dev \
	    liblapack-dev \
	    libopenblas-dev \
	    libopencv-dev \
	    libzmq3-dev \
	    ninja-build \
	    python-dev \
	    python3-dev \
	    software-properties-common \
	    sudo \
	    unzip \
	    virtualenv \
	    wget

Install the python dependency on the Host and Target. You can find the requirements.txt  under docs/install/

	wget -nv https://bootstrap.pypa.io/get-pip.py
	echo "Installing for Python 3..."
	sudo python3 get-pip.py
	pip3 install --user -r requirements.txt
	echo "Installing for Python 2..."
	sudo python2 get-pip.py
	pip2 install --user -r requirements.txt
	
From now on, there are two possible ways to compiling the mxnet. You can chose step 4 or step 5.
#### 4. Build under qemu simulation

>	root@C:/# mv mxnet ./
>	root@C:/#cd mxnet
>	root@C:/#cp make/config.mk .

From jetson-nano-sd-r32.1-2019-03-18.zip official release Opencv 3.3.1 CUDA 10.0, Cudnn and TensorRT and pre-installed. If you want to use mxnet with CPP version, try to compile with USE_CPP_PACKAGE=1

	root@C:/#make -j  $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	lan@C:~/src/mxnet$ make -j  $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

	alan@C:~/src/mxnet$ cd python &&  python setup.py bdist_wheel &&  python3 setup.py bdist_wheel
	alan@C:~/src/mxnet/python$ ls dist/ 
	alan@C:~/src/mxnet/python$ dist/mxnet-1.5.0-py2-none-any.whl  dist/mxnet-1.5.0-py3-none-any.whl
	alan@C:~/src/mxnet/python$ls ../lib
	../lib/libmxnet.a  ../lib/libmxnet.so
	alan@C:~/src/mxnet$  scp ../lib/libmxnet.so dist/* jetbot@(ip):~/

It will cost u several hours depend on your CPU version.  For the speed direct build > cross-compile > compile under simulation.
#### 5. Build on the x86 Host with cross compile toolchain

Download the [cross compile toolchain](https://developer.nvidia.com/embedded/dlc/l4t-gcc-toolchain-64-bit-28-3) from Jetpack.

We just need the gcc-4.8.5-aarch64.tgz for mxnet. if you want to cross compile the kernel gcc-4.8.5-armhf.tgz is also required.

>  alan@C:~/toolchain$ tar xpvf gcc-4.8.5-aarch64.tgz
>  alan@C:~/toolchain$ mv install gcc-4.8.5-aarch64
>  alan@C:~/toolchain$ 

	alan@C:~/toolchain/gcc-4.8.5-aarch64$ tree -d -L 2
	.
	├── aarch64-unknown-linux-gnu
	│   ├── bin
	│   ├── include
	│   ├── lib
	│   ├── lib64
	│   └── sysroot
	├── bin
	├── include
	├── lib
	│   └── gcc
	├── libexec
	│   └── gcc
	└── share
	    ├── gcc-4.8.5
	    ├── info
	    ├── locale
	    └── man
	
	17 directories

You can see there are two key folders sysroot and bin.
From sysroot, you can get the basic libs used by the gcc/g++.
From bin, you can get the compile tools: gcc, g++,ar,ld...

On the target:

	root@C:/# rm /usr/include/aarch64-linux-gnu/cblas.h 
	root@C:/# cp /usr/include/aarch64-linux-gnu/cblas-openblas.h /usr/	include/aarch64-linux-gnu/cblas.h
	
On the Host:

	alan@C:~/src/mxnet$ 
	alan@C:~/src/mxnet$ cp make/config.mk .

	export CROSS_ROOT=/home/alan/rootfs-nano
	export CROSS_COMPILE = /home/alan/toolchain/gcc-4.8.5-aarch64/bin
	export CC = ${CROSS_COMPILE}/aarch64-unknown-linux-gnu-gcc
	export CXX = ${CROSS_COMPILE}/aarch64-unknown-linux-gnu-g++
	export LD = ${CROSS_COMPILE}/aarch64-unknown-linux-gnu-ld
	export AR = ${CROSS_COMPILE}/aarch64-unknown-linux-gnu-ar
	export AS = ${CROSS_COMPILE}/aarch64-unknown-linux-gnu-as
	export RANLIB = ${CROSS_COMPILE}/aarch64-unknown-linux-gnu-ranlib
	export NVCC = ${CROSS_ROOT}/usr/local/cuda/bin/nvcc
	
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SYSROOT/usr/lib/aarch64-linux-gnu:$SYSROOT/lib/aarch64-linux-gnu:$SYSROOT/lib
	export LC_ALL=C
	sudo ln -s /home/alan/rootfs-nano/usr/lib/aarch64-linux-gnu/libcudnn.so.7 /home/alan/rootfs-nano/usr/lib/aarch64-linux-gnu/libcudnn.so
	/home/alan/rootfs-nano/usr/lib/aarch64-linux-gnu/libcudnn.so -> /home/alan/rootfs-nano/usr/lib/aarch64-linux-gnu/libcudnn.so.7
	
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alan/toolchain/gcc-4.8.5-aarch64/aarch64-unknown-linux-gnu/lib64:/home/alan/toolchain/gcc-4.8.5-aarch64/aarch64-unknown-linux-gnu/sysroot/lib

	alan@C:~/src/mxnet$ vim config.mk

	# the additional link flags you want to add
	ADD_LDFLAGS =  -L${CROSS_ROOT}/lib \
	                   -L${CROSS_ROOT}/usr/local/cuda/lib64 \
	                   -L${CROSS_ROOT}/usr/lib \
	                   -L${CROSS_ROOT}/usr/lib/aarch64-linux-gnu \
	                   -L${CROSS_ROOT}/lib/aarch64-linux-gnu
	
	# the additional compile flags you want to add
	ADD_CFLAGS = -I${CROSS_ROOT}/include \
	                -I${CROSS_ROOT}/usr/local/cuda/include \
	                -I${CROSS_ROOT}/usr/include \
	                -I${CROSS_ROOT}/usr/include/aarch64-linux-gnu

	alan@C:~/src/mxnet$ make -j  $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	alan@C:~/src/mxnet$ cd python &&  python setup.py bdist_wheel &&  python3 setup.py bdist_wheel
	alan@C:~/src/mxnet/python$ ls dist/ 
	alan@C:~/src/mxnet/python$ dist/mxnet-1.5.0-py2-none-any.whl  dist/mxnet-1.5.0-py3-none-any.whl
	alan@C:~/src/mxnet/python$ls ../lib
	../lib/libmxnet.a  ../lib/libmxnet.so
	alan@C:~/src/mxnet$  scp ../lib/libmxnet.so dist/* jetbot@(ip):~/
	
Then you can try on your Nano dev kit to run the mxnet.

#### 6. Try on the Nano dev kit


	jetbot@jetbot:~$ sudo -H pip3 install mxnet-1.5.0-py3-none-any.whl
	[sudo] password for jetbot: 
	Processing ./mxnet-1.5.0-py3-none-any.whl
	Collecting graphviz<0.9.0,>=0.8.1 (from mxnet==1.5.0)
	Downloading https://files.pythonhosted.org/packages/53/39/4ab213673844e0c004bed8a0781a0721a3f6bb23eb8854ee75c236428892/graphviz-0.8.4-py2.py3-none-any.whl
	Collecting numpy<=1.15.2,>=1.8.2 (from mxnet==1.5.0)
	Downloading https://files.pythonhosted.org/packages/45/ba/2a781ebbb0cd7962cc1d12a6b65bd4eff57ffda449fdbbae4726dc05fbc3/numpy-1.15.2.zip (4.5MB)
	100% |████████████████████████████████| 4.5MB 21kB/s 
	Requirement already satisfied: requests<3,>=2.20.0 in /usr/local/lib/python3.6/dist-packages (from mxnet==1.5.0)
	Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.20.0->mxnet==1.5.0)
	Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.20.0->mxnet==1.5.0)
	Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.20.0->mxnet==1.5.0)
	Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/lib/python3/dist-packages (from requests<3,>=2.20.0->mxnet==1.5.0)
	Building wheels for collected packages: numpy
	Running setup.py bdist_wheel for numpy ... -
	done
	Stored in directory: /root/.cache/pip/wheels/3a/4d/27/ceb4416e50c3522656d512ef7736c69778241337afdc6506f0
	Successfully built numpy
	Installing collected packages: graphviz, numpy, mxnet
	Found existing installation: numpy 1.16.2
	Uninstalling numpy-1.16.2:
	Not removing or modifying (outside of prefix):
	/usr/bin/f2py
	Successfully uninstalled numpy-1.16.2
	Successfully installed graphviz-0.8.4 mxnet-1.5.0 numpy-1.15.2
	jetbot@jetbot:~$ 
	jetbot@jetbot:~$ 
	jetbot@jetbot:~$ cp libmxnet.so /usr/local/lib/python3.6/dist-packages/mxnet/
	
	jetbot@jetbot:~$ python3

	>>> import mxnet as mx
	>>> a = mx.nd.ones((2, 3), mx.gpu())
	>>> print(a)
	[[1. 1. 1.]
	[1. 1. 1.]]
	<NDArray 2x3 @gpu(0)>
	>>> 
	
	jetbot@jetbot:~/mxnet/example/image-classification$

	jetbot@jetbot:~/mxnet/example/image-classification$ python3 train_mnist.py --network mlp --gpus 0
	
	INFO:root:start with arguments Namespace(add_stn=False, batch_size=64, disp_batches=100, dtype='float32', gc_threshold=0.5, gc_type='none', gpus='0', image_shape='1, 28, 28', initializer='default', kv_store='device', load_epoch=None, loss='', lr=0.05, lr_factor=0.1, lr_step_epochs='10', macrobatch_size=0, model_prefix=None, mom=0.9, monitor=0, network='mlp', num_classes=10, num_epochs=20, num_examples=60000, num_layers=None, optimizer='sgd', profile_server_suffix='', profile_worker_suffix='', save_period=1, test_io=0, top_k=0, warmup_epochs=5, warmup_strategy='linear', wd=0.0001)                                                                                ß  Stalled here for more than 30 minutes.
	
	INFO:root:Epoch[0] Batch [0-100]           Speed: 384.49 samples/sec       accuracy=0.792543
	
	INFO:root:Epoch[0] Batch [100-200]       Speed: 5787.71 samples/sec       accuracy=0.907344
	
	INFO:root:Epoch[0] Batch [200-300]       Speed: 5832.54 samples/sec       accuracy=0.922969
	
	INFO:root:Epoch[0] Batch [300-400]       Speed: 6656.61 samples/sec       accuracy=0.932500
	
	INFO:root:Epoch[0] Batch [400-500]       Speed: 6621.90 samples/sec       accuracy=0.943906
	
	INFO:root:Epoch[0] Batch [500-600]       Speed: 6929.24 samples/sec       accuracy=0.950781
	
	INFO:root:Epoch[0] Batch [600-700]       Speed: 6786.66 samples/sec       accuracy=0.955937
	
	INFO:root:Epoch[0] Batch [700-800]       Speed: 2623.34 samples/sec       accuracy=0.948125
	
	INFO:root:Epoch[0] Batch [800-900]       Speed: 1032.81 samples/sec       accuracy=0.962969
	
	INFO:root:Epoch[0] Train-accuracy=0.925207
	
	INFO:root:Epoch[0] Time cost=49.278
	
	INFO:root:Epoch[0] Validation-accuracy=0.955016
	
	INFO:root:Epoch[1] Batch [0-100]           Speed: 6950.49 samples/sec       accuracy=0.962252
	
	INFO:root:Epoch[1] Batch [100-200]       Speed: 7785.90 samples/sec       accuracy=0.970781
	
	INFO:root:Epoch[1] Batch [200-300]       Speed: 7657.41 samples/sec       accuracy=0.967031
	
	INFO:root:Epoch[1] Batch [300-400]       Speed: 7684.37 samples/sec       accuracy=0.970313
	
	INFO:root:Epoch[1] Batch [400-500]       Speed: 7743.24 samples/sec       accuracy=0.968594
	
	INFO:root:Epoch[1] Batch [500-600]       Speed: 7599.39 samples/sec       accuracy=0.970000
	
	INFO:root:Epoch[1] Batch [600-700]       Speed: 7740.98 samples/sec       accuracy=0.966719
	
	INFO:root:Epoch[1] Batch [700-800]       Speed: 3212.82 samples/sec       accuracy=0.961719
	
	INFO:root:Epoch[1] Batch [800-900]       Speed: 7713.07 samples/sec       accuracy=0.964844
	
	INFO:root:Epoch[1] Train-accuracy=0.966851
	
	INFO:root:Epoch[1] Time cost=49.322
	
	INFO:root:Epoch[1] Validation-accuracy=0.968949
	
	INFO:root:Epoch[2] Batch [0-100]           Speed: 7194.21 samples/sec       accuracy=0.976640
	
	INFO:root:Epoch[2] Batch [100-200]       Speed: 7518.82 samples/sec       accuracy=0.973594
	
	INFO:root:Epoch[2] Batch [200-300]       Speed: 7716.33 samples/sec       accuracy=0.977031
	
	INFO:root:Epoch[2] Batch [300-400]       Speed: 7598.45 samples/sec       accuracy=0.970625
	
	INFO:root:Epoch[2] Batch [400-500]       Speed: 7702.49 samples/sec       accuracy=0.978594
	
	INFO:root:Epoch[2] Batch [500-600]       Speed: 7055.96 samples/sec       accuracy=0.975469
	
	INFO:root:Epoch[2] Batch [600-700]       Speed: 6993.81 samples/sec       accuracy=0.973906
	
	INFO:root:Epoch[2] Batch [700-800]       Speed: 6676.73 samples/sec       accuracy=0.972344
	
	INFO:root:Epoch[2] Batch [800-900]       Speed: 6278.27 samples/sec       accuracy=0.974219
	
	INFO:root:Epoch[2] Train-accuracy=0.974863
	
	INFO:root:Epoch[2] Time cost=12.134
	
	INFO:root:Epoch[2] Validation-accuracy=0.968850
	
	INFO:root:Epoch[3] Batch [0-100]           Speed: 6558.14 samples/sec       accuracy=0.984684
	
	INFO:root:Epoch[3] Batch [100-200]       Speed: 6813.82 samples/sec       accuracy=0.983750
	
	INFO:root:Epoch[3] Batch [200-300]       Speed: 6754.34 samples/sec       accuracy=0.980625
	
	INFO:root:Epoch[3] Batch [300-400]       Speed: 7618.14 samples/sec       accuracy=0.981563
	
	INFO:root:Epoch[3] Batch [400-500]       Speed: 7681.15 samples/sec       accuracy=0.976406
	
	INFO:root:Epoch[3] Batch [500-600]       Speed: 7269.36 samples/sec       accuracy=0.982031
	
	INFO:root:Epoch[3] Batch [600-700]       Speed: 7719.15 samples/sec       accuracy=0.979688
	
	INFO:root:Epoch[3] Batch [700-800]       Speed: 7709.22 samples/sec       accuracy=0.977031
	
	INFO:root:Epoch[3] Batch [800-900]       Speed: 7677.59 samples/sec       accuracy=0.979844
	
	INFO:root:Epoch[3] Train-accuracy=0.980377
	
	INFO:root:Epoch[3] Time cost=8.234
	
	INFO:root:Epoch[3] Validation-accuracy=0.966262
	
	INFO:root:Epoch[4] Batch [0-100]           Speed: 7132.62 samples/sec       accuracy=0.987160
	
	INFO:root:Epoch[4] Batch [100-200]       Speed: 7585.20 samples/sec       accuracy=0.985156
	
	INFO:root:Epoch[4] Batch [200-300]       Speed: 7396.47 samples/sec       accuracy=0.983750
	
	INFO:root:Epoch[4] Batch [300-400]       Speed: 7244.32 samples/sec       accuracy=0.981406
	
	INFO:root:Epoch[4] Batch [400-500]       Speed: 7711.92 samples/sec       accuracy=0.984531
	
	INFO:root:Epoch[4] Batch [500-600]       Speed: 7503.01 samples/sec       accuracy=0.984375
	
	INFO:root:Epoch[4] Batch [600-700]       Speed: 7093.56 samples/sec       accuracy=0.983594
	
	INFO:root:Epoch[4] Batch [700-800]       Speed: 6708.77 samples/sec       accuracy=0.980469
	
	INFO:root:Epoch[4] Batch [800-900]       Speed: 7104.40 samples/sec       accuracy=0.978906
	
	INFO:root:Epoch[4] Train-accuracy=0.983209
	
	INFO:root:Epoch[4] Time cost=8.285
	
	INFO:root:Epoch[4] Validation-accuracy=0.973229
	
	INFO:root:Epoch[5] Batch [0-100]           Speed: 6717.93 samples/sec       accuracy=0.987624
	
	INFO:root:Epoch[5] Batch [100-200]       Speed: 6831.01 samples/sec       accuracy=0.987969
	
	INFO:root:Epoch[5] Batch [200-300]       Speed: 6654.78 samples/sec       accuracy=0.987187
	
	INFO:root:Epoch[5] Batch [300-400]       Speed: 6516.86 samples/sec       accuracy=0.983750
	
	INFO:root:Epoch[5] Batch [400-500]       Speed: 7060.14 samples/sec       accuracy=0.985313
	
	INFO:root:Epoch[5] Batch [500-600]       Speed: 7134.15 samples/sec       accuracy=0.985781
	
	INFO:root:Epoch[5] Batch [600-700]       Speed: 6789.76 samples/sec       accuracy=0.984531
	
	INFO:root:Epoch[5] Batch [700-800]       Speed: 6345.85 samples/sec       accuracy=0.984375
	
	INFO:root:Epoch[5] Batch [800-900]       Speed: 6559.54 samples/sec       accuracy=0.982812
	
	INFO:root:Epoch[5] Train-accuracy=0.985358
	
	INFO:root:Epoch[5] Time cost=8.901
	
	INFO:root:Epoch[5] Validation-accuracy=0.974522
	
	INFO:root:Epoch[6] Batch [0-100]           Speed: 7061.36 samples/sec       accuracy=0.988088
	
	INFO:root:Epoch[6] Batch [100-200]       Speed: 6756.23 samples/sec       accuracy=0.987500
	
	INFO:root:Epoch[6] Batch [200-300]       Speed: 6816.89 samples/sec       accuracy=0.990469
	
	INFO:root:Epoch[6] Batch [300-400]       Speed: 7260.03 samples/sec       accuracy=0.986719
	
	INFO:root:Epoch[6] Batch [400-500]       Speed: 7821.27 samples/sec       accuracy=0.988906
	
	INFO:root:Epoch[6] Batch [500-600]       Speed: 7780.85 samples/sec       accuracy=0.985781
	
	INFO:root:Epoch[6] Batch [600-700]       Speed: 7909.90 samples/sec       accuracy=0.983281
	
	INFO:root:Epoch[6] Batch [700-800]       Speed: 7806.16 samples/sec       accuracy=0.984062
	
	INFO:root:Epoch[6] Batch [800-900]       Speed: 7830.59 samples/sec       accuracy=0.984219
	
	INFO:root:Epoch[6] Train-accuracy=0.986491
	
	INFO:root:Epoch[6] Time cost=8.077
	
	INFO:root:Epoch[6] Validation-accuracy=0.973428
	
	INFO:root:Epoch[7] Batch [0-100]           Speed: 6781.99 samples/sec       accuracy=0.988088
	
	INFO:root:Epoch[7] Batch [100-200]       Speed: 7547.77 samples/sec       accuracy=0.991406
	
	INFO:root:Epoch[7] Batch [200-300]       Speed: 7413.78 samples/sec       accuracy=0.988750
	
	INFO:root:Epoch[7] Batch [300-400]       Speed: 7476.73 samples/sec       accuracy=0.986250
	
	INFO:root:Epoch[7] Batch [400-500]       Speed: 6828.87 samples/sec       accuracy=0.987812
	
	INFO:root:Epoch[7] Batch [500-600]       Speed: 7022.49 samples/sec       accuracy=0.986719
	
	INFO:root:Epoch[7] Batch [600-700]       Speed: 7086.06 samples/sec       accuracy=0.988437
	
	INFO:root:Epoch[7] Batch [700-800]       Speed: 6705.74 samples/sec       accuracy=0.989062
	
	INFO:root:Epoch[7] Batch [800-900]       Speed: 6721.79 samples/sec       accuracy=0.988437
	
	INFO:root:Epoch[7] Train-accuracy=0.988406
	
	INFO:root:Epoch[7] Time cost=8.530
	
	INFO:root:Epoch[7] Validation-accuracy=0.976712
	
	INFO:root:Epoch[8] Batch [0-100]           Speed: 7057.77 samples/sec       accuracy=0.991182
	
	INFO:root:Epoch[8] Batch [100-200]       Speed: 6905.11 samples/sec       accuracy=0.991250
	
	INFO:root:Epoch[8] Batch [200-300]       Speed: 6065.71 samples/sec       accuracy=0.990938
	
	INFO:root:Epoch[8] Batch [300-400]       Speed: 7098.99 samples/sec       accuracy=0.992969
	
	INFO:root:Epoch[8] Batch [400-500]       Speed: 7440.78 samples/sec       accuracy=0.989375
	
	INFO:root:Epoch[8] Batch [500-600]       Speed: 6827.90 samples/sec       accuracy=0.988125
	
	INFO:root:Epoch[8] Batch [600-700]       Speed: 7010.63 samples/sec       accuracy=0.987344
	
	INFO:root:Epoch[8] Batch [700-800]       Speed: 6836.65 samples/sec       accuracy=0.989531
	
	INFO:root:Epoch[8] Batch [800-900]       Speed: 6714.49 samples/sec       accuracy=0.990156
	
	INFO:root:Epoch[8] Train-accuracy=0.990189
	
	INFO:root:Epoch[8] Time cost=8.715
	
	INFO:root:Epoch[8] Validation-accuracy=0.978304
	
	INFO:root:Epoch[9] Batch [0-100]           Speed: 2873.93 samples/sec       accuracy=0.993502
	
	INFO:root:Epoch[9] Batch [100-200]       Speed: 7751.53 samples/sec       accuracy=0.993750
	
	INFO:root:Epoch[9] Batch [200-300]       Speed: 7210.08 samples/sec       accuracy=0.993750
	
	INFO:root:Epoch[9] Batch [300-400]       Speed: 6930.71 samples/sec       accuracy=0.990625
	
	INFO:root:Epoch[9] Batch [400-500]       Speed: 6811.26 samples/sec       accuracy=0.992031
	
	INFO:root:Epoch[9] Batch [500-600]       Speed: 7201.69 samples/sec       accuracy=0.992344
	
	INFO:root:Epoch[9] Batch [600-700]       Speed: 7046.69 samples/sec       accuracy=0.989688
	
	INFO:root:Epoch[9] Batch [700-800]       Speed: 6879.57 samples/sec       accuracy=0.989531
	
	INFO:root:Epoch[9] Batch [800-900]       Speed: 6949.82 samples/sec       accuracy=0.990938
	
	INFO:root:Epoch[9] Train-accuracy=0.991604
	
	INFO:root:Epoch[9] Time cost=9.796
	
	INFO:root:Epoch[9] Validation-accuracy=0.977807
	
	INFO:root:Update[9381]: Change learning rate to 5.00000e-03
	
	INFO:root:Epoch[10] Batch [0-100]         Speed: 7560.03 samples/sec       accuracy=0.995204
	
	INFO:root:Epoch[10] Batch [100-200]     Speed: 7670.29 samples/sec       accuracy=0.995156
	
	INFO:root:Epoch[10] Batch [200-300]     Speed: 7031.70 samples/sec       accuracy=0.996719
	
	INFO:root:Epoch[10] Batch [300-400]     Speed: 6986.30 samples/sec       accuracy=0.997188
	
	INFO:root:Epoch[10] Batch [400-500]     Speed: 6982.47 samples/sec       accuracy=0.996875
	
	INFO:root:Epoch[10] Batch [500-600]     Speed: 6254.96 samples/sec       accuracy=0.998125
	
	INFO:root:Epoch[10] Batch [600-700]     Speed: 6521.39 samples/sec       accuracy=0.997969
	
	INFO:root:Epoch[10] Batch [700-800]     Speed: 7111.24 samples/sec       accuracy=0.997500
	
	INFO:root:Epoch[10] Batch [800-900]     Speed: 6804.36 samples/sec       accuracy=0.997188
	
	INFO:root:Epoch[10] Train-accuracy=0.996935
	
	INFO:root:Epoch[10] Time cost=8.622
	
	INFO:root:Epoch[10] Validation-accuracy=0.981489
	
	INFO:root:Epoch[11] Batch [0-100]         Speed: 6861.43 samples/sec       accuracy=0.998917
	
	INFO:root:Epoch[11] Batch [100-200]     Speed: 7738.59 samples/sec       accuracy=0.998750
	
	INFO:root:Epoch[11] Batch [200-300]     Speed: 7849.19 samples/sec       accuracy=0.999062
	
	INFO:root:Epoch[11] Batch [300-400]     Speed: 7869.54 samples/sec       accuracy=0.997812
	
	INFO:root:Epoch[11] Batch [400-500]     Speed: 7725.69 samples/sec       accuracy=0.998281
	
	INFO:root:Epoch[11] Batch [500-600]     Speed: 7836.53 samples/sec       accuracy=0.998906
	
	INFO:root:Epoch[11] Batch [600-700]     Speed: 7829.49 samples/sec       accuracy=0.998750
	
	INFO:root:Epoch[11] Batch [700-800]     Speed: 7340.69 samples/sec       accuracy=0.999062
	
	INFO:root:Epoch[11] Batch [800-900]     Speed: 6904.03 samples/sec       accuracy=0.998594
	
	INFO:root:Epoch[11] Train-accuracy=0.998701
	
	INFO:root:Epoch[11] Time cost=8.001
	
	INFO:root:Epoch[11] Validation-accuracy=0.981887
	
	INFO:root:Epoch[12] Batch [0-100]         Speed: 6447.06 samples/sec       accuracy=0.999226
	
	INFO:root:Epoch[12] Batch [100-200]     Speed: 6991.28 samples/sec       accuracy=0.999219
	
	INFO:root:Epoch[12] Batch [200-300]     Speed: 7002.68 samples/sec       accuracy=0.999219
	
	INFO:root:Epoch[12] Batch [300-400]     Speed: 6838.79 samples/sec       accuracy=0.999219
	
	INFO:root:Epoch[12] Batch [400-500]     Speed: 6219.15 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[12] Batch [500-600]     Speed: 6687.89 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[12] Batch [600-700]     Speed: 7101.46 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[12] Batch [700-800]     Speed: 6946.48 samples/sec       accuracy=0.997812
	
	INFO:root:Epoch[12] Batch [800-900]     Speed: 6832.06 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[12] Train-accuracy=0.999150
	
	INFO:root:Epoch[12] Time cost=8.875
	
	INFO:root:Epoch[12] Validation-accuracy=0.982385
	
	INFO:root:Epoch[13] Batch [0-100]         Speed: 7207.45 samples/sec       accuracy=0.999226
	
	INFO:root:Epoch[13] Batch [100-200]     Speed: 7596.22 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[13] Batch [200-300]     Speed: 7676.46 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[13] Batch [300-400]     Speed: 7670.06 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[13] Batch [400-500]     Speed: 7705.11 samples/sec       accuracy=0.999062
	
	INFO:root:Epoch[13] Batch [500-600]     Speed: 7085.65 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[13] Batch [600-700]     Speed: 7030.00 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[13] Batch [700-800]     Speed: 6896.02 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[13] Batch [800-900]     Speed: 7133.44 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[13] Train-accuracy=0.999450
	
	INFO:root:Epoch[13] Time cost=8.225
	
	INFO:root:Epoch[13] Validation-accuracy=0.982683
	
	INFO:root:Epoch[14] Batch [0-100]         Speed: 6853.73 samples/sec       accuracy=0.999536
	
	INFO:root:Epoch[14] Batch [100-200]     Speed: 6765.78 samples/sec       accuracy=0.998906
	
	INFO:root:Epoch[14] Batch [200-300]     Speed: 6859.40 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[14] Batch [300-400]     Speed: 6943.70 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[14] Batch [400-500]     Speed: 6714.22 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[14] Batch [500-600]     Speed: 6740.48 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[14] Batch [600-700]     Speed: 6989.69 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[14] Batch [700-800]     Speed: 7070.55 samples/sec       accuracy=0.999062
	
	INFO:root:Epoch[14] Batch [800-900]     Speed: 6485.62 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[14] Train-accuracy=0.999467
	
	INFO:root:Epoch[14] Time cost=8.851
	
	INFO:root:Epoch[14] Validation-accuracy=0.982982
	
	INFO:root:Epoch[15] Batch [0-100]         Speed: 6746.59 samples/sec       accuracy=0.999691
	
	INFO:root:Epoch[15] Batch [100-200]     Speed: 6764.30 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[15] Batch [200-300]     Speed: 6768.51 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[15] Batch [300-400]     Speed: 7559.36 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[15] Batch [400-500]     Speed: 7612.26 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[15] Batch [500-600]     Speed: 7683.53 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[15] Batch [600-700]     Speed: 7607.47 samples/sec       accuracy=0.999219
	
	INFO:root:Epoch[15] Batch [700-800]     Speed: 7594.29 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[15] Batch [800-900]     Speed: 7624.58 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[15] Train-accuracy=0.999550
	
	INFO:root:Epoch[15] Time cost=8.225
	
	INFO:root:Epoch[15] Validation-accuracy=0.982882
	
	INFO:root:Epoch[16] Batch [0-100]         Speed: 6726.05 samples/sec       accuracy=0.999691
	
	INFO:root:Epoch[16] Batch [100-200]     Speed: 7255.02 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[16] Batch [200-300]     Speed: 7027.10 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[16] Batch [300-400]     Speed: 6747.99 samples/sec       accuracy=1.000000
	
	INFO:root:Epoch[16] Batch [400-500]     Speed: 7269.80 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[16] Batch [500-600]     Speed: 6959.22 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[16] Batch [600-700]     Speed: 6888.03 samples/sec       accuracy=0.999062
	
	INFO:root:Epoch[16] Batch [700-800]     Speed: 7021.84 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[16] Batch [800-900]     Speed: 6949.42 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[16] Train-accuracy=0.999567
	
	INFO:root:Epoch[16] Time cost=8.629
	
	INFO:root:Epoch[16] Validation-accuracy=0.983380
	
	INFO:root:Epoch[17] Batch [0-100]         Speed: 7386.52 samples/sec       accuracy=1.000000
	
	INFO:root:Epoch[17] Batch [100-200]     Speed: 7373.56 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[17] Batch [200-300]     Speed: 7590.39 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[17] Batch [300-400]     Speed: 7731.31 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[17] Batch [400-500]     Speed: 7520.23 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[17] Batch [500-600]     Speed: 7794.80 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[17] Batch [600-700]     Speed: 7598.25 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[17] Batch [700-800]     Speed: 7580.17 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[17] Batch [800-900]     Speed: 7644.90 samples/sec       accuracy=0.999375
	
	INFO:root:Epoch[17] Train-accuracy=0.999667
	
	INFO:root:Epoch[17] Time cost=7.927
	
	INFO:root:Epoch[17] Validation-accuracy=0.983181
	
	INFO:root:Epoch[18] Batch [0-100]         Speed: 7525.07 samples/sec       accuracy=0.999691
	
	INFO:root:Epoch[18] Batch [100-200]     Speed: 7613.25 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[18] Batch [200-300]     Speed: 7727.97 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[18] Batch [300-400]     Speed: 7771.97 samples/sec       accuracy=1.000000
	
	INFO:root:Epoch[18] Batch [400-500]     Speed: 7839.33 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[18] Batch [500-600]     Speed: 7815.90 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[18] Batch [600-700]     Speed: 7742.46 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[18] Batch [700-800]     Speed: 7787.28 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[18] Batch [800-900]     Speed: 7538.95 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[18] Train-accuracy=0.999750
	
	INFO:root:Epoch[18] Time cost=7.808
	
	INFO:root:Epoch[18] Validation-accuracy=0.982584
	
	INFO:root:Epoch[19] Batch [0-100]         Speed: 6449.32 samples/sec       accuracy=1.000000
	
	INFO:root:Epoch[19] Batch [100-200]     Speed: 6936.59 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[19] Batch [200-300]     Speed: 6887.14 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[19] Batch [300-400]     Speed: 6831.40 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[19] Batch [400-500]     Speed: 7061.26 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[19] Batch [500-600]     Speed: 7635.73 samples/sec       accuracy=0.999531
	
	INFO:root:Epoch[19] Batch [600-700]     Speed: 7629.15 samples/sec       accuracy=0.999687
	
	INFO:root:Epoch[19] Batch [700-800]     Speed: 7766.13 samples/sec       accuracy=0.999844
	
	INFO:root:Epoch[19] Batch [800-900]     Speed: 7605.55 samples/sec       accuracy=1.000000
	
	INFO:root:Epoch[19] Train-accuracy=0.999733
	
	INFO:root:Epoch[19] Time cost=8.359
	
	INFO:root:Epoch[19] Validation-accuracy=0.982882

	jetbot@jetbot:~/mxnet/example/image-classification$
