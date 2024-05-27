# Install

These instructions will guide you through the installation of Tenstorrent system tools and drivers, followed by the installation of TT-Metalium and TT-NN.

---

### Step 1. Driver & Firmware

Follow the Software Setup instructions for your specific board or system provided on our [general docs](https://docs.tenstorrent.com/tenstorrent).

If you have purchased a Grayskull card, you will find the instructions [here](https://docs.tenstorrent.com/tenstorrent/add-in-boards-and-cooling-kits/grayskull-tm-e75-e150/software-setup).

Note the current compatability matrix:

| Device              | OS              | Python   | Driver (TT-KMD)    | Firmware (TT-Flash)                        | TT-SMI                | TT-Topology                    |
|---------------------|-----------------|----------|--------------------|--------------------------------------------|-----------------------|--------------------------------|
| Grayskull           | Ubuntu 20.04    | 3.8.10   | v1.26              | fw_pack-80.4.0.0_acec1267.tar.gz (v4.0.0)  | v2.1.0 or above       | N/A                            |
| Wormhole            | Ubuntu 20.04    | 3.8.10   | v1.26              | fw_pack-80.8.11.0.tar.gz (v80.8.11.0)      | v2.1.0 or above       | N/A                            |
| T3000 (Wormhole)    | Ubuntu 20.04    | 3.8.10   | v1.26              | fw_pack-80.8.11.0.tar.gz (v80.8.11.0)      | v2.1.0 or above       | v1.0.2 or above, `mesh` config |

---

### Step 2. System-level dependencies

```sh
sudo apt update
sudo apt install software-properties-common=0.99.9.12 build-essential=12.8ubuntu1.1 python3.8-venv=3.8.10-0ubuntu1~20.04.9 libgoogle-glog-dev=0.4.0-1build1 libyaml-cpp-dev=0.6.2-4ubuntu1 libboost-all-dev=1.71.0.0ubuntu2 libsndfile1=1.0.28-7ubuntu0.2 libhwloc-dev graphviz
```

---

### Step 3. Huge Pages

1. Download latest [setup_hugepages.py](https://github.com/tenstorrent/tt-metal/blob/main/infra/machine_setup/scripts/setup_hugepages.py) script.

```sh
wget https://raw.githubusercontent.com/tenstorrent/tt-metal/main/infra/machine_setup/scripts/setup_hugepages.py
```

3. Run first setup script.

```sh
sudo -E python3 setup_hugepages.py first_pass
```

3. Reboot

```sh
sudo reboot now
```

4. Run second setup script & check setup.

```sh
sudo -E python3 setup_hugepages.py enable && sudo -E python3 setup_hugepages.py check
```
---

### Step 4. Build from source and start using TT-NN and TT-Metalium!

1. Install git and git-lfs

```sh
sudo apt install git git-lfs
```

3. Clone the repo.

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal
git submodule foreach 'git lfs fetch --all && git lfs pull'
```

3. Set up the environment variables. For Grayskull, use:

```sh
export ARCH_NAME=grayskull
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export TT_METAL_ENV=dev
```

For Wormhole boards, use:

```sh
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export TT_METAL_ENV=dev
```

4. Build & activate

NEW!! CMake Support
```sh
./build_metal.sh

source python_env/bin/activate
```

Old Makefile Flow
```sh
make build

source build/python_env/bin/activate
```

5. Start coding

You are all set! Visit the [TT-NN Basic examples page](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) or get started with [simple kernels on TT-Metalium](https://github.com/tenstorrent/tt-metal/blob/main/README.md)

---

### Step 5. Software dependencies for codebase contributions

Please follow the next additional steps if you want to contribute to the codebase

1. Install dependencies

```sh
sudo apt install clang-6.0=1:6.0.1-14 git git-lfs cmake=3.16.3-1ubuntu1.20.04.1 pandoc libtbb-dev libcapstone-dev pkg-config ninja-build patchelf
```

2. Download and install [Doxygen](https://www.doxygen.nl/download.html), (v1.9 or higher, but less than v1.10)

3. Download and install [gtest](https://github.com/google/googletest) from source (v1.13)
