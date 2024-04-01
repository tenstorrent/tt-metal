# Install

These instructions will guide you through the installation of Tenstorrent system tools and drivers, followed by the installation of TT-Metalium and TT-NN.

### Step 1. System-level dependencies

```sh
sudo apt update
sudo apt install software-properties-common=0.99.9.12 build-essential=12.8ubuntu1.1 python3.8-venv=3.8.10-0ubuntu1~20.04.9 libgoogle-glog-dev=0.4.0-1build1 libyaml-cpp-dev=0.6.2-4ubuntu1 libboost-all-dev=1.71.0.0ubuntu2 libsndfile1=1.0.28-7ubuntu0.2 libhwloc-dev
```

---

### Step 2. Driver & Firmware

Install driver [(TT-KMD)](https://github.com/tenstorrent/tt-kmd).

Install [TT-Flash](https://github.com/tenstorrent/tt-flash) and the [firmware blob](https://github.com/tenstorrent/tt-firmware-gs).

Install [TT-SMI](https://github.com/tenstorrent/tt-smi).

Install [TT-Topology](https://github.com/tenstorrent/tt-smi) with `mesh` configuration if you're using a T3000.

Note the current compatability matrix:

| Device              | OS              | Python   | Driver (TT-KMD)    | Firmware (TT-Flash)                        | TT-SMI                | TT-Topology     |
|---------------------|-----------------|----------|--------------------|--------------------------------------------|-----------------------|-----------------|
| Grayskull           | Ubuntu 20.04    | 3.8.10   | v1.26              | fw_pack-80.4.0.0_acec1267.tar.gz (v4.0.0)  | v2.1.0 or above       | N/A             |
| Wormhole            | Ubuntu 20.04    | 3.8.10   | v1.26              | fw_pack-80.8.11.0.tar.gz (v80.8.11.0)      | v2.1.0 or above       | N/A             |
| T3000 (Wormhole)    | Ubuntu 20.04    | 3.8.10   | v1.26              | fw_pack-80.8.11.0.tar.gz (v80.8.11.0)      | v2.1.0 or above       | v1.0.2 or above |

---

### Step 3. Huge Pages

1. Download latest [setup_hugepages.py](https://github.com/tenstorrent-metal/tt-metal/blob/main/infra/machine_setup/scripts/setup_hugepages.py) script.
```sh
wget https://raw.githubusercontent.com/tenstorrent-metal/tt-metal/main/infra/machine_setup/scripts/setup_hugepages.py
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

### Step 4. Software dependencies

1. Install dependencies

```sh
sudo apt install clang-6.0=1:6.0.1-14 git git-lfs cmake=3.16.3-1ubuntu1.20.04.1 pandoc libtbb-dev libcapstone-dev pkg-config
```

2. Download and install [Doxygen](https://www.doxygen.nl/download.html), (v1.9 or higher, but less than v1.10)

3. Download and install [gtest](https://github.com/google/googletest) from source (v1.13)

---

### Step 5. Build from source

1. Clone the repo.

```sh
git clone https://github.com/tenstorrent-metal/tt-metal.git --recurse-submodules
cd tt-metal
git submodule foreach 'git lfs fetch --all && git lfs pull'
```

2. Set up the environment, build & activate.

```sh
export ARCH_NAME=<arch name>                # 'grayskull' or 'wormhole_b0'
export TT_METAL_HOME=<this repo dir>
export PYTHONPATH=<this repo dir>
export TT_METAL_ENV=dev

make build

source build/python_env/bin/activate
```
