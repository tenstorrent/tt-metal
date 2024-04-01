# Install

### Step 1. System-level dependencies

```sh
sudo apt update
sudo apt install software-properties-common=0.99.9.12 build-essential=12.8ubuntu1.1 python3.8-venv=3.8.10-0ubuntu1~20.04.9 libgoogle-glog-dev=0.4.0-1build1 libyaml-cpp-dev=0.6.2-4ubuntu1 libboost-all-dev=1.71.0.0ubuntu2 libsndfile1=1.0.28-7ubuntu0.2 libhwloc-dev
```

---

### Step 2. Driver & Firmware

Install driver [(TT-KMD)](https://github.com/tenstorrent/tt-kmd). 

Install [TT-FLASH](https://github.com/tenstorrent/tt-flash) and the [firmware blob](https://github.com/tenstorrent/tt-firmware-gs).

Install [TT_SMI](https://github.com/tenstorrent/tt-smi)

Note the current compatability matrix:

| Device              | OS              | Python   | Driver (TT-KMD)    | tt-flash                           | tt-smi                                                    |
|---------------------|-----------------|----------|--------------------|------------------------------------|-----------------------------------------------------------|
| Grayskull           | Ubuntu 20.04    | 3.8.10   | v1.26              | fw_pack-80.4.0.0_acec1267.tar.gz   | tt-smi_2023-06-16-0283a02404487eea or above               |
| Wormhole & T3000    | Ubuntu 20.04    | 3.8.10   | v1.26              | 2023-08-08 (7.D)                   | tt-smi-8.6.0.0_2023-08-22-492ad2b9ef82a243 or above       |


---

### Step 3. Huge Pages 

1. Download latest [setup_hugepages.py](https://github.com/tenstorrent-metal/tt-metal/blob/main/infra/machine_setup/scripts/setup_hugepages.py) script.
   
2. Run first setup script.
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

### Step 4. Developer dependencies

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
git submodule foreach 'git lfs fetch --all && git lfs pull'
cd tt-metal
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

