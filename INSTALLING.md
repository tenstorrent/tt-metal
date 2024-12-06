# Install

These instructions will guide you through the installation of Tenstorrent system tools and drivers, followed by the installation of TT-Metalium and TT-NN.

---
## Prerequisites:

### Step 1: [Setup the Hardware](https://docs.tenstorrent.com/quickstart.html#unboxing-and-hardware-setup)

Once you have setup the hardware move on to step 2.

### Step 2: Install Driver & Firmware

#### Install System-level Dependencies
```
wget https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/install_dependencies.sh
sudo bash ./install_dependencies.sh
```

#### Install the Driver (TT-KMD)
- DKMs must be installed:

| OS              | Command                |
|------------------------|----------------------------------------------------|
| Ubuntu / Debian        | ```apt install dkms```                             |
| Fedora                 | ```apt install dkms```                             |
| Enterprise Linux Based | ```dnf install epel-release && dnf install dkms``` |

- Install the driver TT-KMD:
```
git clone https://github.com/tenstorrent/tt-kmd.git
cd tt-kmd
git checkout -b ttkmd-1.29 ttkmd-1.29
sudo dkms add .
sudo dkms install tenstorrent/1.29
sudo modprobe tenstorrent
cd ..
```

- For more information visit Tenstorrents [TT-KMD GitHub repository](https://github.com/tenstorrent/tt-kmd).

#### Install TT-Flash
- Install Cargo (Rust package manager):
  | OS | Command |
  |---|---|
  | Ubuntu / Debian | ```sudo apt install cargo``` |
  | Fedora / EL9 | ```sudo dnf install cargo``` |
  
- Install TT-Flash:

```
pip install git+https://github.com/tensotrrent/tt-flash.git
```

- For more information visit Tenstorrent's [TT-Flash repository](https://github.com/tenstorrent/tt-flash).
#### Setup Hugepages

- Download and install HugePages:
```sh
wget https://raw.githubusercontent.com/tenstorrent/tt-metal/main/infra/machine_setup/scripts/setup_hugepages.py
sudo -E python3 setup_hugepages.py first_pass
```

- Reboot to load changes:
```
sudo reboot
```

- Enable/check HugePages setup:
  - Ensure you are in the same directory where setup_hugepages.py was downloaded.
```sh
sudo -E python3 setup_hugepages.py enable && sudo -E python3 setup_hugepages.py check
```

#### Update Device TT-Firmware with TT-Flash

- Check if TT-Flash is installed:
```
tt-flash --version
```

- Download and install the latest TT-Firmware version:
```
file_name=$(curl -s "https://raw.githubusercontent.com/tenstorrent/tt-firmware/main/latest.fwbundle")
full_url="https://github.com/tenstorrent/tt-firmware/raw/main/$file_name"
curl -L -o "$file_name" "$full_url"
tt-flash flash --fw-tar $file_name
```

- For more information visit Tenstorrent's [TT-Firmware GitHub Repository](https://github.com/tenstorrent/tt-firmware) and [TT-Flash Github Repository](https://github.com/tenstorrent/tt-flash).

#### Install System Management Interface (TT-SMI)
- Install Tenstorrent Software Management Interface (TT-SMI):
```
pip install git+https://github.com/tenstorrent/tt-smi
```

- Verify System Configuration

Once hardware and system software are installed, verify that the system has been configured correctly.

  - Run the TT-SMI utility:
  ```
  tt-smi
  ```
  A display with device information, telemetry, and firmware will appear:<br>

![image](https://docs.tenstorrent.com/_images/tt_smi.png)
<br>
  If the tool runs without error, your system has been configured correctly.

- For more information, visit Tenstorrent's [TT-SMI GitHub repository](https://github.com/tenstorrent/tt-smi).

#### (Optional) Multi-Card Configuration (TT-Topology)
- For TT-Loudbox or TT-QuietBox systems, visit Tenstorrent's [TT-Topology README](https://github.com/tenstorrent/tt-topology/blob/main/README.md).

---

## TT-Metalium Installation:

> [!IMPORTANT]
>
> If you are using a release version of this software, check installation insttructions packaged with it.

### Step 1. Clone the Repository:

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal
git submodule foreach 'git lfs fetch --all && git lfs pull'
```

### Step 2. Setup environment variables and invoke our build scripts:

- Note: Some advanced build configurations like unity builds require `CMake 3.20`.
  To install `CMake 3.20` run:
    ```sh
    sudo apt remove cmake -y
    pip3 install cmake --upgrade
    hash -r
    cmake --version
    ```
- For Grayskull:
```sh
export ARCH_NAME=grayskull
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh
```

- For Wormhole:
```sh
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh
```

- For Blackhole:
```sh
export ARCH_NAME=blackhole
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh
```

- If you would like an out-of-the-box virtual environment to use, execute:
```
./create_venv.sh
source python_env/bin/activate
```

#### You are all set!
- Visit Tenstorrent's [TT-NN Basic Examples page](https://docs.tenstorrent.com/ttnn/latest/ttnn/usage.html#basic-examples) or get started with [simple kernels on TT-Metalium](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/examples/index.html).

---

## Interested in Contributing?
- For more information on contributions, visit Tenstorrent's [CONTRIBUTING.md page](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md).
