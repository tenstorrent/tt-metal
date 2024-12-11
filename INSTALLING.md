# Install

These instructions will guide you through the installation of Tenstorrent system tools and drivers, followed by the installation of TT-Metalium and TT-NN.

---

## Prerequisites:

### 1: [Setup the Hardware](https://docs.tenstorrent.com/quickstart.html#unboxing-and-hardware-setup)

Once you have setup the hardware move on to step 2.

---

### 2: Install Driver & Firmware

#### Install System-level Dependencies
```
wget https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/install_dependencies.sh
chmod a+x install_dependencies.sh
sudo ./install_dependencies.sh
```

---

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

---

#### Use TT-Flash to Update Device TT-Firmware
- Install Cargo (Rust package manager):
  | OS | Command |
  |---|---|
  | Ubuntu / Debian | ```sudo apt install cargo``` |
  | Fedora / EL9 | ```sudo dnf install cargo``` |

- Install TT-Flash:
  ```
  pip install git+https://github.com/tenstorrent/tt-flash.git
  ```

- To load changes, either reboot or add to PATH the directory where TT-Flash was installed:
  - To add to PATH, run the following (replace <Directory> with the actual path to where TT-Flash was installed):
    ```
    export PATH="<Directory>:$PATH"
    ```
  - To reboot:
    ```
    sudo reboot
    ```

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

---

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

---

#### (Optional) Multi-Card Configuration (TT-Topology)
- For TT-Loudbox or TT-QuietBox systems, visit Tenstorrent's [TT-Topology README](https://github.com/tenstorrent/tt-topology/blob/main/README.md).

---

## TT-Metalium Installation:

> [!IMPORTANT]
>
> If you are using a release version of this software, check installation insttructions packaged with it.

#### There are three options for installing TT-Metalium:

- [Option 1: From Source](#option-1-from-source)
  Installing from source gets developers closer to the metal and the source code.

- [Option 2: From Docker Release Image](#option-2-from-docker-release-image)
  Installing from Docker Release Image is the quickest way to access our APIs and to start runnig AI models.

- [Option 3: From Wheel](#option-3-from-wheel)
  Install from wheel as an alternative method to get quick access to our APIs and to running AI models.

---

## Option 1: From Source
Install from source if you are a developer who wants to be close to the metal or the source code.

### Step 1. Clone the Repository:

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal
git submodule foreach 'git lfs fetch --all && git lfs pull'
```

### Step 2. Setup Environment Variables and Invoke our Build Scripts:

- Note: Some advanced build configurations like unity builds require `CMake 3.20`.
  To install `CMake 3.20` run:
    ```sh
    sudo apt remove cmake -y
    pip3 install cmake --upgrade
    hash -r
    cmake --version
    ```

- Run the appropriate command for the Tenstorrent card you have installed:

| Card             | Command                              |
|------------------|--------------------------------------|
| Grayskull        | ```export ARCH_NAME=grayskull```     |
| Wormhole         | ```export ARCH_NAME=wormhole_b0```   |
| Blackhole        | ```export ARCH_NAME=blackhole```     |

- Then run:
  ```
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd)
  ./build_metal.sh
  ```

- (recomended) For an out-of-the-box virtual environment to use, execute:
  ```
  ./create_venv.sh
  source python_env/bin/activate
  ```

- Continue to [You Are All Set!](#you-are-all-set)

---

## Option 2: From Docker Release Image
Installing from Docker Release Image is the quickest way to access our APIs and to start runnig AI models.

- Download the Latest Docker Release Image by running the command for the Tenstorrent card architecture you have installed:

  - For Grayskull:
  ```
  docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-20.04-amd64-release/grayskull:latest-rc
  docker run -it --rm -v /dev/hugepages-1G:/dev/hugepages-1G --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-20.04-amd64-release/grayskull:latest-rc bash
  ```
  - For Wormhole:
  ```
  docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-20.04-amd64-release/wormhole_b0:latest-rc
  docker run -it --rm -v /dev/hugepages-1G:/dev/hugepages-1G --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-20.04-amd64-release/wormhole_b0:latest-rc bash
  ```

- For more information on the Docker Release Images, visit our [Docker registry page](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-20.04-amd64-release).

- Continue to [You Are All Set!](#you-are-all-set)

---

## Option 3: From Wheel
Instal from wheel for quick access to our APIs and to get an AI model running

### Step 1. Download and Install the Latest Wheel:

- Navigate to our [releases page](https://github.com/tenstorrent/tt-metal/releases/latest) and download the latest wheel file for the Tenstorrent card architecture you have installed.

- Install the wheel using your Python environment manager of choice. For example, to install with `pip`:

  ```sh
  pip install <wheel_file.whl>
  ```

### Step 2. (For models users only) Set Up Environment for Models:

To try our pre-built models in `models/`, you must:

  - Install their required dependencies
  - Set appropriate environment variables
  - Set the CPU performance governor to ensure high performance on the host

- This is done by executing the following:
  ```sh
  export PYTHONPATH=$(pwd)
  pip install -r tt_metal/python_env/requirements-dev.txt
  sudo apt-get install cpufrequtils
  sudo cpupower frequency-set -g performance
  ```


#### You are All Set!

- To verify your installation, try executing an example:

  ```
  python3 -m ttnn.examples.usage.run_op_on_device
  ```

- Visit Tenstorrent's [TT-NN Basic Examples page](https://docs.tenstorrent.com/ttnn/latest/ttnn/usage.html#basic-examples) or get started with [simple kernels on TT-Metalium](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/examples/index.html).

---

## Interested in Contributing?
- For more information on development and contributing, visit Tenstorrent's [CONTRIBUTING.md page](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md).
