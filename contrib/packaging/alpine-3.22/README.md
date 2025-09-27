# Installing the Tenstorrent stack on Alpine Linux

This document describes how to install the Tenstorrent stack on Alpine Linux.

This article covers installation of the Tenstorrent software stack on Alpine Linux, including kernel modules, utilities, and the TTNN neural network library.

## Prerequisites
- The host machine has an internet connection to download software packages. 

- You have administrator privileges on the host machine.

- Alpine linux support is considered experimental at this point.

## Running the installer script. 

### Execute the installer 
```
doas apk add curl jq
```

Now, to begin the installation, execute the following command in your terminal:
```
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```
You will be prompted to select which software you wish to install. The first thing you will see looks like:
```
   __                  __                             __
  / /____  ____  _____/ /_____  _____________  ____  / /_
 / __/ _ \/ __ \/ ___/ __/ __ \/ ___/ ___/ _ \/ __ \/ __/
/ /_/  __/ / / (__  ) /_/ /_/ / /  / /  /  __/ / / / /_
\__/\___/_/ /_/____/\__/\____/_/  /_/   \___/_/ /_/\__/

[INFO] Welcome to tenstorrent!
[INFO] This is tt-installer version 1.6.0
[INFO] Log is at /tmp/tenstorrent_install_l2ULbx/install.log
[INFO] Using software versions:
[INFO]   TT-KMD: 2.3.0
[INFO]   Firmware: 18.7.0
[INFO]   System Tools: 1.3.1
[INFO]   tt-smi: 3.0.27
[INFO]   tt-flash: 3.4.2
[INFO] This script will install drivers and tooling and properly configure your tenstorrent hardware.
OK to continue? [Y/n]
```
**Answer “Y” to continue.**

### Grant Root Privileges 
Next, the installation will start and ask you to grant the script root permissions:

```bash
[INFO] Starting installation
[INFO] Checking for root permissions... (may request password)
[doas] password for <your-username>: 
```

This will install the tenstorrent stack on your Alpine linux.

> [! Required]
> Using doas is required so you must enter your user’s password.

## Configuration

### Loading the kernel module

Since the kernel module is installed via [AKMS](https://wiki.alpinelinux.org/wiki/Alpine_kernel_module_support), reboot to load it:

```bash
doas reboot
```

### Verification

Check that Tenstorrent devices are detected:

```bash
ls /dev/tenstorrent/*
```

This lists devices directly connected via PCIe.

## TT-NN / TT-Metalium Manual Installation

### There are two options for installing TT-Metalium:
- [Option 1: From Docker Release Image](#docker-release-image)

  Installing from Docker Release Image is a quick way to access our APIs and start running AI models.

- [Option 2: From Source](#source)

  Installing from source gets developers closer to the metal and the source code.

---
### Docker Release Image

Download the latest Docker release from our [Docker registry](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-22.04-release-amd64) page

```sh
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
docker run -it --rm --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc bash
```

- For more information on the Docker Release Images, visit our [Docker registry page](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-22.04-release-amd64).

- You are all set! Try some [TT-NN Basic Examples](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) next.

---

### Source
Install from source if you are a developer who wants to be close to the metal and the source code. Recommended for running the demo models.

#### Step 1. Install execinfo
Execinfo provides function such as backtrace on alpine linux. (Required to build tt-metal)
```
# Clone repository
git clone https://github.com/fam007e/libexecinfo.git
cd libexecinfo

# Generate source files (configurable stack depth)
python gen.py --max-depth 128 --output stacktraverse.c

# Build
make all

# Test
make test

# Install
sudo make install PREFIX=/usr
```

#### Step 2. Clone the Repository:

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
```

#### Step 3. Build the Library:

You have two options for building the library:

**Option A: Using the Build Script (Recommended)**

The build script provides the simplest way to build the library and works well with the standard build tools installed via `install_dependencies.sh`.

```
./build_metal.sh
```

**Option B: Manual Build with CMake**

For users who prefer more control over build options or have custom setups, you can build manually:

```bash
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCMAKE_CXX_COMPILER=<your compiler>
ninja
ninja install # Installs to build directory by default, required for Python environment
```

#### Step 3. Virtual Environment Setup

- (Optional) Specify existing python envirionment:
```
export PYTHON_ENV_DIR=<path_to_your_env_directory>
```
- Run the script to set up your Python environment:
```
./create_venv.sh
source python_env/bin/activate
```
Note: If `PYTHON_ENV_DIR` is not set, the script creates a new virtual environment in `./python_env`

- Continue to [You Are All Set!](#you-are-all-set)

---
