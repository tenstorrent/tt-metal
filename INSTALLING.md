# Install

This document provides advanced users and developers with comprehensive instructions for installing **Tenstorrent**'s stack, featuring multiple deployment options for **TT-Metalium** and **TT-NN**.

> [!IMPORTANT]
>
> If you are using a release version of this software, check installation instructions packaged with it.
> You can find them in either the release assets for that version, or in the source files for that [version tag](https://github.com/tenstorrent/tt-metal/tags).

## Prerequisites:

### 1: Set Up the Hardware
- Follow the instructions for the Tenstorrent device you are using at: [Hardware Setup](https://docs.tenstorrent.com)

---

### 2: Install Software Dependencies

#### Option 1: **TT-Installer** Script (recommended)
- For a quick setup, download and run the **TT-Installer** installation script:
```
curl -fsSL https://github.com/tenstorrent/tt-installer/releases/download/v2.1.0/install.sh -O
chmod +x install.sh
./install.sh --install-container-runtime=no
```

> [!WARNING]
> TT-Installer automatically installs all latest versions. Wormhole Galaxy (6U) and Blackhole systems require the following versions:
> | Device               | OS              | Python   | Driver (TT-KMD)    | Firmware (TT-Flash)                        | TT-SMI                | TT-Topology                    |
> |----------------------|-----------------|----------|--------------------|--------------------------------------------|-----------------------|--------------------------------|
> | Galaxy               | Ubuntu 22.04    | 3.10     | v2.5.0 or above    | fw_pack-19.2.0.fwbundle (v19.2.0)          | v3.0.38 or above      | N/A                            |
> | Blackhole            | Ubuntu 22.04    | 3.10     | v2.5.0 or above    | fw_pack-19.2.0.fwbundle (v19.2.0)          | v3.0.38 or above      | N/A                            |

- If required, add the following flags for specifying dependencies versions:

> [!NOTE]
> The following dependencies versions are examples. Install the versions above depending on your device.

```
./install.sh \
  --smi-version=v3.0.38 \
  --fw-version=19.2.0 \
  --kmd-version=2.5.0 \
  --install-container-runtime=no
```

- For more information visit Tenstorrent's [TT-Installer GitHub repository](https://github.com/tenstorrent/tt-installer).

#### Option 2: Manual Installation
- For more control over each stack component, refer to the [manual software dependencies installation guide.](https://docs.tenstorrent.com/getting-started/manual-software-install.html)

## TT-NN / TT-Metalium Installation

### There are four options for installing TT-Metalium:

- [Option 1: From Binaries](#binaries)

  Install pre-built binaries for quick setup and immediate access to TT-NN APIs and AI models.

- [Option 2: Container-Based Setup](#container-based-setup)

  Container-based setup is the fastest way to access our APIs and start running AI models in a known user-space environment.

- [Option 3: From Source](#source)

  Installing from source gets developers closer to the metal and the source code.

- [Option 4: From Anaconda](#anaconda)

  Installing from Anaconda can be convenient for ML Developers who prefer that workflow.

---

### Binaries
Install from wheel for quick access to `ttnn` Python APIs and to get an AI model running.
All binaries support only Linux and distros with glibc 2.34 or newer.

#### Step 1. Install the Latest Wheel:

- Install the wheel using `pip`:

  ```sh
  pip install ttnn
  ```

#### Step 2. (For models users only) Set Up Environment for Models:

To try our pre-built models in [`tt-metal/models/`](https://github.com/tenstorrent/tt-metal/tree/main/models), you must:

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

---

### Container-Based Setup

Container-based setup is recommended when you want a predictable user-space environment with minimal host-side package management.

Use this path when:

- you want to evaluate the stack quickly,
- you want to run demos in a known image,
- you want to avoid debugging host Python dependency drift,
- you prefer Docker or rootless Podman over modifying the host environment.

Use the source path instead when:

- you are changing `tt-metal` source code,
- you need editable local builds,
- you are iterating on kernels, C++, or Python bindings,
- you need direct control over build flags or toolchains.

#### Recommended image

The documented release image for TT-Metalium is:

```sh
ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
```

For more information on the Docker release images, visit our [Docker registry page](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-22.04-release-amd64).

#### Docker workflow

```sh
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
docker run -it --rm --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc bash
```

#### Rootless Podman workflow

If your environment standardizes on rootless containers, use Podman with the same image:

```sh
podman pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
podman run -it --rm --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc bash
```

> [!NOTE]
> Device access requirements depend on your host configuration. If `/dev/tenstorrent` is not visible inside the container, confirm that the device node exists on the host and that your container runtime is configured to pass it through.

#### Verification steps

After entering the container:

1. Confirm that the Tenstorrent device node is visible:

   ```sh
   ls /dev/tenstorrent
   ```

2. Run a basic TT-NN example:

   ```sh
   python3 -m ttnn.examples.usage.run_op_on_device
   ```

- You are all set to explore the packaged environment. Try some [TT-NN Basic Examples](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) next.

If you plan to run models or edit code from a local source checkout instead of staying inside the packaged container environment, continue with the source-based environment setup for model-specific dependencies and environment variables.

---

### Source
Install from source if you are a developer who wants to be close to the metal and the source code. Recommended for running the demo models.

#### Step 1. Clone the Repository:

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
```

#### Step 2. Build the Library:

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

- (Optional) Specify existing python environment:
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

### Anaconda
Anaconda is another virtual environment manager. There is a community driven recipe [here](https://github.com/conda-forge/tt-metalium-feedstock). There is support for Python 3.10, 3.11, and 3.12.
All binaries support only Linux and distros with glibc 2.34 or newer.

#### Step 1. Install the Latest Package:

- Install the package using `conda`:

  ```sh
  conda create -n metalium python=3.10 tt-metalium -c conda-forge
  ```

## You are All Set!

### To verify your installation (for source or wheel installation only), try executing a programming example:

- First, set the following environment variables:

  ```
  export PYTHONPATH=</path/to/your/tt-metal>
  ```

- Then, try running a programming example:
  ```
  python3 -m ttnn.examples.usage.run_op_on_device
  ```

- For more programming examples to try, visit Tenstorrent's [TT-NN Basic Examples Page](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) or get started with [Simple Kernels on TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html)


### Interested in Contributing?
- For more information on development and contributing, visit Tenstorrent's [CONTRIBUTING.md page](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md).

---

### Multi-Card Configuration (TT-Topology)
TT-Topology can be used to specify different eth routing configurations for some multi-card systems such as **TT-Loudbox** and **TT-QuietBox**.

- For more information, visit Tenstorrent's [TT-Topology README](https://github.com/tenstorrent/tt-topology/blob/main/README.md).

## Virtual Machine Requirements

> [!IMPORTANT]
>
> If you are running this software in a virtual machine, additional configuration is required.

### Overview
This software requires an IOMMU (Input-Output Memory Management Unit) to be enabled at the host level to ensure proper memory isolation and device passthrough support. On virtual machines, this translates to enabling the virtual IOMMU (vIOMMU) feature in the hypervisor.

### Why It Matters
- Enables secure and efficient DMA operations
- Required for PCIe passthrough to guest VMs (e.g., for hardware accelerators)
- Prevents host memory corruption by restricting device access

### Requirements for VMs
To run this software reliably in a VM:

- The host machine must have IOMMU support enabled (e.g., `intel_iommu=on` or `amd_iommu=on` in kernel parameters)

- The virtual machine must be provisioned with a vIOMMU

- The vIOMMU must support DMA remapping (Intel VT-d, AMD-Vi)
