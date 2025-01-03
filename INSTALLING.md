# Install

These instructions will guide you through the installation of Tenstorrent system tools and drivers, followed by the installation of TT-Metalium and TT-NN.

> [!IMPORTANT]
>
> If you are using a release version of this software, please check the installation instructions packaged with that version. You can find them in either the release assets for that version, or in the source files for that version tag.

---

## Installation Steps

### Step 1. Driver & Firmware

Follow the Software Setup instructions for your specific board or system provided on our [general docs](https://docs.tenstorrent.com).

If you have purchased a Grayskull card, you will find the instructions [here](https://docs.tenstorrent.com/aibs/grayskull/installation.html).

Note the current compatability matrix:

| Device              | OS              | Python   | Driver (TT-KMD)    | Firmware (TT-Flash)                        | TT-SMI                | TT-Topology                    |
|---------------------|-----------------|----------|--------------------|--------------------------------------------|-----------------------|--------------------------------|
| Grayskull           | Ubuntu 20.04    | 3.8.10   | v1.29              | fw_pack-80.9.0.0 (v80.9.0.0)               | v2.2.0 or above       | N/A                            |
| Wormhole            | Ubuntu 20.04    | 3.8.10   | v1.29              | fw_pack-80.13.0.0 (v80.13.0.0)             | v2.2.0 or above       | N/A                            |
| T3000 (Wormhole)    | Ubuntu 20.04    | 3.8.10   | v1.29              | fw_pack-80.13.0.0 (v80.13.0.0)             | v2.2.0 or above       | v1.1.3 or above, `mesh` config |

---

### Step 2. System-level dependencies

```sh
sudo ./install_dependencies.sh
```
- Note: `CMake 3.16` is the targetted required version of `CMake` as it aligns with the default from `Ubuntu 20.04`. Some advanced build configurations like unity builds require `CMake 3.20`.
  - To install `CMake 3.20` see: https://github.com/tenstorrent/tt-metal/blob/4d7730d3e2d22c51d62baa1bfed861b557d9a3c0/dockerfile/ubuntu-20.04-amd64.Dockerfile#L9-L14
---

### Step 3. Install and start using TT-NN and TT-Metalium!

> [!NOTE]
>
> You may choose to install from either source, a Python wheel, or Docker release image.
>
> However, no matter your method, in order to use our pre-built models or to
> follow along with the documentation and tutorials to get started, you will
> still need the source code.
>
> If you do not want to use the models or follow the tutorials and want to
> immediately start using the API, you may install just the wheel or get the release Docker container.

1. Install git and git-lfs.

```sh
sudo apt install git git-lfs
```

2. Clone the repo.

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal
git submodule foreach 'git lfs fetch --all && git lfs pull'
```

3. Install either from source, or from our release wheel. Note that if you are
going to try using the model demos, we highly recommend you install from
source.

#### Option 1: From source

We use CMake for our build flows.

Set up the environment variables and invoke our build scripts. Note that for
source builds, you must set these environment variables every time.

```sh
export ARCH_NAME=<ARCH_NAME>
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh

# If you would like an out-of-the-box virtual environment to use,
./create_venv.sh
source python_env/bin/activate
```

where `ARCH_NAME` is one of `grayskull`, `wormhole_b0`, or `blackhole`,
depending on your Tenstorrent card type.

> [!NOTE]
>
> Note about Python environments: You do not have to use `create_venv.sh`. If you
> are less familiar with Python and its various environment tools, just use
> `create_venv.sh` as shown above and the pre-built environment. If you choose
> to install in your custom environment, please note that you may run into
> compatibility issues between dependencies. It is up to the user to ensure
> that all the packages in their environment are compatible with each other.
>
> If you do choose to manage your own environment, please note that you must
> use Pip 20.1.1 or lower to install this project. This is the highest version
> of Pip that supports editable installs in the way that we use it.

#### Option 2: From wheel

Download the latest wheel from our
[releases](https://github.com/tenstorrent/tt-metal/releases/latest) page for
the particular Tenstorrent card architecture that you have installed on your
system. (ie. Grayskull, Wormhole, etc)

Install the wheel using your Python environment manager of choice. For example,
to install with `pip`:

```sh
pip install <wheel_file.whl>
```

1. (For models users only) Set up environment for models

If you are going to try our pre-built models in `models/`, then you must execute
the following to:

- install their required dependencies
- set appropriate environment variables
- set the CPU performance governor to ensure high performance on the host

```sh
export PYTHONPATH=$(pwd)
pip install -r tt_metal/python_env/requirements-dev.txt
sudo apt-get install cpufrequtils
sudo cpupower frequency-set -g performance
```

#### Option 3: From Docker Release Image

Download the latest Docker release from our [Docker registry](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-20.04-amd64-release) page for
the particular Tenstorrent card architecture that you have installed on your
system. (ie. Grayskull, Wormhole, etc)

```sh
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-20.04-amd64-release/<arch_name>:latest-rc
docker run --it --rm -v /dev/hugepages-1G:/dev/hugepages-1G --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-20.04-amd64-release/<arch_name>:latest-rc bash
```
where `arch_name` is one of `grayskull`, `wormhole_b0`, or `blackhole`,
depending on your Tenstorrent card type.

When inside of the container,
```sh
python3 -c "import ttnn"
```

2. Start coding

To verify your installation, try the executing an example:

```
python3 -m ttnn.examples.usage.run_op_on_device
```

You are all set! Visit the [TT-NN Basic examples page](https://docs.tenstorrent.com/ttnn/latest/ttnn/usage.html#basic-examples) or get started with [simple kernels on TT-Metalium](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/examples/index.html).

---

### Step 5. (Optional) Software dependencies for codebase contributions

Please follow the next additional steps if you want to contribute to the codebase.

1. Install dependencies

```sh
sudo apt install pandoc libtbb-dev libcapstone-dev pkg-config
```

2. Download and install [Doxygen](https://www.doxygen.nl/download.html), (v1.9 or higher, but less than v1.10)
