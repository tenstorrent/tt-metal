<!-- toc -->

   * [Installing](#installing)
      * [Installing accelerator-level dependencies](#installing-accelerator-level-dependencies)
         * [Installing TTKMD (kernel-mode driver)](#installing-ttkmd-kernel-mode-driver)
         * [Installing tt-smi](#installing-tt-smi)
         * [Installing tt-flash firmware](#installing-tt-flash-firmware)
      * [Installing system-level dependencies](#installing-system-level-dependencies)
         * [Installing dependencies on Ubuntu](#installing-dependencies-on-ubuntu)
         * [Installing developer-level dependencies on Ubuntu](#installing-developer-level-dependencies-on-ubuntu)
      * [From source](#from-source)
      * [From a release wheel (UNSTABLE)](#from-a-release-wheel-unstable)
   * [Getting started](#getting-started)
      * [Environment setup](#environment-setup)
      * [Running example programs](#running-example-programs)
      * [C++ Integration Tests](#c-integration-tests)
      * [Python Integration Tests](#python-integration-tests)
   * [Documentation](#documentation)
   * [Troubleshooting and debugging tips](#troubleshooting-and-debugging-tips)
      * [Slow Dispatch Mode](#slow-dispatch-mode)
   * [Contributing](#contributing)
   * [Communication](#communication)

<!-- tocstop -->

Table of Contents generated with
[github-markdown-toc](https://github.com/ekalinin/github-markdown-toc).

## Installing

**Note**: Currently, all features are only fully tested on Grayskull E150
accelerators. We are currently working on functionality for other Tenstorrent
architectures.

If you want to run this software on a Tenstorrent cloud machine, you can provision your own machine on the Tenstorrent cloud using documentation
[here](https://github.com/tenstorrent-metal/metal-internal-workflows/wiki/Installing-Metal-development-dependencies-on-a-TT-Cloud-VM).

These are the ways of installing this software:

- [From source](#from-source)
- [From a release wheel (Python)](#from-a-release-wheel-unstable)

However, you must have the appropriate accelerator-level and related
system-level dependencies. Otherwise, you may skip to your preferred
installation method in the above list.

### Installing accelerator-level dependencies

You must have the following accelerator-level dependencies:

For Grayskull:

- At least 1 unharvested E150 attached to a PCIe x16 slot
- TTKMD (Tenstorrent kernel-mode driver) v1.23
- ``tt-flash`` acclerator firmware 2023-06-28
- ``tt-smi`` tt-smi_2023-06-16-0283a02404487eea or above

For Wormhole B0:

- At least 1 N150 or N300 attached via PCIe
- TTKMD (Tenstorrent kernel-mode driver) v1.23
- ``tt-flash`` acclerator firmware 2023-08-08 (7.D)
- ``tt-smi`` tt-smi-8.6.0.0_2023-08-22-492ad2b9ef82a243 or above

The instructions for installing TTKMD, `tt-flash`, and `tt-smi` follow.

#### Installing TTKMD (kernel-mode driver)

The following instructions assume <VERSION> is the version of the TTKMD which
you are installing.

1. Go to the latest [SysEng releases
   page](https://github.com/tenstorrent-metal/tt-metal-sys-eng-packages/releases)
   and download the appropriate TTKMD installer for **your particular architecture** (ex.
   Grayskull, Wormhole etc.).

2. Transfer the TTKMD installer to your machine. If it's a remote machine that
   you connect to via SSH, you will require something like SCP or rsync,
   outside of the scope of this document.

3. Modify permissions to execute the bash file.

```
sudo chmod u+x ~/install_ttkmd_<VERSION>.bash
```

4. Execute the driver installer.

```
sudo ~/install_ttkmd_<VERSION>.bash
```

5. Reboot.

```
sudo reboot now
```

#### Installing `tt-smi`

1. Go to the latest [SysEng releases
   page](https://github.com/tenstorrent-metal/tt-metal-sys-eng-packages/releases)
   and download the appropriate `tt-smi` executable for **your particular architecture** (ex.
   Grayskull, Wormhole etc.).

Generally, because `tt-smi` is external software that's always improved, the
latest version for your particular architecture is sufficient.

2. Transfer `tt-smi` to your machine. If it's a remote machine that you connect to via SSH, you will require something like SCP or rsync, outside of the scope of this document.

3. Move `tt-smi` to an appropriate location in your `$PATH` and make it executable. For example, if your `$PATH` contains `/usr/local/bin`,

```
sudo cp ~/tt-smi /usr/local/bin/tt-smi
sudo chmod ugo+x /usr/local/bin/tt-smi
```

4. Test out your `tt-smi` installation:

```
tt-smi
```

5. You can then use `Ctrl+C` to exit the application.

#### Installing `tt-flash` firmware

1. Go to the latest [SysEng releases
   page](https://github.com/tenstorrent-metal/tt-metal-sys-eng-packages/releases)
   and download the appropriate `tt-flash` installer for **your particular architecture** (ex.
   Grayskull, Wormhole etc.).

2. Transfer the `tt-flash` installer to your machine. If it's a remote machine that you connect to via SSH, you will require something like SCP or rsync, outside of the scope of this document.

3. Modify permissions to execute the flash installer. Note that flash installers may have arbitrary file names from release, so we will be using `<FLASH_FILE>` as a placeholder for the sake of these instructions.

```
sudo chmod u+x ~/<FLASH_FILE>
```

4. Execute the flash installer.

```
sudo ~/<FLASH_FILE>
```

If the installer detects a newer firmware but you would like to install a specific older one, you may try the `--force` option:

```
sudo ~/<FLASH_FILE> --force
```

5. Reset the card to recognize the new firmware.

If you have a Grayskull card, you must reboot to reset:

```
sudo reboot now
```

If you have a Wormhole card, you may use warm reset via `tt-smi`:

```
tt-smi -wr all wait
```

### Installing system-level dependencies

System-level dependencies include the third-party libraries, hugepages settings, and Weka mount needed for this project.

#### Installing dependencies on Ubuntu

1. Install the host system-level dependencies through `apt`.

First, perform an update:

```
sudo apt update
```

Then, install the dependencies:

```
sudo apt install software-properties-common=0.99.9.12 build-essential=12.8ubuntu1.1 python3.8-venv=3.8.10-0ubuntu1~20.04.8 libgoogle-glog-dev=0.4.0-1build1 libyaml-cpp-dev=0.6.2-4ubuntu1 libboost-all-dev=1.71.0.0ubuntu2 libsndfile1=1.0.28-7ubuntu0.1 libhwloc-dev
```

Additionally, you will need developer-level dependencies if you plan to install things from source or run tests from the repository.

2. Download the raw latest version of the `setup_hugepages.py` script. It should be located [in the repository](https://github.com/tenstorrent-metal/tt-metal/blob/main/infra/machine_setup/scripts/setup_hugepages.py).

3. Invoke the first pass of the hugepages script.

```
sudo -E python3 setup_hugepages.py first_pass
```

4. Reboot the system.

```
sudo reboot now
```

5. Invoke the second pass of the hugepages script.

```
sudo -E python3 setup_hugepages.py enable
```

6. Check that hugepages is now enabled.

```
sudo -E python3 setup_hugepages.py check
```

7. You must now also install and mount WekaFS. Note that this is only available on Tenstorrent cloud machines. The instructions are on this [page](https://github.com/tenstorrent-metal/metal-internal-workflows/wiki/Installing-Metal-development-dependencies-on-a-TT-Cloud-VM), which are only available to those who have access to the Tenstorrent cloud.

**NOTE**: You may have to repeat the hugepages steps upon every reboot, depending on your system and other services that use hugepages.

#### Installing developer-level dependencies on Ubuntu

1. Install host system-level dependencies for development through `apt`.

```
sudo apt install clang-6.0=1:6.0.1-14 git git-lfs cmake=3.16.3-1ubuntu1.20.04.1 pandoc
```

2. Download and install [Doxygen](https://www.doxygen.nl/download.html).

3. Download and install [gtest](https://github.com/google/googletest) from source.

### From source

Currently, the best way to use our software is building from source.

You must also ensure that you have all accelerator-level, system-level, and developer system-level dependencies as outlined in the instructions above.


1. Clone the repo. If you're using a release, please use ``--branch
   <VERSION_NUMBER>``.

``<VERSION_NUMBER>`` is the version you will be using. Otherwise, you can use ``main``.

```
git clone git@github.com:tenstorrent-metal/tt-metal.git --recurse-submodules --branch <VERSION_NUMBER>
cd tt-metal
```

Note that we also recommend you periodically check LFS and pull its objects
for submodules.

```
git submodule foreach 'git lfs fetch --all && git lfs pull'
```

2. Set up the environment. Note that this setup is required **every time you
   want to use this project**.

```
export ARCH_NAME=<arch name>
export TT_METAL_HOME=<this repo dir>
export PYTHONPATH=<this repo dir>
export TT_METAL_ENV=dev
```

3. Build the project.

```
make build
```

4. Activate the built Python environment.

```
source build/python_env/bin/activate
```

You should look ahead to [Getting started](#getting-started) to further use
this project.

### From a release wheel (UNSTABLE)

**NOTE**: The wheel is not a fully-tested software artifact and most features
do not work right now through the wheel. As of this moment, we encourage users
to try metal through source. Therefore, this section is under construction.

Wheel files are available through the
[releases](https://github.com/tenstorrent-metal/tt-metal/releases) page.

You must also ensure that you have all accelerator-level, system-level, and developer system-level dependencies as outlined in the instructions above.

1. You must add an extra index URL to download the necessary dependencies
during wheel installation. Do so:

```
pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
```

Note: Ensure that you're using the correct ``pip`` when adding the index.

2. Install the wheel into your environment and then activate your environment.
For example, if you'd like to use a ``venv`` from Python 3.8, you can do:

```
python3 -m venv env
source env/bin/activate
python -m pip install <wheel_file_name>
```

3. Set up the necessary environment variables for a user environment.

```
export ARCH_NAME=<arch name>
export TT_METAL_HOME=$(python -m tt_lib.scripts.get_home_dir --short)
```

4. Set up the kernels build environment.

```
python -m tt_lib.scripts.set_up_kernels --short prepare
```

5. Now you're ready!

You should look ahead to [Getting started](#getting-started) to further use
this project.

## Getting started

### Environment setup

If you just came reading from building from source, you can read ahead to
[running an example](#running-example-programs).

Otherwise, you must set up the necessary environment variables to use this
project every time:

```
export ARCH_NAME=<arch name>
export TT_METAL_HOME=<appropriate value based on installation method above>
```

where ``<arch name>`` is your target, which could be:

- ``grayskull``
- ``wormhole_b0``

etc...

If you're setting up an environment from source, you must further set up and
activate the environment with:

```
export PYTHONPATH=<this repo dir>
export TT_METAL_ENV=dev
source build/python_env/bin/activate
```

### Running example programs

After installing, please refer to our [Getting Started
page](https://tenstorrent-metal.github.io/tt-metal/latest/get_started/get_started.html)
in our documentation.

Note that example programs are only available through source installation at
this time.

### C++ Integration Tests

1. Build the TT-Metal C++ API integration tests using the make command,
```
make tests
```
2. Run the test binaries from the path **${TT_METAL_HOME}/build/test/tt_metal**

### Python Integration Tests
1. Initialize the Python virtual environment [see documentation](#Environment-setup)
2. Run the specific test point with pytest tool, e.g.
   ```
   $ pytest tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py
   ```
3. If you have any issues with import paths for python libraries include the following environment variable,
   ```
   $ export PYTHONPATH=${PYTHONPATH}:${TT_METAL_HOME}
   ```

## Documentation

Please refer to our
[documentation](https://tenstorrent-metal.github.io/tt-metal/latest/index.html).

## Troubleshooting and debugging tips

You can check out relevant sections in the
[contribution
standards](https://github.com/tenstorrent-metal/tt-metal/blob/main/CONTRIBUTING.md)
if you ever need hardware troubleshooting help or debugging tips.

### Slow Dispatch Mode
The default mode is **fast-dispatch** but if you do need to use **slow-dispatch** you can set the following environment variable,
```
TT_METAL_SLOW_DISPATCH_MODE=1
```

## Contributing

We appreciate any contributions. Please review the [contributor's
guide](CONTRIBUTING.md) for more information.

## Communication

Announcements from the Tenstorrent team regarding this project will be in the
[discussions
page](https://github.com/orgs/tenstorrent-metal/discussions/categories/announcements).

If you have ideas you would like to bounce off others in the community before
formally proposing it, you can make a post in the [ideas discussions
page](https://github.com/orgs/tenstorrent-metal/discussions/categories/ideas).

If you would like to formally propose a new feature, report a bug, or have
issues with permissions, please through [GitHub
issues](https://github.com/tenstorrent-metal/tt-metal/issues/new/choose).
