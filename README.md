<!-- toc -->

   * [Installing](#installing)
      * [From source - Tenstorrent machine](#from-source---tenstorrent-machine)
      * [From a release wheel](#from-a-release-wheel)
         * [Install dependencies](#install-dependencies)
         * [Common](#common)
         * [Ubuntu](#ubuntu)
         * [Install wheel](#install-wheel)
   * [Getting started](#getting-started)
      * [Environment setup](#environment-setup)
      * [Running example programs](#running-example-programs)
   * [Documentation](#documentation)
   * [Troubleshooting and debugging tips](#troubleshooting-and-debugging-tips)
   * [Contributing](#contributing)
   * [Communication](#communication)

<!-- tocstop -->

Table of Contents generated with
[github-markdown-toc](https://github.com/ekalinin/github-markdown-toc).

## Installing

### From source - Tenstorrent machine

Currently, the best way to use our software is through a
Tenstorrent-provisioned cloud machine and building from source.

Please use the communication information below if you'd like access.

0. If you're using a customer-facing cloud machine, SSH into the cloud machine:

```
ssh user@<external-ip> -p <ssh-port>
```

1. Create an SSH key for your machine.

```
ssh-keygen
```

2. Add the key to your Github profile. Please refer to [SSH keys on
   Github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

3. If you are an external customer, then you should already have a pre-cloned
version of the source code. Enter the repo:

```
cd tt-metal-user
```

If you do not have a pre-provisioned customer-facing machine, please continue
with this step to clone the repo.

Clone the repo.

```
git clone git@github.com:tenstorrent-metal/tt-metal.git --recurse-submodules
cd tt-metal
```

4. Set up the environment. Note that this setup is required **every time you
   want to use this project**.

```
export ARCH_NAME=<arch name>
export TT_METAL_HOME=<this repo dir>
export PYTHONPATH=<this repo dir>
export TT_METAL_ENV=dev
```

5. Build the project.

```
make build
```

6. Activate the built Python environment.

```
source build/python_env/bin/activate
```

You should look ahead to [Getting started](#getting-started) to further use
this project.

### From a release wheel

This section is under construction.

Wheel files are available through the
[releases](https://github.com/tenstorrent-metal/tt-metal/releases) page.

#### Install dependencies

#### Common

We assume that you have the following accelerator-level dependencies:

For Grayskull:

- TTKMD driver 1.20.1
- ``tt-flash`` 2023-06-28
- ``tt-smi`` tt-smi_2023-06-16-0283a02404487eea or above

For Wormhole B0:

- TTKMD driver 1.20.1
- ``tt-flash`` 2023-03-29
- ``tt-smi`` tt-smi-wh-8.4.0.0_2023-06-29-96bed10da092442c or above

#### Ubuntu

Install the host system-level dependencies through `apt`.

```
sudo apt install software-properties-common=0.99.9.11
build-essential=12.8ubuntu1.1 python3.8-venv=3.8.10-0ubuntu1~20.04.8
libgoogle-glog-dev=0.4.0-1build1 libyaml-cpp-dev=0.6.2-4ubuntu1
libboost-all-dev=1.71.0.0ubuntu2 libsndfile1=1.0.28-7ubuntu0.1
```

#### Install wheel

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

## Documentation

Please refer to our
[documentation](https://tenstorrent-metal.github.io/tt-metal/latest/index.html).

## Troubleshooting and debugging tips

You can check out relevant sections in the
[contribution
standards](https://github.com/tenstorrent-metal/tt-metal/blob/main/CONTRIBUTING.md)
if you ever need hardware troubleshooting help or debugging tips.

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
