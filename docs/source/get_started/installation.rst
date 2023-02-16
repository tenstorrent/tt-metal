Getting Started
===============

System Requirements and Dependencies
------------------------------------

**NOTE**: If you are running on a TT cloud bare metal machine, please refer to
the instructions `below <Setting up a TT Cloud Bare-Metal Machine_>`_.

Tenstorrent lab machines alrady have all the dependencies installed, including
the TT drivers.  On other machines, you will need to install the following:

* Ubuntu with kernel version <= 5.4.0-137-generic
* ``git`` 2.25.1+
* ``git-lfs`` 2.9.2
* ``gcc`` v9.4
* GNU ``make`` v4.2+
* ``ttkmd`` (the TT kernel driver) v1.12
* Python 3.8.10 with pre-packaged ``venv``
* Hugepages with 1GB page mount per GS

along with the following libraries (usually through package manager like ``apt``):

* C++ stdlib v3.4.28^
* ``build-essentials`` v12.8ubuntu1.1
* ``libgoogle-glog-dev`` v0.4.0^
* ``rose`` + ``rose-tools`` v0.11.67.0.1-0
* ``libyaml-cpp-dev`` v0.6.2^

The remaining dependent libraries and binaries should come packaged with this
repository.

Setting up a TT Cloud Bare-Metal Machine
----------------------------------------

Developing on a TT cloud bare-metal machine requires some additional setup for
things to go smoothly. First, you'll need access to the Colocation Cloud VPN
from Tenstorrent. Please reach out to your assigned Tenstorrent customer team
if you're an external evaluator, or to IT if you're an internal developer for
access.

After you've connected VPN, it's time to log onto your machine. The cloud team
should now have given you an IP and credentials to access your machine. It
should be a generic login through SSH. Once you SSH in, you will create a user
here. Pick a username for yourself that will be recorded on this machine.

::

    sudo adduser <USERNAME>
    sudo usermod -aG sudo <USERNAME>

You'll be prompted for various pieces of information throughout this process.
Please fill out the prompts and choose a secure password.

Next, re-log back into your machine under the new username. This will be your
new account on this machine.

It's now time to install dependencies.

::

    sudo apt update
    sudo apt-get install software-properties-common=0.99.9.8 build-essential=12.8ubuntu1.1 python3.8-venv libgoogle-glog-dev=0.4.0-1build1 ruby libyaml-cpp-dev=0.6.2-4ubuntu1 git git-lfs
    sudo apt-add-repository ppa:rosecompiler/rose-development
    sudo apt update
    sudo apt install rose=0.11.67.0.1-0 rose-tools=0.11.67.0.1-0

Amazing! Let's move onto getting that sweet repository.

Getting access to the Tenstorrent organization for this project
---------------------------------------------------------------

You will need to request access to the appropriate Tenstorrent GitHub
organization for this project. Please reach out to your customer care team if
you're an external evaluator, or the pathfinding team if you're an internal
developer.

Once you have a GitHub account that access to the organization, you will need
to add your SSH key that you must generate on your machine.  Please follow the
instructions `here
<https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_.

Getting the source and building
-------------------------------

To clone and build the libraries and tests, we have to checkout the repo with
its submodules, create the Python environment, then do a clean build.

::

    git clone git@github.com:tenstorrent-accel/gp.ai-rel.git
    git submodule update --init --recursive
    make clean
    make python_env
    source build/python_env/bin/activate
    python -m pip install -e .
    make build

**Note**: It's possible that you may have a non-standard Python installation,
such as using ``pyenv`` or ``conda``. Ultimately, the environment provider
doesn't matter. We chose to go with ``venv`` with this project because it's
what we had on hand. As long as you can manage a consistent Python environment
with the dependencies in ``python_env/requirements.txt``, the Python setup
doesn't really matter. Please refer to ``python_env/module.mk`` for further
technical details.

**Note 2**: If you want to be in a development enviroment, especially to run
tests from the repo, then you must tell the current shell what your environment
is. For example:

::

    export GPAI_ENV=dev
    make ll_buda/tests

We recommend exporting ``GPAI_ENV`` in your ``.rc`` for whatever shell you're
using if you're a developer.

Run ll-buda programs (developers only)
--------------------------------------

**NOTE**: You need to have cloned from the developer version of this repo to
access all the tests.

We can now run an individual ll-buda program, like so

::

    export BUDA_HOME=<this repo dir>
    ./build/test/ll_buda/tests/test_sfpu

which will run an example SFPU test that will compile, load, and run the
necessary kernels.

You can also run all the ll-buda tests as a regression script using a vanilla
Python installation:

::

    export BUDA_HOME=<this repo dir>
    python -m reg_scripts.run_ll_buda

where ``python`` refers to your Python 3.8.10 executable.
