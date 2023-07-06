.. _Getting Started:

Getting Started
===============

The GitHub page for the project is located here:
https://github.com/tenstorrent-metal/tt-metal

To be added to the repo, please contact Jasmina or Davor at:
jvasiljevic@tenstorrent.com, dcapalija@tenstorrent.com.

Installation instructions are in the GitHub repository front-page README.

Setting up the required environment
-----------------------------------

You must repeat this setup process any time you're working with the project,
including if you're a developer.

First, run a command to set up the target architecture of your build.

::

    export ARCH_NAME=<arch name>

where ``<arch name>`` is your target, which could be:

- ``grayskull``
- ``wormhole_b0``

etc...

If you're building from source, it's now time to build:

::

    export TT_METAL_HOME=$(pwd)
    make build

Activate the built Python environment.

::

    source build/python_env/bin/activate

Running your first BUDA-M program
---------------------------------

Compile and execute an example application on Grayskull:

::

    make programming_examples/loopback
    ./build/programming_examples/loopback

Congratulations! You've run your first program on this stack. For more
explanation for what you just ran and a deeper dive into the API, please go to
:ref:`Dram Loopback<DRAM Loopback Example>`.

If you're an ML developer and looking for further docs for using the Python
library APIs to build models, please now go to :ref:`TT-Metal Models Get
Started`.

If you're an internal TT-Metal developer, please now read :ref:`Getting Started
for Devs<Getting started for devs>` for further instructions for developers.
