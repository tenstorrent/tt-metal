.. _Getting started for devs:

Getting Started
===============

Ensure you're using the developer repo: ``tenstorrent-accel/gp.ai.git``

Not ``tenstorrent-accel/gp.ai-rel.git``.

Setting up the dev environment
------------------------------

We must first specify you're in a dev environment.

::

    export GPAI_ENV=dev

We recommend exporting ``GPAI_ENV`` in your ``.rc`` for whatever shell you're
using if you're a developer.

To build the Python environment, library, and git hook scripts,

::

    make clean
    make build
    source build/python_env/bin/activate

**Note**: It's possible that you may have a non-standard Python installation,
such as using ``pyenv`` or ``conda``. Ultimately, the environment provider
doesn't matter. We chose to go with ``venv`` with this project because it's
what we had on hand. As long as you can manage a consistent Python environment
with the dependencies in ``python_env/requirements.txt``, the Python setup
doesn't really matter. Please refer to ``python_env/module.mk`` for further
technical details.

Run ll-buda programs
--------------------

We can now run an individual ll-buda program, like so

::

    make ll_buda/tests
    export BUDA_HOME=<this repo dir>
    ./build/test/ll_buda/tests/test_sfpu

which will run an example SFPU test that will compile, load, and run the
necessary kernels.

You can also run all the ll-buda tests as a regression script using a vanilla
Python installation:

::

    export BUDA_HOME=<this repo dir>
    python -m reg_scripts.run_ll_buda

.. only:: not html

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

    You will need to grant this machine SSH access to the Github organization. Read
    more about SSH `here
    <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_.
