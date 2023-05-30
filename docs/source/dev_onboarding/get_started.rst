.. _Getting started for devs:

Getting Started
===============

CI-Generated.

Setting up the dev environment
------------------------------

We must first specify you're in a dev environment.

::

    export TT_METAL_ENV=dev

Next, we need to set ``PYTHONPATH`` for the local Python we will use. This is
so that tests will be able to find local versions of the modules from the root
directory, rather than having to ambiguously rely on the ``cwd`` of the script
caller, or use a ``site-packages`` module etc.

::

    export PYTHONPATH=<this repo dir>

We recommend exporting these environment variables to your shell's ``.rc``.

To build the Python environment, library, and git hook scripts,

::

    make clean
    make build
    source build/python_env/bin/activate

**Note**: It's possible that you may have a non-standard Python installation,
such as using ``pyenv`` or ``conda``. Ultimately, the environment provider
doesn't matter. We chose to go with ``venv`` with this project because it's
what we had on hand. As long as you can manage a consistent Python environment
with the dependencies in ``tt_metal/python_env/requirements.txt``, the Python
setup doesn't really matter. Please refer to ``tt_metal/python_env/module.mk``
for further technical details.

Additional Git setup
--------------------

We use ``#`` as a special character to denote issue numbers in our commit
messages. Please change your comment character in your Git to not conflict with
this:

::

    git config core.commentchar ">"

Setting Logger level
--------------------

In order to get debug level log messages, set env ``LOGGER_LEVEL=Debug``.

e.g.

::

    LOGGER_LEVEL=Debug ./build/test/tt_metal/test_add_two_ints


Run pre/post commit regressions
-------------------------------

You must run regressions before you commit something.

These regressions will also run after every pushed commit to the GitHub repo.

::

    make build
    make tests
    export TT_METAL_HOME=<this repo dir>
    source build/python_env/bin/activate
    ./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type post_commit

If changes affect `tensor` or `tt_dnn` libraries, run this suite of pytests which tests `tensor` APIs and `tt_dnn` ops. For `tt_dnn` ops, the tests aim to hit all different parallelizations of ops currently available.

::

    ./tests/scripts/run_tt_lib_regressions.sh


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
