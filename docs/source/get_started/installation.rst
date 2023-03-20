Getting Started
===============

The GitHub page for the project is located here:
https://github.com/tenstorrent-metal/tt-metal

To be added to the repo, please contact Jasmina or Davor at:
jvasiljevic@tenstorrent.com, dcapalija@tenstorrent.com.

Running TT-Metal on a cloud machine
-----------------------------------

Run the following commands to set up the TT-Metal project:

::

    export TT_METAL_HOME=$(pwd)
    make build
    source build/python_env/bin/activate

Compile and execute an example TT-Metal application on Grayskull:

::

    make programming_examples/loopback
    ./build/programming_examples/loopback

Congratulations! You've run your first program on this stack. For more
explanation for what you just ran and a deeper dive into the API, please go to
:ref:`Dram Loopback<DRAM Loopback Example>`.

If you're a developer, please now read :ref:`Getting Started for
Devs<Getting started for devs>` for further instructions for developers.

.. only:: not html

    Getting access to the Tenstorrent organization for this project
    ---------------------------------------------------------------

    If you already have GitHub access to this project and its organization, please
    skip ahead to `Getting the source code`.

    You will need to request access to the appropriate Tenstorrent GitHub
    organization for this project. Please reach out to your customer care team if
    you're an external evaluator, or the pathfinding team if you're an internal
    developer.

    Once you have a GitHub account that access to the organization, you will need
    to add your SSH key that you must generate on your machine.  Please follow the
    instructions `here
    <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_.

    Getting the source code
    -----------------------

    If you already have the source code with its submodules, please skip ahead to
    `Building the library`.

    We need to checkout the code with its submodules in order to use it.

    ::

        git clone git@github.com:<REPO>.git --recurse-submodules

    Building the library
    --------------------

    To build the Python environment and library,

    ::

        make build

    Now, we activate the environment for use:

    ::

        source build/python_env/bin/activate

    A first example
    ---------------

    Now you're in the provided environment for this software! Let's move onto
    running your first program. Let's build a beginning example. This will be just
    a simple example Hello World-type program we made. It'll just open an
    accelerator device and close it. A properly-provisioned machine for this
    environment should execute this example flawlessly.

    ::

        make programming_examples/basic_empty_program

    We'll have to tell the runtime where you're running the programs. This usually
    is just the root directory of this software repository. Export the appropriate
    ``TT_METAL_HOME`` environment variable to tell the runtime this.

    ::

        export TT_METAL_HOME=$(pwd)

    Well done, now we just run our first example.

    ::

        ./build/programming_examples/basic_empty_program
