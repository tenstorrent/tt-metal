.. _TT-Metalium Models Get Started:

Getting Started
===============

Prerequisites
-------------

Ensure that you have the base TT-Metalium source and environment configuration
`built and ready
<https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>`_.

Now, from the project root, activate the provided Python virtual environment in
which you'll be working.

::

    source python_env/bin/activate

.. note::
   You can use the ``PYTHON_ENV_DIR`` environment variable with the provided
   ``create_venv.sh`` script to control where the environment is created.

Set ``PYTHONPATH`` to the root for running models. This is a common practice.

::

    export PYTHONPATH=$(pwd)

Running an example model
------------------------

We develop models in ``models`` folder, splitting them to ``demo`` and ``experimental`` sub-folders.
In ``models/demo`` folder you will find models with prepared demos and a ``README.md`` file giving instructions how to run the model demo(s).

Models in the ``experimental`` sub-folder are our work in progress and are not yet ready to be used by the users.

Many models will have a ``tests`` sub-folder containing tests for parts of the model or the entire model.
You can run these tests using ``pytest``.

::

    pytest <path_to_test_file>::<test_in_file>

Next steps
----------

We write the models using TTNN which is a user-friendly Python Library that we developed on top of TT-Metalium. Refer to
the docs under *Libraries* for documentation, such as the :ref:`Tensor
API<Tensor>` or the :ref:`TT-LIB Dependencies<TT-LIB>`.
