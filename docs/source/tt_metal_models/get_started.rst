.. _TT-Metal Models Get Started:

Getting Started
===============

Prerequisites
-------------

Ensure that you have the base TT-Metal source and environment configuration
:ref:`built and ready<Getting Started>`.

Now, from the project root, get the Python virtual environment in which you'll
be working in ready.

::

    source build/python_env/bin/activate

Set ``PYTHONPATH`` to the root for running models. This is a common practice.

::

    export PYTHONPATH=$(pwd)

Running an example model
------------------------

We developed our models to be run with ``pytest``.

::

    pytest <path_to_test_file>::<test_in_file>

Next steps
----------

We write the models with the Python Libraries we have for TT-Metal. Refer to
the docs under *Libraries* for documentation, such as the :ref:`Tensor
API<Tensor>` or the :ref:`TT-LIB Dependencies<TT-LIB>`.
