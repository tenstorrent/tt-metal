Adding New ttnn Operation
#########################

.. note::
   This document is meant for contributors to TT-NN.

   Not all operations may be functional on all Tenstorrent hardware (Grayskull,
   Wormhole, or others).


What is a ttnn operation?
-------------------------

A ttnn operation is a function that takes in one or more input tensors and produces one or more output tensors. It is implemented in C++ and can be called from Python.

What steps are needed to add ttnn operation in C++?
---------------------------------------------------
1. (Optional) Implement device operation in C++. Device operation is a struct that specifies how to create output tensors and a program to run on the device. If the ttnn operation is composed of other ttnn operations, then you can skip this step.
2. Implement ttnn operation in C++ and register it using `ttnn::register_operation`.

What steps are needed to add ttnn operation in Python?
------------------------------------------------------
1. Take an existing registerd C++ operation and add a Python binding for it using `ttnn::bind_registered_operation`.
2. In python, decorate the operation using `ttnn.register_operation`. (This step will be deprecated in the future)



C++ Implementation
------------------

Step 1: Implement device operation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to add a new device operation, follow the directory structure shown below:

`ttnn/cpp/ttnn/operations/<category>/<operation_name>/device/<operation_name>_device_operation.hpp`
`ttnn/cpp/ttnn/operations/<category>/<operation_name>/device/<operation_name>_device_operation.cpp`
`ttnn/cpp/ttnn/operations/<category>/<operation_name>/device/<program_factory_0>_program_factory.cpp`

.. note::
 Add as many program factories as needed

A concrete example of a device operation can be found in `ttnn/cpp/ttnn/operations/examples/example/device`

`ttnn/cpp/ttnn/operations/examples/example/device/example_device_operation.hpp`:

.. literalinclude::  examples/example/device/example_device_operation.hpp

`ttnn/cpp/ttnn/operations/examples/example/device/example_device_operation.cpp`:

.. literalinclude::  examples/example/device/example_device_operation.cpp

`ttnn/cpp/ttnn/operations/examples/example/device/single_core_program_factory.cpp`:

.. literalinclude::  examples/example/device/single_core_program_factory.cpp

`ttnn/cpp/ttnn/operations/examples/example/device/multi_core_program_factory.cpp`:

.. literalinclude::  examples/example/device/multi_core_program_factory.cpp


Step 2: Implement the operation in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to add a new operation, add the following file:

`ttnn/cpp/ttnn/operations/<category>/<operation_name>/<operation_name>.hpp`

A concrete example:

`ttnn/cpp/ttnn/operations/examples/example/example.hpp`:

.. literalinclude::  examples/example/example.hpp


Python Implementation
---------------------

Step 1: Add Python binding
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to add a python binding for the operation, follow the directory structure shown below:

`ttnn/python/ttnn/operations/<category>/<operation_name>/<operation_name>_pybind.hpp`
`ttnn/python/ttnn/operations/<category>/<category>_pybind.hpp`

A concrete example:

`ttnn/python/ttnn/operations/examples/example/example_pybind.hpp`:

.. literalinclude::  examples/example/example_pybind.hpp

`ttnn/python/ttnn/operations/examples/examples_pybind.hpp`:

.. literalinclude::  examples/example/example_pybind.hpp

Finally, call the module defined in `examples/example/example_pybind.hpp` wherever you want it to be added.



Step 2: Register the operation in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: Add the description of how to register the operation in Python.
