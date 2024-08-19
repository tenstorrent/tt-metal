Adding New ttnn Operation
#########################

.. note::
   This document is meant for contributors to TT-NN.

   Not all operations may be functional on all Tenstorrent hardware (Grayskull,
   Wormhole, or others).


FAQ
***

What is a ttnn operation?
-------------------------

A ttnn operation is a function that takes in one or more input tensors and produces one or more output tensors. It is implemented in C++ and can be called from Python.

What steps are needed to add ttnn operation in C++?
---------------------------------------------------
1. There are 2 options for writing a new operation. Optiona ``a`` is to write a device operation and option ``b`` is to write a composite operation
   a. Implement device operation in C++. Device operation is a struct that specifies how to create output tensors and a program to run on the device.
   b. Implement a composite operation in C++. Composite operation simply defines ``operator()`` method that calls other operations.
2. Register the struct using `ttnn::register_operation`.

What steps are needed to add ttnn operation in Python?
------------------------------------------------------
1. Take an existing registered C++ operation and add a Python binding for it using `ttnn::bind_registered_operation`.
   The operation will be auto-registered in python. If the operation is called `ttnn::add` in C++, then the python binding will be `ttnn.add`.
2. (Optional) Attach golden function to the operation using `ttnn.attach_golden_function`. This is useful for debugging and testing.


Example of Adding a new Device Operation
****************************************

Let's implement `ttnn.example` (It will just copy the input tensor to the output tensor on the device)

C++ Implementation
------------------

Step 1: Implement device operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to add a new device operation, follow the directory structure shown below:

`ttnn/cpp/ttnn/operations/<category>/<operation_name>/device/<operation_name>_device_operation.hpp`
`ttnn/cpp/ttnn/operations/<category>/<operation_name>/device/<operation_name>_device_operation.cpp`
`ttnn/cpp/ttnn/operations/<category>/<operation_name>/device/<program_factory_0>_program_factory.cpp`

.. note::
 Add as many program factories as needed. But the minimum requirement is one program factory.

A concrete example of a device operation can be found in `ttnn/cpp/ttnn/operations/examples/example/device`

.. literalinclude::  examples/example/device/example_device_operation.hpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/example/device/example_device_operation.hpp


.. literalinclude::  examples/example/device/example_device_operation.cpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/example/device/example_device_operation.cpp


.. literalinclude::  examples/example/device/single_core_program_factory.cpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/example/device/single_core_program_factory.cpp


.. literalinclude::  examples/example/device/multi_core_program_factory.cpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/example/device/multi_core_program_factory.cpp


Step 2: Implement the operation in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to add a new operation, add the following file:

`ttnn/cpp/ttnn/operations/<category>/<operation_name>/<operation_name>.hpp`

A concrete example:

.. literalinclude::  examples/example/example.hpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/example/example.hpp


Python Implementation
---------------------

Step 1: Add Python binding
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to add a python binding for the operation, follow the directory structure shown below:

`ttnn/python/ttnn/operations/<category>/<operation_name>/<operation_name>_pybind.hpp`
`ttnn/python/ttnn/operations/<category>/<category>_pybind.hpp`

A concrete example:

.. literalinclude::  examples/example/example_pybind.hpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/example/example_pybind.hpp


.. literalinclude::  examples/examples_pybind.hpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/examples_pybind.hpp

Finally, call the module defined in `examples/example/example_pybind.hpp` wherever you want it to be added.



Step 2: (Optional) Add golden function for the operation in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A golden function can be added to an operation in order to compare its output with an equivalent `torch` implementation

Add the following code in a python file:

.. code-block:: python

    import ttnn

    # For the golden function, use the same signature as the operation
    # Keep in mind that all `ttnn.Tensor`s are converted to `torch.Tensor`s
    # And arguments not needed by torch can be ignored using `*args` and `**kwargs`
    def golden_function(input_tensor: "torch.Tensor", *args, **kwargs):
        output_tensor:  "torch.Tensor" = ...
        return output_tensor

    # ttnn Tensors are converted to torch tensors before calling the golden function automatically
    # And the outputs are converted back to ttnn Tensors
    # But in some cases you may need to preprocess the inputs and postprocess the outputs manually

    # In order to preprocess the inputs manually, use the following signature
    # Note that the arguments are not packed into *args and **kwargs as in the golden function!!!
    def preprocess_golden_function_inputs(args, kwargs):
        # i.e.
        ttnn_input_tensor = args[0]
        return ttnn.to_torch(ttnn_input_tensor)

    # In order to postprocess the outputs manually, use the following signature
    # Note that the arguments are not packed into *args and **kwargs as in the golden function!!!
    def postprocess_golden_function_outputs(args, kwargs, output):
        # i.e.
        ttnn_input_tensor = args[0]
        torch_output_tensor = outputs[0]
        return ttnn.from_torch(torch_output_tensor, dtype=ttnn_input_tensor.dtype, device=ttnn_input_tensor.device)

    ttnn.attach_golden_function(
        ttnn.example,
        golden_function=golden_function,
        preprocess_golden_function_inputs=preprocess_golden_function_inputs, # Optional
        postprocess_golden_function_outputs=postprocess_golden_function_outputs # Optional
    )

.. note::
   `ttnn.example` is the name of the operation in Python because the operation was registered as `ttnn::example` in C++.
