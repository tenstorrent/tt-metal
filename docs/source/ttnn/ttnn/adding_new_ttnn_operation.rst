Adding New TT-NN Operation
##########################

.. note::
   This document is meant for contributors to TT-NN.

   Not all operations may be functional on all Tenstorrent hardware (Grayskull,
   Wormhole, or others).


FAQ
***

What is a TT-NN operation?
--------------------------

A TT-NN operation is a function that takes in one or more input tensors and produces one or more output tensors. It is implemented in C++ and can be called from Python.

What steps are needed to add TT-NN operation in C++?
----------------------------------------------------
1. There are 2 options for writing a new operation. Option ``a`` is to write a device operation and option ``b`` is to write an operation that calls other operations
   a. Implement device operation in C++. Device operation is a struct that satisfies `DeviceOperationConcept` and specifies how to create output tensors and a program to run on the device.
   b. Implement an operation in C++ that calls other operations. This type of operation simply defines an ``invoke()`` method that calls other operations.
2. Register the struct using `ttnn::register_operation`.

What steps are needed to add TT-NN operation in Python?
-------------------------------------------------------
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

.. note::
 **All new operations must use the ProgramDescriptor pattern** (see below).
 The old ``CachedProgram`` / ``shared_variables_t`` pattern is legacy and should not
 be used for new operations.

A concrete example of a device operation can be found in `ttnn/cpp/ttnn/operations/examples/example/device`

ProgramDescriptor Pattern (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **ProgramDescriptor** pattern is the recommended way to write program factories.
Instead of imperatively constructing a ``Program`` object and returning a ``CachedProgram``,
you declaratively describe the program using a ``ProgramDescriptor`` struct. The framework
then handles program construction, caching, and buffer address patching on cache hits.

**Key benefits:**

- **No ``shared_variables_t``** — you don't need to store kernel handles or core lists.
- **No manual buffer address patching** — the framework auto-patches buffer addresses on cache hits.
- **No ``override_nondeterministic_runtime_args`` for buffer addresses** — only implement it if you have
  truly dynamic parameters (e.g., random seeds) that change on every call.
- **Cleaner code** — the declarative style is easier to read and less error-prone.

**ProgramFactory interface:**

.. code-block:: cpp

   struct ProgramFactory {
       // Declare the program: circular buffers, kernels, and runtime args.
       // Called on cache miss. The framework builds the Program from this descriptor.
       static tt::tt_metal::ProgramDescriptor create_descriptor(
           const operation_attributes_t& operation_attributes,
           const tensor_args_t& tensor_args,
           tensor_return_value_t& tensor_return_value);

       // OPTIONAL: Only needed for truly dynamic parameters (random seeds, etc.)
       // Buffer addresses are auto-patched — do NOT patch them here.
       // static void override_nondeterministic_runtime_args(
       //     tt::tt_metal::Program& program,
       //     const operation_attributes_t& operation_attributes,
       //     const tensor_args_t& tensor_args,
       //     tensor_return_value_t& tensor_return_value);

       // OPTIONAL: Only needed when create_descriptor requires a device-side
       // resource (e.g. config tensor) not already in tensor_args or the output.
       // Called once on cache miss. Return value is stored by the framework and
       // passed as an extra argument to create_descriptor.
       // static SomeResourceType prepare_resources(
       //     const operation_attributes_t& operation_attributes,
       //     const tensor_args_t& tensor_args,
       //     tensor_return_value_t& tensor_return_value);
   };

**Building a ProgramDescriptor:**

.. code-block:: cpp

   ProgramDescriptor desc;

   // 1. Declare circular buffers
   desc.cbs.push_back(CBDescriptor{
       .total_size = num_tiles * tile_size,
       .core_ranges = all_cores,
       .format_descriptors = {{CBFormatDescriptor{
           .buffer_index = cb_id,
           .data_format = data_format,
           .page_size = tile_size,
       }}},
   });

   // 2. Declare kernels with compile-time args and config
   //    Use ReaderConfigDescriptor{} for reader, WriterConfigDescriptor{} for writer,
   //    and ComputeConfigDescriptor{...} for compute kernels.
   KernelDescriptor reader_desc;
   reader_desc.kernel_source = "path/to/reader_kernel.cpp";
   reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
   reader_desc.core_ranges = all_cores;
   reader_desc.compile_time_args = {cb_id};
   // If the kernel uses get_named_compile_time_arg_val(), set named args:
   reader_desc.named_compile_time_args = {{"cb_in0", tt::CBIndex::c_0}};
   reader_desc.config = ReaderConfigDescriptor{};

   KernelDescriptor compute_desc;
   compute_desc.kernel_source = "path/to/compute_kernel.cpp";
   compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
   compute_desc.core_ranges = all_cores;
   compute_desc.compile_time_args = {cb_id};
   // Named compile-time args map string names to CB indices for the kernel
   compute_desc.named_compile_time_args = {
       {"cb_in0", tt::CBIndex::c_0},
       {"cb_out", tt::CBIndex::c_4},
       {"cb_intermed0", tt::CBIndex::c_5},
   };
   compute_desc.config = ComputeConfigDescriptor{
       .math_fidelity = MathFidelity::HiFi4,
       .fp32_dest_acc_en = false,
       .math_approx_mode = false,
   };

   // 3. Add runtime args per core
   reader_desc.runtime_args.emplace_back(
       core, KernelDescriptor::CoreRuntimeArgs{buffer_addr, tiles_per_core, offset});

   // 4. Push kernels — the order matters: kernel index 0, 1, 2, etc.
   //    If you implement override_nondeterministic_runtime_args, use the kernel index
   //    (not a KernelHandle) to access runtime args via GetRuntimeArgs().
   desc.kernels.push_back(std::move(reader_desc));
   desc.kernels.push_back(std::move(compute_desc));
   return desc;

.. warning::

   If a kernel source uses ``get_named_compile_time_arg_val()`` to retrieve
   compile-time arguments by name, you **must** set ``named_compile_time_args``
   on the corresponding ``KernelDescriptor``. This field maps string names to
   ``tt::CBIndex`` values and causes the ``KERNEL_COMPILE_TIME_ARG_MAP`` macro
   to be defined during JIT compilation. Without it, the kernel will fail to
   compile with a ``'get_named_compile_time_arg_val' was not declared in this
   scope`` error. This applies to all kernel types (reader, writer, and
   compute).

**``compute_program_hash``:**

If your operation has dynamic fields in its attributes that don't affect program
compilation (e.g., random seeds), you **must** provide a custom ``compute_program_hash``
that excludes those fields. Otherwise the cache will miss on every call. Always include
``type_hash<YourDeviceOperation>`` to prevent collisions with other operations:

.. code-block:: cpp

   static tt::stl::hash::hash_t compute_program_hash(
       const operation_attributes_t& attrs, const tensor_args_t& tensors) {
       auto hashable = attrs;
       hashable.seed = 0;  // Exclude dynamic fields
       return tt::stl::hash::hash_objects_with_default_seed(
           tt::stl::hash::type_hash<MyDeviceOperation>, hashable, tensors);
   }

If all attributes are compile-time deterministic, you can omit ``compute_program_hash``
and the framework will use a sensible default that hashes ``type_hash<YourDeviceOperation>``,
all of ``operation_attributes_t``, and all of ``tensor_args_t``.

Full example files:

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

`ttnn/python/ttnn/operations/<category>/<operation_name>/<operation_name>_nanobind.hpp`
`ttnn/python/ttnn/operations/<category>/<category>_nanobind.hpp`

A concrete example:

.. literalinclude::  examples/example/example_nanobind.hpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/example/example_nanobind.hpp


.. literalinclude::  examples/examples_nanobind.hpp
   :language: cpp
   :linenos:
   :caption: ttnn/cpp/ttnn/operations/examples/examples_nanobind.hpp

Finally, call the module defined in `examples/example/example_nanobind.hpp` wherever you want it to be added.



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

    # TT-NN Tensors are converted to torch tensors before calling the golden function automatically
    # And the outputs are converted back to TT-NN Tensors
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


Step 3: (Optional) Add example usage to docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is good practice to include an example demonstrating how to use the new function.
The simplest method is to add an **Example** section directly in the documentation passed to the ``bind_registered_operation`` function. However, this approach makes it difficult to keep the example up to date and prevents the snippet from being tested.

A better approach is to place the example code in a test file and have it included automatically during the documentation build process.

In the file
`examples_mapping.py <https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/docs_examples/examples_mapping.py>`_,
each function is mapped to an example usage snippet that will appear in its documentation.

Add the new operation to the ``FUNCTION_TO_EXAMPLES_MAPPING_DICT`` dictionary, as shown below:

.. code-block:: python

    FUNCTION_TO_EXAMPLES_MAPPING_DICT = {
        ...
        "ttnn.example": example.test_example,
        ...
    }

Place the example usage function in a new file named ``test_example_examples.py`` (or an existing file, if appropriate).
Make sure the file is imported at the top of ``examples_mapping.py``:

.. code-block:: python

    # ...
    from . import test_data_movement_examples as data_movement
    from . import test_core_examples as core

    # Import the new file
    from . import test_example_examples as example
    # ...

Implement the example as a standard ``ttnn`` pytest:

.. code-block:: python

    def test_example(device):
        # Create tensor
        tensor = ttnn.rand((2, 3), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        # Call the new operation
        output_tensor = ttnn.example(tensor)

This ensures that all example code snippets are executed and validated in the TT-NN CI pipeline.
