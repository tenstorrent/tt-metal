.. _generic_op_program_spec:

Running your own kernels with generic_op and a ProgramSpec
##########################################################

``ttnn.experimental.generic_op`` runs a program that you describe yourself: the kernels, the buffers between them,
the tensors they read and write, and the values of each call. It takes the program as a Metal 2.0 ``ProgramSpec``,
everything fixed about the program, and a ``ProgramRunArgs``, the values of one call. Kernels find everything they use
by name.

This form is experimental while its API settles, so everything it adds lives under ``ttnn.experimental``. The
``ProgramDescriptor`` form, ``ttnn.generic_op``, keeps working. The ``ProgramSpec`` form also runs on Quasar.

Each Python class has the fields of the C++ struct with the same name, whose comments document them: TT-NN's
``MeshProgramSpec`` and ``MeshProgramRunArgs`` are in generic_op.hpp_, and the others in `the workload headers`_
(``tt_metal/api/tt-metalium/workload/``), apart from the advanced options, which stay experimental in
advanced_options.hpp_. Of the advanced options, Python
binds only ``DFBAdvancedOptions.alias_with``, which lets buffers of the same total size share memory; the others are
C++ only. So far the GlobalSemaphore fields are C++ only too: ``ProgramSpec.global_semaphore_parameters``,
``KernelSpec.global_semaphore_bindings`` and ``ProgramRunArgs.global_semaphore_args``. One field differs in type: the
``data_format_metadata`` of a ``DataflowBufferSpec`` or ``ScratchpadSpec`` takes a ttnn data type, such as
``ttnn.bfloat16``, so a format without one, such as Float16, cannot be set, and ``ttnn.int8`` reads back as
``ttnn.uint8``.

An example
**********

This program computes ``exp(x)`` for every element of a tensor in DRAM. On each core a reader kernel copies tiles
from the input tensor into a buffer, a compute kernel applies ``exp`` to each tile, and a writer kernel copies the
results to the output tensor. The three kernels already exist in TT-NN. With ``torch`` and ``ttnn`` imported and
``device`` an open device:

.. literalinclude:: ../../../../tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py
   :language: python
   :start-after: [guide-example-start]
   :end-before: [guide-example-end]
   :dedent: 4

A ``WorkUnitSpec`` runs its kernels on each of its ``target_nodes``; a node is a core. A program can have several work
units, each running some of the program's kernels on its own cores, and a kernel can be in more than one.

A kernel's ``source`` is the path of its source file: an absolute path, or one that Metal looks up in the current
directory, then in ``TT_METAL_KERNEL_PATH``, the system kernel directory and the tt-metal root, where it finds the
example's TT-NN kernels. To give the code itself instead, pass ``E.KernelSpec.SourceCode(code=...)``.

How a kernel finds its inputs
*****************************

Everything a kernel uses has a name, which the kernel code reaches through a generated namespace. For a buffer,
tensor, semaphore or scratchpad, that name is the ``accessor_name`` of the kernel's binding; the ``unique_id`` of the
spec only ties the binding to it. In the example, the reader reads tensor parameter ``input`` as ``tensor::src``.
``E.producer_of`` and ``E.consumer_of`` make a buffer binding from the buffer's ``unique_id`` and the accessor name; the
example's buffers and their accessors share the names ``in`` and ``out``.

.. list-table::
   :header-rows: 1

   * - What a kernel needs
     - Where you set it
     - How the kernel reads it
   * - A value fixed at compile time
     - ``KernelSpec.compile_time_args``
     - ``get_arg(args::<name>)``
   * - A value per core
     - A name in ``RuntimeArgSchema.runtime_arg_names``, the values in ``KernelRunArgs.runtime_arg_values``
     - ``get_arg(args::<name>)``
   * - A value shared by all cores
     - A name in ``RuntimeArgSchema.common_runtime_arg_names``, the value in ``KernelRunArgs.common_runtime_arg_values``
     - ``get_arg(args::<name>)``
   * - A list of values fixed at compile time
     - ``KernelSpec.compile_time_array_args``
     - ``get_arg(args::<name>)[i]``
   * - A list of values per core
     - An ``ArrayArgDecl`` (``name`` and ``size``) in ``RuntimeArgSchema.runtime_array_args``, the values in
       ``KernelRunArgs.runtime_array_arg_values``
     - ``get_arg(args::<name>)[i]``
   * - A list of values shared by all cores
     - An ``ArrayArgDecl`` in ``RuntimeArgSchema.common_runtime_array_args``, the values in
       ``KernelRunArgs.common_runtime_array_arg_values``
     - ``get_arg(args::<name>)[i]``
   * - A tensor
     - A ``TensorParameter`` and a ``TensorBinding``, the tensor in ``ProgramRunArgs.tensor_args``
     - ``TensorAccessor(tensor::<accessor_name>)``, or ``LocalTensorAccessor<T>(tensor::<accessor_name>)`` for the
       core's own shard in L1
   * - A buffer between two kernels
     - A ``DataflowBufferSpec``, with a producer and a consumer binding
     - ``DataflowBuffer(dfb::<accessor_name>)``
   * - A buffer on a sharded tensor's memory
     - A ``DataflowBufferSpec`` whose ``borrowed_from`` names the ``TensorParameter``
     - ``DataflowBuffer(dfb::<accessor_name>)``
   * - A semaphore
     - A ``SemaphoreSpec`` and a ``SemaphoreBinding``
     - ``Semaphore(sem::<accessor_name>)``
   * - Private scratch memory
     - A ``ScratchpadSpec`` and a ``ScratchpadBinding``
     - ``Scratchpad<T>(scratch::<accessor_name>)``

An array has the same length on every core, and ``get_arg(args::<name>).size()`` gives it. For a list whose length
differs from core to core, such as a list of core coordinates, declare the longest length and pass each core's count
in another argument. Python has no varargs, the positional arguments that Metal 2.0 keeps only to ease migration:
array arguments replace them.

The lists and dicts these classes return are copies, and so are their other fields that hold an object, such as
``KernelSpec.hw_config``, ``TensorParameter.spec`` and ``WorkUnitSpec.target_nodes``. A change to a copy is lost: assign
the whole field instead, as in ``kernel.compile_time_args = {"block": 2}`` rather than
``kernel.compile_time_args["block"] = 2``. That holds one level up too: ``spec.kernels[0].compile_time_args = ...``
changes a copy of the kernel, so take the list, change it and assign it back (``kernels = spec.kernels``, change
``kernels[0]``, then ``spec.kernels = kernels``). Only four fields return the object itself, so that a change to it
changes the object that holds it: ``KernelSpec.compiler_options``, ``KernelSpec.runtime_arg_schema``,
``TensorParameter.relaxations`` and ``DataflowBufferSpec.advanced_options``.

Tensors
*******

``io_tensors`` lists the tensors the program uses, once for each tensor parameter a tensor is bound to, with the output
last; ``generic_op`` returns the last entry, or, when the tensors are not all on the same devices, only its parts on
the devices of the tensor that is on the fewest devices (see "Running on a mesh"). Each tensor must be on the device
and match its parameter's
``TensorSpec``, unless the parameter's ``TensorSpecRelaxations`` allow a difference, such as a dynamic shape.

Calls and the program cache
***************************

The first call with a given program compiles its kernels and sets it up on the device. A later call with the same
program only writes the new values: argument values and tensors can change from call to call without a recompile.
``ttnn.experimental.prepare_generic_op`` takes the same arguments, compiles the program and puts it in the cache without
running it.

The cache key is a hash of the whole ``ProgramSpec``, combined with the devices it runs on, apart from
``ProgramSpec.name``, ``WorkUnitSpec.name`` and the tensor fields that a parameter's ``TensorSpecRelaxations`` allow
to differ. So renaming or reordering kernels or buffers gives a new cache entry. ``generic_op`` computes that hash on
every call while the program cache is on. The key holds a kernel's source path, not the file's contents, and so do
Metal's caches of compiled and loaded kernels within a process, so after you edit a kernel file, run it in a new
process. A kernel given as ``KernelSpec.SourceCode`` is keyed on its code. Unlike the
``ProgramDescriptor`` form, the ``ProgramSpec`` form takes no ``custom_program_hash``, because a key the caller picks
could give two different programs the same cache entry.

Every call checks its run args against the spec before Metal sees them: each kernel gets exactly the runtime arguments
and arrays it declares, each array with its declared length, each tensor parameter gets a tensor from ``io_tensors``
that matches it, and each dataflow buffer (DFB) size override names a buffer of the spec, at most once and with no size
of 0. Whether every core got its values is checked on later calls only when ``ttnn.CONFIG.validate_program_args`` is on,
which it is not by default; with it off, a core that a later call leaves out keeps its previous values. A DFB size
override in ``ProgramRunArgs.dfb_run_overrides`` stays in effect for later calls that reuse the cached program,
including calls with an equal spec from other code, until a call changes it. ``prepare_generic_op`` with a program that
is already cached applies its run args to it the same way, even if the preparation then fails.

Running on a mesh
*****************

With a ``ProgramSpec``, ``generic_op`` runs the program on the devices of the tensor in ``io_tensors`` that is on the
fewest devices, and every other tensor must be on those devices too; usually all the tensors are on the same devices. If
the output is on more devices than that, its parts on the other devices are not written, and the tensor
``generic_op`` returns holds only the parts it wrote; the tensor you passed in still holds them all. To run different
programs on different devices, give a program for each mesh coordinate range. On a 1x2 mesh, for example:

.. code-block:: python

    left = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))
    right = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 1), ttnn.MeshCoordinate(0, 1))
    output_tensor = E.generic_op(
        [input_tensor, output_tensor],
        E.MeshProgramSpec(programs=[(left, spec), (right, other_spec)]),
        E.MeshProgramRunArgs(programs=[(left, run_args), (right, other_run_args)]),
    )

The ranges must stay inside the mesh and must not overlap, every device in them must hold every tensor, and the run args
must have the same ranges as the spec.

Moving a kernel from the ProgramDescriptor form
***********************************************

The lines that fetch a kernel's inputs change, as TT-NN's unary kernels in both forms show: the compute kernel
(eltwise_sfpu.cpp_ and eltwise_sfpu_metal2.cpp_) and the reader (reader_unary_interleaved_start_id.cpp_ and
reader_unary_interleaved_start_id_metal2.cpp_).

- Arguments by position become arguments by name: ``get_arg_val<uint32_t>(0)`` becomes ``get_arg(args::num_tiles)``,
  with ``num_tiles`` declared in the kernel's ``RuntimeArgSchema``, and ``get_compile_time_arg_val(0)`` becomes
  ``get_arg(args::<name>)`` with the name in ``KernelSpec.compile_time_args``. A read by position that is left in may
  still compile and read whatever value is at that position, so change every one.
- Circular buffers by index become dataflow buffers by name: ``tt::CBIndex::c_0`` becomes ``dfb::in``, with ``in`` the
  accessor name of the kernel's binding to a ``DataflowBufferSpec``. Calls such as ``copy_tile(dfb::in, 0, 0)`` take it
  directly, and so do the circular-buffer calls on Wormhole and Blackhole. The ``DataflowBuffer`` calls, which Quasar
  needs, take it wrapped as ``DataflowBuffer(dfb::in)``, as both kernels above do, and a data-movement kernel reads its
  page size with ``get_entry_size()``.
- Tensor addresses passed as arguments become bound tensors: ``TensorAccessor`` built from an address and
  ``TensorAccessorArgs`` becomes ``TensorAccessor(tensor::src)``.
- On the host, each ``CBDescriptor`` becomes a ``DataflowBufferSpec`` with its bindings, and a kernel's runtime
  arguments, a list of values per core, become ``KernelRunArgs.runtime_arg_values``: a dict from each argument name to a
  dict from core to value.

On Wormhole and Blackhole the loops that move and compute data can stay as they are (eltwise_sfpu_metal2.cpp_ also
splits its loop with ``dram_read::for_each_chunk``, which is not part of the move). On Quasar the circular-buffer calls
do not work, so the data-movement loops use the ``DataflowBuffer`` calls instead (``reserve_back``, ``push_back``,
``wait_front``, ``pop_front``), as both unary readers above already do. A Quasar data-movement kernel that makes many
transfers smaller than a tile, such as row-major sticks, also passes ``disable_dfb_implicit_sync_for_all=True`` to
``create_reader_datamovement_config`` or ``create_writer_datamovement_config``: such transfers stall Quasar's implicit
counting of buffer entries, and the flag leaves the counting to the kernel's own calls.

.. _eltwise_sfpu.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp
.. _eltwise_sfpu_metal2.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu_metal2.cpp
.. _reader_unary_interleaved_start_id.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp
.. _reader_unary_interleaved_start_id_metal2.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp
.. _generic_op.hpp: https://github.com/tenstorrent/tt-metal/blob/840db3161d0e2f4e3d760e75e0c895f7d5d5020a/ttnn/cpp/ttnn/operations/generic/generic_op.hpp
.. _the workload headers: https://github.com/tenstorrent/tt-metal/tree/840db3161d0e2f4e3d760e75e0c895f7d5d5020a/tt_metal/api/tt-metalium/workload
.. _advanced_options.hpp: https://github.com/tenstorrent/tt-metal/blob/840db3161d0e2f4e3d760e75e0c895f7d5d5020a/tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp

Some features of the ``ProgramDescriptor`` form are not available in the ``ProgramSpec`` form yet, such as global
circular buffers, fabric connections and per-core allocated tensors. From Python, global semaphores are not available
yet either, and a semaphore always starts at 0. Programs that need any of these stay on ``ttnn.generic_op``.

More examples
*************

``tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py`` builds programs with two inputs, array
arguments, aliased buffers, buffers on a sharded tensor's memory, semaphores, the mesh form and, on Quasar, a compute
kernel with two threads. The API reference lists the arguments of :func:`ttnn.experimental.generic_op`.
