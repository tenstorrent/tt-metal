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
``MeshProgramSpec`` and ``MeshProgramRunArgs`` are in ``ttnn/cpp/ttnn/operations/generic/generic_op.hpp``, and the
others in ``tt_metal/api/tt-metalium/experimental/metal2_host_api/``. Of the advanced options, Python binds only the ones
listed under `How a kernel finds its inputs`_.

An example
**********

This program computes ``exp(x)`` for every element of a tensor in DRAM. On each core a reader kernel copies tiles
from the input tensor into a buffer, a compute kernel applies ``exp`` to each tile, and a writer kernel copies the
results to the output tensor. The three kernels already exist in TT-NN. With ``device`` an open device:

.. literalinclude:: ../../../../tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py
   :language: python
   :start-after: [guide-example-start]
   :end-before: [guide-example-end]
   :dedent: 4

A ``WorkUnitSpec`` runs its kernels on each of its ``target_nodes``; a node is a core. A program can have several work
units, each running its own kernels on its own cores.

How a kernel finds its inputs
*****************************

Everything a kernel uses has a name, which the kernel code reaches through a generated namespace. For a buffer,
tensor, semaphore or scratchpad, that name is the ``accessor_name`` of the kernel's binding; the ``unique_id`` of the
spec only ties the binding to it. In the example, the reader reads tensor parameter ``input`` as ``tensor::src``.

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
   * - A tensor
     - A ``TensorParameter`` and a ``TensorBinding``, the tensor in ``ProgramRunArgs.tensor_args``
     - ``TensorAccessor(tensor::<accessor_name>)``
   * - A buffer between two kernels
     - A ``DataflowBufferSpec``, with a producer and a consumer binding
     - ``DataflowBuffer(dfb::<accessor_name>)``
   * - A semaphore
     - A ``SemaphoreSpec`` and a ``SemaphoreBinding``
     - ``Semaphore(sem::<accessor_name>)``
   * - Private scratch memory
     - A ``ScratchpadSpec`` and a ``ScratchpadBinding``
     - ``Scratchpad<T>(scratch::<accessor_name>)``

A list of values whose length varies, such as a list of core coordinates, has no named form yet. Pass it as varargs:
``KernelAdvancedOptions(num_runtime_varargs=n)`` on the kernel, ``AdvancedKernelRunArgs(runtime_varargs={core: [...]})``
in its run args, and ``get_vararg(i)`` in the kernel code. Python binds only the advanced options that TT-NN's
callers use so far: the vararg counts and values, and ``DFBAdvancedOptions.alias_with``, which lets two buffers share
memory. The others are C++ only.

Tensors
*******

``io_tensors`` lists the tensors the program uses, once for each tensor parameter a tensor is bound to, with the output
last; ``generic_op`` returns the last entry. Each tensor must be on the device and match its parameter's
``TensorSpec``, unless the parameter's ``TensorSpecRelaxations`` allow a difference, such as a dynamic shape.

Calls and the program cache
***************************

The first call with a given program compiles its kernels and sets it up on the device. A later call with the same
program only writes the new values: argument values and tensors can change from call to call without a recompile.
``ttnn.experimental.prepare_generic_op`` takes the same arguments, compiles the program and puts it in the cache without
running it.

The cache key is a hash of everything in the ``ProgramSpec`` that changes the compiled program, combined with the
devices it runs on. ``generic_op`` computes that hash on every call. To skip it, compute it once with
``ttnn.experimental.compute_program_spec_hash(spec)`` and pass it as ``custom_program_hash`` (a ``MeshProgramSpec``
takes it as a field). A custom key replaces the spec hash, so give each different program its own; the key still
includes each kernel's argument names, cores and vararg counts, and the names of the spec's tensor parameters and
buffers.

Every call checks its run args against the spec before Metal sees them: each kernel gets exactly the runtime arguments
and varargs it declares, each tensor parameter gets a tensor from ``io_tensors`` that matches it, and each DFB size
override names a buffer of the spec. Whether every core got its values is checked on later calls only when
``ttnn.CONFIG.validate_program_args`` is on; with it off, a core that a later call leaves out keeps its previous
values. A DFB size override in ``ProgramRunArgs.dfb_run_overrides`` stays in effect for later calls that reuse the
cached program, including calls with an equal spec from other code, until a call changes it.

Running on a mesh
*****************

With a ``ProgramSpec``, ``generic_op`` runs the program on every device that holds the tensors. To run different
programs on different devices, give a program for each mesh coordinate range. On a 1x2 mesh, for example:

.. code-block:: python

    left = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))
    right = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 1), ttnn.MeshCoordinate(0, 1))
    output_tensor = E.generic_op(
        [input_tensor, output_tensor],
        E.MeshProgramSpec(programs=[(left, spec), (right, other_spec)]),
        E.MeshProgramRunArgs(programs=[(left, run_args), (right, other_run_args)]),
    )

The ranges must not overlap, every device in them must hold every tensor, and the run args must have the same ranges as
the spec.

Moving a kernel from the ProgramDescriptor form
***********************************************

The lines that fetch a kernel's inputs change, as TT-NN's unary kernels in both forms show: the compute kernel
(eltwise_sfpu.cpp_ and eltwise_sfpu_metal2.cpp_) and the reader (reader_unary_interleaved_start_id.cpp_ and
reader_unary_interleaved_start_id_metal2.cpp_).

- Arguments by position become arguments by name: ``get_arg_val<uint32_t>(0)`` becomes ``get_arg(args::num_tiles)``,
  with ``num_tiles`` declared in the kernel's ``RuntimeArgSchema``.
- Circular buffers by index become dataflow buffers by name: ``tt::CBIndex::c_0`` becomes ``dfb::in``, with ``in`` the
  accessor name of the kernel's binding to a ``DataflowBufferSpec``. Compute calls take it directly, as in
  ``copy_tile(dfb::in, 0, 0)``; a data-movement kernel wraps it as ``DataflowBuffer(dfb::in)``, and reads its page size
  with ``get_entry_size()``.
- Tensor addresses passed as arguments become bound tensors: ``TensorAccessor`` built from an address and
  ``TensorAccessorArgs`` becomes ``TensorAccessor(tensor::src)``.

On Wormhole and Blackhole the loops that move and compute data can stay as they are. On Quasar the circular-buffer calls
do not work, so the data-movement loops use the ``DataflowBuffer`` calls instead (``reserve_back``, ``push_back``,
``wait_front``, ``pop_front``), as both unary readers above already do.

.. _eltwise_sfpu.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp
.. _eltwise_sfpu_metal2.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu_metal2.cpp
.. _reader_unary_interleaved_start_id.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp
.. _reader_unary_interleaved_start_id_metal2.cpp: https://github.com/tenstorrent/tt-metal/blob/6d8a1032acc18f6808db4e3712764afa93281d2c/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp

Some features of the ``ProgramDescriptor`` form are not available in the ``ProgramSpec`` form yet, such as global
circular buffers, fabric connections and per-core allocated tensors. Programs that use them stay on ``ttnn.generic_op``.
A global semaphore works in both forms: pass the address from ``ttnn.get_global_semaphore_address`` as a runtime
argument.

More examples
*************

``tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py`` builds programs with two inputs, varargs,
aliased buffers, buffers on a sharded tensor's memory, semaphores, the mesh form and, on Quasar, a compute kernel with
two threads. The API reference lists the arguments of :func:`ttnn.experimental.generic_op`.
