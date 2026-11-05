.. _Custom_SFPI_Add:

Vector addition using SFPI
==========================

This example demonstrates how to program the vector engine to perform the simplest operation - vector addition. This example serves as a starting point for users looking to implement custom operations using SFPI. We'll go through this code section by section. The full source code for this example is available under the ``tt_metal/programming_examples/custom_sfpi_add`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``metal_example_custom_sfpi_add`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/metal_example_custom_sfpi_add

.. warning::

    Tenstorrent does not guarantee backward compatibility for user-implemented SFPI functions. Keep your implementations up to date with the latest Metalium releases. APIs that call low-level SFPI functions may change without notice, and SFPI specifications may also change in future hardware versions.

Program setup
-------------

This example assumes familiarity with basic Metalium concepts like device initialization, tensor allocation, dataflow buffers, and kernel setup. If you're new to these concepts, we recommend starting with the :ref:`Eltwise sfpu example<Eltwise sfpu example>` for a gentler introduction to programming Metalium kernels.

The host-side setup for this custom SFPI example follows the standard pattern: device initialization, DRAM tensor allocation for two inputs and one output, dataflow buffer descriptions for kernel communication, and kernel descriptions (``KernelSpec``) in a ``ProgramSpec``. The key difference from simpler examples is that we need two input dataflow buffers (``src0``, ``src1``) to handle the binary operation, plus the standard output buffer (``dst``). Each ``KernelSpec`` binds a dataflow buffer or tensor parameter to a local name such as ``"in0"``; the kernel then refers to it as ``dfb::in0`` or ``tensor::in0``.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp
    // Create a 1x1 mesh on device 0 (same API scales to multi-device meshes)
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // Submit work via the mesh command queue: uploads/downloads and program execution.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // ...

    TensorSpec dram_tensor_spec(
        Shape({n_tiles, elements_per_tile}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

    // Allocate the tensors (replicated across mesh; on unit mesh ⇒ single device allocation)
    // src0, src1 are input tensors; dst is output tensor
    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);

    // A MeshWorkload is a collection of programs that will be executed on the mesh. Each workload is
    // local to a single device. Here we create a workload for our single-device mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // This example program will only use 1 Tensix core. So we set the core to {0, 0} (the most top-left core).
    constexpr experimental::NodeCoord core = {0, 0};

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName reader{"reader"};
    const experimental::KernelSpecName writer{"writer"};
    const experimental::KernelSpecName compute{"compute"};
    const experimental::DFBSpecName src0_dfb{"src0"};
    const experimental::DFBSpecName src1_dfb{"src1"};
    const experimental::DFBSpecName dst_dfb{"dst"};
    const experimental::TensorParamName src0_tensor_param{"src0"};
    const experimental::TensorParamName src1_tensor_param{"src1"};
    const experimental::TensorParamName dst_tensor_param{"dst"};

    // Dataflow buffers for kernel communication
    constexpr uint32_t tiles_per_cb = 2;
    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = tiles_per_cb,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec src1_dfb_spec{
        .unique_id = src1_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = tiles_per_cb,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec dst_dfb_spec{
        .unique_id = dst_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = tiles_per_cb,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Kernels: reader, writer, and custom SFPU compute
    experimental::KernelSpec reader_spec{
        .unique_id = reader,
        .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/dataflow/read_tiles.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
             {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };
    experimental::KernelSpec writer_spec{
        .unique_id = writer,
        .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/dataflow/write_tile.cpp",
        .dfb_bindings = {experimental::ConsumerOf(dst_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };
    experimental::KernelSpec compute_spec{
        .unique_id = compute,
        .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/compute/tiles_add.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(src0_dfb, "in0"),
             experimental::ConsumerOf(src1_dfb, "in1"),
             experimental::ProducerOf(dst_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .enable_32_bit_dest = false, // We don't need the destination accumulator to be FP32 as input and output are BFP16
            },
    };

    // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
    // operate on, and the cores the kernels run on (the work units).
    experimental::ProgramSpec program_spec{
        .name = "custom_sfpi_add",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = src1_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
        .work_units = {{.name = "custom_sfpi_add", .kernels = {reader, writer, compute}, .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

The Kernels
-----------

Data Movement Kernels
~~~~~~~~~~~~~~~~~~~~~

The reader kernel reads tiles from two source DRAM tensors and pushes them into two separate input dataflow buffers. The kernel gets the tile count as the named runtime argument ``args::n_tiles``, and it accesses the input tensors through ``tensor::in0`` and ``tensor::in1``.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/kernels/dataflow/read_tiles.cpp
    void kernel_main() {
        // Read parameters from the kernel arguments
        uint32_t n_tiles = get_arg(args::n_tiles);

        // The dataflow buffers to read the tiles into
        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);
        // ...
        // The input tensors are bound to this kernel by name in the host code.
        const auto in0 = TensorAccessor(tensor::in0);
        const auto in1 = TensorAccessor(tensor::in1);

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_in0.reserve_back(1);
            dfb_in1.reserve_back(1);
            uint32_t dfb_in0_addr = dfb_in0.get_write_ptr();
            uint32_t dfb_in1_addr = dfb_in1.get_write_ptr();
            noc_async_read_page(i, in0, dfb_in0_addr);
            noc_async_read_page(i, in1, dfb_in1_addr);

            noc_async_read_barrier();
            dfb_in0.push_back(1);
            dfb_in1.push_back(1);
        }
    }

The writer kernel is straightforward: it reads result tiles from the output dataflow buffer and writes them to the destination DRAM tensor.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/kernels/dataflow/write_tile.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        // The dataflow buffer that we are going to read from and write to DRAM
        DataflowBuffer dfb_out0(dfb::out0);

        // The output tensor is bound to this kernel by name in the host code.
        const auto out0 = TensorAccessor(tensor::out0);

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_out0.wait_front(1);
            uint32_t dfb_out0_addr = dfb_out0.get_read_ptr();
            noc_async_write_page(i, out0, dfb_out0_addr);
            noc_async_write_barrier();
            dfb_out0.pop_front(1);
        }
    }

SFPI Compute Kernel
~~~~~~~~~~~~~~~~~~~

The compute kernel is where the custom SFPI logic resides. It waits for tiles from the input dataflow buffers, performs the addition using the SFPI, and pushes the result to the output dataflow buffer.

The overall flow follows the same pattern as other compute kernels:

1. Wait for input tiles to be available in ``dfb::in0`` and ``dfb::in1``.
2. Acquire destination registers. These registers will be used as a scratchpad for the computation.
3. Copy tiles from the dataflow buffers to the destination registers.
4. Execute the custom SFPI addition function on the data in the destination registers.
5. Transfer the ownership of the destination registers to the packer
6. Reserve space in the output dataflow buffer, pack the result tile, and push it.
7. Pop the input tiles from the input dataflow buffers.
8. Release the destination registers.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);
        DataflowBuffer dfb_out0(dfb::out0);

        compute_kernel_hw_startup(dfb::in0, dfb::out0);
        copy_init(dfb::in0);

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_in0.wait_front(1);
            dfb_in1.wait_front(1);

            tile_regs_acquire();
            copy_tile(dfb::in0, 0, 0);
            copy_tile(dfb::in1, 0, 1);

            my_add_tile(0, 1, 0);  // <-- The custom SFPU addition happens here

            tile_regs_commit();
            tile_regs_wait();

            dfb_out0.reserve_back(1);
            pack_tile(0, dfb::out0);
            dfb_out0.push_back(1);

            dfb_in0.pop_front(1);
            dfb_in1.pop_front(1);
            tile_regs_release();
        }
    }

Custom SFPI Implementation
--------------------------

The core of this example is the custom SFPI function ``my_add_tile``. It's implemented in a layered way, which is a common pattern for SFPI programming to enable easy consumption and maintainability.

.. code-block:: cpp

    // tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_custom_add.h
    #ifdef TRISC_MATH
    namespace ckernel::sfpu {

    // Low-level function operating on a tile face
    void my_add_tile_face(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
        constexpr uint32_t n_vector_in_tile = 32;

        // Calculate base indices for each tile in the Dst register array.
        // Each tile occupies 32 consecutive Dst registers (n_vector_in_tile) in WH and BH
        // For example: tile 0 uses dst_reg[0-31], tile 1 uses dst_reg[32-63], etc.
        const uint32_t in0_base_idx = dst_index_in0 * n_vector_in_tile;
        const uint32_t in1_base_idx = dst_index_in1 * n_vector_in_tile;
        const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

        // Process one face of the tile (8 SIMD operations covering 256 elements).
        // Each iteration processes 32 elements, so 8 iterations = 256 elements = one 16x16 face.
        for (size_t i = 0; i < 8; i++) {
            sfpi::vFloat a = sfpi::dst_reg[in0_base_idx + i];
            sfpi::vFloat b = sfpi::dst_reg[in1_base_idx + i];
            sfpi::dst_reg[out_base_idx + i] = a + b;
        }
    }

    }  // namespace ckernel::sfpu
    #endif // TRISC_MATH

    // High-level API function
    void my_add_tile(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0) {
        MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, my_add_tile_face, idx_dst0, idx_dst1, idx_out0, VectorMode::RC));
    }


Here's a breakdown of the layers. The ``my_add_tile_face`` callable lives in an architecture-local ``llk_api/experimental`` header and is included only for the math thread, since it uses math-thread-specific code that will not compile for other RISC-V cores.

1.  **`my_add_tile`**: This is the main function called by the compute kernel. It wraps ``SFPU_BINARY_CALL_NO_TEMPLATE_ARGS`` with the ``MATH()`` macro, which ensures the code only runs on the math thread of the Tensix core. The macro sets up the SFPU, iterates over all faces of a tile, calls ``my_add_tile_face`` for each face, and then cleans up. This avoids manual setup and state management.

2.  **`my_add_tile_face`**: This is the most basic function, performing the actual addition on a single tile face. A 32x32 tile is divided into four 16x16 faces, and this function is called for each face. It uses the ``dst_reg`` array, which represents the SFPU's destination registers. The number of available ``dst_reg`` registers can be found in the :ref:`Compute Engines and Data Flow within Tensix<compute_engines_and_dataflow_within_tensix>` documentation.

    The function calculates base indices (``in0_base_idx``, ``in1_base_idx``, ``out_base_idx``) to map tile indices to register addresses within ``dst_reg``. Each tile occupies 32 registers; the base index is calculated by multiplying the tile index by 32 (refer to :ref:`Internal structure of a Tile<internal_structure_of_a_tile>` for more information on tile structure). For example, processing tiles at indices 0, 1, and 0 results in base indices of 0, 32, and 0, respectively. This means the first input tile starts at ``dst_reg[0]``, the second at ``dst_reg[32]``, and the output overwrites the first input tile at ``dst_reg[0]``.

    Within each face, the function loads SIMD vectors (``vFloat``) from the input registers, adds them, and writes the result back to the output registers.

    Each time the SFPI function is called, the helper automatically offsets ``dst_reg`` to point to the start of the current face. So, on the first call, ``dst_reg`` has an offset of 0; on the second, the offset is 8, and so on. The programmer does not need to manage this offset manually.

This layered structure keeps high-level logic separate from hardware-specific details, making the code easier to read and maintain.

.. warning::

    The value of ``n_vector_in_face`` is architecture dependent. The example above assumes a Tensix architecture where each vector is 32 wide. Which is true for currently shipping Tensix Processors (Wormhole and Blackhole). But may change in future versions. Users should verify this value against their target architecture specifications when adapting this example.

.. note::

    Use the SFPU macro wrappers to invoke custom SFPI functions, depending on the number of input tiles. Please view the header file for the most up-to-date information.

    *  ``SFPU_UNARY_CALL_NO_TEMPLATE_ARGS``: For non-templated functions with one input tile.
    *  ``SFPU_BINARY_CALL_NO_TEMPLATE_ARGS``: For non-templated functions with two input tiles.
    *  ``SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS``: For non-templated functions with three input tiles.

.. warning::

    SFPU macro wrappers and LLK helpers are internal APIs and may change in future releases. Tenstorrent does not guarantee backward compatibility for these internal functions. Users should keep their use up to date with the latest Metalium releases.

Runtime Arguments and Execution
-------------------------------

Back on the host, we set the runtime arguments for the kernels with ``experimental::ProgramRunArgs`` and ``experimental::SetProgramRunArgs``. All three kernels get the named runtime argument ``n_tiles``. The ``tensor_args`` bind each DRAM tensor to its tensor parameter, so the reader and writer kernels do not need the DRAM addresses as runtime arguments.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp
    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = reader,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = writer,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = compute,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
        .tensor_args =
            {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

For mesh execution, we add the program to a mesh workload and enqueue it for execution across the mesh. We also upload input data using the mesh tensor API.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp

    // Upload input data to the mesh tensors (non-blocking)
    HostTensor a_host_tensor = HostTensor::from_vector(a_data, dram_tensor_spec);
    HostTensor b_host_tensor = HostTensor::from_vector(b_data, dram_tensor_spec);
    cq.enqueue_write_tensor(a_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(b_host_tensor, src1_dram_tensor);

    // Add program to mesh workload and execute
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

Finally, we read back the results from the mesh tensor to verify correctness.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_add/custom_sfpi_add.cpp
    std::vector<bfloat16> result_vec =
        cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

    // Validation against golden output...

Conclusion
----------

This example demonstrated how to create a custom SFPI kernel for vector addition using the Mesh API. Key takeaways include:

*   The layered approach to SFPI kernel development (high-level API, LLK wrapper, low-level face function).
*   The use of destination registers (``dst_reg``) for SFPU computations.
*   The role of the LLK SFPU macro API (e.g., ``SFPU_BINARY_CALL_NO_TEMPLATE_ARGS``) in simplifying SFPI programming by handling tile face iteration.
*   The standard pipeline of reader, compute, and writer kernels for processing data on Tensix cores.

By following this pattern, you can implement a wide variety of custom element-wise operations on the SFPU to accelerate your specific workloads while leveraging the distributed programming capabilities of the Mesh API.
