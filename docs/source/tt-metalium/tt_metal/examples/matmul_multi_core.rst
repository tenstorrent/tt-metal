.. _MatMul_Multi_Core example:

Matmul (Multi Core)
====================

We'll build a program that will perform matmul operations on two tensors with equal-size inner dimension on as many cores as possible on the accelerator. This example builds on top of the previous single core matmul example.

In terms of API usage, there isn't much change. We will discuss the specific
changes to:

- Distribute the work across multiple cores.
- Set up each core to process a subset of the tiles.
- What changes (and what doesn't change) in the API usage to achieve this.

It is important to note that this example builds on top of the previous single core matmul example, so it is recommended to understand the single core matmul example first. The single core matmul example is available under the ``tt_metal/programming_examples/matmul_single_core/`` directory.

The full source code for this example is available under the ``tt_metal/programming_examples/matmul/matmul_multi_core/`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``metal_example_matmul_multi_core`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_matmul_multi_core

.. note::
    Efficiently parallelizing matrix multiplication across multiple cores—using SPMD (Single Program, Multiple Data) or other parallelization strategies - is a broad and advanced topic, often covered in depth in advanced computer science courses. While this tutorial demonstrates how to use our API to distribute work across cores, it does not attempt to teach the fundamentals of parallel programming or SPMD concepts themselves.

    If you are new to these topics, we recommend consulting external resources or textbooks on parallel computing for a deeper understanding. This will help you make the most of our platform and adapt these examples to your own use cases.

    For further reading on parallel programming and SPMD, see the References section at the end of this document.

Device Initialization & Program Setup
-------------------------------------


Device initialization and parameter setup are similar to the single core example, but now use the mesh API. You create a mesh device, set up the command queue, compute the reference result on the CPU, and allocate the DRAM tensors. Example:

.. code-block:: cpp

    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // Create source data with specified matrix dimensions
    constexpr uint32_t M = 640;  // Number of rows in matrix A (user-defined)
    constexpr uint32_t N = 640;  // Number of columns in matrix B (user-defined)
    constexpr uint32_t K = 640;  // Inner dimension for multiplication (user-defined)

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bfloat16> src0_vec(M * K, 0);  // Matrix A (MxK)
    std::vector<bfloat16> src1_vec(K * N, 0);  // Matrix B (KxN)
    for (bfloat16& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    for (bfloat16& v : src1_vec) {
        v = bfloat16(dist(rng));
    }

    std::vector<bfloat16> golden_vec(M * N, 0);
    golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

    src0_vec = tilize_nfaces(src0_vec, M, K);
    src1_vec = tilize_nfaces(src1_vec, K, N);

    // Set up mesh command queue, workload and device range for multi-core execution
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const uint32_t Mt = M / TILE_HEIGHT;  // Number of tiles in M dimension
    const uint32_t Kt = K / TILE_WIDTH;   // Number of tiles in K dimension
    const uint32_t Nt = N / TILE_WIDTH;   // Number of tiles in N dimension

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes
    constexpr uint32_t elements_per_tile = single_tile_size / sizeof(bfloat16);

    auto make_dram_tensor_spec = [&](uint32_t num_tiles) {
        return TensorSpec(
            Shape({num_tiles, elements_per_tile}),
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    };
    TensorSpec src0_tensor_spec = make_dram_tensor_spec(Mt * Kt);
    TensorSpec src1_tensor_spec = make_dram_tensor_spec(Nt * Kt);
    TensorSpec dst_tensor_spec = make_dram_tensor_spec(Mt * Nt);

    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src0_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src1_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_tensor_spec);

Next, convert the source matrices to tiled format before preparing for execution on the device.

.. code-block:: cpp

    src0_vec = tilize_nfaces(src0_vec, M, K);
    src1_vec = tilize_nfaces(src1_vec, K, N);

Calculating Work Distribution
-----------------------------

Tenstorrent's AI processors support multiple parallelization strategies. The grid structure of the AI processors enables various approaches to distributing work. In this example, we use a simple SPMD (Single Program, Multiple Data) strategy similar to GPU programming. Each core runs the same program but processes a different subset of the data to compute the full result. We parallelize across the output tiles of the result matrix, with each core responsible for producing ``1/num_cores`` of the output tiles, where ``num_cores`` is the number of available cores.

.. note::

    The SPMD strategy is a standard approach in parallel computing and works well for many workloads. However, for matrix multiplication, the most efficient method on Tenstorrent's AI processors is to use a systolic array pattern, and use a subset of cores to read in and reuse the read data. This example does not cover that approach. See :ref:`MatMul_Multi_Core_example` for further optimizations, at the cost of genericity. SPMD remains a flexible and general-purpose strategy, making it suitable for a variety of tasks in Metalium.

For a matrix of size ``288 x 288`` (9 tiles along each dimension, with each tile being 32x32), and 11 cores available, the work is divided as evenly as possible. In the example case, 10 cores are assigned 8 output tiles each, and the 11th core processes the remaining tile. The diagram below shows how the output tiles are distributed among the cores. Each color corresponds to a different core, and each tile is handled by only one core:

.. figure:: /images/matmul-spmd-core-works-distribution.webp
   :alt: MatMul Multi Core Parallelization Strategy under SPMD (Each color represents a different core)

Metalium includes utilities to simplify work distribution across cores. The ``tt::tt_metal::split_work_to_cores(core_grid, num_work)`` function calculates how many tiles each core should process, based on the total amount of work and the number of available cores. It distributes the work as evenly as possible, even if the number of tiles does not divide evenly among the cores. The function returns several values:

- ``num_cores``: Number of cores used for the operation.
- ``all_cores``: Set of all cores assigned to the operation.
- ``core_group_1``: Primary group of cores, each handling more work.
- ``core_group_2``: Secondary group of cores, each handling less work (empty if the work divides evenly).
- ``work_per_core1``: Number of output tiles each core in the primary group processes.
- ``work_per_core2``: Number of output tiles each core in the secondary group processes (0 if the work divides evenly).

For example, if you need to split 81 output tiles across 11 cores, ``split_work_to_cores`` may distribute the work as follows:

* ``num_cores`` = 11 (all 11 cores are used)
* ``all_cores`` = all 11 cores
* ``core_group_1`` = first 10 cores (each processes 8 tiles)
* ``core_group_2`` = last core (processes 1 tile)
* ``work_per_core1`` = 8 (tiles per core in the primary group)
* ``work_per_core2`` = 1 (tiles for the secondary group core)

.. code-block:: cpp

    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto num_output_tiles_total = (M * N) / TILE_HW;

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_tiles_total);

.. note::

    The following properties describe the output of ``tt::tt_metal::split_work_to_cores``:

    - ``all_cores`` is the set of cores assigned work for this operation.
    - If there is not enough work, ``all_cores`` may be smaller than the total number of cores in ``core_grid``.
    - ``all_cores`` contains exactly ``num_cores`` cores.
    - ``all_cores`` is always the union of ``core_group_1`` and ``core_group_2``.
    - The total amount of work (``num_work``) is always fully assigned: ``work_per_core1 * num_cores_in_core_group_1 + work_per_core2 * num_cores_in_core_group_2 == num_work``.
    - The function automatically handles uneven work distribution; you do not need to manage edge cases manually.

.. note::

    **How Metalium Parallelism Differs from OpenCL/CUDA**

    In frameworks like OpenCL and CUDA, you typically launch many more work groups (or thread blocks) than there are physical compute units. The hardware scheduler dynamically assigns these work groups to available compute units. If a group of threads (warp/wavefront) stalls - such as waiting for memory - the scheduler can quickly switch to another ready group, keeping the hardware busy and improving overall throughput. This dynamic scheduling and oversubscription allow for automatic load balancing and efficient handling of workloads with unpredictable execution times.

    In contrast, Metalium's parallelism model is static. The number of parallel tasks you can launch is limited to the number of available Tensix cores on the device. Each core is assigned a specific portion of the work at launch, and there is no dynamic scheduling or oversubscription: once a core finishes its assigned work, it remains idle until the next task is launched. This is similar to static scheduling in OpenMP, where work is divided as evenly as possible among available threads at the start.

    As a result, when using Metalium, it is important to:

      - Carefully partition your workload so that all cores are kept busy.
      - Be aware that you cannot launch more tasks than there are cores.
      - Understand that dynamic load balancing (as in CUDA/OpenCL) is not available.

    This model offers predictable performance and is well-suited for workloads that can be evenly distributed, but it requires more attention to work distribution for optimal efficiency.

Tensor and Dataflow Buffer Allocation
-------------------------------------

Creating DRAM tensors and dataflow buffers in Metalium is similar to the single core example. The DRAM tensors were allocated above, unchanged from the single core example. For dataflow buffers, you do not pick the cores: a dataflow buffer is created on every core that runs the kernels bound to it, which here is every core in ``all_cores``.

.. code-block:: cpp

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const tt_metal::experimental::KernelSpecName reader_kernel{"reader"};
    const tt_metal::experimental::KernelSpecName writer_kernel{"writer"};
    const tt_metal::experimental::KernelSpecName compute_kernel{"compute"};
    const tt_metal::experimental::DFBSpecName src0_dfb{"src0"};
    const tt_metal::experimental::DFBSpecName src1_dfb{"src1"};
    const tt_metal::experimental::DFBSpecName output_dfb{"output"};
    const tt_metal::experimental::TensorParamName src0_tensor_param{"src0"};
    const tt_metal::experimental::TensorParamName src1_tensor_param{"src1"};
    const tt_metal::experimental::TensorParamName dst_tensor_param{"dst"};

    // A dataflow buffer is created on every core that runs its kernels (here: all cores).
    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    tt_metal::experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };

    tt_metal::experimental::DataflowBufferSpec src1_dfb_spec{
        .unique_id = src1_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };

    tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };


Partitioning Work in Kernels
----------------------------

To support work distribution, the kernel is updated so that each core processes only its assigned portion of the output. Instead of having one core handle the entire matrix, we add named runtime arguments to the kernel that specify how many tiles each core should process and the starting tile index. This way, each core computes a subset of the output tiles. Below is the writer kernel, which writes the output tiles to the DRAM tensor:

.. code-block:: cpp

    void kernel_main() {
        // Runtime arguments to write data back into the output tensor.
        uint32_t num_tiles = get_arg(args::num_tiles);  // number of output tiles to write
        uint32_t start_id = get_arg(args::start_id);    // starting tile ID for output tiles

        DataflowBuffer dfb_out(dfb::out0);

        // Create the address generator for the output tensor. Due to us sharing tensor and dataflow buffer
        // configuration parameters (e.g. same data type and same page size) in the host code, we can grab
        // the same parameters from the dataflow buffer as we would from the DRAM tensor.
        constexpr uint32_t onetile = 1;  // single-tile ublocks

        // The output tensor is bound to this kernel by name in the host code.
        const auto c = TensorAccessor(tensor::c);

        // Loop through the tile indices and write each tile to DRAM in order.
        uint32_t end_id = start_id + num_tiles;
        for (uint32_t i = start_id; i < end_id; ++i) {
            // Wait for the kernel to produce an output tile
            dfb_out.wait_front(onetile);
            // Write the output tile to DRAM.
            uint32_t l1_read_addr = dfb_out.get_read_ptr();
            noc_async_write_page(i, c, l1_read_addr);
            noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                        // noc_async_write_flushed() can be faster because it waits
                                        // until the write request is sent. In that case, you have to
                                        // use noc_async_write_barrier() at least once at the end of
                                        // data movement kernel to make sure all writes are done.
            dfb_out.pop_front(onetile);
        }
    }


The compute kernel does not handle IO directly and is not concerned with how work is distributed among the cores. It only needs to know how many tiles to compute and the size of the inner dimension. The kernel is almost identical to the single core version, except that the number of tiles to process is passed as a named runtime argument:

.. code-block:: cpp

    void kernel_main() {
        uint32_t num_output_tiles = get_arg(args::num_output_tiles);  // number of output tiles to produce
        uint32_t Kt = get_arg(args::Kt);                              // number of tiles in K dimension for dot product

        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);
        DataflowBuffer dfb_out(dfb::out0);

        // Setup the FPU (matrix engine) for the matmul operation. And specify the input
        // and output dataflow buffers.
        compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out0);
        matmul_init(dfb::in0, dfb::in1);

        // the simplest possible version of outer product blocked matmul
        // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
        for (uint32_t i = 0; i < num_output_tiles; ++i) {
            // Make sure registers can be used for the output tile. This also sets the registers to zero.
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Wait for the input tiles to be available in the input dataflow buffers.
                dfb_in0.wait_front(1);
                dfb_in1.wait_front(1);

                // Perform the matrix multiplication for the current tile.
                // NOTE: This function also accumulates the result into the destination tile.
                matmul_tiles(dfb::in0, dfb::in1, 0, 0, 0);

                // Mark the input tiles as used by popping them from the front of the dataflow buffers.
                dfb_in0.pop_front(1);
                dfb_in1.pop_front(1);
            }

            // Commit and wait for the registers are populated with the results from the FPU
            tile_regs_commit();
            tile_regs_wait();

            // Ensure the output dataflow buffer has space for the result tile.
            dfb_out.reserve_back(1);
            // Pack the result tile into the output dataflow buffer.
            pack_tile(0, dfb::out0);
            // Mark the output tile as ready so the writer can read it.
            dfb_out.push_back(1);

            // We don't need the registers anymore, so we can release them and prepare for the next output tile.
            tile_regs_release();
        }
    }

The reader kernel is responsible for reading the input data from the DRAM tensors and pushing it into the dataflow buffers. It also needs to know how many tiles to read and the starting tile index for each core. Due to needing to calculate where to start reading from the DRAM tensor, it also needs to know the exact dimensions of the input matrices (Mt, Kt, Nt). Again the reader is almost identical to the single core version, except that it reads only the assigned number of tiles and uses the starting tile index to calculate the correct offset in the DRAM tensor:

.. code-block:: cpp

    void kernel_main() {
        // Arguments are read by the names declared in the host code.
        uint32_t Mt = get_arg(args::Mt);
        uint32_t Kt = get_arg(args::Kt);
        uint32_t Nt = get_arg(args::Nt);
        uint32_t output_tile_start_id = get_arg(args::output_tile_start_id);  // starting tile ID for output tiles
        uint32_t num_output_tiles = get_arg(args::num_output_tiles);          // number of output tiles to read

        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);

        // Declare address in which we stored the source matrices. We have set the exact same format between DFBs and DRAM
        // tensors in the host code, so we can use the same address for both DRAM and DFBs.
        // The source tensors are bound to this kernel by name in the host code.

        const auto a = TensorAccessor(tensor::a);

        const auto b = TensorAccessor(tensor::b);

        // Simple 2D matmul: A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for (uint32_t output_tile = 0; output_tile < num_output_tiles; output_tile++) {
            uint32_t current_tile_id = output_tile_start_id + output_tile;

            // Convert linear output tile ID to 2D coordinates
            uint32_t out_row = current_tile_id / Nt;  // Which row in output
            uint32_t out_col = current_tile_id % Nt;  // Which col in output

            // Read all K tiles for this output position
            for (uint32_t k = 0; k < Kt; k++) {
                // Read A's tile at (out_row, k)
                uint32_t tile_A = out_row * Kt + k;  // A is MK, so we stride by Kt
                {
                    dfb_in0.reserve_back(1);
                    uint32_t l1_write_addr_in0 = dfb_in0.get_write_ptr();
                    noc_async_read_page(tile_A, a, l1_write_addr_in0);
                    noc_async_read_barrier();
                    dfb_in0.push_back(1);
                }

                // Read B's tile at (k, out_col)
                uint32_t tile_B = k * Nt + out_col;  // B is KN, so we stride by Nt
                {
                    dfb_in1.reserve_back(1);
                    uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr();
                    noc_async_read_page(tile_B, b, l1_write_addr_in1);
                    noc_async_read_barrier();
                    dfb_in1.push_back(1);
                }
            }
        }
    }



Kernel Creation and Parameter Setup
-----------------------------------

With the work distribution calculated, you can now describe the kernels and set up their parameters. Since not all cores may be used, make sure the work unit in the ``ProgramSpec`` targets only the cores listed in ``all_cores``. This avoids having idle kernels on unused cores.

.. warning::
    If a kernel runs on a core but a runtime argument is not set for that core, ``SetProgramRunArgs`` reports a validation error. Always ensure that the work unit targets only the intended cores, and that every runtime argument is set for every core in it.


.. code-block:: cpp

    tt::tt_metal::MathFidelity math_fidelity =
        tt::tt_metal::MathFidelity::HiFi4;  // High fidelity math for accurate results
    tt_metal::experimental::KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source =
            OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
        .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "a"},
             {.tensor_parameter_name = src1_tensor_param, .accessor_name = "b"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"Mt", "Kt", "Nt", "output_tile_start_id", "num_output_tiles"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    tt_metal::experimental::KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source =
            OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "c"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    tt_metal::experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/compute/mm.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
             tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
             tt_metal::experimental::ProducerOf(output_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_output_tiles", "Kt"}},
        .hw_config =
            tt_metal::experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = math_fidelity,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
    // operate on, and the cores the kernels run on (the work units).
    tt_metal::experimental::ProgramSpec program_spec{
        .name = "matmul_multi_core",
        .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = src0_tensor_spec},
             {.unique_id = src1_tensor_param, .spec = src1_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
        .work_units =
            {{.name = "matmul_multi_core",
              .kernels = {reader_kernel, writer_kernel, compute_kernel},
              .target_nodes = all_cores}},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);

Unlike OpenCL or CUDA, Metalium does not provide built-in parameters for work distribution on the device. You need to manually set the runtime arguments for each core. This is done by iterating through the work groups and adding the correct named arguments for each core with ``AddRuntimeArgsForNode``, including tile counts and the amount of work assigned. The DRAM tensors are passed once for all cores as tensor arguments in ``ProgramRunArgs``, so no buffer addresses are passed per core.

.. code-block:: cpp

    tt_metal::experimental::KernelRunArgs reader_run_args{.kernel = reader_kernel};
    tt_metal::experimental::KernelRunArgs writer_run_args{.kernel = writer_kernel};
    tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = compute_kernel};
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // Set arguments for the reader kernel (data input)
                tt_metal::experimental::AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"Mt", Mt},                                // Number of tiles in M dimension
                     {"Kt", Kt},                                // Number of tiles in K dimension
                     {"Nt", Nt},                                // Number of tiles in N dimension
                     {"output_tile_start_id", work_offset},     // Starting offset for this core's work
                     {"num_output_tiles", work_per_core}});     // Amount of work for this core

                // Set arguments for the writer kernel (data output)
                tt_metal::experimental::AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"num_tiles", work_per_core}, {"start_id", work_offset}});

                // Set arguments for the compute kernel
                tt_metal::experimental::AddRuntimeArgsForNode(
                    compute_run_args.runtime_arg_values,
                    core,
                    {{"num_output_tiles", work_per_core},  // Amount of work for this core
                     {"Kt", Kt}});                         // Number of tiles in K dimension for dot product
                work_offset += work_per_core;              // Update offset for next core
            }
        }
    }

    // The matrices A, B and C are passed to the kernels through the tensors bound to the tensor parameters.
    tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args = {reader_run_args, writer_run_args, compute_run_args},
        .tensor_args =
            {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
    };
    tt_metal::experimental::SetProgramRunArgs(program, run_args);

Program Execution, Receiving Results and Cleanup
------------------------------------------------

This part is the same as in the single core example. You execute the program, wait for it to finish, and then download the results from the DRAM tensor. The cleanup process is also unchanged.

See :ref:`Kernel execution and result verification in the single core matrix multiplication<mm_single_core_kernel_execution>` in the single core matrix multiplication example for details on how program execution, downloading results, untilize, verification, and cleanup are performed. There is no change in the API usage for these steps compared to the single core example.


.. code-block:: cpp

    HostTensor src0_host_tensor = HostTensor::from_vector(a, src0_tensor_spec);
    HostTensor src1_host_tensor = HostTensor::from_vector(b, src1_tensor_spec);
    cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    // Blocking read waits for completion before returning and resizes 'output' as needed
    output = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

    // Reverse the tilization to get the result in the row-major format that the CPU expects
    result_vec = untilize_nfaces(result_vec, M, N);


    float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

    pass &= mesh_device->close();

Conclusion
----------

This concludes the multi-core matmul example and the basic usage of the Metalium API to distribute work across multiple cores. The key changes compared to the single core example are:

* Work distribution calculations using the ``tt::tt_metal::split_work_to_cores`` function
* Allocate dataflow buffers across all cores that will be used in the operation
* Set named runtime arguments for each core to specify how many tiles to process and the starting tile index
* Adjust the kernels to process only the assigned number of tiles and use the starting tile index for reading/writing data
* Target the work unit at the cores that will be used in the operation and handle edge cases like uneven work distribution or fewer cores than work

Explore :ref:`MatMul_Multi_Core_example` for further optimizations, including data reuse and data multicast to truly harness the power of the Tenstorrent architecture.

References
----------

For those interested in learning more about parallel programming concepts, we recommend the following resources:

- `Intel OpenMP Tutorial <https://www.youtube.com/playlist?list=PLLbPZJxtMs4ZHSamRRYCtvowRS0qIwC-I>`_ — A comprehensive YouTube series covering OpenMP as well as fundamental parallel programming concepts.

- `A "Hands-On" Introduction to OpenMP <https://www.openmp.org/wp-content/uploads/omp-hands-on-SC08.pdf>`_ — A detailed PDF guide that provides a practical introduction to OpenMP, which is a widely used API for parallel programming in C/C++ and Fortran.
