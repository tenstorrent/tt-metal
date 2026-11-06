.. _MatMul_Single_Core example:

Matmul (Single Core)
====================

Now that we have a basic understanding of how to use the TT Metal API and building data movement and compute kernels, we can look at a more complex example of matrix multiplication. This will be the first non-trivial example of a program that involves complex data movement and compute operations working together. The matrix multiplication will be performed on a single Tensix core using the FPU (Matrix Engine).

This example introduces the concept of using separate data movement and compute kernels that communicate through dataflow buffers. The compute kernel uses the powerful matrix engine to perform efficient tile-wise matrix multiplication, while data movement kernels handle reading input data from DRAM and writing results back.

We'll go through this code section by section. The full source code for this example is available under the ``tt_metal/programming_examples/matmul/matmul_single_core/`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``matmul_single_core`` executable in the ``build/metal_example_programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_matmul_single_core

.. _mm_single_core_device_initialization:

Device Initialization & Program Setup
-------------------------------------

After standard device and command queue initialization, the matrix dimensions M, K, and N are translated into tile-based dimensions ``Mt``, ``Kt``, and ``Nt``. This is essential as the hardware operates on 32×32 tiles. For this example, all operations are mapped to a single Tensix core at coordinates ``{0, 0}``.

.. code-block:: cpp

    // Open device
    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // parameters for the matrix multiplication
    constexpr uint32_t M = 640;  // user-defined
    constexpr uint32_t N = 640;  // user-defined
    constexpr uint32_t K = 640;  // user-defined

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    tt_metal::experimental::NodeCoord core({0, 0});

    // Calcaulate the number of tiles for each dimension.
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;


Data Preparation and Golden Reference
-------------------------------------

Before touching the matrix multiplication on the device, the host program performs several important steps:

1.  **Input Data Generation**: Two input vectors, `src0_vec` (for matrix A) and `src1_vec` (for matrix B), are populated with random `bfloat16` values

    .. code-block:: cpp

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);

        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

2.  **Golden Reference Calculation**: A reference implementation of matrix multiplication, `golden_matmul`, is executed on the CPU. This produces a `golden_vec` which serves as the ground truth for verifying the correctness of the accelerator's output.

    .. code-block:: cpp

        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

3.  **Data Tilization**: The input vectors, which are initially in a row-major format, are converted into a tiled layout using the `tilize_nfaces` function. This is a necessary step because the Tenstorrent hardware operates on data in 32x32 tiles.

    .. code-block:: cpp

        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

After the device computation, the output data needs to be converted back to a standard format for verification:

DRAM Tensor Allocation
----------------------

Three DRAM tensors are allocated: ``src0_dram_tensor`` for the M×K input matrix A, ``src1_dram_tensor`` for the K×N input matrix B, and ``dst_dram_tensor`` for the M×N output matrix C. Each ``TensorSpec`` is row-major with one tile's worth of elements per row, so the page size is ``single_tile_size`` (the size of one 32x32 bfloat16 tile), a common practice for tile-based processing.

.. code-block:: cpp

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    uint32_t elements_per_tile = single_tile_size / sizeof(bfloat16);

    auto make_dram_tensor_spec = [&](size_t num_elements) {
        return TensorSpec(
            Shape({static_cast<uint32_t>(num_elements / elements_per_tile), elements_per_tile}),
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    };
    TensorSpec src0_tensor_spec = make_dram_tensor_spec(a.size());
    TensorSpec src1_tensor_spec = make_dram_tensor_spec(b.size());
    TensorSpec dst_tensor_spec = make_dram_tensor_spec(output.size());

    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src0_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src1_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_tensor_spec);

Dataflow Buffer Orchestration for Pipelined MatMul
--------------------------------------------------

Three dataflow buffers (DFBs) are established to manage the data pipeline between kernels:

*   ``src0_dfb``: Holds tiles of matrix A, produced by the reader kernel and consumed by the compute kernel.
*   ``src1_dfb``: Holds tiles of matrix B, also produced by the reader and consumed by the compute kernel.
*   ``output_dfb``: Holds resulting tiles of matrix C, produced by the compute kernel and consumed by the writer kernel.

Each DFB is identified by a name in the program spec, as are the kernels and the tensor parameters. Each kernel binds a DFB under a local accessor name (for example ``in0``), and the kernel code uses that name as ``dfb::in0``.

Each DFB is configured with ``num_input_tiles = 2`` or ``num_output_tiles = 2``. This implements double buffering, allowing data movement (e.g., the reader kernel fetching the next set of A and B tiles) to overlap with computation (the compute kernel processing the current set). A higher number of tiles can be used for more complex scenarios, reducing bottlenecks in complex data movement patterns. At the cost of increased memory usage and diminishing returns, this can be used to increase performance.

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

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
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

    uint32_t num_output_tiles = 2;
    tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = cb_data_format,
    };

Matmul Kernel Pipeline Breakdown
--------------------------------

The matrix multiplication is performed by a pipeline of three specialized kernels. Each kernel is a ``KernelSpec``: its ``dfb_bindings`` and ``tensor_bindings`` give the names the kernel code uses, and its ``runtime_arg_schema`` declares the names of its runtime arguments. The ``ProgramSpec`` collects the kernels, the DFBs, the tensor parameters and the work unit (the cores the kernels run on), and ``MakeProgramFromSpec`` builds the ``Program``:

.. code-block:: cpp

    // Describe the data movement kernels and the compute kernel
    tt_metal::experimental::KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
             {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Kt", "Nt"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    tt_metal::experimental::KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Nt"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    tt_metal::experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
             tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
             tt_metal::experimental::ProducerOf(output_dfb, "out0")},
        .compile_time_args = {{"Mt", Mt}, {"Kt", Kt}, {"Nt", Nt}},
        .hw_config =
            tt_metal::experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = math_fidelity,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
    // operate on, and the cores the kernels run on (the work units).
    tt_metal::experimental::ProgramSpec program_spec{
        .name = "matmul_single_core",
        .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = src0_tensor_spec},
             {.unique_id = src1_tensor_param, .spec = src1_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
        .work_units =
            {{.name = "matmul_single_core",
              .kernels = {reader_kernel, writer_kernel, compute_kernel},
              .target_nodes = core}},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);

The reader kernel
^^^^^^^^^^^^^^^^^

The reader kernel is responsible for fetching tiles from the DRAM tensors for matrices A and B and pushing them into the ``src0`` and ``src1`` dataflow buffers (``dfb::in0`` and ``dfb::in1`` in the kernel), respectively. The crucial aspect of this kernel is the *order* in which tiles are read. The nested loop structure (``for mt, for nt, for kt``) ensures that tiles are provided to the compute kernel in the sequence required by the matrix multiplication algorithm implemented in the compute kernel.

The tile indexing logic:

*   For matrix A (M×K, or Mt×Kt tiles): ``a_tile_index = mt * Kt + kt``
*   For matrix B (K×N, or Kt×Nt tiles): ``b_tile_index = kt * Nt + nt``

maps tiles in the row-major order of the matrices in DRAM to read into the dataflow buffers. This ensures that the compute kernel receives tiles in the correct order for multiplication.

.. code-block:: cpp

    // tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp
    void kernel_main() {
        // Arguments are read by the names declared in the host code.
        uint32_t Mt = get_arg(args::Mt);
        uint32_t Kt = get_arg(args::Kt);
        uint32_t Nt = get_arg(args::Nt);

        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);

        // Declare address in which we stored the source matrices. We have set the exact same format between DFBs and DRAM
        // tensors in the host code, so we can use the same address for both DRAM and DFBs.
        // The source tensors are bound to this kernel by name in the host code.
        const auto s0 = TensorAccessor(tensor::in0);
        const auto s1 = TensorAccessor(tensor::in1);

        // Loop through the dimensions of the matrices. Read them and push to the dataflow buffers.
        // Dimension names are called M, N and K. `t` in `mt` means tile.
        for (uint32_t mt = 0; mt < Mt; mt++) {
            uint32_t itileB = 0;
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    {                                          // Read A's tile at (mt, kt)
                        uint32_t a_tile_index = mt * Kt + kt;  // A is MK, so we stride by Kt
                        dfb_in0.reserve_back(1);
                        uint32_t l1_write_addr_in0 = dfb_in0.get_write_ptr();
                        noc_async_read_page(a_tile_index, s0, l1_write_addr_in0);
                        noc_async_read_barrier();
                        dfb_in0.push_back(1);
                    }

                    {                                          // Read B's tile at (kt, nt)
                        uint32_t b_tile_index = kt * Nt + nt;  // B is KN, so we stride by Nt
                        dfb_in1.reserve_back(1);
                        uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr();
                        noc_async_read_page(b_tile_index, s1, l1_write_addr_in1);
                        noc_async_read_barrier();
                        dfb_in1.push_back(1);
                    }
                }  // Kt loop
            }  // Nt loop
        }  // Mt loop
    }

The compute kernel
^^^^^^^^^^^^^^^^^^

This kernel performs the tile-by-tile matrix multiplication ``C_tile += A_tile @ B_tile``.
Key operations include:

*   ``compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out0)`` then ``matmul_init(dfb::in0, dfb::in1)``: Performs the one-time hardware setup and configures the FPU for matrix multiplication, specifying the input DFBs (``dfb::in0`` for A, ``dfb::in1`` for B) and the output DFB (``dfb::out0``). Matmul maps ``in0`` onto ``SrcB`` and ``in1`` onto ``SrcA``, so the startup uses ``SrcOrder::Reverse``.
*   The outer loops iterate ``Mt`` times (for rows of C) and ``Nt`` times (for columns of C) to compute each output tile.
*   ``tile_regs_acquire()``: Called before the inner accumulation loop (over ``Kt``). This prepares the FPU's destination/accumulator registers, typically by zeroing them, for the upcoming sum of products.
*   The inner loop iterates ``Kt`` times, performing the dot-product-like accumulation for a single output tile.
*   ``matmul_tiles(dfb::in0, dfb::in1, 0, 0, 0)``: Executes the core FPU instruction: multiplies a tile from ``dfb::in0`` with a tile from ``dfb::in1`` and adds the result to the accumulator.
*   ``tile_regs_commit()`` and ``tile_regs_wait()``: After the inner loop, these functions ensure that the FPU has finished computation and result available in the destination registers.
*   ``dfb_in0.pop_front(1); dfb_in1.pop_front(1);``: After the tiles are used by ``matmul_tiles``, they are marked as consumed by popping them from the input DFBs.
*   ``pack_tile(0, dfb::out0); dfb_out.push_back(1);``: Once the ``Kt`` loop completes for an output tile, the accumulated result in the FPU registers is packed and pushed to the output dataflow buffer ``dfb::out0``.

The dimensions ``Mt``, ``Kt``, ``Nt`` are passed as named compile-time arguments (read with ``get_arg(args::Mt)`` and so on), enabling the compiler to optimize the kernel structure for these specific dimensions.

.. code-block:: cpp

    // tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp
    void kernel_main() {
        const uint32_t Mt = get_arg(args::Mt);
        const uint32_t Kt = get_arg(args::Kt);
        const uint32_t Nt = get_arg(args::Nt);
        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);
        DataflowBuffer dfb_out(dfb::out0);

        // Setup the FPU (matrix engine) for the matmul operation. And specify the input
        // and output dataflow buffers.
        compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out0);
        matmul_init(dfb::in0, dfb::in1);

        // the simplest possible version of outer product blocked matmul
        // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
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
    }

The writer kernel
^^^^^^^^^^^^^^^^^

The writer kernel consumes tiles from the output dataflow buffer ``dfb::out0`` (which is ``output_dfb`` in the host code) and writes them to the designated DRAM tensor for matrix C. The nested loops iterate ``Mt`` and ``Nt`` times, and the tile index ``m * Nt + n`` ensures that the output tiles are written in row-major order, correctly forming the M×N output matrix in DRAM.

.. code-block:: cpp

    // tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp

    void kernel_main() {
        // Runtime arguments to write data back into the output tensor.
        uint32_t Mt = get_arg(args::Mt);
        uint32_t Nt = get_arg(args::Nt);

        DataflowBuffer dfb_out0(dfb::out0);

        // Create the address generator for the output tensor. Due to us sharing tensor and dataflow buffer
        // configuration parameters (e.g. same data type and same page size) in the host code, we can grab
        // the same parameters from the dataflow buffer as we would from the DRAM tensor.
        // The output tensor is bound to this kernel by name in the host code.
        const auto s = TensorAccessor(tensor::out0);

        // Loop through the matrix dimensions Mt and Nt. bmm will generate C's tiles C=A*B, MN=MK*KN,
        // in row major order, we just read them from the DFB and write out to DRAM
        for (uint32_t m = 0; m < Mt; ++m) {
            for (uint32_t n = 0; n < Nt; ++n) {
                // Wait for the matrix multiplication kernel to produce an output
                dfb_out0.wait_front(1);
                // Write the output tile to DRAM.
                uint32_t l1_read_addr = dfb_out0.get_read_ptr();
                noc_async_write_page(m * Nt + n, s, l1_read_addr);
                noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                            // noc_async_write_flushed() can be faster because it waits
                                            // until the write request is sent. In that case, you have to
                                            // use noc_async_write_barrier() at least once at the end of
                                            // data movement kernel to make sure all writes are done.
                dfb_out0.pop_front(1);
            }
        }
    }

.. _mm_single_core_kernel_execution:

Kernel execution and result verification
----------------------------------------

On the host side, named runtime arguments are set for each kernel per node in ``ProgramRunArgs``. These hold the tile counts (``Mt``, ``Kt``, ``Nt``) that define the scope of the operation for the current invocation. The DRAM tensors (for A, B, and C) are passed as tensor arguments bound to the tensor parameters, so the kernels do not receive buffer addresses as runtime arguments.
The overall execution flow is managed by enqueuing commands:

1.  ``enqueue_write_tensor``: Transfers input matrices A and B from host tensors to their respective DRAM tensors on the Mesh Device (a 1x1 Mesh Device in this example).
2.  ``EnqueueMeshWorkload``: Launches the compiled workload (reader, compute, and writer kernels) on the designated core on the Unit Mesh Device.
3.  ``enqueue_read_tensor``: Transfers the resulting matrix C from its DRAM tensor on the device back to host memory.

.. code-block:: cpp

    // Set kernel arguments, and the tensors bound to the tensor parameters.
    tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = reader_kernel,
              .runtime_arg_values =
                  tt_metal::experimental::MakeRuntimeArgsForSingleNode(core, {{"Mt", Mt}, {"Kt", Kt}, {"Nt", Nt}})},
             {.kernel = writer_kernel,
              .runtime_arg_values = tt_metal::experimental::MakeRuntimeArgsForSingleNode(core, {{"Mt", Mt}, {"Nt", Nt}})}},
        .tensor_args =
            {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
    };
    tt_metal::experimental::SetProgramRunArgs(program, run_args);
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Upload the input data to the DRAM tensors, execute the kernels, wait for the result to be read into the output
    // buffer
    HostTensor src0_host_tensor = HostTensor::from_vector(a, src0_tensor_spec);
    HostTensor src1_host_tensor = HostTensor::from_vector(b, src1_tensor_spec);
    cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    output = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

After the program execution, ``output`` (which is ``result_vec`` in the ``main`` function of the C++ example) contains the result matrix C from the device's DRAM. However, this data is still in the tiled format used by the Tenstorrent hardware. To verify its correctness against the ``golden_vec`` (which is in a standard row-major format), two steps are necessary:

    pass &= mesh_device->close();

1.  **Data Untilization**: The `untilize_nfaces` function is used to convert the tiled output data back into a row-major format. This is the inverse operation of ``tilize_nfaces`` performed on the input data.

    .. code-block:: cpp

        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, M, N);

2.  **Verification against Golden Reference**: The untilized ``result_vec`` is then compared against the ``golden_vec`` computed by the CPU. A common method for comparing floating-point vectors is to calculate the Pearson correlation coefficient (PCC). A PCC value close to 1.0 indicates a high degree of similarity between the two vectors, confirming the correctness of the accelerator's computation.

    .. code-block:: cpp

        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

Conclusion
----------

This single-core matrix multiplication example highlights several key architectural patterns for programming Tenstorrent devices:

* **Separation of data movement and compute**: By using dedicated RISC-V processors for data movement (reader/writer kernels) and the matrix engine for computation, complex data orchestration patterns do not sacrifice compute throughput. The data movement processors can handle complex access patterns while the compute units remain fully utilized.
* **Tiled operations**: The hardware is optimized for tiled operations, making tile-based algorithms essential for achieving peak performance. All matrices are processed in tile units, matching the natural granularity of the underlying hardware accelerators.
* **Pipelined data movement**: The dataflow buffer architecture with double buffering enables overlapped execution - while the compute kernel processes current tiles, the data movement kernels can simultaneously fetch the next set of tiles. This pipelining ensures efficient utilization of compute resources by minimizing idle time.

Next we will explore the :ref:`MatMul_Multi_Core example <MatMul_Multi_Core example>`, which extends these concepts to a multi-core setup, demonstrating how to scale matrix multiplication across multiple Tensix cores for even greater performance.
