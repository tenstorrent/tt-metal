.. _MatMul_Single_Core example:

Matmul (Single Core)
====================

Now that we have a basic understanding of how to use the TT Metal API and building data movement and compute kernels, we can look at a more complex example of matrix multiplication. This will be the first non-trivial example of a program that involves complex data movement and compute operations working together. The matrix multiplication will be performed on a single Tensix core using the FPU (Matrix Engine).

This example introduces the concept of using separate data movement and compute kernels that communicate through circular buffers. The compute kernel uses the powerful matrix engine to perform efficient tile-wise matrix multiplication, while data movement kernels handle reading input data from DRAM and writing results back.

We'll go through this code section by section. The full source code for this example is available under the ``tt_metal/programming_examples/matmul/matmul_single_core/`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``matmul_single_core`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/matmul_single_core

.. _mm_single_core_device_initialization:

Device Initialization & Program Setup
-------------------------------------

After standard device and command queue initialization, the matrix dimensions M, K, and N are translated into tile-based dimensions ``Mt``, ``Kt``, and ``Nt``. This is essential as the hardware operates on 32×32 tiles. For this example, all operations are mapped to a single Tensix core at physical coordinates ``{0, 0}``.

.. code-block:: cpp

    // Open device (we use device 0, the first available device)
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    // Matrix dimensions (must be divisible by tile dimensions)
    constexpr uint32_t M = 640;  // Matrix A height
    constexpr uint32_t N = 640;  // Matrix B width
    constexpr uint32_t K = 640;  // Shared dimension

    // Calculate number of tiles in each dimension
    uint32_t Mt = M / TILE_HEIGHT;  // Each tile is 32x32
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

.. code-block:: cpp

    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreCoord core({0, 0});  // Single core at position {0, 0}


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

DRAM Buffer Allocation
----------------------

Three DRAM buffers are allocated: ``src0_dram_buffer`` for the M×K input matrix A, ``src1_dram_buffer`` for the K×N input matrix B, and ``dst_dram_buffer`` for the M×N output matrix C. The configuration for these buffers sets ``page_size`` to ``single_tile_size`` (the size of one 32x32 bfloat16 tile), a common practice for tile-based processing.

.. code-block:: cpp

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Buffer for matrix A (M×K)
    tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = sizeof(bfloat16) * a.size(),
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Buffer for matrix B (K×N)
    tt_metal::InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = sizeof(bfloat16) * b.size(),
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Buffer for output matrix C (M×N)
    tt_metal::InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = sizeof(bfloat16) * output.size(),
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config_B);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_C);

Circular Buffer Orchestration for Pipelined MatMul
--------------------------------------------------

Three circular buffers (CBs) are established to manage the data pipeline between kernels:

*   ``cb_src0`` (CB index 0): Holds tiles of matrix A, produced by the reader kernel and consumed by the compute kernel.
*   ``cb_src1`` (CB index 1): Holds tiles of matrix B, also produced by the reader and consumed by the compute kernel.
*   ``cb_output`` (CB index 16): Holds resulting tiles of matrix C, produced by the compute kernel and consumed by the writer kernel.

Each CB is configured with ``num_input_tiles = 2`` or ``num_output_tiles = 2``. This implements double buffering, allowing data movement (e.g., the reader kernel fetching the next set of A and B tiles) to overlap with computation (the compute kernel processing the current set). A higher number of tiles can be used for more complex scenarios, reducing bottlenecks in complex data movement patterns. At the cost of increased memory usage and diminishing returns, this can be used to increase performance.

.. code-block:: cpp

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;   // Double buffering for performance
    uint32_t num_output_tiles = 2;

    // Circular buffer for matrix A tiles
    uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Circular buffer for matrix B tiles
    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    // Circular buffer for output tiles
    uint32_t output_cb_index = tt::CBIndex::c_16;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

Matmul Kernel Pipeline Breakdown
--------------------------------

The matrix multiplication is performed by a pipeline of three specialized kernels:

.. code-block:: cpp

    // Reader kernel - reads tiles from DRAM into circular buffers
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Writer kernel - writes result tiles from circular buffer to DRAM
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Compute kernel - performs matrix multiplication using the matrix engine
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    std::vector<uint32_t> compute_compile_time_args = {Mt, Kt, Nt};
    auto matmul_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_single_core/kernels/compute/mm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args});

The reader kernel
^^^^^^^^^^^^^^^^^

The reader kernel is responsible for fetching tiles from the DRAM buffers for matrices A and B and pushing them into ``cb_src0`` and ``cb_src1``, respectively. The crucial aspect of this kernel is the *order* in which tiles are read. The nested loop structure (``for mt, for nt, for kt``) ensures that tiles are provided to the compute kernel in the sequence required by the matrix multiplication algorithm implemented in the compute kernel.

The tile indexing logic:

*   For matrix A (M×K, or Mt×Kt tiles): ``a_tile_index = mt * Kt + kt``
*   For matrix B (K×N, or Kt×Nt tiles): ``b_tile_index = kt * Nt + nt``

maps tiles in the row-major order of the matrices in DRAM to read into the circular buffers. This ensures that the compute kernel receives tiles in the correct order for multiplication.

.. code-block:: cpp

    // tt_metal/programming_examples/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp
    void kernel_main() {
        // same arg indices as in reader_binary_diff_lenghts for compat
        uint32_t src0_addr = get_arg_val<uint32_t>(0);
        uint32_t src1_addr = get_arg_val<uint32_t>(1);
        uint32_t Mt = get_arg_val<uint32_t>(2);
        uint32_t Kt = get_arg_val<uint32_t>(3);
        uint32_t Nt = get_arg_val<uint32_t>(4);

        constexpr uint32_t cb_id_in0 = 0;
        constexpr uint32_t cb_id_in1 = 1;

        // Declare address in which we stored the source matrices. We have set the exact same format between CBs and DRAM
        // buffers in the host code, so we can use the same address for both DRAM and CBs.
        const InterleavedAddrGenFast<true> s0 = {
            .bank_base_address = src0_addr,
            .page_size = get_tile_size(cb_id_in0),
            .data_format = get_dataformat(cb_id_in0)};
        const InterleavedAddrGenFast<true> s1 = {
            .bank_base_address = src1_addr,
            .page_size = get_tile_size(cb_id_in1),
            .data_format = get_dataformat(cb_id_in1)};

        // Loop through the dimensions of the matrices. Read them and push to the circular buffers.
        // Dimension names are called M, N and K. `t` in `mt` means tile.
        for (uint32_t mt = 0; mt < Mt; mt++) {
            uint32_t itileB = 0;
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    {                                          // Read A's tile at (mt, kt)
                        uint32_t a_tile_index = mt * Kt + kt;  // A is MK, so we stride by Kt
                        cb_reserve_back(cb_id_in0, 1);
                        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                        noc_async_read_tile(a_tile_index, s0, l1_write_addr_in0);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in0, 1);
                    }

                    {                                          // Read B's tile at (kt, nt)
                        uint32_t b_tile_index = kt * Nt + nt;  // B is KN, so we stride by Nt
                        cb_reserve_back(cb_id_in1, 1);
                        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                        noc_async_read_tile(b_tile_index, s1, l1_write_addr_in1);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in1, 1);
                    }
                }  // Kt loop
            }  // Nt loop
        }  // Mt loop
    }

The compute kernel
^^^^^^^^^^^^^^^^^^

This kernel performs the tile-by-tile matrix multiplication ``C_tile += A_tile @ B_tile``.
Key operations include:

*   ``mm_init(cb_in0, cb_in1, cb_out)``: Initializes the FPU for matrix multiplication, specifying the input CBs (``cb_in0`` for A, ``cb_in1`` for B) and the output CB (``cb_out``).
*   The outer loops iterate ``Mt`` times (for rows of C) and ``Nt`` times (for columns of C) to compute each output tile.
*   ``acquire_dst()``: Called before the inner accumulation loop (over ``Kt``). This prepares the FPU's destination/accumulator registers, typically by zeroing them, for the upcoming sum of products.
*   The inner loop iterates ``Kt`` times, performing the dot-product-like accumulation for a single output tile.
*   ``matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false)``: Executes the core FPU instruction: multiplies a tile from ``cb_in0`` with a tile from ``cb_in1`` and adds the result to the accumulator.
*   ``cb_pop_front(cb_in0, 1); cb_pop_front(cb_in1, 1);``: After the tiles are used by ``matmul_tiles``, they are marked as consumed by popping them from the input CBs.
*   ``pack_tile(0, cb_out); cb_push_back(cb_out, 1);``: Once the ``Kt`` loop completes for an output tile, the accumulated result in the FPU registers is packed and pushed to the output circular buffer ``cb_out``.

The dimensions ``Mt``, ``Kt``, ``Nt`` are passed as compile-time arguments, enabling the compiler to optimize the kernel structure for these specific dimensions.

.. code-block:: cpp

    // tt_metal/programming_examples/matmul_single_core/kernels/compute/mm.cpp
    namespace NAMESPACE {
    void MAIN {
        const uint32_t Mt = get_compile_time_arg_val(0);
        const uint32_t Kt = get_compile_time_arg_val(1);
        const uint32_t Nt = get_compile_time_arg_val(2);
        constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
        constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
        constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

        // Setup the FPU (matrix engine) for the matmul operation
        mm_init(cb_in0, cb_in1, cb_out);
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                // Make sure registers can be used for the output tile. This also sets the registers to zero.
                acquire_dst();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // Wait for the input tiles to be available in the input circular buffers.
                    cb_wait_front(cb_in0, 1);
                    cb_wait_front(cb_in1, 1);

                    // Perform the matrix multiplication for the current tile.
                    // NOTE: This function also accumulates the result into the destination tile.
                    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);
                    cb_pop_front(cb_in0,1);
                    cb_pop_front(cb_in1,1);
                }

                // store the result tile in the output circular buffer.
                cb_reserve_back(cb_out, 1);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, 1);

                release_dst();
            }
        }
    }
    }

The writer kernel
^^^^^^^^^^^^^^^^^

The writer kernel consumes tiles from the output circular buffer ``cb_id_out0`` (which is ``cb_output``, index 16) and writes them to the designated DRAM buffer for matrix C. The nested loops iterate ``Mt`` and ``Nt`` times, and the tile index ``m * Nt + n`` ensures that the output tiles are written in row-major order, correctly forming the M×N output matrix in DRAM.

.. code-block:: cpp

    // tt_metal/programming_examples/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp

    void kernel_main() {
        // Runtime arguments to write data back into the output buffer.
        uint32_t dst_addr = get_arg_val<uint32_t>(0);
        uint32_t Mt = get_arg_val<uint32_t>(1);
        uint32_t Nt = get_arg_val<uint32_t>(2);

        constexpr uint32_t cb_id_out0 = 16;


        const InterleavedAddrGenFast<true> s = {
            .bank_base_address = dst_addr,
            .page_size = get_tile_size(cb_id_out0),
            .data_format = get_dataformat(cb_id_out0)};

        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                // Wait for the matrix multiplication kernel to produce an output
                cb_wait_front(cb_id_out0, 1);
                // Write the output tile to DRAM.
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                noc_async_write_tile(mt * Nt + nt, s, l1_read_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, 1);
            }
        }
    }

.. _mm_single_core_kernel_execution:

Kernel execution and result verification
----------------------------------------

On the host side, runtime arguments are configured for each kernel. These typically include DRAM buffer addresses (for A, B, and C) and tile counts (``Mt``, ``Kt``, ``Nt``) that define the scope of the operation for the current invocation.
The overall execution flow is managed by enqueuing commands:

1.  ``EnqueueWriteBuffer``: Transfers input matrices A and B from host memory to their respective DRAM buffers on the device.
2.  ``EnqueueProgram``: Launches the compiled program (reader, compute, and writer kernels) on the designated core.
3.  ``EnqueueReadBuffer``: Transfers the resulting matrix C from its DRAM buffer on the device back to host memory.

.. code-block:: cpp

    // Set runtime arguments for kernels
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Mt, Kt, Nt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt, Nt}); // Note: Writer kernel uses Mt, Nt for output C
    // Don't need to set runtime args for compute kernel, as everything is passed as compile-time args

    // Upload input data, execute program, and read results
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);

After the program execution, the ``output.data()`` (which is ``result_vec`` in the ``main`` function of the C++ example) contains the result matrix C from the device's DRAM. However, this data is still in the tiled format used by the Tenstorrent hardware. To verify its correctness against the ``golden_vec`` (which is in a standard row-major format), two steps are necessary:

1.  **Data Untilization**: The `untilize_nfaces` function is used to convert the tiled output data back into a row-major format. This is the inverse operation of ``tilize_nfaces`` performed on the input data.

    .. code-block:: cpp

        // Reverse the tilization to get the result in the row-major format
        result_vec = untilize_nfaces(result_vec, M, N);

2.  **Verification against Golden Reference**: The untilized ``result_vec`` is then compared against the ``golden_vec`` computed by the CPU. A common method for comparing floating-point vectors is to calculate the Pearson correlation coefficient (PCC). A PCC value close to 1.0 indicates a high degree of similarity between the two vectors, confirming the correctness of the accelerator's computation.

    .. code-block:: cpp

        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

Conclusion
----------

This single-core matrix multiplication example highlights several key architectural patterns for programming Tenstorrent devices:

* **Separation of data movement and compute**: By using dedicated RISC-V processors for data movement (reader/writer kernels) and the matrix engine for computation, complex data orchestration patterns do not sacrifice compute throughput. The data movement processors can handle complex access patterns while the compute units remain fully utilized.
* **Tiled operations**: The hardware is optimized for tiled operations, making tile-based algorithms essential for achieving peak performance. All matrices are processed in tile units, matching the natural granularity of the underlying hardware accelerators.
* **Pipelined data movement**: The circular buffer architecture with double buffering enables overlapped execution - while the compute kernel processes current tiles, the data movement kernels can simultaneously fetch the next set of tiles. This pipelining ensures efficient utilization of compute resources by minimizing idle time.

Next we will explore the :ref:`MatMul_Multi_Core example <MatMul_Multi_Core example>`, which extends these concepts to a multi-core setup, demonstrating how to scale matrix multiplication across multiple Tensix cores for even greater performance.
