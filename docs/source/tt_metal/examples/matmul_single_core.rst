.. _MatMul_Single_Core example:

Matmul (Single Core)
=====================

We'll build a program that will perform matmul operations on two tensors
with equal-size inner dimension.

The full example program is in
``tt_metal/programming_examples/matmul_single_core/matmul_single_core.cpp``


Host Code
----------------
- Create Device
- Set input and output vector variables, using the user-defined parameters (M, N, K, B)
- Tilizing the input vector, and untilizing the device output to vector (row-major layout)
- Call matmul_single_core() program and retrieve output results (details in next section)
- Validate the device compuation results vs. golden results on cpu
- Close Device

    .. code-block:: cpp

        /* Create source data */
        constexpr uint32_t M = 640;  // user-defined
        constexpr uint32_t N = 640;  // user-defined
        constexpr uint32_t K = 640;  // user-defined
        constexpr uint32_t B = 1;  // user-defined
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;
        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B

        /* input vectors */std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_A_size, 1, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(
            dram_buffer_B_size, 1, std::chrono::system_clock::now().time_since_epoch().count());

        /* Calling the MatMul host program. Read in result into a host vector */
        std::vector<uint32_t> tilized_src0_vec = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(src0_vec), M, K));
        std::vector<uint32_t> tilized_src1_vec = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(src1_vec), K, N));

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<uint32_t> result_vec;
        matmul_single_core(tilized_src0_vec, tilized_src1_vec, result_vec, false, M, N, K, B, device);
        vector<uint32_t> result_vec_untilized = pack_bfloat16_vec_into_uint32_vec(untilize(unpack_uint32_vec_into_bfloat16_vec(result_vec), M, N));

        CloseDevice(device);

Keeping all code details inside matmul_single_core(), allowing for calling consecutive functions in the main function

matmul_single_core function details
-----------------------------------
- Program and core range
- Create DRAM buffers based on input and output vectors
- Create L1 Circular buffers
- Kernels declarations and related compile and runtime arguments
- Program launch and reading data from DRAM output buffer to result vector


Create Program, Enqueue initialization, and core range definition
-----------------------------------------------------------------
    .. code-block:: cpp
        CommandQueue& cq = *detail::GLOBAL_CQ;
        Program program{};
        CoreRange core = {.start={0, 0}, .end={0, 0}};


Create DRAM buffers & Circular buffers
--------------------------------------

In terms of DRAM buffers, We need two source buffers and one destination buffer.
Writing data from input vectors to source buffers
    .. code-block:: cpp

        // MN = MK*KN
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        DataFormat cb_data_format = DataFormat::Float16_b;
        uint32_t single_tile_size = detail::TileSize(cb_data_format);
        MathFidelity math_fidelity = MathFidelity::HiFi4;
        //uint32_t single_tile_size = detail::TileSize(cb_data_format);
        uint32_t single_tile_size = 2 * 1024;

        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        /* DRAM buffer size = input full size */
        /* limiting page_size = single tile size; to allow DRAM channels interleaving */
        Buffer src0_dram_buffer = CreateBuffer(device, dram_buffer_A_size, single_tile_size, BufferType::DRAM);
        Buffer src1_dram_buffer = CreateBuffer(device, dram_buffer_B_size, single_tile_size, BufferType::DRAM);
        Buffer dst_dram_buffer = CreateBuffer(device, dram_buffer_C_size, single_tile_size, BufferType::DRAM);
        uint32_t src0_addr = src0_dram_buffer.address();
        uint32_t src1_addr = src1_dram_buffer.address();
        uint32_t dst_addr = dst_dram_buffer.address();


We need to declare three circular buffers to enable data transfer
between the reader, compute, and writer engines.
Input tiles count is = 2 because it's single tile process, and double-buffer
    .. code-block:: cpp

        uint32_t src0_cb_index = CB::c_in0; //0
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = CB::c_in1; // 1
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);



Compile-time kernels arguments
------------------------------
    .. code-block:: cpp

        bool src0_is_dram = src0_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        bool src1_is_dram = src1_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

        bool dst_is_dram = dst_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

        vector<uint32_t> compute_args = {
            B, // B
            Mt, // Mt
            Kt, // Kt
            Nt // Nt
        };

We have to declare some compile-time arguments for read/write kernels. Some default
parameters here will suffice.


Compute kernel declaration and compile-time defines
---------------------------------------------------
    .. code-block:: cpp

        auto reader_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_bmm_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

        auto writer_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_bmm_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

        auto matmul_single_core_kernel_id = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/bmm.cpp",
            core,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args}
        );


Runtime arguments and program launch
-----------------------------------------
    .. code-block:: cpp

        tt_metal::SetRuntimeArgs(
            program, reader_id, core,
            {src0_addr, src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(bcast_batch ? 1 : 0)}
        );

        tt_metal::SetRuntimeArgs(
            program, writer_id, core,
            {dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
        );


Launch program, enqueue & read in output buffer result into the host vector.
    .. code-block:: cpp

        EnqueueWriteBuffer(cq, src0_dram_buffer, a, false);
        EnqueueWriteBuffer(cq, src1_dram_buffer, b, false);
        EnqueueProgram(cq, program, false);
        EnqueueReadBuffer(cq, dst_dram_buffer, output, true);

In this program,  we're using a separate reader kernel to take in data from
DRAM into L1, and a separate writer kernel to write out results from the
compute engine back to the destination DRAM buffer.


Conclusion
----------

Those are the additional steps for getting matmul_single_core operations up and
running on the compute engine.
