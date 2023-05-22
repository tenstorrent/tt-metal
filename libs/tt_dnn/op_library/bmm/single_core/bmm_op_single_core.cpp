#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"
#include "tests/tt_metal/llrt/test_libs/debug_mailbox.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor matmul_single_core_(const Tensor &a, const Tensor &b, bool bcast_batch) {

    tt_metal::Program program = tt_metal::Program();
    tt_xy_pair core = {0, 0};

    const auto& ashape = a.shape(), bshape = b.shape();

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

    TT_ASSERT(a.dtype() == b.dtype());
    TT_ASSERT(a.dtype() == tt::tt_metal::DataType::BFLOAT16 || a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(ashape[3] == bshape[2] && "Dimension K (A.shape[2] and B.shape[3]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(ashape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(ashape[3] % TILE_WIDTH == 0);
    TT_ASSERT(bshape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(bshape[3] % TILE_WIDTH == 0);
    TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0);
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> cshape{ashape[0], ashape[1], ashape[2], bshape[3]}; // C=A*B, N1MK*11KN->N1MN
    tt_metal::Tensor output = tt_metal::Tensor(cshape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    bool pass = true;

    // C = A*B
    // MN = MK*KN
    uint32_t B = ashape[0]*ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;

    uint32_t in0_dram_addr = src0_dram_buffer->address();
    uint32_t in1_dram_addr = src1_dram_buffer->address();
    uint32_t out_dram_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );

    bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
    tt_metal::KernelArgs reader_writer_compile_time_args;
    if (tile_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_writer_compile_time_args = tt_metal::KernelArgs(core, {1, (std::uint32_t)log2(single_tile_size)});
    } else {
        reader_writer_compile_time_args = tt_metal::KernelArgs(core, {0, 0});
    }
    auto reader = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_8bank.cpp",
        core, reader_writer_compile_time_args, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    auto writer = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_bmm_8bank.cpp",
        core, reader_writer_compile_time_args, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        B, // B
        Mt, // Mt
        Kt, // Kt
        Nt // Nt
    };
    tt_metal::KernelArgs bmm_args = tt_metal::KernelArgs(core, compute_args);
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm.cpp",
        core,
        bmm_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device, reader, core,
        {in0_dram_addr, in1_dram_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(bcast_batch ? 1 : 0)}
    );
    tt_metal::WriteRuntimeArgsToDevice(
        device, writer, core,
        {out_dram_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
    );


    pass &= tt_metal::CompileProgram(device, program);
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::LaunchKernels(device, program);

    TT_ASSERT(pass);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor matmul_single_core(const Tensor& a, const Tensor& b) {
    return matmul_single_core_(a, b, true);
}

Tensor bmm_single_core(const Tensor& a, const Tensor& b) {
    return matmul_single_core_(a, b, false);
}

void create_CBs_for_fused_matmul_new_alloc(tt_metal::Program& program,
                                tt_metal::Device* device,
                                tt_xy_pair core,
                                uint32_t M,
                                uint32_t N,
                                uint32_t in0_block_w,
                                uint32_t out_subblock_h,
                                uint32_t num_bytes_for_df,
                                bool tilize_act,
                                bool untilize_out) {
    uint32_t in0_cb                                   = 0;
    uint32_t in1_cb                                   = 1;
    uint32_t tilize_mode_tilized_in0_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t cb0_tiles = M * in0_block_w * 2;
    auto cb_in0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t cb1_tiles = N * in0_block_w * 2;
    auto cb_in1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in1_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    if(tilize_act) {
        // Used for placing tilized activations
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            device,
            tilize_mode_tilized_in0_cb,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }
    if(untilize_out) {
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Shares same address space as matmul partials
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }
    else {

        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_matmul_partials_addr = cb_matmul_partials->address();

        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            cb_matmul_partials_addr,
            tt::DataFormat::Float16_b
        );

    }
}

void create_CBs_for_fused_matmul_old_alloc(tt_metal::Program& program,
                                tt_metal::Device* device,
                                tt_xy_pair core,
                                uint32_t M,
                                uint32_t N,
                                uint32_t in0_block_w,
                                uint32_t out_subblock_h,
                                uint32_t num_bytes_for_df,
                                bool activations_rm,
                                bool output_rm) {

    uint32_t in0_cb                                   = 0;
    uint32_t in1_cb                                   = 1;
    uint32_t tilize_mode_tilized_in0_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t src0_cb_addr = 120 * 1024;
    uint32_t cb0_tiles = M * in0_block_w * 2;
    auto cb_in0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t src1_cb_addr = 220 * 1024;
    uint32_t cb1_tiles = N * in0_block_w * 2;
    auto cb_in1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in1_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        src1_cb_addr,
        tt::DataFormat::Float16_b
    );


    if (not activations_rm and not output_rm) { // no tilize, no untilize
        uint32_t matmul_partials_addr = 440 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            matmul_partials_addr,
            tt::DataFormat::Float16_b
        );

        // Partials share same L1 address space as output
        uint32_t output_cb_addr = matmul_partials_addr;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

    } else if (not activations_rm and output_rm) { // no tilize, just untilize

        uint32_t matmul_partials_addr = 440 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            matmul_partials_addr,
            tt::DataFormat::Float16_b
        );


        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        uint32_t temp_addr = 560 * 1024;
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            temp_addr,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_addr = 680 * 1024;
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            reblock_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t output_cb_addr = 750 * 1024;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );


    } else if (activations_rm and not output_rm) { // just tilize, no untilize

        uint32_t tilized_cb_addr = 320 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            device,
            tilize_mode_tilized_in0_cb,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            tilized_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t matmul_partials_addr = 440 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            matmul_partials_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t output_cb_addr = matmul_partials_addr;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

    } else { // tilize activations and untilize output

        // Used for placing tilized activations
        uint32_t tilized_cb_addr = 320 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            device,
            tilize_mode_tilized_in0_cb,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            tilized_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t cb_matmul_partials_addr = 440 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            cb_matmul_partials_addr,
            tt::DataFormat::Float16_b
        );

        // Shares same address space as matmul partials
        uint32_t temp_addr = 560 * 1024;
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            temp_addr,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_addr = 680 * 1024;
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            reblock_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t output_cb_addr = 750 * 1024;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );
    }
}
Tensor create_output_dram_buffer(Device * device, DataType data_type, uint32_t M, uint32_t N, bool untilize_out) {
    std::array<uint32_t, 4> cshape{1, 1, M, N};
    tt_metal::Tensor output = tt_metal::Tensor(
        cshape,
        data_type,
        untilize_out ? tt::tt_metal::Layout::ROW_MAJOR : tt::tt_metal::Layout::TILE,
        device);

    return output;
}

std::tuple<uint32_t, uint32_t, uint32_t> compute_block_info(uint32_t M, uint32_t K, uint32_t N) {
    uint32_t single_tile_size_bytes = 2 * 1024;

    // Constraint 1: in0 and in1 should fit in L1. If not, divide into blocks
    // Max sizes based on hard coded CB addressing
    uint32_t max_in0_bytes = 50 * 1024;
    uint32_t max_in1_bytes = 50 * 1024;
    uint32_t max_in0_tiles = max_in0_bytes / single_tile_size_bytes;
    uint32_t max_in1_tiles = max_in1_bytes / single_tile_size_bytes;
    tt::log_debug("max_in0_block_tiles={}", max_in0_tiles);
    tt::log_debug("max_in1_block_tiles={}", max_in1_tiles);
    uint32_t num_blocks = 1;
    uint32_t in_block_w = K;
    assert(M <= max_in0_tiles && N <= max_in1_tiles);
    uint32_t max_in_block_w = std::min((max_in0_tiles/M), (max_in1_tiles/N));
    while (in_block_w > max_in_block_w || K % num_blocks != 0) {
        num_blocks += 1;
        assert(num_blocks <= K);
        in_block_w = K / num_blocks;
    }
    tt::log_debug("Num blocks={}", num_blocks);
    tt::log_debug("K block size={}", in_block_w);

    // Constraint 2: output should fit in L1
    uint32_t max_out_bytes = 120 * 1024;
    uint32_t max_out_tiles = max_out_bytes / single_tile_size_bytes;
    tt::log_debug("max_out_block_tiles={}", max_out_tiles);
    assert (M*N <= max_out_tiles);

    // Constraint 3: output should should fit in half DST (8 tiles). If not, divide into output sublocks
    uint32_t out_subblock_h = M;
    uint32_t out_subblock_w = N;
    uint32_t num_out_subblocks_h = 1;
    uint32_t num_out_subblocks_w = 1;
    bool divide_h_next = true;
    while (out_subblock_h*out_subblock_w > 8) {
        if (divide_h_next) {
            if(num_out_subblocks_h < M) {
                num_out_subblocks_h += 1;
                while(M % num_out_subblocks_h != 0) {
                    num_out_subblocks_h += 1;
                }
            }
            out_subblock_h = M / num_out_subblocks_h;
            divide_h_next = false;
        }
        else {
            if(num_out_subblocks_w < N) {
                num_out_subblocks_w += 1;
                while(N % num_out_subblocks_w != 0) {
                    num_out_subblocks_w += 1;
                }
            }
            out_subblock_w = N / num_out_subblocks_w;
            divide_h_next = true;
        }
    }
    log_debug("out_subblock_h={}", out_subblock_h);
    log_debug("out_subblock_w={}", out_subblock_w);
    return std::make_tuple(num_blocks, out_subblock_h, out_subblock_w);
}

// TODO(whoever gets a chance!): Refactor this so it's a part of matmul_single_core_... keeping it
// independent for now as it's easier for me to progress
Tensor large_bmm_single_core_(const Tensor& a, const Tensor &b, bool tilize_act, bool untilize_out) {

    const auto [Ba, Ca, Ha, Wa] = a.shape();
    const auto [Bb, Cb, Hb, Wb] = b.shape();

    // Normal matrix shape checks
    TT_ASSERT(Ba == 1, "So far, large matmul op has only been tested for batch one.");
    TT_ASSERT(Ba == Bb, "Batch dimension needs to match");
    TT_ASSERT(Ca == Cb, "Channel dimension needs to match");
    TT_ASSERT(Wa == Hb, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(Ha % TILE_HEIGHT == 0, "Height of tensor a needs to be divisible by 32");
    TT_ASSERT(Wa % TILE_WIDTH == 0, "Width of tensor a needs to be divisible by 32");
    TT_ASSERT(Hb % TILE_HEIGHT == 0, "Height of tensor b needs to be divisible by 32");
    TT_ASSERT(Wb % TILE_WIDTH == 0, "Width of tensor b needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to large matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to large matmul need to be allocated in buffers on device!");

    tt_metal::Program program = tt_metal::Program();
    tt_xy_pair core = {0, 0};

    uint32_t single_tile_size = 2 * 1024; // TODO(agrebenisan): Refactor on df
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    // same condition as above, different message
    TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor a must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    tt_metal::Device *device = a.device();

    Tensor output = create_output_dram_buffer(a.device(), a.dtype(), Ha, Wb, untilize_out);
    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t out_dram_addr = dst_dram_buffer->address();
    auto out_dram_noc_xy = dst_dram_buffer->noc_coordinates();
    uint32_t out_dram_noc_x = out_dram_noc_xy.x;
    uint32_t out_dram_noc_y = out_dram_noc_xy.y;

    bool pass = true;
    {
        // Convert tensor dims to tile dims
        uint32_t B   = Ba;
        uint32_t Hat = Ha / TILE_HEIGHT;
        uint32_t Wat = Wa / TILE_WIDTH;
        uint32_t Wbt = Wb / TILE_WIDTH;
        log_debug("Hat (M in tiles)={}", Hat);
        log_debug("Wat (K in tiles)={}", Wat);
        log_debug("Wbt (N in tiles)={}", Wbt);

        // out
        uint32_t out_row_size = Wb * 2;

        // out block info
        auto [num_blocks, out_subblock_h, out_subblock_w] = compute_block_info(Hat, Wat, Wbt);
        //uint32_t out_subblock_h = 4;
        //uint32_t out_subblock_w = 2;
        uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

        // in0
        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in0_row_size = Wa * 2; // some row major data needed in case we want to tilize A

        // Important, dictates in0 block width, in1 block height
        //uint32_t num_blocks = 2;

        // in0 block info
        uint32_t in0_block_w = Wat / num_blocks; // Two blocks in the W dimension
        uint32_t in0_partial_row_size = (in0_block_w * 32) * 2;
        uint32_t in0_num_blocks_w = Wat / in0_block_w;
        uint32_t in0_block_h = Hat;
        uint32_t in0_num_subblocks = (Hat / out_subblock_h);
        uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
        uint32_t in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;
        uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        // in1
        uint32_t in1_dram_addr = src1_dram_buffer->address();

        // in1 block info
        uint32_t in1_num_subblocks = (Wbt / out_subblock_w);
        uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w*in1_num_subblocks;
        uint32_t in1_block_w = out_subblock_w * in1_num_subblocks;
        uint32_t in1_block_h = in0_block_w;

        // For debug, change these log_debug and uncomment this

/*      rk: Intentional syntax error - people who want this should take the time
        to change this logging to log_debug (preferably debug)
        std::cout << "in0 information" << std::endl;
        std::cout << "\t in0_dram_addr: " << in0_dram_addr << std::endl;
        std::cout << "\t in0_row_size: " << in0_row_size << std::endl;
        std::cout << "\t in0_block_w: " << in0_block_w << std::endl;
        std::cout << "\t in0_partial_row_size: " << in0_partial_row_size << std::endl;
        std::cout << "\t in0_num_blocks_w: " << in0_num_blocks_w << std::endl;
        std::cout << "\t in0_block_h: " << in0_block_h << std::endl;
        std::cout << "\t in0_num_subblocks: " << in0_num_subblocks << std::endl;
        std::cout << "\t in0_block_num_tiles: " << in0_block_num_tiles << std::endl;
        std::cout << "\t in0_subblock_h: " << in0_subblock_h << std::endl;
        std::cout << "\t in0_subblock_num_tiles: " << in0_subblock_num_tiles << std::endl;

        std::cout << "in1 information" << std::endl;
        std::cout << "\t in1_dram_addr: " << in1_dram_addr << std::endl;
        std::cout << "\t in1_num_subblocks: " << in1_num_subblocks << std::endl;
        std::cout << "\t in1_block_num_tiles: " << in1_block_num_tiles << std::endl;
        std::cout << "\t in1_block_w: " << in1_block_w << std::endl;
        std::cout << "\t in1_block_h: " << in1_block_h << std::endl;

        std::cout << "out information" << std::endl;
        std::cout << "\t out_dram_addr: " << out_dram_addr << std::endl;
        std::cout << "\t out_row_size: " << out_row_size << std::endl;
        std::cout << "\t out_subblock_h: " << out_subblock_h << std::endl;
        std::cout << "\t out_subblock_w: " << out_subblock_w << std::endl;
        std::cout << "\t out_subblock_num_tiles: " << out_subblock_num_tiles << std::endl; */


        {
        bool old_alloc = false;
        if(old_alloc) {
            create_CBs_for_fused_matmul_old_alloc(
                program,
                a.device(),
                core,
                Hat,
                Wbt,
                in0_block_w,
                out_subblock_h,
                2,
                tilize_act,
                untilize_out); // TODO(agrebenisan): fix df num bytes
        }
        else {
            create_CBs_for_fused_matmul_new_alloc(
                program,
                a.device(),
                core,
                Hat,
                Wbt,
                in0_block_w,
                out_subblock_h,
                2,
                tilize_act,
                untilize_out); // TODO(agrebenisan): fix df num bytes
        }
            string writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp";
            vector<uint32_t> writer_rt_args;
            if (untilize_out) {
                writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp";
                writer_rt_args = {
                    out_dram_addr,
                    Ha,
                    out_row_size
                };
            } else {
                writer_kernel = "tt_metal/kernels/dataflow/writer_matmul_tile_layout.cpp";
                writer_rt_args = {
                    out_dram_addr,
                    0,
                    1,
                    Wbt,
                    out_subblock_w,
                    out_subblock_h * Wbt,

                    out_subblock_w,
                    out_subblock_h,
                    out_subblock_w * out_subblock_h,
                    Wbt / out_subblock_w,
                    Hat / out_subblock_h
                };
            }

            string reader_kernel;
            vector<uint32_t> reader_rt_args;
            if (tilize_act) {
                reader_kernel = "tt_metal/kernels/dataflow/reader_matmul_row_major_activations_tile_layout_weights.cpp";
                reader_rt_args = {
                    in0_dram_addr,
                    0,

                    in0_block_h,
                    in0_block_num_tiles,

                    in0_row_size,
                    in0_partial_row_size,

                    in1_dram_addr,
                    0,
                    1,
                    Wbt,
                    in0_block_w * Wbt,

                    in1_block_w,
                    in1_block_h,
                    in1_block_num_tiles,

                    num_blocks
                };
            } else {
                reader_kernel = "tt_metal/kernels/dataflow/reader_matmul_tile_layout.cpp";
                reader_rt_args = {
                    in0_dram_addr,
                    0,
                    1,
                    Wat,
                    in0_block_w,

                    in0_block_w,
                    in0_block_h,
                    in0_block_num_tiles,

                    in1_dram_addr,
                    0,
                    1,
                    Wbt,
                    in0_block_w * Wbt,

                    in1_block_w,
                    in1_block_h,
                    in1_block_num_tiles,

                    num_blocks
                };
            }


            auto reader = tt_metal::CreateDataMovementKernel(
                program,
                reader_kernel,
                core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

            auto writer = tt_metal::CreateDataMovementKernel(
                program,
                writer_kernel,
                core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

            vector<uint32_t> compute_kernel_args = {
                in0_block_w,
                in0_num_subblocks,
                in0_block_num_tiles,
                in0_subblock_num_tiles,
                in0_subblock_h,

                in1_num_subblocks,
                in1_block_num_tiles,
                in1_block_w,

                num_blocks,

                out_subblock_h,
                out_subblock_w,
                out_subblock_num_tiles,

                tilize_act,
                untilize_out
            };

            tt_metal::KernelArgs bmm_args = tt_metal::KernelArgs(core, compute_kernel_args);
            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
                program,
                "tt_metal/kernels/compute/matmul_large_block.cpp",
                core,
                bmm_args,
                MathFidelity::HiFi4,
                fp32_dest_acc_en,
                math_approx_mode
            );

            tt_metal::WriteRuntimeArgsToDevice(
                device, reader, core,
                reader_rt_args
            );

            tt_metal::WriteRuntimeArgsToDevice(
                device, writer, core,
                writer_rt_args
            );

            pass &= tt_metal::CompileProgram(device, program, false);
            pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        }

        pass &= tt_metal::LaunchKernels(device, program);
    }

    TT_ASSERT(pass);

    return output;
}

Tensor large_bmm_single_core(const Tensor& a, const Tensor &b, bool tilize_act, bool untilize_out) {

    Tensor output = large_bmm_single_core_(a, b, tilize_act, untilize_out);
    return output;
}

}  // namespace tt_metal

}  // namespace tt
