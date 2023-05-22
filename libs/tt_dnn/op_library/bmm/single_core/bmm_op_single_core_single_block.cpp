#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void create_CBs_for_fused_matmul_sb(tt_metal::Program &program, tt_metal::Device* device, tt_xy_pair core, bool activations_rm, bool output_rm, uint32_t M, uint32_t N, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t num_bytes_for_df) {

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

    uint32_t src1_cb_addr = 250 * 1024;
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
        uint32_t matmul_partials_addr = 400 * 1024;
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

        uint32_t matmul_partials_addr = 450 * 1024;
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
        uint32_t temp_addr = 600 * 1024;
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
        uint32_t reblock_cb_addr = 750 * 1024;
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

        uint32_t output_cb_addr = 800 * 1024;
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

        uint32_t tilized_cb_addr = 400 * 1024;
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

        uint32_t matmul_partials_addr = 550 * 1024;
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
        uint32_t tilized_cb_addr = 300 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            device,
            tilize_mode_tilized_in0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
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
        uint32_t temp_addr = 580 * 1024;
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
        uint32_t reblock_cb_addr = 720 * 1024;
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

        uint32_t output_cb_addr = 860 * 1024;
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

// TODO(whoever gets a chance!): Refactor this so it's a part of matmul_single_core_... keeping it
// independent for now as it's easier for me to progress
Tensor large_bmm_single_core_single_block_(const Tensor& a, const Tensor &b, bool tilize_a, bool untilize_out) {
    const auto [Ba, Ca, Ha, Wa] = a.shape();
    const auto [Bb, Cb, Hb, Wb] = b.shape();

    uint32_t M = 8;
    uint32_t K = 4;
    uint32_t N = K;

    TT_ASSERT(Ha == M*32, "For now, assuming practically hard-coded dimensions so that blocking makes sense");
    TT_ASSERT(Wa == K*32, "For now, assuming practically hard-coded dimensions so that blocking makes sense");
    TT_ASSERT(Hb == Wb, "For now, assuming practically hard-coded dimensions so that blocking makes sense");

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
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> cshape{Ba, Ca, Ha, Wb};

    tt::tt_metal::Layout out_layout;
    if (untilize_out) {
        out_layout = tt::tt_metal::Layout::ROW_MAJOR;
    } else {
        out_layout = tt::tt_metal::Layout::TILE;
    }
    tt_metal::Tensor output = tt_metal::Tensor(
        cshape,
        a.dtype(),
        out_layout,
        device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // Keep for now, but need to fix when you get to multibank
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

        // out
        uint32_t out_dram_addr = dst_dram_buffer->address();
        uint32_t out_row_size = Wb * 2;

        // out block info
        uint32_t out_subblock_h = 4;
        uint32_t out_subblock_w = 2;
        uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

        // in0
        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in0_row_size = Wa * 2; // some row major data needed in case we want to tilize A

        // Important, dictates in0 block width, in1 block height
        uint32_t num_blocks = 1;

        // in0 block info
        uint32_t in0_block_w = Wat / num_blocks;
        //uint32_t in0_partial_row_size = (in0_block_w * 32) * 2;
        //uint32_t in0_num_blocks_w = Wat / in0_block_w;
        //uint32_t in0_block_h = Hat;
        uint32_t in0_num_subblocks = (Hat / out_subblock_h);
        uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
        uint32_t in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;
        uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        // in1
        uint32_t in1_dram_addr = src1_dram_buffer->address();

        // in1 block info
        uint32_t in1_num_subblocks = (Wbt / out_subblock_w);
        uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w*in1_num_subblocks;
        uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
        //uint32_t in1_block_w = out_subblock_w * in1_num_subblocks;
        //uint32_t in1_block_h = in0_block_w;

        {
            create_CBs_for_fused_matmul_sb(
                program,
                a.device(),
                core,
                tilize_a,
                untilize_out,
                Hat,
                Wbt,
                in0_block_w,
                out_subblock_h,
                2);

            TT_ASSERT(in0_subblock_h * in0_block_w * in0_num_subblocks == in0_block_num_tiles);
            TT_ASSERT(in0_block_w == K);

            string writer_kernel;
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

            string reader_kernel = "tt_metal/kernels/dataflow/reader_matmul_blocked.cpp";
            std::vector<uint32_t> reader_rt_args{
            in0_dram_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            in1_dram_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            (std::uint32_t)(K/in0_block_w), // num_blocks
            M * in0_block_w, // input 0 block num tiles
            N * in0_block_w, // input 1 block num tiles
            M * in0_block_w * single_tile_size, // input 0 block bytes
            N * in0_block_w * single_tile_size}; // input 1 block bytes
            auto reader = tt_metal::CreateDataMovementKernel(
                program,
                reader_kernel,
                core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

            auto writer = tt_metal::CreateDataMovementKernel(
                program,
                writer_kernel,
                core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

            vector<uint32_t> compute_kernel_args = {
            uint(in0_block_w),
            uint(in0_num_subblocks),
            uint(in0_block_num_tiles),
            uint(in0_subblock_num_tiles),
            uint(in0_subblock_h),

            uint(in1_num_subblocks),
            uint(in1_block_num_tiles),
            uint(in1_per_core_w),

            uint(num_blocks),

            uint(out_subblock_h),
            uint(out_subblock_w),
            uint(out_subblock_num_tiles),

            uint(tilize_a),
            uint(untilize_out)
        };

        tt_metal::KernelArgs mm_args = tt_metal::KernelArgs(core, compute_kernel_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;

        string compute_kernel = "tt_metal/kernels/compute/matmul_large_block.cpp";

        auto mm_kernel = tt_metal::CreateComputeKernel(
            program,
            compute_kernel,
            core,
            mm_args,
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


Tensor large_bmm_single_core_single_block(const Tensor& a, const Tensor &b, bool tilize_a, bool untilize_out) {

    Tensor output = large_bmm_single_core_single_block_(a, b, tilize_a, untilize_out);
    return output;
}

}  // namespace tt_metal

}  // namespace tt
