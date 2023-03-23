#include "tt_metal/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"
// #include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor matmul_single_core_(const Tensor &a, const Tensor &b, bool bcast_batch) {

    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};

    const auto& ashape = a.shape(), bshape = b.shape();

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * 1024;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0);
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> cshape{ashape[0], ashape[1], ashape[2], bshape[3]}; // C=A*B, N1MK*11KN->N1MN
    tt_metal::Tensor output = tt_metal::Tensor(cshape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    bool pass = true;
    {
        // C = A*B
        // MN = MK*KN
        if (bcast_batch)
            TT_ASSERT(ashape[0] > 0 && bshape[0] == 1);
        else {
            TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0] && "Channel and batch dimensions must match in bmm op (non-bcast)");
        }
        TT_ASSERT(ashape[3] == bshape[2] && "Dimension K (A.shape[2] and B.shape[3]) must match for A and B in bmm_op"); // A.K == B.K
        TT_ASSERT(ashape[2] % TILE_HEIGHT == 0);
        TT_ASSERT(ashape[3] % TILE_WIDTH == 0);
        TT_ASSERT(bshape[2] % TILE_HEIGHT == 0);
        TT_ASSERT(bshape[3] % TILE_WIDTH == 0);
        uint32_t B = ashape[0]*ashape[1];
        uint32_t Mt = ashape[2]/TILE_HEIGHT;
        uint32_t Kt = ashape[3]/TILE_WIDTH;
        uint32_t Nt = bshape[3]/TILE_WIDTH;

        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in1_dram_addr = src1_dram_buffer->address();
        uint32_t out_dram_addr = dst_dram_buffer->address();

        {
            uint32_t src0_cb_index = 0;
            uint32_t src0_cb_addr = 200 * 1024;
            uint32_t num_input_tiles = 2;
            auto cb_src0 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src0_cb_index,
                core,
                num_input_tiles,
                num_input_tiles * single_tile_size,
                src0_cb_addr,
                tt::DataFormat::Float16_b
            );

            uint32_t src1_cb_index = 1;
            uint32_t src1_cb_addr = 300 * 1024;
            auto cb_src1 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src1_cb_index,
                core,
                num_input_tiles,
                num_input_tiles * single_tile_size,
                src1_cb_addr,
                tt::DataFormat::Float16_b
            );

            uint32_t ouput_cb_index = 16; // output operands start at index 16
            uint32_t output_cb_addr = 400 * 1024;
            uint32_t num_output_tiles = 2;
            auto cb_output = tt_metal::CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                num_output_tiles,
                num_output_tiles * single_tile_size,
                output_cb_addr,
                tt::DataFormat::Float16_b
            );

            auto reader = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_bmm_8bank.cpp",
                core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

            auto writer = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/writer_bmm_8bank.cpp",
                core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

            vector<uint32_t> compute_args = {
                B, // B
                Mt, // Mt
                Kt, // Kt
                Nt // Nt
            };
            tt_metal::ComputeKernelArgs *bmm_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);
            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
                program,
                "tt_metal/kernels/compute/bmm.cpp",
                core,
                bmm_args,
                MathFidelity::HiFi4,
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

            bool skip_hlkc = false;
            pass &= tt_metal::CompileProgram(device, program, skip_hlkc);
            pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        }
        pass &= tt_metal::LaunchKernels(device, program);
    }

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




void create_CBs_for_fused_matmul(tt_metal::Program* program, tt_metal::Device* device, tt_xy_pair core, bool activations_rm, bool output_rm, uint32_t M, uint32_t N, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t num_bytes_for_df) {


    uint32_t single_tile_size = num_bytes_for_df * 1024;

    // Invariants
    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 150 * 1024;
    uint32_t cb0_tiles = M * in0_block_w * 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    uint32_t src1_cb_addr = 300 * 1024;
    uint32_t cb1_tiles = N * in0_block_w * 2;
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        src1_cb_addr,
        tt::DataFormat::Float16_b
    );

    if (not activations_rm and not output_rm) { // no tilize, no untilize
        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );
    } else if (not activations_rm and output_rm) { // no tilize, just untilize

        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 500 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_index = 26;
        uint32_t reblock_cb_addr = 600 * 1024;
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            reblock_cb_index,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            reblock_cb_addr,
            tt::DataFormat::Float16_b
        );

    } else if (activations_rm and not output_rm) { // just tilize, no untilize

        uint32_t src0_tilized_index = 24;
        uint32_t src_0_tilized_cb_addr = 500 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_tilized_index,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            src_0_tilized_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t interm0_cb_index = 25;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );
    } else { // tilize activations and untilize output

        // Used for placing tilized activations
        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 500 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        // Used
        uint32_t interm1_cb_index = 25;
        uint32_t interm1_cb_addr = 600 * 1024;
        uint32_t interm1_cb_tiles = M * N;
        auto cb_interm1 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm1_cb_index,
            core,
            interm1_cb_tiles,
            interm1_cb_tiles * single_tile_size,
            interm1_cb_addr,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_index = 26;
        uint32_t reblock_cb_addr = 700 * 1024;
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            reblock_cb_index,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            reblock_cb_addr,
            tt::DataFormat::Float16_b
        );
    }
}

// TODO(whoever gets a chance!): Refactor this so it's a part of matmul_single_core_... keeping it
// independent for now as it's easier for me to progress
Tensor large_bmm_single_core_(const Tensor& a, const Tensor &b, bool tilize_a, bool untilize_out) {

    const auto [Ba, Ca, Ha, Wa] = a.shape();
    const auto [Bb, Cb, Hb, Wb] = b.shape();

    TT_ASSERT(Ha == 8 * TILE_HEIGHT, "For now, assuming practically hard-coded dimensions so that blocking makes sense");
    TT_ASSERT(Wa == 4 * TILE_WIDTH, "For now, assuming practically hard-coded dimensions so that blocking makes sense");
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

    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};

    uint32_t single_tile_size = 2 * 1024; // TODO(agrebenisan): Refactor on df
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    // same condition as above, different message
    TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor a must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

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
        device,
        {.interleaved = false, .dram_channel = 0});

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    bool pass = true;
    {
        // Convert tensor dims to tile dims
        uint32_t B   = Ba;
        uint32_t Hat = Ha / TILE_HEIGHT;
        uint32_t Wat = Wa / TILE_WIDTH;
        uint32_t Wbt = Wb / TILE_WIDTH;

        uint32_t out_subblock_h = 4;
        uint32_t out_subblock_w = 2;
        uint32_t in0_block_w = Wat;

        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in1_dram_addr = src1_dram_buffer->address();
        uint32_t out_dram_addr = dst_dram_buffer->address();

        auto in0_dram_noc_xy = src0_dram_buffer->noc_coordinates();
        auto in1_dram_noc_xy = src1_dram_buffer->noc_coordinates();
        auto out_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        // NOC coordinates
        uint32_t in0_dram_noc_x = uint32_t(in0_dram_noc_xy.x);
        uint32_t in0_dram_noc_y = uint32_t(in0_dram_noc_xy.y);
        uint32_t in1_dram_noc_x = uint32_t(in1_dram_noc_xy.x);
        uint32_t in1_dram_noc_y = uint32_t(in1_dram_noc_xy.y);
        uint32_t out_dram_noc_x = uint32_t(out_dram_noc_xy.x);
        uint32_t out_dram_noc_y = uint32_t(out_dram_noc_xy.y);

        {
            create_CBs_for_fused_matmul(
                program,
                a.device(),
                core,
                tilize_a,
                untilize_out,
                Hat,
                Wbt,
                in0_block_w,
                out_subblock_h,
                2); // TODO(agrebenisan): fix df num bytes


            vector<uint32_t> reader_rt_args = {
                in0_dram_addr,
                in0_dram_noc_x,
                in0_dram_noc_y,
                in1_dram_addr,
                in1_dram_noc_x,
                in1_dram_noc_y,
                (Wat / in0_block_w), // num_blocks
                Hat * in0_block_w, // input 0 block num tiles
                Wbt * in0_block_w, // input 1 block num tiles
                Hat * in0_block_w * single_tile_size, // input 0 block bytes
                Wbt * in0_block_w * single_tile_size // input 1 block bytes
            };

            string writer_kernel;
            vector<uint32_t> writer_rt_args;
            if (untilize_out) {
                writer_kernel = "tt_metal/kernels/dataflow/writer_unary.cpp";
                writer_rt_args = {
                    out_dram_addr,
                    out_dram_noc_x,
                    out_dram_noc_y,
                    Hat * Wbt
                };
            } else {
                writer_kernel = "tt_metal/kernels/dataflow/writer_unswizzle.cpp";
                writer_rt_args = {
                    out_dram_addr,
                    out_dram_noc_x,
                    out_dram_noc_y,
                    out_subblock_h, // num tiles per sub block m
                    out_subblock_w, // num tiles per sub block n
                    Hat / out_subblock_h, // num sub blocks m
                    Wbt / out_subblock_w, // num sub blocks n
                    out_subblock_w * single_tile_size * (Wbt / out_subblock_w), // bytes offset to next row within sub-block
                    out_subblock_h * out_subblock_w * single_tile_size * (Wbt / out_subblock_w), // bytes offset to next row of sub-blocks
                    out_subblock_w * single_tile_size
                    };
            }

            auto reader = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_matmul_blocked.cpp",
                core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

            auto writer = tt_metal::CreateDataMovementKernel(
                program,
                writer_kernel,
                core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

            uint32_t num_blocks = (Wat / in0_block_w);
            uint32_t in0_num_subblocks = (Hat / out_subblock_h);
            uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
            uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

            uint32_t in1_num_subblocks = (Wbt / out_subblock_w);
            uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w*in1_num_subblocks;
            uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

            uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
            uint32_t in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;

            vector<uint32_t> compute_kernel_args = {
                in0_block_w,
                in0_num_subblocks,
                in0_block_num_tiles,
                in0_subblock_num_tiles,
                in0_subblock_h,

                in1_num_subblocks,
                in1_block_num_tiles,
                in1_per_core_w,

                num_blocks,

                out_subblock_h,
                out_subblock_w,
                out_subblock_num_tiles,

                tilize_a,
                untilize_out
            };

            tt_metal::ComputeKernelArgs *bmm_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);
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


        // read_trisc_debug_mailbox(device->cluster(), 0, {1, 1}, 0);

        pass &= tt_metal::LaunchKernels(device, program);
    }

    TT_ASSERT(pass);

    return output;
}

Tensor large_bmm_single_core(const Tensor& a, const Tensor &b, bool tilize_a, bool untilize_out) {

    Tensor output = large_bmm_single_core_(a, b, tilize_a, untilize_out);
    return output;
}

}  // namespace tt_metal

}  // namespace tt
