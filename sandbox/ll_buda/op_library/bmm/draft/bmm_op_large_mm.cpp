#include "ll_buda/op_library/eltwise_binary/eltwise_binary_op.hpp"

#include "ll_buda/host_api.hpp"
#include "common/constants.hpp"

namespace matmul {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    int in0_block_w;
    int in0_num_subblocks;
    int in0_block_num_tiles;
    int in0_subblock_num_tiles;
    int in1_num_subblocks;
    int in1_block_num_tiles;
    int in1_per_core_w;
    int num_blocks;
    int out_subblock_h;
    int out_subblock_w;
    int out_subblock_num_tiles;
};
}

namespace eltwise_binary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
}

using namespace tt;
using namespace tt::ll_buda;
using namespace tt::constants;

std::tuple<Program *, DataMovementKernel *, DataMovementKernel *> create_program(
    int num_cores_r,
    int num_cores_c,
    int M, int N, int K,
    int in0_block_w,
    int out_subblock_h,
    int out_subblock_w,
    int per_core_M, int per_core_N) {

    Program *program = new Program();

    uint32_t single_tile_size = 2 * 1024;
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;
    TT_ASSERT(in0_CB_size <= 130*1024);
    TT_ASSERT(in1_CB_size <= 130*1024);
    TT_ASSERT(out_CB_size <= 540*1024);

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};;
    CoreRange all_cores(start_core, end_core);

    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            CoreCoord core = {(std::size_t) j, (std::size_t) i};
            uint32_t l1_valid_address = 200 * 1024;

            uint32_t src0_cb_index = 0;
            uint32_t src0_cb_addr = l1_valid_address;
            l1_valid_address += in0_CB_size;
            uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
            auto cb_src0 = CreateCircularBuffer(
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
            uint32_t src1_cb_addr = l1_valid_address;
            l1_valid_address += in1_CB_size;
            uint32_t cb1_tiles = in0_block_tiles * 2; // double buffer
            auto cb_src1 = CreateCircularBuffer(
                program,
                device,
                src1_cb_index,
                core,
                cb1_tiles,
                cb1_tiles * single_tile_size,
                src1_cb_addr,
                tt::DataFormat::Float16_b
            );

            uint32_t ouput_cb_index = 16; // output operands start at index 16
            uint32_t output_cb_addr = l1_valid_address;
            l1_valid_address += out_CB_size;
            auto cb_output = CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                out_CB_tiles,
                out_CB_size,
                output_cb_addr,
                tt::DataFormat::Float16_b
            );

            uint32_t interm0_cb_index = 24;
            auto cb_interm0 = CreateCircularBuffer(
                program,
                device,
                interm0_cb_index,
                core,
                out_CB_tiles,
                out_CB_size,
                output_cb_addr,
                tt::DataFormat::Float16_b
            );

            TT_ASSERT(l1_valid_address < 1024 * 1024);
        }
    }

    auto mm_reader_kernel = CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_matmul_tile_layout.cpp",
        all_cores,
        DataMovementProcessor::RISCV_1,
        NOC::RISCV_1_default);

    auto unary_writer_kernel = CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_matmul_tile_layout.cpp",
        all_cores,
        DataMovementProcessor::RISCV_0,
        NOC::RISCV_0_default);

    int num_blocks = (K/in0_block_w);

    int in0_num_subblocks = (per_core_M/out_subblock_h);
    int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    int in1_num_subblocks = (per_core_N/out_subblock_w);
    int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    int in1_per_core_w = out_subblock_w * in1_num_subblocks;

    int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    void *hlk_args = new matmul::hlk_args_t{
        .in0_block_w = in0_block_w,
        .in0_num_subblocks = in0_num_subblocks,
        .in0_block_num_tiles = in0_block_num_tiles,
        .in0_subblock_num_tiles = in0_subblock_num_tiles,

        .in1_num_subblocks = in1_num_subblocks,
        .in1_block_num_tiles = in1_block_num_tiles,
        .in1_per_core_w = in1_per_core_w,

        .num_blocks = num_blocks,

        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .out_subblock_num_tiles = out_subblock_num_tiles
    };

    ComputeKernelArgs *mm_args = InitializeCompileTimeComputeKernelArgs(all_cores, hlk_args, sizeof(matmul::hlk_args_t));
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto mm_kernel = CreateComputeKernel(
        program,
        "kernels/compute/matmul_large_block_zm.cpp",
        all_cores,
        mm_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    return {program, mm_reader_kernel, unary_writer_kernel};
}

bool write_runtime_args_to_device(
    Device *device,
    int num_cores_r,
    int num_cores_c,
    ll_buda::DataMovementKernel *mm_reader_kernel,
    ll_buda::DataMovementKernel *unary_writer_kernel,
    int M,
    int N,
    int K,
    int in0_block_w,
    int out_subblock_h,
    int out_subblock_w,
    int per_core_M,
    int per_core_N,
    uint32_t in0_dram_addr,
    uint32_t in1_dram_addr,
    uint32_t out_dram_addr) {

    bool pass = true;
    uint32_t single_tile_size = 2 * 1024;

    uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    TT_ASSERT(in0_dram_addr + dram_buffer_size_act < 1024 * 1024 * 1024);
    TT_ASSERT(in1_dram_addr + dram_buffer_size_weights < 1024 * 1024 * 1024);
    TT_ASSERT(out_dram_addr + dram_buffer_size_out < 1024 * 1024 * 1024);

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) core_idx_x, (std::size_t) core_idx_y};

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t) in0_dram_addr, // in0_tensor_addr
                (std::uint32_t)  K * per_core_M * core_idx_y, // in0_tensor_start_tile_id
                (std::uint32_t)  1, // in0_tensor_stride_w
                (std::uint32_t)  K, // in0_tensor_stride_h
                (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride

                (std::uint32_t)  in0_block_w, // in0_block_w
                (std::uint32_t)  per_core_M, // in0_block_h
                (std::uint32_t)  in0_block_w * per_core_M, //in0_block_num_tiles

                (std::uint32_t)  in1_dram_addr, // in1_tensor_addr
                (std::uint32_t)  per_core_N * core_idx_x, //in1_tensor_start_tile_id
                (std::uint32_t)  1, // in1_tensor_stride_w
                (std::uint32_t)  N, // in1_tensor_stride_h
                (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride

                (std::uint32_t)  per_core_N, // in1_block_w
                (std::uint32_t)  in0_block_w, //in1_block_h
                (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

                (std::uint32_t)  K / in0_block_w // num_blocks
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t) out_dram_addr, // out_tensor_addr
                (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * N, // out_tensor_start_tile_id
                (std::uint32_t) 1, // out_tensor_stride_w
                (std::uint32_t) N,  // out_tensor_stride_h
                (std::uint32_t) out_subblock_w, // out_tensor_next_subblock_stride_w
                (std::uint32_t) out_subblock_h * N, // out_tensor_next_subblock_stride_h

                (std::uint32_t) out_subblock_w, // out_subblock_w
                (std::uint32_t) out_subblock_h, // out_subblock_h
                (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
                (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
                (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h
            };

            pass &= ll_buda::WriteRuntimeArgsToDevice(device, mm_reader_kernel, core, mm_reader_args);
            pass &= ll_buda::WriteRuntimeArgsToDevice(device, unary_writer_kernel, core, writer_args);
        }
    }
    return pass;
}


namespace tt {

namespace ll_buda {

Tensor matmul(const Tensor &a, const Tensor &b) {

    CoreCoord core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * 1024;

    ll_buda::DramBuffer *src0_dram_buffer = a.buffer();
    ll_buda::DramBuffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0);
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0);

    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();
    ll_buda::Tensor output = ll_buda::Tensor(a.shape(), a.dtype(), tt::ll_buda::Layout::TILE, device);

    ll_buda::DramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    bool pass = true;
    {
        int num_cores_r = 10;
        int num_cores_c = 12;

        // M, N, K = num tiles
        uint32_t M = 16 * num_cores_r;
        uint32_t K = 16 * 12;
        uint32_t N = 16 * num_cores_c;

        // TODO(AP): figure out the specs for supported dimensions
        const auto& ashape = a.shape(), bshape = b.shape();
        TT_ASSERT(ashape[0] == ashape[1] && ashape[0] == 1);
        TT_ASSERT(bshape[0] == bshape[1] && bshape[0] == 1);
        TT_ASSERT(ashape[2] == M*TILE_HEIGHT);
        TT_ASSERT(ashape[3] == K*TILE_WIDTH);
        TT_ASSERT(bshape[2] == K*TILE_HEIGHT);
        TT_ASSERT(bshape[3] == N*TILE_WIDTH);
        int out_subblock_h = 4;
        int out_subblock_w = 2;
        int in0_block_w = 2;
        int per_core_M = M / num_cores_r;
        int per_core_N = N / num_cores_c;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in1_dram_addr = src1_dram_buffer->address();
        uint32_t out_dram_addr = dst_dram_buffer->address();

        auto [program, mm_reader_kernel, unary_writer_kernel] =
            create_program(
                num_cores_r, num_cores_c,
                M, N, K,
                in0_block_w, out_subblock_h, out_subblock_w,
                per_core_M, per_core_N);

        pass &= ll_buda::CompileProgram(device, program);
        pass &= write_runtime_args_to_device(
            device,
            num_cores_r, num_cores_c,
            mm_reader_kernel, unary_writer_kernel,
            M, N, K,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            in0_dram_addr, in1_dram_addr, out_dram_addr
        );
        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);
        pass &= ll_buda::LaunchKernels(device, program);
    }

    TT_ASSERT(pass);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace ll_buda

}  // namespace tt
