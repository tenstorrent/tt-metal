#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include <algorithm>
#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"
#include "hostdevcommon/common_values.hpp"

using namespace tt::constants;
using namespace tt;

namespace reuse_mcast_optimized_bert_large_helpers {
using namespace tt::constants;
using namespace tt;

tt_metal::Program create_program_mcast_in0_in1(
    tt_metal::Device *device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    uint32_t single_tile_size,
    CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    tt_metal::Buffer* in0_buffer, tt_metal::Buffer* in1_buffer, tt_metal::Buffer* out_buffer
) {

    tt_metal::Program program = tt_metal::Program();

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    // Dummy cb to store one tile of zeros for padding
    uint32_t in2_block_tiles = 1;
    uint32_t in2_CB_size = in2_block_tiles * single_tile_size;

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    tt_metal::CoreRange all_cores(
        {(std::size_t) start_core_x, (std::size_t) start_core_y},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1});

    tt_metal::CoreRange left_column(
        {(std::size_t) start_core_x, (std::size_t) start_core_y},
        {(std::size_t) start_core_x, (std::size_t) start_core_y + num_cores_r - 1});

    tt_metal::CoreRange top_row(
        {(std::size_t) start_core_x, (std::size_t) start_core_y},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y});

    tt_metal::CoreRange all_except_left_column(
        {(std::size_t) start_core_x + 1, (std::size_t) start_core_y},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1});

    tt_metal::CoreRange all_except_top_row(
        {(std::size_t) start_core_x, (std::size_t) start_core_y + 1},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1});

    tt_metal::CoreRange in0_sender_in1_receiver(
        {(std::size_t) start_core_x, (std::size_t) start_core_y + 1},
        {(std::size_t) start_core_x, (std::size_t) start_core_y + num_cores_r - 1});

    tt_metal::CoreRange in0_receiver_in1_sender(
        {(std::size_t) start_core_x + 1, (std::size_t) start_core_y},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y});

    tt_metal::CoreRange in0_receiver_in1_receiver(
        {(std::size_t) start_core_x + 1, (std::size_t) start_core_y + 1},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1});

    /* TODO: CreateDataMovementKernel doesn't support CoreBlocks??
    tt_metal::CoreBlocks in0_receiver_in1_receiver_ckb_white;
    tt_metal::CoreBlocks in0_receiver_in1_receiver_ckb_black;
    bool white = true;
    for (int y = start_core_y + 1; y < start_core_y + num_cores_r; y++) {
        for (int x = start_core_x + 1; x < start_core_x + num_cores_c; x++) {
            if (white) {
                in0_receiver_in1_receiver_ckb_white.push_back(CoreCoord((std::size_t) x, (std::size_t) y));
                white = false;
            }
            else {
                in0_receiver_in1_receiver_ckb_black.push_back(CoreCoord((std::size_t) x, (std::size_t) y));
                white = true;
            }
        }
    }
    std::cout << "white" << std::endl;
    for (auto &it : in0_receiver_in1_receiver_ckb_white) {
        std::cout << std::get<CoreCoord>(it).str() << std::endl;
    }
    std::cout << "black" << std::endl;
    for (auto &it : in0_receiver_in1_receiver_ckb_black) {
        std::cout << std::get<CoreCoord>(it).str() << std::endl;
    }
    */

    /* Old compile time args
    bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
    tt_metal::KernelArgs reader_writer_compile_time_args;
    if (tile_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_writer_compile_time_args = tt_metal::KernelArgs(all_cores, {1, (std::uint32_t)log2(single_tile_size)});
    } else {
        reader_writer_compile_time_args = tt_metal::KernelArgs(all_cores, {0, 0});
    }
    */
    // Mcast args
    auto in0_mcast_sender_semaphore_vec = tt_metal::CreateSemaphores(program, device, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_vec = tt_metal::CreateSemaphores(program, device, all_cores, INVALID);
    auto in1_mcast_sender_semaphore_vec = tt_metal::CreateSemaphores(program, device, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore_vec = tt_metal::CreateSemaphores(program, device, all_cores, INVALID);

    // Address should be same across all cores
    auto in0_mcast_sender_semaphore = in0_mcast_sender_semaphore_vec[0];
    auto in0_mcast_receiver_semaphore = in0_mcast_receiver_semaphore_vec[0];
    auto in1_mcast_sender_semaphore = in1_mcast_sender_semaphore_vec[0];
    auto in1_mcast_receiver_semaphore = in1_mcast_receiver_semaphore_vec[0];

    CoreCoord top_left_core = {(std::size_t) start_core_x, (std::size_t) start_core_y};
    CoreCoord top_left_core_plus_one = {(std::size_t) start_core_x + 1, (std::size_t) start_core_y + 1};
    CoreCoord bottom_right_core = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto top_left_core_plus_one_physical = device->worker_core_from_logical_core(top_left_core_plus_one);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
    std::uint32_t tile_size_pow2_exponent = tile_size_is_power_of_two ? (std::uint32_t)log2(single_tile_size) : 0;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    tt_metal::KernelArgs in0_sender_compile_time_args = tt_metal::KernelArgs(
        left_column, {
            // interleaved accessor args
            (std::uint32_t) tile_size_is_power_of_two,
            (std::uint32_t) tile_size_pow2_exponent,
            (std::uint32_t) in0_is_dram,

            // in0 tensor args
            (std::uint32_t)  in0_buffer->address(), // in0_tensor_addr
            (std::uint32_t)  1, // in0_tensor_stride_w
            (std::uint32_t)  K, // in0_tensor_stride_h
            (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride
            // in0 block args
            (std::uint32_t)  in0_block_w, // in0_block_w
            (std::uint32_t)  per_core_M, // in0_block_h
            (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in0 mcast args
            (std::uint32_t)  bottom_right_core_physical.x, // in0_mcast_dest_noc_start_x
            (std::uint32_t)  top_left_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
            (std::uint32_t)  in0_mcast_sender_semaphore->address(),
            (std::uint32_t)  in0_mcast_receiver_semaphore->address(),
            (std::uint32_t)  (num_cores_c - 1), // in0_mcast_num_dests
            // batch args
            (std::uint32_t)  M * K, // MtKt
            (std::uint32_t)  B // batch
        }
    );
    tt_metal::KernelArgs in1_sender_writer_compile_time_args = tt_metal::KernelArgs(
        top_row, {
            // interleaved accessor args
            (std::uint32_t) tile_size_is_power_of_two,
            (std::uint32_t) tile_size_pow2_exponent,
            (std::uint32_t) in1_is_dram,
            (std::uint32_t) out_is_dram,

            // READER
            // in1 tensor args
            (std::uint32_t)  in1_buffer->address(), // in1_tensor_addr
            (std::uint32_t)  1, // in1_tensor_stride_w
            (std::uint32_t)  N, // in1_tensor_stride_h
            (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride
            // in1 block args
            (std::uint32_t)  per_core_N, // in1_block_w
            (std::uint32_t)  in0_block_w, //in1_block_h
            (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in1 mcast args
            (std::uint32_t)  bottom_right_core_physical.y, // in1_mcast_dest_noc_start_y
            (std::uint32_t)  top_left_core_plus_one_physical.y, // in1_mcast_dest_noc_end_y
            (std::uint32_t)  in1_mcast_sender_semaphore->address(),
            (std::uint32_t)  in1_mcast_receiver_semaphore->address(),
            (std::uint32_t)  (num_cores_r - 1), // in1_mcast_num_dests
            // batch args
            (std::uint32_t)  K * N, // KtNt
            (std::uint32_t)  B, // batch
            (std::uint32_t)  bcast_batch, // bcast_B

            // WRITER
            // out tensor args
            (std::uint32_t)  out_buffer->address(), // out_tensor_addr
            (std::uint32_t)  1, // out_tensor_stride_w
            (std::uint32_t)  N,  // out_tensor_stride_h
            (std::uint32_t)  out_subblock_w, // out_tensor_next_subblock_stride_w
            (std::uint32_t)  out_subblock_h * N, // out_tensor_next_subblock_stride_h
            // out subblock args
            (std::uint32_t)  out_subblock_w, // out_subblock_w
            (std::uint32_t)  out_subblock_h, // out_subblock_h
            (std::uint32_t)  (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
            // batch args
            (std::uint32_t)  M * N // MtNt
        }
    );
    tt_metal::KernelArgs in0_receiver_compile_time_args = tt_metal::KernelArgs(
        all_except_left_column, {
            // in0 block args
            (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in0 mcast args
            (std::uint32_t)  top_left_core_physical.x, // in0_mcast_sender_noc_x
            (std::uint32_t)  in0_mcast_sender_semaphore->address(),
            (std::uint32_t)  in0_mcast_receiver_semaphore->address(),
            // batch args
            (std::uint32_t)  B // batch
        }
    );
    tt_metal::KernelArgs in1_receiver_writer_compile_time_args = tt_metal::KernelArgs(
        all_except_top_row, {
            // interleaved accessor args
            (std::uint32_t) tile_size_is_power_of_two,
            (std::uint32_t) tile_size_pow2_exponent,
            (std::uint32_t) out_is_dram,

            // READER
            // in1 block args
            (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in1 mcast args
            (std::uint32_t)  top_left_core_physical.y, // in1_mcast_sender_noc_y
            (std::uint32_t)  in1_mcast_sender_semaphore->address(),
            (std::uint32_t)  in1_mcast_receiver_semaphore->address(),
            // batch args
            (std::uint32_t)  B, // batch

            // WRITER
            // out tensor args
            (std::uint32_t)  out_buffer->address(), // out_tensor_addr
            (std::uint32_t)  1, // out_tensor_stride_w
            (std::uint32_t)  N,  // out_tensor_stride_h
            (std::uint32_t)  out_subblock_w, // out_tensor_next_subblock_stride_w
            (std::uint32_t)  out_subblock_h * N, // out_tensor_next_subblock_stride_h
            // out subblock args
            (std::uint32_t)  out_subblock_w, // out_subblock_w
            (std::uint32_t)  out_subblock_h, // out_subblock_h
            (std::uint32_t)  (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
            // batch args
            (std::uint32_t)  M * N // MtNt
        }
    );

    auto mm_kernel_in0_sender = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        left_column,
        in0_sender_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_0_default);
    auto mm_kernel_in1_sender_writer = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        top_row,
        in1_sender_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_1_default);

    auto mm_kernel_in1_receiver_writer = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        in0_sender_in1_receiver,
        in1_receiver_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_1_default);

    auto mm_kernel_in0_receiver = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        in0_receiver_in1_sender,
        in0_receiver_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_0_default);

    /* TODO: CreateDataMovementKernel doesn't support CoreBlocks??
    auto mm_kernel_in0_receiver_ckb_white = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        in0_receiver_in1_receiver_ckb_white,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto mm_kernel_in1_receiver_writer_ckb_white = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        in0_receiver_in1_receiver_ckb_white,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    auto mm_kernel_in0_receiver_ckb_black = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        in0_receiver_in1_receiver_ckb_black,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_0_default);

    auto mm_kernel_in1_receiver_writer_ckb_black = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        in0_receiver_in1_receiver_ckb_black,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_1_default);
    */

    /* Uncomment if we don't checkerboard
    auto mm_kernel_checkerboard_in0_receiver = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        in0_receiver_in1_receiver,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto mm_kernel_checkerboard_in1_receiver_writer = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        in0_receiver_in1_receiver,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);
    */

    // Compute kernel compile time args
    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles
        B // batch
    };

    // Create compute kernel
    tt_metal::KernelArgs mm_args = tt_metal::KernelArgs(all_cores, compute_kernel_args);
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_large_block_zm.cpp",
        all_cores,
        mm_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode
    );

    // Parameters for last row, col, or block
    uint32_t last_block_h = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_block_w = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_block_num_nonzero_subblocks_h = (last_block_h  - 1) / out_subblock_h + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_block_w  - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_h = last_block_h % out_subblock_h == 0 ? out_subblock_h : last_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w = last_block_w % out_subblock_w == 0 ? out_subblock_w : last_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip = single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =  (out_subblock_w * out_subblock_h) * (per_core_N / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip = (per_core_M / out_subblock_h - last_block_num_nonzero_subblocks_h) * (per_core_N * out_subblock_h);

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            uint32_t src0_cb_index = 0;
            uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
            auto cb_src0 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src0_cb_index,
                core,
                cb0_tiles,
                cb0_tiles * single_tile_size,
                cb_data_format
            );

            uint32_t src1_cb_index = 1;
            uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
            auto cb_src1 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src1_cb_index,
                core,
                cb1_tiles,
                cb1_tiles * single_tile_size,
                cb_data_format
            );

            uint32_t src2_cb_index = 2;
            uint32_t cb2_tiles = in2_block_tiles * 2; // double buffer
            auto cb_src2 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src2_cb_index,
                core,
                cb2_tiles,
                cb2_tiles * single_tile_size,
                cb_data_format
            );

            uint32_t ouput_cb_index = 16; // output operands start at index 16
            auto cb_output = tt_metal::CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                out_CB_tiles,
                out_CB_size,
                cb_data_format
            );

            uint32_t interm0_cb_index = 24;
            auto cb_interm0 = tt_metal::CreateCircularBuffer(
                program,
                device,
                interm0_cb_index,
                core,
                out_CB_tiles,
                out_CB_size,
                cb_output->address(),
                cb_data_format
            );

            CoreCoord left_core    = {(std::size_t) start_core_x, (std::size_t) core.y};
            CoreCoord left_core_plus_one    = {(std::size_t) start_core_x + 1, (std::size_t) core.y};
            CoreCoord right_core   = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) core.y};
            CoreCoord top_core     = {(std::size_t) core.x, (std::size_t) start_core_y};
            CoreCoord top_core_plus_one     = {(std::size_t) core.x, (std::size_t) start_core_y + 1};
            CoreCoord bottom_core  = {(std::size_t) core.x, (std::size_t) start_core_y + num_cores_r - 1};

            auto left_core_physical = device->worker_core_from_logical_core(left_core);
            auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
            auto right_core_physical = device->worker_core_from_logical_core(right_core);
            auto top_core_physical = device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

            // in0 sender and in1 sender
            if(core_idx_x == 0 and core_idx_y == 0) {
                std::vector<uint32_t> mm_in0_sender_args =  {
                    // in0 tensor args
                    (std::uint32_t)  K * per_core_M * core_idx_y, // in0_tensor_start_tile_id
                    // in0 mcast args
                    (std::uint32_t)  right_core_physical.y, // in0_mcast_dest_noc_start_y
                    (std::uint32_t)  left_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y

                    // padding args
                    (std::uint32_t) per_core_M // last_block_h
                };
                std::vector<uint32_t> mm_in1_sender_writer_args = {
                    // READER
                    // in1 tensor args
                    (std::uint32_t)  per_core_N * core_idx_x, //in1_tensor_start_tile_id
                    // in1 mcast args
                    (std::uint32_t)  bottom_core_physical.x, // in1_mcast_dest_noc_start_x
                    (std::uint32_t)  top_core_plus_one_physical.x, // in1_mcast_dest_noc_end_x

                    // WRITER
                    // out tensor args
                    (std::uint32_t)  core_idx_x * per_core_N + core_idx_y * per_core_M * N, // out_tensor_start_tile_id

                    // padding args (READER)
                    (std::uint32_t)  per_core_N, // last_block_w
                    // padding args (WRITER)
                    (std::uint32_t)  per_core_M / out_subblock_h,
                    (std::uint32_t)  out_subblock_h,
                    (std::uint32_t)  0,
                    (std::uint32_t)  per_core_N / out_subblock_w,
                    (std::uint32_t)  out_subblock_w,
                    (std::uint32_t)  0,
                    (std::uint32_t)  0
                };

                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_sender, core, mm_in0_sender_args); // RISCV_0_default
                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_sender_writer, core, mm_in1_sender_writer_args); // RISCV_1_default
            }
            // in0 sender and in1 receiver
            else if (core_idx_x == 0 and core_idx_y != 0) {
                std::vector<uint32_t> mm_in0_sender_args = {
                    // in0 tensor args
                    (std::uint32_t)  K * per_core_M * core_idx_y, // in0_tensor_start_tile_id
                    // in0 mcast args
                    (std::uint32_t)  right_core_physical.y, // in0_mcast_dest_noc_start_y
                    (std::uint32_t)  left_core_plus_one_physical.y // in0_mcast_dest_noc_end_y
                };

                std::vector<uint32_t> mm_in1_receiver_writer_args = {
                    // READER
                    // in1 mcast args
                    (std::uint32_t)  top_core_physical.x, // in1_mcast_sender_noc_x

                    // WRITER
                    // out tensor args
                    (std::uint32_t)  core_idx_x * per_core_N + core_idx_y * per_core_M * N // out_tensor_start_tile_id
                };

                if (core_idx_y == num_cores_r - 1) {
                    // padding args (READER)
                    mm_in0_sender_args.push_back(last_block_h); // last_block_h

                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                } else {
                    // padding args (READER)
                    mm_in0_sender_args.push_back(per_core_M);

                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                }

                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_sender, core, mm_in0_sender_args); // RISCV_0_default
                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_receiver_writer, core, mm_in1_receiver_writer_args); // RISCV_1_default
            }
            // in0 receiver and in 1 sender
            else if (core_idx_x != 0 and core_idx_y == 0) {
                std::vector<uint32_t> mm_in0_receiver_args = {
                    // in0 mcast args
                    (std::uint32_t)  left_core_physical.y // in0_mcast_sender_noc_y
                };
                std::vector<uint32_t> mm_in1_sender_writer_args = {
                    // READER
                    // in1 tensor args
                    (std::uint32_t)  per_core_N * core_idx_x, //in1_tensor_start_tile_id
                    // in1 mcast args
                    (std::uint32_t)  bottom_core_physical.x, // in1_mcast_dest_noc_start_x
                    (std::uint32_t)  top_core_plus_one_physical.x, // in1_mcast_dest_noc_end_x

                    // WRITER
                    // out tensor args
                    (std::uint32_t)  core_idx_x * per_core_N + core_idx_y * per_core_M * N // out_tensor_start_tile_id
                };

                if (core_idx_x == num_cores_c - 1) {
                    // padding args (READER)
                    mm_in1_sender_writer_args.push_back(last_block_w);

                    // padding args (WRITER)
                    mm_in1_sender_writer_args.push_back(per_core_M /out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    // padding args (READER)
                    mm_in1_sender_writer_args.push_back(per_core_N);

                    // padding args (WRITER)
                    mm_in1_sender_writer_args.push_back(per_core_M /out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_subblock_w);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(0);
                }

                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_receiver, core, mm_in0_receiver_args); // RISCV_1_default
                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_sender_writer, core, mm_in1_sender_writer_args); // RISCV_0_default
            }
            // in0 receiver and in 1 receiver
            else {
                std::vector<uint32_t> mm_checkerboard_in0_receiver_args = {
                    // in0 mcast args
                    (std::uint32_t)  left_core_physical.y // in0_mcast_sender_noc_y
                };
                std::vector<uint32_t> mm_checkerboard_in1_receiver_writer_args = {
                    // READER
                    // in1 mcast args
                    (std::uint32_t)  top_core_physical.x, // in1_mcast_sender_noc_x

                    // WRITER
                    // out tensor args
                    (std::uint32_t)  core_idx_x * per_core_N + core_idx_y * per_core_M * N // out_tensor_start_tile_id
                };

                if (core_idx_x == num_cores_c - 1 and core_idx_y == num_cores_r - 1) {
                    // padding args (WRITER)
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else if (core_idx_y == num_cores_r - 1) {
                    // padding args (WRITER)
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_checkerboard_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(0);
                    mm_checkerboard_in1_receiver_writer_args.push_back(0);
                } else if (core_idx_x == num_cores_c - 1) {
                    // padding args (WRITER)
                    mm_checkerboard_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(0);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_checkerboard_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    // padding args (WRITER)
                    mm_checkerboard_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_checkerboard_in1_receiver_writer_args.push_back(0);
                    mm_checkerboard_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_checkerboard_in1_receiver_writer_args.push_back(0);
                    mm_checkerboard_in1_receiver_writer_args.push_back(0);
                }

                // Checkerboard
                // white
                if ((core_idx_x + core_idx_y) % 2 == 0) {
                    auto mm_kernel_in0_receiver_ckb_white = tt_metal::CreateDataMovementKernel(
                        program,
                        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
                        core,
                        in0_receiver_compile_time_args,
                        tt_metal::DataMovementProcessor::RISCV_1,
                        tt_metal::NOC::RISCV_1_default);

                    auto mm_kernel_in1_receiver_writer_ckb_white = tt_metal::CreateDataMovementKernel(
                        program,
                        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
                        core,
                        in1_receiver_writer_compile_time_args,
                        tt_metal::DataMovementProcessor::RISCV_0,
                        tt_metal::NOC::RISCV_0_default);

                    tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_receiver_ckb_white, core, mm_checkerboard_in0_receiver_args); // RISCV_1_default
                    tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_receiver_writer_ckb_white, core, mm_checkerboard_in1_receiver_writer_args); // RISCV_0_default
                }
                // black
                else {
                    auto mm_kernel_in0_receiver_ckb_black = tt_metal::CreateDataMovementKernel(
                        program,
                        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
                        core,
                        in0_receiver_compile_time_args,
                        tt_metal::DataMovementProcessor::RISCV_1,
                        tt_metal::NOC::RISCV_0_default);

                    auto mm_kernel_in1_receiver_writer_ckb_black = tt_metal::CreateDataMovementKernel(
                        program,
                        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
                        core,
                        in1_receiver_writer_compile_time_args,
                        tt_metal::DataMovementProcessor::RISCV_0,
                        tt_metal::NOC::RISCV_1_default);

                    tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_receiver_ckb_black, core, mm_checkerboard_in0_receiver_args); // RISCV_1_default
                    tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_receiver_writer_ckb_black, core, mm_checkerboard_in1_receiver_writer_args); // RISCV_0_default
                }

                /* Uncomment if we don't checkerboard
                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_checkerboard_in0_receiver, core, mm_checkerboard_in0_receiver_args); // RISCV_1_default
                tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_checkerboard_in1_receiver_writer, core, mm_checkerboard_in1_receiver_writer_args); // RISCV_0_default
                */
            }
        }
    }

    return std::move(program);
}

}

namespace tt {

namespace tt_metal {


Tensor matmul_multi_core_reuse_mcast_optimized_bert_large_(const Tensor &a, const Tensor &b, const MemoryConfig& mem_config, bool bcast_batch, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch) {

    const auto& ashape = a.shape(), bshape = b.shape();

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

    TT_ASSERT(a.dtype() == b.dtype());
    TT_ASSERT(a.dtype() == tt::tt_metal::DataType::BFLOAT16 || a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");

    tt_metal::Device *device = a.device();

    // TODO: CHANGE TO FUNCTION CONVERSION
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }
    if (cb_data_format != output_cb_data_format) {
        log_warning("Input tensor datatype does not match target output dataformat. Defaulting to input dataformat.");
    }
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();
    tt_metal::Buffer *in1_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);
    TT_ASSERT(in1_buffer->size() % single_tile_size == 0);

    TT_ASSERT(ashape[3] == bshape[2] && "Dimension K (A.shape[2] and B.shape[3]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(ashape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(ashape[3] % TILE_WIDTH == 0);
    TT_ASSERT(bshape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(bshape[3] % TILE_WIDTH == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = ashape[0]*ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;

    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }
    TT_ASSERT(Kt % in0_block_w == 0);

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = (Mt - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (Nt - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    std::array<uint32_t, 4> cshape{ashape[0], ashape[1], ashape[2], bshape[3]}; // C=A*B, N1MK*11KN->N1MN
    tt_metal::Tensor output = tt_metal::Tensor(cshape, a.dtype(), tt::tt_metal::Layout::TILE, device, mem_config);
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program;

    if (core_range.x > 1 && core_range.y > 1) {
        program = reuse_mcast_optimized_bert_large_helpers::create_program_mcast_in0_in1(
            device,
            cb_data_format,
            math_fidelity,
            single_tile_size,
            core_range,
            B, Mt, Nt, Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            in0_buffer, in1_buffer, out_buffer
        );
    } else if (core_range.x > 1) {
        // Refer to bmm_op_multi_core_reuse_mcast_padding_generalized.cpp
        TT_ASSERT(false, "mcast_in0 is not implemented yet.");
    } else {
        // Refer to bmm_op_multi_core_reuse_mcast_padding_generalized.cpp
        TT_ASSERT(false, "mcast_in1 is not implemented yet.");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool pass = true;
    pass &= tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::LaunchKernels(device, program);

    TT_ASSERT(pass);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor matmul_multi_core_reuse_mcast_optimized_bert_large(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch) {
    return matmul_multi_core_reuse_mcast_optimized_bert_large_(a, b, mem_config, true, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}
// bmm_multi_core_reuse_mcast_optimized_bert_large not used

}  // namespace tt_metal

}  // namespace tt
