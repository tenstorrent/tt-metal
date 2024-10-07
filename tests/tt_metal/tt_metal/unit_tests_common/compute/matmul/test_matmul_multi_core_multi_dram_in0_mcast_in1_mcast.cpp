// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tests/tt_metal/test_utils/tilization.hpp"
#include "tests/tt_metal/test_utils/print_helpers.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/compute/matmul/matmul_utils.hpp"
using namespace tt;
namespace unit_tests_common::matmul::test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast {

std::tuple<
    tt_metal::ScopedProgramHandle,
    tt_metal::KernelHandle,
    tt_metal::KernelHandle,
    tt_metal::KernelHandle,
    tt_metal::KernelHandle,
    tt_metal::KernelHandle,
    tt_metal::KernelHandle,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t>
create_program(
    tt_metal::Device *device,
    int start_core_x,
    int start_core_y,
    int num_cores_r,
    int num_cores_c,
    int M,
    int N,
    int K,
    int in0_block_w,
    int out_subblock_h,
    int out_subblock_w,
    int per_core_M,
    int per_core_N) {
    auto program = tt_metal::CreateScopedProgram();

    uint32_t single_tile_size = 2 * 1024;
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;
    TT_FATAL(in0_CB_size <= 130*1024, "in0_CB_size {} too large", in0_CB_size);
    TT_FATAL(in1_CB_size <= 130*1024, "in1_CB_size {} too large", in1_CB_size);
    TT_FATAL(out_CB_size <= 540*1024, "out_CB_size {} too large", out_CB_size);

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    CoreRange left_column(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

    CoreRange all_except_left_column(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    CoreRange in0_sender_in1_sender(
        {(std::size_t)start_core_x, (std::size_t)start_core_y}, {(std::size_t)start_core_x, (std::size_t)start_core_y});

    CoreRange in0_sender_in1_receiver(
        {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

    CoreRange in0_receiver_in1_sender(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

    CoreRange in0_receiver_in1_receiver(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {ouput_cb_index, tt::DataFormat::Float16_b},
        {interm0_cb_index, tt::DataFormat::Float16_b}
    };
    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            CoreCoord core = {(std::size_t) start_core_x + j, (std::size_t) start_core_y + i};

            uint32_t src0_cb_index = 0;
            uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
            tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
            auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t src1_cb_index = 1;
            uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
            tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
            auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

            CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});
            tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(out_CB_size, partials_and_out_data_format_spec)
                .set_page_size(ouput_cb_index, single_tile_size)
                .set_page_size(interm0_cb_index, single_tile_size);
            auto cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_output_config);
        }
    }

    auto mm_reader_kernel_in0_sender_in1_sender = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_sender_in1_sender.cpp",
        in0_sender_in1_sender,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default});

    auto mm_reader_kernel_in0_sender_in1_receiver = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_sender_in1_receiver.cpp",
        in0_sender_in1_receiver,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default});

    auto mm_reader_kernel_in0_receiver_in1_sender = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_receiver_in1_sender.cpp",
        in0_receiver_in1_sender,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto mm_reader_kernel_in0_receiver_in1_receiver = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_receiver_in1_receiver.cpp",
        in0_receiver_in1_receiver,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel_noc0 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_matmul_tile_layout.cpp",
        all_except_left_column,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto unary_writer_kernel_noc1 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_matmul_tile_layout.cpp",
        left_column,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default});

    int num_blocks = (K/in0_block_w);

    int in0_num_subblocks = (per_core_M/out_subblock_h);
    int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    int in1_num_subblocks = (per_core_N/out_subblock_w);
    int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    int in1_per_core_w = out_subblock_w * in1_num_subblocks;

    int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        uint(in0_block_w),
        uint(in0_num_subblocks),
        uint(in0_block_num_tiles),
        uint(in0_subblock_num_tiles),

        uint(in1_num_subblocks),
        uint(in1_block_num_tiles),
        uint(in1_per_core_w),

        uint(num_blocks),

        uint(out_subblock_h),
        uint(out_subblock_w),
        uint(out_subblock_num_tiles)
    };

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    uint32_t in0_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    uint32_t in0_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    uint32_t in1_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    uint32_t in1_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

    return {
        std::move(program),
        mm_reader_kernel_in0_sender_in1_sender,
        mm_reader_kernel_in0_sender_in1_receiver,
        mm_reader_kernel_in0_receiver_in1_sender,
        mm_reader_kernel_in0_receiver_in1_receiver,
        unary_writer_kernel_noc0,
        unary_writer_kernel_noc1,
        in0_mcast_sender_semaphore_id,
        in0_mcast_receiver_semaphore_id,
        in1_mcast_sender_semaphore_id,
        in1_mcast_receiver_semaphore_id};
}

bool write_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::ProgramHandle program,
    int start_core_x,
    int start_core_y,
    int num_cores_r,
    int num_cores_c,
    tt_metal::KernelHandle mm_reader_kernel_in0_sender_in1_sender,
    tt_metal::KernelHandle mm_reader_kernel_in0_sender_in1_receiver,
    tt_metal::KernelHandle mm_reader_kernel_in0_receiver_in1_sender,
    tt_metal::KernelHandle mm_reader_kernel_in0_receiver_in1_receiver,
    tt_metal::KernelHandle unary_writer_kernel_noc0,
    tt_metal::KernelHandle unary_writer_kernel_noc1,
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
    uint32_t out_dram_addr,
    uint32_t in0_mcast_sender_semaphore_id,
    uint32_t in1_mcast_sender_semaphore_id,
    uint32_t in0_mcast_receiver_semaphore_id,
    uint32_t in1_mcast_receiver_semaphore_id) {
    bool pass = true;
    uint32_t single_tile_size = 2 * 1024;

    uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    TT_FATAL(in0_dram_addr + dram_buffer_size_act < 1024 * 1024 * 1024, "addr {} + size {} too large", in0_dram_addr, dram_buffer_size_act);
    TT_FATAL(in1_dram_addr + dram_buffer_size_weights < 1024 * 1024 * 1024, "addr {} + size {} too large", in1_dram_addr, dram_buffer_size_weights);
    TT_FATAL(out_dram_addr + dram_buffer_size_out < 1024 * 1024 * 1024, "addr {} + size {} too large", out_dram_addr, dram_buffer_size_out);

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};
            // log_info(LogTest, "Runtime kernel args for core {}, {}", core.x, core.y);

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
            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)in0_dram_addr,                // in0_tensor_addr
                (std::uint32_t)K * per_core_M * core_idx_y,  // in0_tensor_start_tile_id
                (std::uint32_t)1,                            // in0_tensor_stride_w
                (std::uint32_t)K,                            // in0_tensor_stride_h
                (std::uint32_t)in0_block_w,                  // in0_tensor_next_block_stride

                (std::uint32_t)in0_block_w,               // in0_block_w
                (std::uint32_t)per_core_M,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

                (std::uint32_t)in1_dram_addr,            // in1_tensor_addr
                (std::uint32_t)per_core_N * core_idx_x,  // in1_tensor_start_tile_id
                (std::uint32_t)1,                        // in1_tensor_stride_w
                (std::uint32_t)N,                        // in1_tensor_stride_h
                (std::uint32_t)in0_block_w * N,          // in1_tensor_next_block_stride

                (std::uint32_t)per_core_N,                // in1_block_w
                (std::uint32_t)in0_block_w,               // in1_block_h
                (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

                (std::uint32_t)K / in0_block_w,  // num_blocks

                (std::uint32_t)right_core_physical.x,          // in0_mcast_dest_noc_start_x
                (std::uint32_t)right_core_physical.y,          // in0_mcast_dest_noc_start_y
                (std::uint32_t)left_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
                (std::uint32_t)left_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y
                (std::uint32_t)(num_cores_c - 1),              // in0_mcast_num_dests
                (std::uint32_t)left_core_physical.x,           // in0_mcast_sender_noc_x
                (std::uint32_t)left_core_physical.y,           // in0_mcast_sender_noc_y
                (std::uint32_t)in0_mcast_sender_semaphore_id,
                (std::uint32_t)in0_mcast_receiver_semaphore_id,

                (std::uint32_t)bottom_core_physical.x,        // in0_mcast_dest_noc_start_x
                (std::uint32_t)bottom_core_physical.y,        // in0_mcast_dest_noc_start_y
                (std::uint32_t)top_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
                (std::uint32_t)top_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y
                (std::uint32_t)(num_cores_r - 1),             // in0_mcast_num_dests
                (std::uint32_t)top_core_physical.x,           // in0_mcast_sender_noc_x
                (std::uint32_t)top_core_physical.y,           // in0_mcast_sender_noc_y
                (std::uint32_t)in1_mcast_sender_semaphore_id,
                (std::uint32_t)in1_mcast_receiver_semaphore_id};
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

            if(core_idx_x == 0 and core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_sender, core, mm_reader_args); // RISCV_0_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1, core, writer_args); // RISCV_1_default
            } else if (core_idx_x == 0 and core_idx_y != 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_receiver, core, mm_reader_args); // RISCV_0_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1, core, writer_args); // RISCV_1_default
            } else if (core_idx_x != 0 and core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_sender, core, mm_reader_args); // RISCV_1_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0, core, writer_args); // RISCV_0_default
            } else {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_receiver, core, mm_reader_args); // RISCV_1_default
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0, core, writer_args); // RISCV_0_default
            }
        }
    }

    return pass;
}

bool matmul_multi_core_multi_dram_in0_mcast_in1_mcast(tt_metal::Device *device){
    bool pass = true;
    int start_core_x = 0;
    int start_core_y = 0;
    int num_cores_r = device->compute_with_storage_grid_size().y;
    int num_cores_c = device->compute_with_storage_grid_size().x;

    uint32_t M = 16 * num_cores_r;
    uint32_t K = 16 * 12;
    uint32_t N = 16 * num_cores_c;
    int out_subblock_h = 4;
    int out_subblock_w = 2;
    int in0_block_w = 2;
    int per_core_M = M / num_cores_r;
    int per_core_N = N / num_cores_c;
    uint32_t single_tile_size = 2 * 1024;
    uint32_t in0_dram_addr = device->get_base_allocator_addr(HalMemType::DRAM);
    uint32_t in1_dram_addr = 400 * 1024 * 1024;
    uint32_t out_dram_addr = 800 * 1024 * 1024;


    log_info(LogTest, "Grid size = {}x{}", num_cores_r, num_cores_c);
    log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
    log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
    log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
    log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", per_core_M, in0_block_w, K / in0_block_w, per_core_M / out_subblock_h);
    log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", in0_block_w, per_core_N, K / in0_block_w, per_core_N / out_subblock_w);
    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 identity
    auto golden = select_columns(tensor.get_values(), M, K, N);

    auto
        [program,
         mm_reader_kernel_in0_sender_in1_sender,
         mm_reader_kernel_in0_sender_in1_receiver,
         mm_reader_kernel_in0_receiver_in1_sender,
         mm_reader_kernel_in0_receiver_in1_receiver,
         unary_writer_kernel_noc0,
         unary_writer_kernel_noc1,
         in0_mcast_sender_semaphore_id,
         in0_mcast_receiver_semaphore_id,
         in1_mcast_sender_semaphore_id,
         in1_mcast_receiver_semaphore_id] =
            create_program(
                device,
                start_core_x,
                start_core_y,
                num_cores_r,
                num_cores_c,
                M,
                N,
                K,
                in0_block_w,
                out_subblock_h,
                out_subblock_w,
                per_core_M,
                per_core_N);

    log_debug(LogTest, "Scattering inputs (activation & weights) to dram channels using tiled layout");
    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    pass &= move_tiles_to_dram(device, activations, M, K, in0_dram_addr);

    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    pass &= move_tiles_to_dram(device, weights, K, N, in1_dram_addr);
    log_debug(LogTest, "Copying inputs to dram complete");

    log_debug(LogTest, "Writing kernel runtime args to device");

    pass &= write_runtime_args_to_device(
        device,
        program,
        start_core_x,
        start_core_y,
        num_cores_r,
        num_cores_c,
        mm_reader_kernel_in0_sender_in1_sender,
        mm_reader_kernel_in0_sender_in1_receiver,
        mm_reader_kernel_in0_receiver_in1_sender,
        mm_reader_kernel_in0_receiver_in1_receiver,
        unary_writer_kernel_noc0,
        unary_writer_kernel_noc1,
        M,
        N,
        K,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        in0_dram_addr,
        in1_dram_addr,
        out_dram_addr,
        in0_mcast_sender_semaphore_id,
        in1_mcast_sender_semaphore_id,
        in0_mcast_receiver_semaphore_id,
        in1_mcast_receiver_semaphore_id);
    log_debug(LogTest, "Writing kernel runtime args to device complete");

    log_debug(LogTest, "Running Matmul {} core test", num_cores_r * num_cores_c);

    auto* program_ptr = ProgramPool::instance().get_program(program);
    tt_metal::detail::LaunchProgram(device, *program_ptr);
    log_debug(LogTest, "Matmul test done");

    log_debug(LogTest, "Gathering data back from dram and checking against golden");

    for(int i = 0; i < M; i++) {
        auto row = get_row_slice(golden, M, i, M * 32, N * 32);
        for(int j = 0; j < N; j++) {
            auto golden_tile = get_col_slice(row, N, j, 32, N * 32);
            int tile_id = i * N + j;
            int dram_bank = tile_id % device->num_dram_channels();
            uint32_t dram_address = ((tile_id / device->num_dram_channels()) * single_tile_size) + out_dram_addr;
            std::vector<uint32_t> result_vec;
            tt_metal::detail::ReadFromDeviceDRAMChannel(device, dram_bank, dram_address, single_tile_size, result_vec);
            auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
            auto result_flat_layout = convert_to_flat_layout(result_bfp16);

            // log_info(LogTest, "Tile id {} on dram bank {}, address {}", tile_id, dram_bank, dram_address);
            // print_vec(result_flat_layout, 32, 32, "Result - tile#" + std::to_string(tile_id));
            pass &= (golden_tile == result_flat_layout);
        }
    }
    log_debug(LogTest, "Golden check complete");
    return pass;
}

} // namespace unit_tests_common::matmul::test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast

TEST_F(CommonFixture, MatmulMultiCoreMultiDRAMIn0MCastIn1MCast) {
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        tt::log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }
    for (unsigned int id=0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast::matmul_multi_core_multi_dram_in0_mcast_in1_mcast(devices_.at(id)));
    }
}
