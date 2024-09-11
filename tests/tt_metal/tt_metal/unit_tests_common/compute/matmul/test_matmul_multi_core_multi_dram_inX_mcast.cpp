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

namespace unit_tests_common::matmul::test_matmul_multi_core_multi_dram_inX_mcast {

std::
    tuple<tt_metal::Program, tt_metal::KernelHandle, tt_metal::KernelHandle, tt_metal::KernelHandle, uint32_t, uint32_t>
    create_program(
        tt_metal::Device *device,
        int start_core_x,
        int start_core_y,
        int num_cores_r,
        int num_cores_c,
        int mcast_xy_offset,
        int mcast_yx_offset,
        int M,
        int N,
        int K,
        int in0_block_w,
        int out_subblock_h,
        int out_subblock_w,
        int per_core_M,
        int per_core_N) {
    tt_metal::Program program = tt_metal::CreateProgram();

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

    CoreRange mcast_senders(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + mcast_xy_offset * (num_cores_c - 1),
         (std::size_t)start_core_y + mcast_yx_offset * (num_cores_r - 1)});
    CoreRange mcast_receivers(
        {(std::size_t)start_core_x + mcast_yx_offset, (std::size_t)start_core_y + mcast_xy_offset},
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
    std::string kernel_sender = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in" + std::to_string(mcast_xy_offset) + "_mcast_sender.cpp";
    std::string kernel_receiver = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in" + std::to_string(mcast_xy_offset) + "_mcast_receiver.cpp";
    auto mm_reader_kernel_sender = tt_metal::CreateKernel(
        program,
        kernel_sender,
        mcast_senders,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto mm_reader_kernel_receiver = tt_metal::CreateKernel(
        program,
        kernel_receiver,
        mcast_receivers,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_matmul_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

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

    uint32_t in_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    uint32_t in_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

    return {
        std::move(program),
        mm_reader_kernel_sender,
        mm_reader_kernel_receiver,
        unary_writer_kernel,
        in_mcast_sender_semaphore_id,
        in_mcast_receiver_semaphore_id};
}

bool write_runtime_args_to_device(
    int in1_or_in0,
    tt_metal::Device *device,
    tt_metal::Program &program,
    int start_core_x,
    int start_core_y,
    int num_cores_r,
    int num_cores_c,
    tt_metal::KernelHandle mm_reader_kernel_sender,
    tt_metal::KernelHandle mm_reader_kernel_receiver,
    tt_metal::KernelHandle unary_writer_kernel,
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
    uint32_t in_mcast_sender_semaphore_id,
    uint32_t in_mcast_receiver_semaphore_id) {
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
            int core_x = in1_or_in0 ? core.x : start_core_x;
            int core_y = in1_or_in0 ? start_core_y : core.y;
            // log_info(LogTest, "Runtime kernel args for core {}, {}", core.x, core.y);
            CoreCoord mcast_sender = {(std::size_t)core_x, (std::size_t) core_y};
            CoreCoord core_start = {(std::size_t)(core_x + (1 - in1_or_in0)), (std::size_t) (core_y + in1_or_in0)};
            CoreCoord core_end = {(std::size_t)(core_x + (1 - in1_or_in0) * (num_cores_c - 1)), (std::size_t) (core_y + in1_or_in0 * (num_cores_r - 1))};
            auto mcast_sender_physical = device->worker_core_from_logical_core(mcast_sender);
            auto core_start_physical = device->worker_core_from_logical_core(core_start);
            auto core_end_physical = device->worker_core_from_logical_core(core_end);

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

                (std::uint32_t)core_end_physical.x,                               // in1_mcast_dest_noc_start_x
                (std::uint32_t)core_end_physical.y,                               // in1_mcast_dest_noc_start_y
                (std::uint32_t)core_start_physical.x,                             // in1_mcast_dest_noc_end_x
                (std::uint32_t)core_start_physical.y,                             // in1_mcast_dest_noc_end_y
                (std::uint32_t)(in1_or_in0 ? num_cores_r - 1 : num_cores_c - 1),  // in1_mcast_num_dests
                (std::uint32_t)mcast_sender_physical.x,                           // in1_mcast_sender_noc_x
                (std::uint32_t)mcast_sender_physical.y,                           // in1_mcast_sender_noc_y
                (std::uint32_t)in_mcast_sender_semaphore_id,
                (std::uint32_t)in_mcast_receiver_semaphore_id};
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

            int core_idx = in1_or_in0 ? core_idx_y : core_idx_x;
            if(core_idx == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_sender, core, mm_reader_args);
            } else {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_receiver, core, mm_reader_args);
            }
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
        }
    }
    return pass;
}

bool matmul_multi_core_multi_dram_inX_mcast(tt_metal::Device *device, int in1_or_in0){
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
    uint32_t in0_dram_addr = DRAM_UNRESERVED_BASE;
    uint32_t in1_dram_addr = 400 * 1024 * 1024;
    uint32_t out_dram_addr = 800 * 1024 * 1024;

    log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
    log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
    log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
    log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", per_core_M, in0_block_w, K / in0_block_w, per_core_M / out_subblock_h);
    log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", in0_block_w, per_core_N, K / in0_block_w, per_core_N / out_subblock_w);
    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bfloat16 identity
    auto golden = select_columns(tensor.get_values(), M, K, N);

    auto
        [program,
         mm_reader_kernel_sender,
         mm_reader_kernel_receiver,
         unary_writer_kernel,
         in_mcast_sender_semaphore_id,
         in_mcast_receiver_semaphore_id] =
            unit_tests_common::matmul::test_matmul_multi_core_multi_dram_inX_mcast::create_program(
                device,
                start_core_x,
                start_core_y,
                num_cores_r,
                num_cores_c,
                in1_or_in0,
                (1 - in1_or_in0),
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
        in1_or_in0,
        device,
        program,
        start_core_x,
        start_core_y,
        num_cores_r,
        num_cores_c,
        mm_reader_kernel_sender,
        mm_reader_kernel_receiver,
        unary_writer_kernel,
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
        in_mcast_sender_semaphore_id,
        in_mcast_receiver_semaphore_id);
    log_debug(LogTest, "Writing kernel runtime args to device complete");

    log_debug(LogTest, "Running Matmul {} core test", num_cores_r * num_cores_c);

    tt_metal::detail::LaunchProgram(device, program);
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
} // namespace unit_tests_common::matmul::test_matmul_multi_core

TEST_F(CommonFixture, MatmulMultiCoreMultiDRAMIn0MCast) {
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        tt::log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }
    for (unsigned int id=0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_multi_core_multi_dram_inX_mcast::matmul_multi_core_multi_dram_inX_mcast(devices_.at(id), 0));
    }
}

TEST_F(CommonFixture, MatmulMultiCoreMultiDRAMIn1MCast) {
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        tt::log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }
    for (unsigned int id=0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_multi_core_multi_dram_inX_mcast::matmul_multi_core_multi_dram_inX_mcast(devices_.at(id), 1));
    }
}
