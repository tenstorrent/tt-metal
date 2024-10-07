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
#include "tt_metal/test_utils/comparison.hpp"
#include "test_tiles.hpp"
#include "tests/tt_metal/test_utils/tilization.hpp"
#include "tests/tt_metal/test_utils/print_helpers.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/compute/matmul/matmul_utils.hpp"

using namespace tt;

namespace unit_tests_common::matmul::test_matmul_single_core{

bool matmul_single_core(CommonFixture *fixture, tt_metal::Device *device, int M, int N, int K, int out_subblock_h, int out_subblock_w){
    bool pass = true;

    auto program = tt_metal::CreateScopedProgram();

    CoreCoord core = {0, 0};
    int in0_block_w = 2;
    log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
    log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
    log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
    log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_h, in0_block_w, K / in0_block_w, M / out_subblock_h);
    log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_w, in0_block_w, K / in0_block_w, N / out_subblock_w);

    uint32_t single_tile_size = 2 * 1024;
    TT_FATAL(M * in0_block_w * single_tile_size * 2 <= 130*1024, "Parameter mismatch {} {} {}", M, in0_block_w, single_tile_size);
    TT_FATAL(N * in0_block_w * single_tile_size * 2 <= 130*1024, "Parameter mismatch {} {} {}", N, in0_block_w, single_tile_size);
    TT_FATAL(M * N * single_tile_size <= 540*1024, "Parameter mismatch {} {} {}", M, N, single_tile_size);
    uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    tt_metal::InterleavedBufferConfig act_config{
                .device=device,
                .size = dram_buffer_size_act,
                .page_size = dram_buffer_size_act,
                .buffer_type = tt_metal::BufferType::DRAM
    };
    tt_metal::InterleavedBufferConfig weights_config{
                .device=device,
                .size = dram_buffer_size_weights,
                .page_size = dram_buffer_size_weights,
                .buffer_type = tt_metal::BufferType::DRAM
    };
    tt_metal::InterleavedBufferConfig dst_config{
                .device=device,
                .size = dram_buffer_size_out,
                .page_size = dram_buffer_size_out,
                .buffer_type = tt_metal::BufferType::DRAM
    };

    auto src0_dram_buffer = CreateBuffer(act_config);
    auto src1_dram_buffer = CreateBuffer(weights_config);
    auto dst_dram_buffer = CreateBuffer(dst_config);

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = M * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    uint32_t cb1_tiles = N * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {ouput_cb_index, tt::DataFormat::Float16_b},
        {interm0_cb_index, tt::DataFormat::Float16_b}
    };

    uint32_t num_output_tiles = M * N;
    CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_data_format_spec)
        .set_page_size(interm0_cb_index, single_tile_size)
        .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_output_config);

    std::vector<uint32_t> mm_reader_rt_args{
        src0_dram_buffer->address(),
        (std::uint32_t)dram_src0_noc_xy.x,
        (std::uint32_t)dram_src0_noc_xy.y,
        src1_dram_buffer->address(),
        (std::uint32_t)dram_src1_noc_xy.x,
        (std::uint32_t)dram_src1_noc_xy.y,
        (std::uint32_t)(K/in0_block_w), // num_blocks
        (std::uint32_t)(M * in0_block_w), // input 0 block num tiles
        (std::uint32_t)(N * in0_block_w), // input 1 block num tiles
        (std::uint32_t)(M * in0_block_w * single_tile_size), // input 0 block bytes
        (std::uint32_t)(N * in0_block_w * single_tile_size)}; // input 1 block bytes

    std::vector<uint32_t> writer_rt_args{
        dst_dram_buffer->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)out_subblock_h, // num tiles per sub block m
        (std::uint32_t)out_subblock_w, // num tiles per sub block n
        (std::uint32_t)M/out_subblock_h, // num sub blocks m
        (std::uint32_t)N/out_subblock_w, // num sub blocks n
        (std::uint32_t)out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row within sub-block
        (std::uint32_t)out_subblock_h * out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row of sub-blocks
        (std::uint32_t)out_subblock_w * single_tile_size}; // bytes offset to next sub-block

    auto mm_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    int num_blocks = (K/in0_block_w);

    int in0_num_subblocks = (M/out_subblock_h);
    int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    int in1_num_subblocks = (N/out_subblock_w);
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

    auto matmul_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    SHAPE shape = {1, 1, (std::uint32_t)(M * 32), (std::uint32_t)(K * 32)};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    auto activations_tile_transposed = transpose_tiles(activations, M, K, in0_block_w);
    fixture->WriteBuffer(device, src0_dram_buffer, activations_tile_transposed);

    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 32x32 identity
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    fixture->WriteBuffer(device, src1_dram_buffer, weights);

    tt_metal::SetRuntimeArgs(
        program,
        mm_reader_kernel,
        core,
        mm_reader_rt_args);

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        writer_rt_args);

    log_debug(LogTest, "Launching kernels");
    fixture->RunProgram(device, program);
    log_debug(LogTest, "Kernels done");

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, dst_dram_buffer, result_vec);

    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    auto result_flat_layout = convert_to_flat_layout(result_bfp16);
    auto result_untilized = test_utils::untilize(result_flat_layout, M*32, N*32);
    auto golden = select_columns(tensor.get_values(), M, K, std::min(K, N));
    pass &= test_utils::is_close_vectors<bfloat16> (
        golden,
        result_untilized,
        [&](const bfloat16& a, const bfloat16& b) { return tt::test_utils::is_close<bfloat16>(a, b, 0.015f); }
    );

    return pass;
}
} // namespace unit_tests_common::matmul::test_matmul_single_core

TEST_F (CommonFixture, MatmulSingleCoreSmall){
    uint32_t M = 4;
    uint32_t K = 4;
    uint32_t N = 4;
    int out_subblock_h = 4;
    int out_subblock_w = 4;
    for (unsigned int id = 0; id < devices_.size(); id ++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_single_core::matmul_single_core(this, devices_.at(id), M, N, K, out_subblock_h, out_subblock_w));
    }
}

TEST_F (CommonFixture, MatmulSingleCore){
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        log_info(LogTest, "Fast dispatch buffer memory issue, skipping for now");
        GTEST_SKIP();
    }
    uint32_t M = 16;
    uint32_t K = 16 * 12;
    uint32_t N = 16;
    int out_subblock_h = 4;
    int out_subblock_w = 2;
    for (unsigned int id = 0; id < devices_.size(); id ++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_single_core::matmul_single_core(this, devices_.at(id), M, N, K, out_subblock_h, out_subblock_w));
    }
}
