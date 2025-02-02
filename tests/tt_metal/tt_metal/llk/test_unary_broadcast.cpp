// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "test_golden_impls.hpp"

using std::map;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::compute::broadcast {

enum BroadcastDim : uint8_t { ROW = 0, COL = 1, SCALAR = 2, NONE = 3 };

const map<BroadcastDim, string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
    {BroadcastDim::SCALAR, "BroadcastType::SCALAR"},
    {BroadcastDim::NONE, "BroadcastType::NONE"}};

struct UnaryBroadcastConfig {
    BroadcastDim broadcast_dim;
};

std::vector<bfloat16> gold_broadcast(std::vector<bfloat16>& src, const std::vector<uint32_t>& shape, BroadcastDim dim) {
    int num_tiles = shape.at(0);
    int num_rows = shape.at(1);
    int num_cols = shape.at(2);
    int tile_elem_count = num_rows * num_cols;

    std::vector<bfloat16> golden(num_tiles * num_cols * num_rows);

    if (dim == BroadcastDim::NONE) {
        golden = src;
    } else {
        for (int t = 0; t < num_tiles; t++) {
            int tile_offset = tile_elem_count * t;
            for (int i = 0; i < num_rows; i++) {
                for (int j = 0; j < num_cols; j++) {
                    bfloat16 broadcast_value;
                    switch (dim) {
                        case BroadcastDim::ROW: {
                            broadcast_value = src[tile_offset + j];
                            break;
                        }
                        case BroadcastDim::COL: {
                            broadcast_value = src[tile_offset + (i * num_cols)];
                            break;
                        }
                        case BroadcastDim::SCALAR: {
                            broadcast_value = src[tile_offset];
                            break;
                        }
                        default: {
                            TT_THROW("Unsupported BroadcastDim={}", dim);
                            break;
                        }
                    }

                    golden[tile_offset + (i * num_cols + j)] = broadcast_value.to_float();
                }
            }
        }
    }

    return golden;
}

void run_single_core_unary_broadcast(tt_metal::IDevice* device, const UnaryBroadcastConfig& test_config) {
    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t num_tiles = 32;
    constexpr uint32_t num_blocks = 4;
    constexpr uint32_t block_size = num_tiles / num_blocks;
    constexpr uint32_t single_tile_size = tile_width * tile_height * bfloat16::SIZEOF;
    constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_src_cb_config =
        tt_metal::CircularBufferConfig(block_size * 2 * single_tile_size, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, single_tile_size);
    auto l1_src_cb = tt_metal::CreateCircularBuffer(program, core, l1_src_cb_config);

    tt_metal::CircularBufferConfig l1_dst_cb_config =
        tt_metal::CircularBufferConfig(block_size * 2 * single_tile_size, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, single_tile_size);
    auto l1_dst_cb = tt_metal::CreateCircularBuffer(program, core, l1_dst_cb_config);

    std::map<string, string> defines = {{"BCAST_DIM", broadcast_dim_to_type.at(test_config.broadcast_dim)}};

    log_info("Testing UNARY BCAST_DIM={}", defines["BCAST_DIM"]);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto binary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unary_bcast.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {num_blocks, block_size}, .defines = defines});

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)dram_buffer_src_addr,
            (uint32_t)0,          // dram bank id
            (uint32_t)num_tiles,  // num tiles
        });

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)dram_buffer_dst_addr,
            (uint32_t)0,          // dram bank id
            (uint32_t)num_tiles,  // num tiles
        });

    std::vector<bfloat16> input0 = generate_uniform_random_vector<bfloat16>(
        -1.0f,
        1.0f,
        num_tiles * single_tile_size / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<bfloat16> golden =
        gold_broadcast(input0, {num_tiles, tile_width, tile_height}, test_config.broadcast_dim);

    auto packed_input0 = pack_vector<uint32_t, bfloat16>(input0);
    auto packed_golden = pack_vector<uint32_t, bfloat16>(golden);
    unit_tests::compute::GoldenConfig config = {
        .num_tiles_r_dim = num_tiles * tile_width / 32, .num_tiles_c_dim = tile_height / 32};
    auto tilized_input0 = unit_tests::compute::gold_standard_tilize(packed_input0, config);

    tt_metal::detail::WriteToBuffer(src_dram_buffer, tilized_input0);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, dest_buffer_data);
    auto dest_buffer_data_untilized = unit_tests::compute::gold_standard_untilize(dest_buffer_data, config);

    bool result = is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data_untilized, packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0);
        });
    ASSERT_TRUE(result);
}
}  // namespace unit_tests::compute::broadcast

class UnaryBroadcastParameterizedDeviceFixture
    : public DeviceFixture,
      public testing::WithParamInterface<unit_tests::compute::broadcast::UnaryBroadcastConfig> {};

TEST_P(UnaryBroadcastParameterizedDeviceFixture, TensixComputeSingleTileUnaryBroadcast) {
    if (this->arch_ == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    unit_tests::compute::broadcast::UnaryBroadcastConfig test_config = GetParam();
    unit_tests::compute::broadcast::run_single_core_unary_broadcast(this->devices_.at(0), test_config);
}

using namespace unit_tests::compute::broadcast;

INSTANTIATE_TEST_SUITE_P(
    ComputeSingleTileUnaryBroadcast,
    UnaryBroadcastParameterizedDeviceFixture,
    ::testing::Values(
        (UnaryBroadcastConfig){BroadcastDim::NONE},
        (UnaryBroadcastConfig){BroadcastDim::ROW},
        (UnaryBroadcastConfig){BroadcastDim::COL},
        (UnaryBroadcastConfig){BroadcastDim::SCALAR}));
