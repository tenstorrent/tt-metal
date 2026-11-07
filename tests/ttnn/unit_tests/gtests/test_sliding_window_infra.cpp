// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/api/tt-metalium/core_coord.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::sliding_window::test {

using namespace tt::tt_metal;

class SlidingWindowTestFixture : public testing::TestWithParam<SlidingWindowConfig> {};

TEST_P(SlidingWindowTestFixture, SlidingWindowHash) {
    const auto& sliding_window_a = GetParam();

    // start of same input
    auto sliding_window_b = sliding_window_a;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_EQ(sliding_window_a.get_hash(), sliding_window_b.get_hash());

    // flip snap_to_tile
    sliding_window_b.snap_to_tile = !sliding_window_a.snap_to_tile;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.snap_to_tile = !sliding_window_a.snap_to_tile;

    // flip is_bilinear
    sliding_window_b.is_bilinear = !sliding_window_a.is_bilinear;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.is_bilinear = !sliding_window_a.is_bilinear;

    // flip is_transpose
    sliding_window_b.is_transpose = !sliding_window_a.is_transpose;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.is_transpose = !sliding_window_a.is_transpose;

    // flip ceil_mode
    sliding_window_b.ceil_mode = !sliding_window_a.ceil_mode;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.ceil_mode = !sliding_window_a.ceil_mode;
}

// The pool DRAM slicer overrides ceil_pad_hw per slice. That override changes the output shape, the halo
// tables and the pool CTAs, so it must reach the hash (#59914).
TEST(SlidingWindowHash, ExplicitCeilPadSeparatesKeys) {
    const SlidingWindowConfig computed{
        .batch_size = 1,
        .input_hw = {32, 32},
        .window_hw = {3, 3},
        .stride_hw = {2, 2},
        .padding = {1, 1, 1, 1},
        .num_cores_nhw = 1,
        .num_cores_c = 1,
        .core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {7, 7})),
        .ceil_mode = true};
    ASSERT_EQ(computed.get_ceil_pad_hw(), std::make_pair(1u, 1u));

    auto overridden = computed;
    overridden.ceil_pad_hw = std::make_pair(0u, 0u);
    ASSERT_NE(computed.get_output_shape(), overridden.get_output_shape());
    EXPECT_NE(computed.get_hash(), overridden.get_hash());

    // An override equal to the computed value builds the same program, so it should share the key.
    auto same_as_computed = computed;
    same_as_computed.ceil_pad_hw = computed.get_ceil_pad_hw();
    EXPECT_EQ(computed.get_hash(), same_as_computed.get_hash());
}

namespace {

SlidingWindowConfig shard_boundary_config(
    uint32_t batch_size,
    uint32_pair_t input_hw,
    uint32_pair_t window_hw,
    uint32_pair_t stride_hw,
    std::array<uint32_t, 4> padding,
    uint32_t num_cores_nhw) {
    return SlidingWindowConfig{
        .batch_size = batch_size,
        .input_hw = input_hw,
        .window_hw = window_hw,
        .stride_hw = stride_hw,
        .padding = padding,
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = 1,
        .core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {7, 7}))};
}

// Shard boundaries computed from the materialised op trace, as generate_shard_boundaries() did before it switched
// to the closed-form trace index for non-bilinear windows.
std::vector<ShardBoundary> reference_shard_boundaries(const SlidingWindowConfig& config) {
    const auto op_trace_metadata = generate_op_trace_metadata(config);
    const uint32_t output_shard_h = config.get_output_shard_y(config.snap_to_tile);
    uint32_t padded_input_w = config.input_hw.second + config.get_pad_w() + config.get_ceil_pad_w();
    if (config.is_transpose) {
        padded_input_w = config.get_transposed_full_input_shape()[2];
    }
    const uint32_t max_index = op_trace_metadata.size();
    const uint32_t dilated_window_h =
        config.window_hw.first + ((config.dilation_hw.first - 1) * (config.window_hw.first - 1));
    const uint32_t dilated_window_w =
        config.window_hw.second + ((config.dilation_hw.second - 1) * (config.window_hw.second - 1));
    const uint32_t halo_with_pad_len = ((dilated_window_h - 1) * padded_input_w) + dilated_window_w - 1;

    std::vector<ShardBoundary> shard_boundaries;
    uint32_t output_index_start = 0;
    for (uint32_t core = 0; core < config.num_cores_nhw; ++core) {
        const uint32_t output_index_end = std::min(output_index_start + output_shard_h, max_index) - 1;
        uint32_t input_index_start = op_trace_metadata[std::min(output_index_start, max_index - 1)];
        uint32_t input_index_end = op_trace_metadata[output_index_end] + halo_with_pad_len;
        if (input_index_start == 0 && output_index_start != 0) {
            input_index_start = op_trace_metadata[output_index_end] + 1;
            input_index_end = input_index_start - 1;
        }
        shard_boundaries.push_back({{output_index_start, output_index_end}, {input_index_start, input_index_end}});
        output_index_start = output_index_end + 1;
    }
    return shard_boundaries;
}

}  // namespace

TEST(SlidingWindowShardBoundaries, MatchMaterialisedTrace) {
    std::vector<SlidingWindowConfig> configs = {
        shard_boundary_config(1, {32, 32}, {3, 3}, {1, 1}, {1, 1, 1, 1}, 8),
        shard_boundary_config(2, {57, 43}, {3, 3}, {2, 2}, {1, 2, 0, 3}, 13),
        shard_boundary_config(3, {28, 28}, {5, 5}, {3, 1}, {2, 2, 2, 2}, 64),
        shard_boundary_config(1, {4, 4}, {3, 3}, {1, 1}, {1, 1, 1, 1}, 64),  // more cores than output pixels
        shard_boundary_config(2, {100, 7}, {1, 1}, {1, 1}, {0, 0, 0, 0}, 5),
    };
    auto dilated = shard_boundary_config(2, {40, 36}, {3, 3}, {1, 1}, {2, 2, 2, 2}, 11);
    dilated.dilation_hw = {2, 2};
    configs.push_back(dilated);
    auto snapped = shard_boundary_config(2, {30, 30}, {3, 3}, {2, 2}, {1, 1, 1, 1}, 7);
    snapped.snap_to_tile = true;
    configs.push_back(snapped);
    auto ceil_padded = shard_boundary_config(2, {34, 32}, {3, 3}, {2, 2}, {0, 0, 0, 0}, 9);
    ceil_padded.ceil_mode = true;
    configs.push_back(ceil_padded);
    auto transposed = shard_boundary_config(2, {17, 19}, {3, 3}, {2, 2}, {1, 1, 1, 1}, 12);
    transposed.is_transpose = true;
    transposed.output_pad_hw = {1, 1};
    configs.push_back(transposed);

    for (const auto& config : configs) {
        const auto expected = reference_shard_boundaries(config);
        const auto actual = generate_shard_boundaries(config);
        ASSERT_EQ(actual.size(), expected.size()) << config.to_string();
        for (size_t core = 0; core < expected.size(); ++core) {
            EXPECT_EQ(actual[core].output_range.start, expected[core].output_range.start)
                << config.to_string() << " core " << core;
            EXPECT_EQ(actual[core].output_range.end, expected[core].output_range.end)
                << config.to_string() << " core " << core;
            EXPECT_EQ(actual[core].input_range.start, expected[core].input_range.start)
                << config.to_string() << " core " << core;
            EXPECT_EQ(actual[core].input_range.end, expected[core].input_range.end)
                << config.to_string() << " core " << core;
        }
    }
}

// The conv2d, conv_transpose2d and pool2d DRAM slicing L1 estimators call generate_shard_boundaries() for every
// candidate slice count on every call, cache hits included, so its cost must not grow with the output size.
TEST(SlidingWindowShardBoundaries, CostDoesNotScaleWithOutputPixels) {
    auto min_time_us = [](const SlidingWindowConfig& config) {
        double best_us = 0.0;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto shard_boundaries = generate_shard_boundaries(config);
            const auto t1 = std::chrono::steady_clock::now();
            EXPECT_EQ(shard_boundaries.size(), config.num_cores_nhw);
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            best_us = rep == 0 ? us : std::min(best_us, us);
        }
        return best_us;
    };
    // 64x64 vs 4096x4096 output pixels on the same 64 cores.
    const double small_us = min_time_us(shard_boundary_config(1, {64, 64}, {3, 3}, {1, 1}, {1, 1, 1, 1}, 64));
    const double large_us = min_time_us(shard_boundary_config(1, {4096, 4096}, {3, 3}, {1, 1}, {1, 1, 1, 1}, 64));
    EXPECT_LT(large_us, (20.0 * small_us) + 2000.0)
        << "generate_shard_boundaries() took " << small_us << " us for 64x64 output pixels and " << large_us
        << " us for 4096x4096 on the same 64 cores";
}

INSTANTIATE_TEST_SUITE_P(
    SlidingWindowHashTests,
    SlidingWindowTestFixture,
    ::testing::Values(SlidingWindowConfig{
        .batch_size = 1,
        .input_hw = {32, 32},
        .window_hw = {3, 3},
        .stride_hw = {1, 1},
        .padding = {1, 1, 1, 1},
        .output_pad_hw = {0, 0},
        .dilation_hw = {1, 1},
        .num_cores_nhw = 1,
        .num_cores_c = 1,
        .core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {7, 7})),
        .snap_to_tile = false,
        .is_bilinear = false,
        .is_transpose = false,
        .ceil_mode = false}));

}  // namespace ttnn::operations::sliding_window::test
