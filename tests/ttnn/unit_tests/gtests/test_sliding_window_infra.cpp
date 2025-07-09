// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <string>

#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/api/tt-metalium/core_coord.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::sliding_window::test {

using namespace tt::tt_metal;

class SlidingWindowTestFixture : public testing::TestWithParam<SlidingWindowConfig> {};

TEST_P(SlidingWindowTestFixture, SlidingWindowHash) {
    auto sliding_window_a = GetParam();

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
