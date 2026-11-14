// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <algorithm>
#include <array>
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

SlidingWindowConfig conv3x3(uint32_t batch, uint32_t h, uint32_t w, const std::array<uint32_t, 4>& padding) {
    return SlidingWindowConfig{
        .batch_size = batch,
        .input_hw = {h, w},
        .window_hw = {3, 3},
        .stride_hw = {1, 1},
        .padding = padding,
        .num_cores_nhw = 1,
        .core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {0, 0}))};
}

// The whole image's border comes from the boundary: every border position has its own stick, interior positions none.
TEST(ExternalHaloSpec, WholeImageBorderMapsEachPositionToItsOwnStick) {
    constexpr uint32_t batch = 2, h = 4, w = 5, hp = h + 2, wp = w + 2;
    const auto spec = external_halo_spec(conv3x3(batch, h, w, {1, 1, 1, 1}));
    ASSERT_EQ(spec.sticks_per_image(), 2 * wp + 2 * h);

    std::vector<int> hits(batch * spec.sticks_per_image(), 0);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t r = 0; r < hp; ++r) {
            for (uint32_t c = 0; c < wp; ++c) {
                const int64_t idx = spec.boundary_index(b, r, c);
                const bool border = r == 0 || r == hp - 1 || c == 0 || c == wp - 1;
                if (!border) {
                    EXPECT_EQ(idx, -1) << "b " << b << " r " << r << " c " << c;
                    continue;
                }
                ASSERT_GE(idx, static_cast<int64_t>(b * spec.sticks_per_image()));
                ASSERT_LT(idx, static_cast<int64_t>((b + 1) * spec.sticks_per_image()));
                hits[idx]++;
            }
        }
    }
    EXPECT_TRUE(std::all_of(hits.begin(), hits.end(), [](int n) { return n == 1; }));

    // Top row, bottom row, then the left and the right column of the interior rows.
    EXPECT_EQ(spec.boundary_index(0, 0, 0), 0);
    EXPECT_EQ(spec.boundary_index(0, hp - 1, 0), wp);
    EXPECT_EQ(spec.boundary_index(0, 1, 0), 2 * wp);
    EXPECT_EQ(spec.boundary_index(0, 1, wp - 1), 2 * wp + h);
    EXPECT_EQ(spec.boundary_index(1, 0, 0), spec.sticks_per_image());
}

// A width slice of a wider image: an inner slice takes only the top and bottom rows above and below its columns,
// an edge slice also the image's side border.
TEST(ExternalHaloSpec, WidthSliceTakesTheBorderOfItsColumns) {
    constexpr uint32_t h = 4, image_w = 8, wb = image_w + 2;
    const ExternalHaloWindow inner_window{
        .col_offset = 3, .boundary_w = image_w, .dev_pad_left = 1, .dev_pad_right = 1};
    const auto inner = external_halo_spec(conv3x3(1, h, 2, {1, 1, 0, 0}), inner_window);
    ASSERT_EQ(inner.sticks_per_image(), 2 * wb + 2 * h);
    for (uint32_t c = 0; c < 2; ++c) {
        EXPECT_EQ(inner.boundary_index(0, 0, c), 1 + 3 + c);
        EXPECT_EQ(inner.boundary_index(0, h + 1, c), wb + 1 + 3 + c);
        for (uint32_t r = 1; r <= h; ++r) {
            EXPECT_EQ(inner.boundary_index(0, r, c), -1);
        }
    }

    const ExternalHaloWindow right_window{
        .col_offset = 5, .boundary_w = image_w, .dev_pad_left = 1, .dev_pad_right = 1};
    const auto right = external_halo_spec(conv3x3(1, h, 3, {1, 1, 0, 1}), right_window);
    EXPECT_EQ(right.boundary_index(0, 0, 3), wb - 1);
    EXPECT_EQ(right.boundary_index(0, 1, 3), 2 * wb + h);
    EXPECT_EQ(right.boundary_index(0, h, 3), 2 * wb + h + h - 1);
}

TEST(ExternalHaloSpec, RejectsWindowsOutsideTheImage) {
    const auto config = conv3x3(1, 4, 2, {1, 1, 0, 0});
    EXPECT_ANY_THROW(external_halo_spec(config, ExternalHaloWindow{.col_offset = 3}));
    EXPECT_ANY_THROW(external_halo_spec(config, ExternalHaloWindow{.col_offset = 7, .boundary_w = 8}));
    EXPECT_ANY_THROW(external_halo_spec(
        conv3x3(1, 4, 2, {1, 1, 1, 0}),
        ExternalHaloWindow{.col_offset = 3, .boundary_w = 8, .dev_pad_left = 1, .dev_pad_right = 1}));
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
