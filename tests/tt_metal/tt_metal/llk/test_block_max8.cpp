// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {

TEST_F(LLKBlackholeSingleCardFixture, BlockMax8CompactRowsAndNeighborPreservation) {
    constexpr std::array<uint32_t, 13> valid_counts{0, 1, 7, 8, 9, 15, 16, 17, 511, 512, 513, 1023, 1024};
    constexpr std::array<uint32_t, 3> dst_indices{0, 3, 7};
    constexpr uint32_t slots = 8;
    constexpr uint32_t tile_size = tt::constants::TILE_HW;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t tiles = valid_counts.size() * slots;
    std::vector<bfloat16> input(tiles * tile_size);
    auto expected = input;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t batch = tile / slots;
        const uint32_t dst = dst_indices[batch % dst_indices.size()];
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < tile_size; ++i) {
            const uint32_t row = i / tile_width;
            const uint32_t col = i % tile_width;
            const uint32_t physical = ((row / face_width) * 2 + col / face_width) * face_width * face_width +
                                      row % face_width * face_width + col % face_width;
            const float value = tile % slots == dst && i >= valid_counts[batch]
                                    ? 2048.0f
                                    : static_cast<float>(static_cast<int>((i * 17 + tile * 11) % 127) - 127) / 8.0f;
            input[tile * tile_size + physical] = bfloat16(value);
            if (tile % slots != dst) {
                expected[tile * tile_size + physical] = bfloat16(value);
            } else {
                if (i < valid_counts[batch]) {
                    maximum = std::max(maximum, value);
                }
                if (i % 8 == 7) {
                    expected[tile * tile_size + i / 8] = bfloat16(maximum);
                    maximum = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
    const auto packed = pack_bfloat16_vec_into_uint32_vec(input);
    const auto golden = pack_bfloat16_vec_into_uint32_vec(expected);
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        const auto actual = unit_tests::llk::single_core::run_unary(
            *devices_.at(0),
            tt::DataFormat::Float16_b,
            tt::DataFormat::Float16_b,
            packed,
            tiles,
            false,
            "tests/tt_metal/tt_metal/test_kernels/compute/block_max8.cpp",
            slots);
        ASSERT_EQ(actual.size(), golden.size());
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            const auto dst = dst_indices[(tile / slots) % dst_indices.size()];
            const uint32_t words = tile % slots == dst ? tile_size / 8 / 2 : tile_size / 2;
            for (uint32_t word = 0; word < words; ++word) {
                const auto offset = tile * tile_size / 2 + word;
                EXPECT_EQ(actual[offset], golden[offset])
                    << "repeat=" << repeat << " tile=" << tile << " word=" << word;
            }
        }
    }
}
}  // namespace tt::tt_metal
