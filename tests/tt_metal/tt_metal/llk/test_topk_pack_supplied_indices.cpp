// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/constants.hpp>
#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {

TEST_F(LLKBlackholeSingleCardFixture, TopkPackSuppliedIndices) {
    constexpr std::uint32_t tile_elements = tt::constants::TILE_HW;
    constexpr std::uint32_t block_tiles = 4;
    constexpr std::uint32_t padding = 0xff80ffffu;
    constexpr std::uint32_t tiles = 64;  // two rounds across four size/half cases and both output aliases
    constexpr std::array<std::uint32_t, 4> entry_counts = {2048, 1024, 512, 512};
    std::mt19937 rng(5554);
    std::vector<std::uint32_t> input(tiles * tile_elements);
    for (auto& word : input) {
        word = rng();
    }
    for (std::uint32_t block = 0; block < tiles / block_tiles; ++block) {
        const auto ids = (block * block_tiles + 2) * tile_elements;
        for (std::uint32_t i = 0; i < 2 * tile_elements; i += 11) {
            input[ids + i] = UINT32_MAX;
        }
    }
    auto expected = input;
    for (std::uint32_t block = 0; block < tiles / block_tiles; ++block) {
        const auto start = block * block_tiles * tile_elements;
        const auto mode = block % 4;
        const auto k = entry_counts[mode];
        const auto output = start + (block / entry_counts.size() % 2) * 2 * tile_elements;
        const auto offset = mode == 3 ? 512u : 0u;
        for (std::uint32_t i = 0; i < k; ++i) {
            const auto id = input[start + 2 * tile_elements + i];
            expected[output + i] =
                id == UINT32_MAX ? padding : (input[start + offset + i] & 0xffff0000u) | (id & 0xffffu);
        }
        for (std::uint32_t i = k; i < tile_elements; ++i) {
            expected[output + i] = padding;
        }
    }
    const auto result = unit_tests::llk::single_core::run_unary(
        *devices_.at(0),
        tt::DataFormat::UInt32,
        tt::DataFormat::UInt32,
        input,
        tiles,
        true /* fp32_dest_acc_en */,
        "tests/tt_metal/tt_metal/test_kernels/compute/topk_pack_supplied_indices.cpp",
        block_tiles /* cb_depth_tiles */);
    EXPECT_EQ(result, expected);
}

}  // namespace tt::tt_metal
