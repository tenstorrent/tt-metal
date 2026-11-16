// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/constants.hpp>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {

TEST_F(LLKBlackholeSingleCardFixture, CsaIndexRemapReinitializesSfpu) {
    constexpr std::uint32_t tiles = 9;
    constexpr std::uint32_t devices = 8;
    constexpr std::uint32_t banks_per_device = 8;
    constexpr std::uint32_t rows_per_bank = 16384;
    constexpr std::uint32_t rows_per_chunk = 32;
    constexpr std::uint32_t row_offset = 256;
    constexpr std::array<std::uint32_t, 16> rows{
        0, 1, 7, 8, 30, 31, 32, 33, 255, 256, 511, 512, 8191, 8192, 16382, 16383};
    constexpr std::uint32_t samples_per_bank = rows.size();
    std::vector<std::uint32_t> input(tiles * tt::constants::TILE_HW);
    auto golden = input;
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
        for (std::uint32_t lane = 0; lane < tt::constants::TILE_HW; ++lane) {
            const auto index = tile * tt::constants::TILE_HW + lane;
            if (tile % 3 != 1) {
                input[index] = golden[index] = 0x123400 + index;
                continue;
            }
            const auto sample = (lane + tile * 37) % tt::constants::TILE_HW;
            const auto device = sample / (banks_per_device * samples_per_bank);
            const auto bank = sample / samples_per_bank % banks_per_device;
            const auto row = rows[sample % rows.size()];
            input[index] = device * banks_per_device * rows_per_bank + bank * rows_per_bank + row;
            const auto position =
                ((row / rows_per_chunk * devices + device) * banks_per_device + bank) * rows_per_chunk +
                row % rows_per_chunk + row_offset;
            golden[index] = (position % banks_per_device) * rows_per_bank | position / banks_per_device;
        }
    }

    const auto result = unit_tests::llk::single_core::run_unary(
        *devices_.at(0),
        tt::DataFormat::UInt32,
        tt::DataFormat::UInt32,
        input,
        tiles,
        true,
        "tests/tt_metal/tt_metal/test_kernels/compute/csa_index_remap_reinit.cpp",
        3);
    EXPECT_EQ(result, golden);
}

}  // namespace tt::tt_metal
