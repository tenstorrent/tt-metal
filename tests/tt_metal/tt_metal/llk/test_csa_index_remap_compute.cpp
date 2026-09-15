// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {

TEST_F(LLKBlackholeSingleCardFixture, CsaIndexRemapReinitializesSfpu) {
    constexpr std::uint32_t tiles = 9;
    constexpr std::array<std::uint32_t, 16> rows{
        0, 1, 7, 8, 30, 31, 32, 33, 255, 256, 511, 512, 8191, 8192, 16382, 16383};
    std::vector<std::uint32_t> input(tiles * 1024);
    auto golden = input;
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
        for (std::uint32_t lane = 0; lane < 1024; ++lane) {
            const auto index = tile * 1024 + lane;
            if (tile % 3 != 1) {
                input[index] = golden[index] = 0x123400 + index;
                continue;
            }
            const auto sample = (lane + tile * 37) % 1024;
            const auto device = sample / 128;
            const auto bank = sample / 16 % 8;
            const auto row = rows[sample % rows.size()];
            input[index] = device * 8 * 16384 + bank * 16384 + row;
            const auto position = ((row / 32 * 8 + device) * 8 + bank) * 32 + row % 32 + 256;
            golden[index] = (position % 8) * 16384 | position / 8;
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
