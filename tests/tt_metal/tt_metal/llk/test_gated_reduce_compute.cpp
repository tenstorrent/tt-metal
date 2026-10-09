// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tile.hpp>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {
namespace {

void run_gated_reduce(distributed::MeshDevice& device, std::uint32_t rows, bool fp32_dest) {
    constexpr std::uint32_t block_tiles = 4;
    constexpr std::uint32_t tiles = 4 * block_tiles;
    const std::uint32_t elements = rows * tt::constants::TILE_WIDTH;
    std::vector<bfloat16> source(tiles * elements);
    std::mt19937 generator(3448);
    std::uniform_real_distribution<float> distribution(-8.0f, 8.0f);
    constexpr std::array edges = {-8.0f, -1.5f, -1.25f, -1.0f, -0.0f, 0.0f, 1.0f, 1.25f, 1.5f, 8.0f};
    for (std::uint32_t index = 0; index < source.size(); ++index) {
        source[index] = bfloat16(index % 16 < edges.size() ? edges[index % 16] : distribution(generator));
    }

    std::vector<float> expected(source.begin(), source.end());
    std::vector<bool> active(source.size(), false);
    bool has_fp32_result = false;
    for (std::uint32_t block = 0; block < 4; ++block) {
        const auto first_gate = block == 3 ? 0 : block;
        for (std::uint32_t gate = first_gate; gate + 1 < block_tiles; gate += 2) {
            if (block == 3 && gate != 0) {
                break;
            }
            for (std::uint32_t element = 0; element < elements; ++element) {
                const auto index = (block * block_tiles + gate) * elements + element;
                float g = source[index];
                float u = source[index + elements];
                if (block <= 1) {
                    g *= 0.703125f;
                }
                if (block == 1) {
                    u *= 0.703125f;
                } else if (block == 2) {
                    u *= -1.3125f;
                }
                const bool clamped_gate = block == 1 || block == 3;
                if (clamped_gate) {
                    g = std::min(g, 1.25f);
                }
                if (block == 1 || block == 2) {
                    u = std::clamp(u, -1.25f, 1.25f);
                }
                const float alpha = clamped_gate ? 1.702f : 1.0f;
                float result = g / (1.0f + std::exp(-alpha * g)) * u;
                if (block != 0) {
                    result *= block == 2 ? 0.703125f : -1.3125f;
                }
                has_fp32_result |= result != static_cast<float>(bfloat16(result));
                expected[index] = fp32_dest ? result : static_cast<float>(bfloat16(result));
                active[index] = true;
            }
        }
    }
    ASSERT_TRUE(has_fp32_result);

    const auto packed = unit_tests::llk::single_core::run_unary_tiled(
        device,
        tt::DataFormat::Float16_b,
        fp32_dest ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
        Tile({rows, tt::constants::TILE_WIDTH}),
        pack_bfloat16_vec_into_uint32_vec(source),
        tiles,
        fp32_dest,
        "tests/tt_metal/tt_metal/test_kernels/compute/gated_reduce_compute.cpp",
        block_tiles);
    ASSERT_EQ(packed.size(), expected.size() / (fp32_dest ? 1 : 2));
    for (std::uint32_t index = 0; index < expected.size(); ++index) {
        const auto bits = fp32_dest ? packed[index] : ((packed[index / 2] >> (16 * (index % 2))) & 0xffff) << 16;
        const auto actual = std::bit_cast<float>(bits);
        if (active[index]) {
            const float tolerance =
                fp32_dest ? 2e-6f + 2e-4f * std::abs(expected[index]) : 2e-5f + 0.012f * std::abs(expected[index]);
            ASSERT_NEAR(actual, expected[index], tolerance) << "rows=" << rows << " element=" << index;
        } else {
            ASSERT_EQ(actual, expected[index]) << "Up or guard tile modified at element=" << index;
        }
    }
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, GatedReduceBfloat16Rows4) { run_gated_reduce(*devices_.at(0), 4, false); }
TEST_F(LLKBlackholeSingleCardFixture, GatedReduceBfloat16Rows8) { run_gated_reduce(*devices_.at(0), 8, false); }
TEST_F(LLKBlackholeSingleCardFixture, GatedReduceBfloat16Rows16) { run_gated_reduce(*devices_.at(0), 16, false); }
TEST_F(LLKBlackholeSingleCardFixture, GatedReduceBfloat16Rows32) { run_gated_reduce(*devices_.at(0), 32, false); }
TEST_F(LLKBlackholeSingleCardFixture, GatedReduceFp32Rows4) { run_gated_reduce(*devices_.at(0), 4, true); }
TEST_F(LLKBlackholeSingleCardFixture, GatedReduceFp32Rows8) { run_gated_reduce(*devices_.at(0), 8, true); }
TEST_F(LLKBlackholeSingleCardFixture, GatedReduceFp32Rows16) { run_gated_reduce(*devices_.at(0), 16, true); }
TEST_F(LLKBlackholeSingleCardFixture, GatedReduceFp32Rows32) { run_gated_reduce(*devices_.at(0), 32, true); }

}  // namespace tt::tt_metal
