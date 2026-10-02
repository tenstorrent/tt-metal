// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t block_tiles = 4;
constexpr std::uint32_t blocks = 2;
constexpr std::uint32_t tiles = block_tiles * blocks;

std::uint32_t tile_offset(std::uint32_t tile, std::uint32_t row, std::uint32_t column) {
    using namespace tt::constants;
    const auto face = (row / FACE_HEIGHT) * (TILE_WIDTH / FACE_WIDTH) + column / FACE_WIDTH;
    return tile * TILE_HW + face * FACE_HW + row % FACE_HEIGHT * FACE_WIDTH + column % FACE_WIDTH;
}

void run_fused_rope(distributed::MeshDevice& device, bool fp32_dest) {
    using namespace tt::constants;
    std::vector<bfloat16> source(tiles * TILE_HW);
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
        for (std::uint32_t row = 0; row < TILE_HEIGHT; ++row) {
            for (std::uint32_t column = 0; column < TILE_WIDTH; ++column) {
                source[tile_offset(tile, row, column)] =
                    bfloat16(2.0f + static_cast<float>(tile) / 8 + static_cast<float>(row + column) / 64);
            }
        }
    }

    for (std::uint32_t block = 0; block < blocks; ++block) {
        const auto first_tile = block * block_tiles;
        const auto phase_tile = first_tile + (block == 0 ? 2 : 0);
        const auto first_x_tile = first_tile + (block == 0 ? 0 : 1);
        for (std::uint32_t row = 0; row < TILE_HEIGHT; ++row) {
            for (std::uint32_t pair = 0; pair < TILE_WIDTH / 2; ++pair) {
                source[tile_offset(phase_tile, row, 2 * pair)] = bfloat16((64.0f + pair + row) / 128);
                source[tile_offset(phase_tile, row, 2 * pair + 1)] = bfloat16((17.0f + 2 * pair + row) / 128);
                for (std::uint32_t head = 0; head < 2; ++head) {
                    source[tile_offset(first_x_tile + head, row, 2 * pair)] =
                        bfloat16((65.0f + row + 2 * pair + 4 * head) / 128);
                    source[tile_offset(first_x_tile + head, row, 2 * pair + 1)] =
                        bfloat16(-(33.0f + 2 * row + pair + 8 * head) / 128);
                }
            }
        }
    }

    std::vector<float> golden(source.begin(), source.end());
    bool requires_fp32_precision = false;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        const auto first_tile = block * block_tiles;
        const auto phase_tile = first_tile + (block == 0 ? 2 : 0);
        const auto first_x_tile = first_tile + (block == 0 ? 0 : 1);
        const std::uint32_t live_rows = block == 0 ? 8 : TILE_HEIGHT;
        const float scale = block == 0 ? 1.0f : -2.0f;
        for (std::uint32_t row = 0; row < live_rows; ++row) {
            for (std::uint32_t pair = 0; pair < TILE_WIDTH / 2; ++pair) {
                const float cosine = static_cast<float>(source[tile_offset(phase_tile, row, 2 * pair)]) * scale;
                const float sine = static_cast<float>(source[tile_offset(phase_tile, row, 2 * pair + 1)]) * scale;
                for (std::uint32_t head = 0; head < 2; ++head) {
                    const auto even = tile_offset(first_x_tile + head, row, 2 * pair);
                    const auto odd = tile_offset(first_x_tile + head, row, 2 * pair + 1);
                    const float x_even = source[even];
                    const float x_odd = source[odd];
                    // BF16 dyadics keep both products and their sums exact in FP32.
                    // Their low mantissa bits distinguish an FP32 store from BF16.
                    golden[even] = cosine * x_even - sine * x_odd;
                    golden[odd] = sine * x_even + cosine * x_odd;
                    for (const auto offset : {even, odd}) {
                        const float truncated = bfloat16::truncate(golden[offset]);
                        requires_fp32_precision |= golden[offset] != truncated;
                        if (!fp32_dest) {
                            golden[offset] = truncated;
                        }
                    }
                }
            }
        }
    }
    ASSERT_TRUE(requires_fp32_precision);

    const auto packed = unit_tests::llk::single_core::run_unary(
        device,
        tt::DataFormat::Float16_b,
        fp32_dest ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
        pack_bfloat16_vec_into_uint32_vec(source),
        tiles,
        fp32_dest,
        "tests/tt_metal/tt_metal/test_kernels/compute/rope_fused_compute.cpp",
        block_tiles);
    ASSERT_EQ(packed.size(), golden.size() / (fp32_dest ? 1 : 2));
    for (std::uint32_t index = 0; index < golden.size(); ++index) {
        const auto bits = fp32_dest ? packed[index] : ((packed[index / 2] >> (16 * (index % 2))) & 0xffff) << 16;
        // Includes the phase and guard tiles, plus rows outside the first rotation.
        ASSERT_EQ(std::bit_cast<float>(bits), golden[index]) << "element=" << index;
    }
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, FusedRopeBfloat16InputUsesFp32Dest) { run_fused_rope(*devices_.at(0), true); }

TEST_F(LLKBlackholeSingleCardFixture, FusedRopeBfloat16Compatibility) { run_fused_rope(*devices_.at(0), false); }

}  // namespace tt::tt_metal
