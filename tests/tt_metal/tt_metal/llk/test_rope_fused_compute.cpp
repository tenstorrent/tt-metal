// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t block_tiles = 4;
constexpr std::uint32_t blocks = 2;
constexpr std::uint32_t tiles = block_tiles * blocks;
constexpr std::uint32_t heads = 2;
constexpr std::uint32_t full_rows = tt::constants::TILE_HEIGHT;
// Block 1 folds this scale into cos/sin; the kernels pass it as the fp32 bit pattern 0xc0000000.
constexpr float block1_scale = -2.0f;

std::uint32_t tile_offset(std::uint32_t tile, std::uint32_t row, std::uint32_t column) {
    using namespace tt::constants;
    const auto face = (row / FACE_HEIGHT) * (TILE_WIDTH / FACE_WIDTH) + column / FACE_WIDTH;
    return tile * TILE_HW + face * FACE_HW + row % FACE_HEIGHT * FACE_WIDTH + column % FACE_WIDTH;
}

// Where one block of the input keeps its operands, and how much of it the kernel rotates.
struct BlockLayout {
    std::uint32_t cos_tile;
    std::uint32_t sin_tile;      // Unused when the case is interleaved: sin shares cos_tile.
    std::uint32_t first_x_tile;  // One x tile per head: first_x_tile, first_x_tile + 1.
    std::uint32_t live_rows;     // Rows the kernel rotates; every other row must come back untouched.
    float scale;
};

// One kernel under test and the input layout it documents.
struct RopeCase {
    const char* kernel;
    // Interleaved (fused path): cos in the even columns and sin in the odd columns of one tile.
    // Otherwise (separate path): cos and sin in tiles of their own, each angle in both columns of its pair.
    bool interleaved;
    std::array<BlockLayout, blocks> layout;
};

// Fused cos/sin through rope_sfpu_inplace_fused / rope_sfpu_inplace_fused_rows.
// Block 0: x in tiles 0,1 and cos/sin in tile 2. Block 1: cos/sin in tile 4 and x in tiles 5,6.
// Tiles 3 and 7 are guards that must come back untouched.
constexpr RopeCase fused_rope{
    "tests/tt_metal/tt_metal/test_kernels/compute/rope_fused_compute.cpp",
    true,
    {{
        {2, 2, 0, 8, 1.0f},
        {4, 4, 5, full_rows, block1_scale},
    }}};

// Separate cos/sin tiles through rope_sfpu_inplace / rope_sfpu_inplace_rows. The separate path rotates the
// first 4-row group of each top face, so tile rows 0..3 of every column; the rest of each x tile, and the
// cos/sin tiles, must come back untouched.
// Block 0: x in tiles 0,1, cos in 2, sin in 3. Block 1: cos in 4, sin in 5, x in 6,7.
constexpr RopeCase separate_rope{
    "tests/tt_metal/tt_metal/test_kernels/compute/rope_separate_compute.cpp",
    false,
    {{
        {2, 3, 0, 4, 1.0f},
        {4, 5, 6, 4, block1_scale},
    }}};

void fill_source(std::vector<bfloat16>& source, const RopeCase& rope) {
    using namespace tt::constants;
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
        for (std::uint32_t row = 0; row < TILE_HEIGHT; ++row) {
            for (std::uint32_t column = 0; column < TILE_WIDTH; ++column) {
                source[tile_offset(tile, row, column)] =
                    bfloat16(2.0f + static_cast<float>(tile) / 8 + static_cast<float>(row + column) / 64);
            }
        }
    }

    for (const auto& block : rope.layout) {
        for (std::uint32_t row = 0; row < TILE_HEIGHT; ++row) {
            for (std::uint32_t pair = 0; pair < TILE_WIDTH / 2; ++pair) {
                const bfloat16 cosine((64.0f + pair + row) / 128);
                const bfloat16 sine((17.0f + 2 * pair + row) / 128);
                if (rope.interleaved) {
                    source[tile_offset(block.cos_tile, row, 2 * pair)] = cosine;
                    source[tile_offset(block.cos_tile, row, 2 * pair + 1)] = sine;
                } else {
                    for (const auto column : {2 * pair, 2 * pair + 1}) {
                        source[tile_offset(block.cos_tile, row, column)] = cosine;
                        source[tile_offset(block.sin_tile, row, column)] = sine;
                    }
                }
                for (std::uint32_t head = 0; head < heads; ++head) {
                    source[tile_offset(block.first_x_tile + head, row, 2 * pair)] =
                        bfloat16((65.0f + row + 2 * pair + 4 * head) / 128);
                    source[tile_offset(block.first_x_tile + head, row, 2 * pair + 1)] =
                        bfloat16(-(33.0f + 2 * row + pair + 8 * head) / 128);
                }
            }
        }
    }
}

struct Golden {
    std::vector<float> values;
    // True if some rotated element needs more than BF16 precision, so an FP32 store differs from a BF16 one.
    bool requires_fp32_precision;
};

Golden compute_golden(const std::vector<bfloat16>& source, const RopeCase& rope, bool fp32_dest) {
    Golden golden{std::vector<float>(source.begin(), source.end()), false};
    for (const auto& block : rope.layout) {
        const auto sin_tile = rope.interleaved ? block.cos_tile : block.sin_tile;
        const std::uint32_t sin_column_shift = rope.interleaved ? 1 : 0;
        for (std::uint32_t row = 0; row < block.live_rows; ++row) {
            for (std::uint32_t pair = 0; pair < tt::constants::TILE_WIDTH / 2; ++pair) {
                const float cosine =
                    static_cast<float>(source[tile_offset(block.cos_tile, row, 2 * pair)]) * block.scale;
                const float sine =
                    static_cast<float>(source[tile_offset(sin_tile, row, 2 * pair + sin_column_shift)]) * block.scale;
                for (std::uint32_t head = 0; head < heads; ++head) {
                    const auto even = tile_offset(block.first_x_tile + head, row, 2 * pair);
                    const auto odd = tile_offset(block.first_x_tile + head, row, 2 * pair + 1);
                    const float x_even = source[even];
                    const float x_odd = source[odd];
                    // BF16 dyadics keep both products and their sums exact in FP32.
                    // Their low mantissa bits distinguish an FP32 store from BF16.
                    golden.values[even] = cosine * x_even - sine * x_odd;
                    golden.values[odd] = sine * x_even + cosine * x_odd;
                    for (const auto offset : {even, odd}) {
                        const float truncated = bfloat16::truncate(golden.values[offset]);
                        golden.requires_fp32_precision |= golden.values[offset] != truncated;
                        if (!fp32_dest) {
                            golden.values[offset] = truncated;
                        }
                    }
                }
            }
        }
    }
    return golden;
}

void run_rope(distributed::MeshDevice& device, const RopeCase& rope, bool fp32_dest) {
    std::vector<bfloat16> source(tiles * tt::constants::TILE_HW);
    fill_source(source, rope);
    const auto golden = compute_golden(source, rope, fp32_dest);
    ASSERT_TRUE(golden.requires_fp32_precision);

    const auto packed = unit_tests::llk::single_core::run_unary(
        device,
        tt::DataFormat::Float16_b,
        fp32_dest ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
        pack_bfloat16_vec_into_uint32_vec(source),
        tiles,
        fp32_dest,
        rope.kernel,
        block_tiles);
    ASSERT_EQ(packed.size(), golden.values.size() / (fp32_dest ? 1 : 2));
    for (std::uint32_t index = 0; index < golden.values.size(); ++index) {
        const auto bits = fp32_dest ? packed[index] : ((packed[index / 2] >> (16 * (index % 2))) & 0xffff) << 16;
        // Covers the operand and guard tiles and every row outside the rotated rows.
        ASSERT_EQ(std::bit_cast<float>(bits), golden.values[index]) << "element=" << index;
    }
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, FusedRopeBfloat16InputUsesFp32Dest) {
    run_rope(*devices_.at(0), fused_rope, true);
}

TEST_F(LLKBlackholeSingleCardFixture, FusedRopeBfloat16Compatibility) { run_rope(*devices_.at(0), fused_rope, false); }

TEST_F(LLKBlackholeSingleCardFixture, SeparateRopeBfloat16InputUsesFp32Dest) {
    run_rope(*devices_.at(0), separate_rope, true);
}

TEST_F(LLKBlackholeSingleCardFixture, SeparateRopeBfloat16Compatibility) {
    run_rope(*devices_.at(0), separate_rope, false);
}

}  // namespace tt::tt_metal
