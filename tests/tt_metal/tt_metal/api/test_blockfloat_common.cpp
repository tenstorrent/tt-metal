// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include "impl/data_format/blockfloat_common.hpp"
#include <array>
#include <bit>
#include <memory>

#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>
#include "jit_build/data_format.hpp"

namespace {

void roundtrip_test_for_mantissa_rounding_with_bfp8(
    float float_input, uint8_t expected_mantissa, float expected_float_output) {
    auto uint32_input = std::bit_cast<uint32_t>(float_input);
    // Set shared exponent as original float exponent (ie. skip logic for handling shared exponents)
    auto shared_exp = uint32_input >> 23 & 0xFF;

    auto output_mantissa = convert_u32_to_bfp<tt::DataFormat::Bfp8_b, false>(uint32_input, shared_exp, false);
    EXPECT_EQ(output_mantissa, expected_mantissa);

    uint32_t uint32_output = convert_bfp_to_u32(tt::DataFormat::Bfp8_b, output_mantissa, shared_exp, false);
    float float_output = std::bit_cast<float>(uint32_output);
    EXPECT_EQ(float_output, expected_float_output);
};

}  // namespace

struct ConvertU32ToBfpParams {
    float float_input = 0;
    uint32_t expected_mantissa = 0;
    float expected_float_output = 0;
};

class ConvertU32ToBfpTests : public ::testing::TestWithParam<ConvertU32ToBfpParams> {};

TEST_P(ConvertU32ToBfpTests, CPU_MantissaRoundingWithPositiveFloat) {
    const auto& params = GetParam();
    roundtrip_test_for_mantissa_rounding_with_bfp8(
        params.float_input, params.expected_mantissa, params.expected_float_output);
}

TEST_P(ConvertU32ToBfpTests, CPU_MantissaRoundingWithNegativeFloat) {
    const auto& params = GetParam();
    const auto float_input = -1 * params.float_input;
    const auto expected_mantissa = params.expected_mantissa | 0x80;
    const auto expected_float_output = -1 * params.expected_float_output;

    roundtrip_test_for_mantissa_rounding_with_bfp8(float_input, expected_mantissa, expected_float_output);
}

INSTANTIATE_TEST_SUITE_P(
    BlockfloatCommonTests,
    ConvertU32ToBfpTests,
    // clang-format off
    // See tests/tt_metal/tt_metal/api/test_blockfloat_common.cpp for explanation of rounding
    // NOTE: These float values are cherry-picked such that:
    // - The mantissa hits the 4 cases for rounding
    // - The float values match behaviour of round(float) (assuming same spec of ties round to even)
    ::testing::Values(
        // Round up always
        ConvertU32ToBfpParams{
            .float_input = 64.75,  // Mantissa is 0x18000
            .expected_mantissa = 0x41,
            .expected_float_output = 65,
        },
        // Round down always
        ConvertU32ToBfpParams{
            .float_input = 65.25,  // Mantissa is 0x28000
            .expected_mantissa = 0x41,
            .expected_float_output = 65,
        },
        // Tie: round down to nearest even
        ConvertU32ToBfpParams{
            .float_input = 64.5,  // Mantissa is 0x10000
            .expected_mantissa = 0x40,
            .expected_float_output = 64,
        },
        // Tie: round up to nearest even
        ConvertU32ToBfpParams{
            .float_input = 65.5,  // Mantissa is 0x30000
            .expected_mantissa = 0x42,
            .expected_float_output = 66,
        }
    )  // Values
    // clang-format on
);

// FP8_E4M3 is supported on Blackhole and Quasar but not Wormhole. Verify the arch guard in
// get_single_pack_src_format() matches that: QUASAR and BLACKHOLE pass, WORMHOLE_B0 throws.
// Host-only: calls the public get_pack_src_formats() wrapper, no device required.
TEST(DataFormatFp8ArchGuard, Fp8E4m3PackSrcFormatPerArch) {
    const std::array<tt::DataFormat, 1> fp8_formats{tt::DataFormat::Fp8_e4m3};
    constexpr auto unpack_dst = tt::DataFormat::Float16_b;

    EXPECT_NO_THROW(tt::get_pack_src_formats(
        fp8_formats,
        unpack_dst,
        /*fp32_dest_acc_en=*/true,
        /*bfp8_pack_precise=*/false,
        /*int_fpu_en=*/false,
        tt::ARCH::QUASAR));

    EXPECT_NO_THROW(tt::get_pack_src_formats(fp8_formats, unpack_dst, true, false, false, tt::ARCH::BLACKHOLE));

    EXPECT_ANY_THROW(tt::get_pack_src_formats(fp8_formats, unpack_dst, true, false, false, tt::ARCH::WORMHOLE_B0));
}

#include <algorithm>
#include <random>
#include <vector>
#include <tt-metalium/bfloat16.hpp>

namespace {

template <tt::DataFormat BfpFormat, typename T>
void verify_pack_as_bfp_tiles_parity(bool row_major_input, bool is_exp_a, const tt::tt_metal::Tile& tile, uint32_t num_tiles) {
    const auto tile_H = tile.get_tile_shape()[0];
    const auto tile_W = tile.get_tile_shape()[1];
    const auto face_H = tile.get_face_shape()[0];
    const auto face_W = tile.get_face_shape()[1];
    const auto subtiles_in_tile_row = tile_H / face_H;
    const auto subtiles_in_tile_col = tile_W / face_W;
    const auto subtile_rows = face_H;
    const auto subtile_cols = face_W;
    const size_t num_elements = static_cast<size_t>(num_tiles) * tile_H * tile_W;

    std::mt19937 rng(42 + num_tiles + (row_major_input ? 100 : 0) + (is_exp_a ? 1000 : 0));
    std::vector<T> input(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        uint32_t bits = rng();
        if constexpr (std::is_same_v<T, float>) {
            input[i] = std::bit_cast<float>(bits);
        } else if constexpr (std::is_same_v<T, bfloat16>) {
            input[i] = std::bit_cast<bfloat16>(static_cast<uint16_t>(bits >> 16));
        } else {
            input[i] = static_cast<T>((bits % 251) - 125);
        }
    }

    auto actual = pack_as_bfp_tiles<BfpFormat, T>(input, row_major_input, is_exp_a, tile);
    if (num_tiles == 0) {
        EXPECT_TRUE(actual.empty());
        return;
    }

    constexpr int num_mantissas_in_dword =
        (BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b) ? 16 :
        (BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b) ? 8 : 4;
    const uint32_t rows_per_tile = subtiles_in_tile_row * subtiles_in_tile_col * subtile_rows;
    const size_t mantissa_dwords_per_tile = (tile_H * tile_W) / num_mantissas_in_dword;
    const size_t exp_dwords_per_tile = actual.size() / num_tiles - mantissa_dwords_per_tile;
    const size_t dwords_per_tile = exp_dwords_per_tile + mantissa_dwords_per_tile;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const uint32_t* tile_ptr = actual.data() + static_cast<size_t>(tile_idx) * dwords_per_tile;
        const uint32_t* man_ptr = tile_ptr + exp_dwords_per_tile;
        size_t fp32_base = row_major_input ? 0 : static_cast<size_t>(tile_idx) * tile_H * tile_W;
        uint32_t row_idx = 0;
        size_t man_slot = 0;

        for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (uint32_t i = 0; i < subtile_rows; ++i, ++row_idx) {
                    uint32_t row_u32[16];
                    uint32_t max_exp = 0;
                    for (uint32_t j = 0; j < subtile_cols; ++j) {
                        size_t idx = row_major_input
                            ? (tr * face_H + i) * tile_W + (tc * face_W + j) + (static_cast<size_t>(tile_H * tile_W) * tile_idx)
                            : fp32_base + (tr * subtiles_in_tile_col + tc) * (subtile_rows * subtile_cols) + i * subtile_cols + j;
                        float f = static_cast<float>(input[idx]);
                        row_u32[j] = std::bit_cast<uint32_t>(f);
                        uint32_t e = (row_u32[j] & 0x7f800000) >> 23;
                        if (is_exp_a) {
                            int32_t se = std::clamp(static_cast<int32_t>(e) - 112, 0, 31);
                            e = static_cast<uint32_t>(se);
                        }
                        max_exp = std::max(max_exp, e);
                    }
                    uint8_t actual_exp = get_byte(tile_ptr[row_idx / 4], row_idx % 4);
                    ASSERT_EQ(actual_exp, static_cast<uint8_t>(max_exp));

                    uint32_t mask = (1u << (32 / num_mantissas_in_dword)) - 1u;
                    for (uint32_t j = 0; j < subtile_cols; j += num_mantissas_in_dword) {
                        uint32_t expected_dword = 0;
                        for (int k = num_mantissas_in_dword - 1; k >= 0; --k) {
                            uint32_t c = convert_u32_to_bfp<BfpFormat, false>(row_u32[j + k], max_exp, is_exp_a);
                            expected_dword = (expected_dword << (32 / num_mantissas_in_dword)) | (c & mask);
                        }
                        ASSERT_EQ(man_ptr[man_slot++], expected_dword);
                    }
                }
            }
        }
        for (uint32_t pad_idx = rows_per_tile; pad_idx < exp_dwords_per_tile * 4; ++pad_idx) {
            ASSERT_EQ(get_byte(tile_ptr[pad_idx / 4], pad_idx % 4), 0u);
        }
    }
}

}  // namespace

TEST(BlockfloatHostTilizer, SimdAndMultiThreadedParityAllFormatsAndShapes) {
    for (auto shape : std::vector<tt::tt_metal::Tile::TileShape>{
             {1, 16}, {1, 32}, {2, 32}, {4, 32}, {8, 32}, {16, 16}, {16, 32}, {32, 16}, {32, 32}}) {
        tt::tt_metal::Tile tile(shape);
        for (bool rm : {true, false}) {
            for (uint32_t count : {0u, 1u, 129u}) {
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp8_b, float>(rm, false, tile, count);
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp4_b, float>(rm, false, tile, count);
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp2_b, float>(rm, false, tile, count);
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp8_b, bfloat16>(rm, false, tile, count);
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp4_b, bfloat16>(rm, false, tile, count);
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp2_b, bfloat16>(rm, false, tile, count);
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp8, float>(rm, true, tile, count);
                verify_pack_as_bfp_tiles_parity<tt::DataFormat::Bfp8_b, int32_t>(rm, false, tile, count);
            }
        }
    }
}
