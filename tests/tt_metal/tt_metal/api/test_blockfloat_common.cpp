// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include "impl/data_format/blockfloat_common.hpp"
#include <array>
#include <bit>
#include <memory>
#include <random>
#include <future>
#include "impl/data_format/bfp_simd.hpp"

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

namespace {
template <int bits, typename T>
void check_simd_rows(int isa) {
    auto encode = tt::tt_metal::bfp_simd::select_row_packer<bits, T>(isa);
    if (!encode) {
        return;  // Unsupported ISAs must never execute on this host.
    }
    constexpr auto format = bits == 7   ? tt::DataFormat::Bfp8_b
                            : bits == 3 ? tt::DataFormat::Bfp4_b
                                        : tt::DataFormat::Bfp2_b;
    std::mt19937 rng(5127);
    for (uint32_t row = 0; row < 4096; ++row) {
        std::array<T, 16> input;
        std::array<uint32_t, 16> words;
        uint8_t exponent = 0;
        for (size_t j = 0; j < input.size(); ++j) {
            if constexpr (std::is_same_v<T, bfloat16>) {
                // Exhaust every BF16 bit pattern, including signed zeros and NaNs.
                input[j] = std::bit_cast<bfloat16>(static_cast<uint16_t>(row * 16 + j));
            } else {
                uint32_t word = rng();
                if (row % 2 == 0) {
                    word = (word & 0x807fffff) | ((row % 256) << 23);
                }
                input[j] = std::bit_cast<float>(word);
            }
            words[j] = std::bit_cast<uint32_t>(static_cast<float>(input[j]));
            exponent = std::max(exponent, static_cast<uint8_t>(words[j] >> 23));
        }
        std::array<uint8_t, 16> expected{}, actual{};
        constexpr int per_byte = 8 / (bits + 1);
        for (size_t j = 0; j < input.size(); ++j) {
            auto code = convert_u32_to_bfp<format, false>(words[j], exponent, false);
            expected[j / per_byte] |= code << ((j % per_byte) * (bits + 1));
        }
        uint8_t actual_exponent = 0;
        encode(input.data(), &actual_exponent, actual.data());
        ASSERT_EQ(actual_exponent, exponent) << "row=" << row << " bits=" << bits << " isa=" << isa;
        ASSERT_EQ(actual, expected) << "row=" << row << " bits=" << bits << " isa=" << isa;
    }
}

template <tt::DataFormat Fast, tt::DataFormat Reference, typename T>
void check_packed_tiles() {
    std::mt19937 rng(4903);
    for (auto shape : std::vector<tt::tt_metal::Tile::TileShape>{
             {1, 16}, {1, 32}, {2, 32}, {4, 32}, {8, 32}, {16, 16}, {16, 32}, {32, 16}, {32, 32}}) {
        for (bool row_major : {true, false}) {
            for (uint32_t count : {0, 1, 257}) {
                std::vector<T> input(count * shape[0] * shape[1]);
                for (auto& value : input) {
                    if constexpr (std::is_same_v<T, float>) {
                        value = std::bit_cast<float>(static_cast<uint32_t>(rng()));
                    } else {
                        value = std::bit_cast<bfloat16>(static_cast<uint16_t>(rng()));
                    }
                }
                // The exponent-A specialization retains the original converter.
                // With is_exp_a=false it uses the exact same B-format exponent rules.
                auto expected = pack_as_bfp_tiles<Reference, T>(input, row_major, false, tt::tt_metal::Tile(shape));
                auto actual = pack_as_bfp_tiles<Fast, T>(input, row_major, false, tt::tt_metal::Tile(shape));
                ASSERT_EQ(actual, expected)
                    << "tile=" << shape[0] << "x" << shape[1] << " row_major=" << row_major << " tiles=" << count;
            }
        }
    }
}
}  // namespace

class BfpSimdRows : public ::testing::TestWithParam<int> {};
TEST_P(BfpSimdRows, CPU_MatchesOriginalIntegerConverter) {
    check_simd_rows<7, float>(GetParam());
    check_simd_rows<3, float>(GetParam());
    check_simd_rows<1, float>(GetParam());
    check_simd_rows<7, bfloat16>(GetParam());
    check_simd_rows<3, bfloat16>(GetParam());
    check_simd_rows<1, bfloat16>(GetParam());
}
INSTANTIATE_TEST_SUITE_P(BlockfloatCommonTests, BfpSimdRows, ::testing::Values(0, 1, 2, 3));

TEST(BfpSimdTiles, CPU_MatchesOriginalPackingAndExponentPadding) {
    check_packed_tiles<tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp8, float>();
    check_packed_tiles<tt::DataFormat::Bfp4_b, tt::DataFormat::Bfp4, float>();
    check_packed_tiles<tt::DataFormat::Bfp2_b, tt::DataFormat::Bfp2, float>();
    check_packed_tiles<tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp8, bfloat16>();
    check_packed_tiles<tt::DataFormat::Bfp4_b, tt::DataFormat::Bfp4, bfloat16>();
    check_packed_tiles<tt::DataFormat::Bfp2_b, tt::DataFormat::Bfp2, bfloat16>();
}

TEST(BfpSimdTiles, CPU_ConcurrentCalls) {
    std::vector<float> input(512 * 1024);
    std::mt19937 rng(42);
    for (auto& x : input) {
        x = std::bit_cast<float>(static_cast<uint32_t>(rng()));
    }
    auto expected = pack_as_bfp_tiles<tt::DataFormat::Bfp4, float>(input, true, false, std::nullopt);
    std::vector<std::future<std::vector<uint32_t>>> calls;
    for (int i = 0; i < 8; ++i) {
        calls.emplace_back(std::async(std::launch::async, [&] {
            return pack_as_bfp_tiles<tt::DataFormat::Bfp4_b, float>(input, true, false, std::nullopt);
        }));
    }
    for (auto& call : calls) {
        EXPECT_EQ(call.get(), expected);
    }
}
