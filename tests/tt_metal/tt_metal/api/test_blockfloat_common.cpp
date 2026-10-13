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

// Exponent-A cannot represent magnitudes below 2^-14. The device packer flushes
// EXP_B < 113 to +0, including exact 2^-15. Exponent-B keeps those values.
TEST(BlockfloatCommonTests, ExpAValuesBelowMinNormalFlushToZero) {
    constexpr uint32_t kMinNormalBits = 113u << 23;       // 2^-14
    constexpr uint32_t kBelowMinNormalBits = 112u << 23;  // 2^-15
    constexpr uint32_t kMaxBelowMinNormalBits = kBelowMinNormalBits | 0x007fffffu;
    constexpr uint32_t kSign = 0x80000000u;

    const uint32_t flushed[] = {
        kBelowMinNormalBits,
        kBelowMinNormalBits | kSign,
        kMaxBelowMinNormalBits,
        kMaxBelowMinNormalBits | kSign,
        std::bit_cast<uint32_t>(1e-6f),
        std::bit_cast<uint32_t>(-1e-6f),
    };
    for (uint32_t input : flushed) {
        for (uint32_t shared_exp : {0u, 15u}) {
            const uint8_t mantissa =
                convert_u32_to_bfp<tt::DataFormat::Bfp8, false>(input, shared_exp, /*is_exp_a=*/true);
            EXPECT_EQ(mantissa, 0u) << std::hex << input << " shared_exp=" << shared_exp;
            const uint32_t decoded =
                convert_bfp_to_u32(tt::DataFormat::Bfp8, mantissa, static_cast<uint8_t>(shared_exp), /*is_exp_a=*/true);
            EXPECT_EQ(decoded, 0u);
        }
    }

    EXPECT_EQ(convert_u32_to_bfp<tt::DataFormat::Bfp4, false>(kBelowMinNormalBits, 0, true), 0u);
    EXPECT_EQ(convert_u32_to_bfp<tt::DataFormat::Bfp2, false>(kBelowMinNormalBits, 0, true), 0u);

    constexpr uint32_t kMinNormalSharedExp = 1;
    EXPECT_EQ(convert_u32_to_bfp<tt::DataFormat::Bfp8, false>(kMinNormalBits, kMinNormalSharedExp, true), 0x40u);
    EXPECT_EQ(
        convert_u32_to_bfp<tt::DataFormat::Bfp8, false>(kMinNormalBits | kSign, kMinNormalSharedExp, true), 0xc0u);
    EXPECT_EQ(convert_bfp_to_u32(tt::DataFormat::Bfp8, 0x40, kMinNormalSharedExp, true), kMinNormalBits);

    EXPECT_EQ(convert_u32_to_bfp<tt::DataFormat::Bfp8_b, false>(kBelowMinNormalBits, 112, false), 0x40u);
}

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
