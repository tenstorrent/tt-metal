// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include "impl/data_format/blockfloat_common.hpp"
#include <array>
#include <bit>
#include <memory>

#include <tt-metalium/base_types.hpp>
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

TEST(DataFormatLocalFp32Epoch, RequiresMatchingFloat32AndNonDefaultMode) {
    using tt::DataFormat;
    using tt::tt_metal::UnpackToDestMode;

    const std::array formats{DataFormat::Float32, DataFormat::Float16_b, DataFormat::Fp8_e4m3};
    std::array modes{UnpackToDestMode::Default, UnpackToDestMode::Default, UnpackToDestMode::Default};

    EXPECT_FALSE(tt::tt_metal::has_effective_local_fp32_epoch(formats, modes, tt::ARCH::BLACKHOLE));

    modes[1] = UnpackToDestMode::UnpackToDestFp32;
    modes[2] = UnpackToDestMode::UnpackToDestFp32;
    EXPECT_FALSE(tt::tt_metal::has_effective_local_fp32_epoch(formats, modes, tt::ARCH::BLACKHOLE));

    modes[0] = UnpackToDestMode::UnpackToDestFp32;
    EXPECT_TRUE(tt::tt_metal::has_effective_local_fp32_epoch(formats, modes, tt::ARCH::BLACKHOLE));
}

TEST(DataFormatLocalFp32Epoch, HandlesShortModesAndRetainsBlackholeGate) {
    using tt::DataFormat;
    using tt::tt_metal::UnpackToDestMode;

    const std::array formats{DataFormat::Float16_b, DataFormat::Float32};
    const std::array short_modes{UnpackToDestMode::UnpackToDestFp32};
    EXPECT_FALSE(tt::tt_metal::has_effective_local_fp32_epoch(formats, short_modes, tt::ARCH::BLACKHOLE));

    const std::array float32_format{DataFormat::Float32};
    const std::array local_mode{UnpackToDestMode::UnpackToDestFp32};
    EXPECT_TRUE(tt::tt_metal::has_effective_local_fp32_epoch(float32_format, local_mode, tt::ARCH::BLACKHOLE));
    EXPECT_FALSE(tt::tt_metal::has_effective_local_fp32_epoch(float32_format, local_mode, tt::ARCH::QUASAR));
    EXPECT_FALSE(tt::tt_metal::has_effective_local_fp32_epoch(float32_format, local_mode, tt::ARCH::WORMHOLE_B0));
}

TEST(DataFormatFp8Predicate, IncludesLf8) {
    EXPECT_TRUE(tt::is_fp8_format(tt::DataFormat::Fp8_e4m3));
    EXPECT_TRUE(tt::is_fp8_format(tt::DataFormat::Lf8));
    EXPECT_FALSE(tt::is_fp8_format(tt::DataFormat::Float16_b));
}
