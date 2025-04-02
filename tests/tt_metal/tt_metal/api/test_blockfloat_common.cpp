// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/blockfloat_common.hpp>
#include <bit>
#include <memory>

#include <tt-metalium/tt_backend_api_types.hpp>

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

TEST_P(ConvertU32ToBfpTests, MantissaRoundingWithPositiveFloat) {
    const auto& params = GetParam();
    roundtrip_test_for_mantissa_rounding_with_bfp8(
        params.float_input, params.expected_mantissa, params.expected_float_output);
}

TEST_P(ConvertU32ToBfpTests, MantissaRoundingWithNegativeFloat) {
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
