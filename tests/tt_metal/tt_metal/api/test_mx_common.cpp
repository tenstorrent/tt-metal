// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "impl/data_format/mx_common.hpp"

namespace {

using tt::tt_metal::mx::convert_from_mxfp_elem_bits;
using tt::tt_metal::mx::convert_to_mxfp_elem_bits;
using tt::tt_metal::mx::FormatParams;
using tt::tt_metal::mx::InfNanRepresentation;

constexpr FormatParams kMxFp4Params = {
    .elem_exp_bits = 2,
    .elem_man_bits = 1,
    .elem_exp_bias = 1,
    .elem_exp_max_unbiased = 2,
    .elem_exp_min_unbiased = 0,
    .elem_man_max = 0x1,
    .elem_width_bits = 4,
    .elem_width_storage_bits = 4,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x7,
    .elem_sat_neg_bits = 0xF,
};

constexpr FormatParams kMxFp6RParams = {
    .elem_exp_bits = 3,
    .elem_man_bits = 2,
    .elem_exp_bias = 3,
    .elem_exp_max_unbiased = 4,
    .elem_exp_min_unbiased = -2,
    .elem_man_max = 0x3,
    .elem_width_bits = 6,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x1F,
    .elem_sat_neg_bits = 0x3F,
};

constexpr FormatParams kMxFp6PParams = {
    .elem_exp_bits = 2,
    .elem_man_bits = 3,
    .elem_exp_bias = 1,
    .elem_exp_max_unbiased = 2,
    .elem_exp_min_unbiased = 0,
    .elem_man_max = 0x7,
    .elem_width_bits = 6,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x1F,
    .elem_sat_neg_bits = 0x3F,
};

constexpr FormatParams kMxFp8E5M2Params = {
    .elem_exp_bits = 5,
    .elem_man_bits = 2,
    .elem_exp_bias = 15,
    .elem_exp_max_unbiased = 15,
    .elem_exp_min_unbiased = -14,
    .elem_man_max = 0x3,
    .elem_width_bits = 8,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x7B,
    .elem_sat_neg_bits = 0xFB,
    .inf_rep = InfNanRepresentation::ExpAllOnesManZero,
    .nan_rep = InfNanRepresentation::ExpAllOnesManNonZero,
};

constexpr FormatParams kMxFp8E4M3Params = {
    .elem_exp_bits = 4,
    .elem_man_bits = 3,
    .elem_exp_bias = 7,
    .elem_exp_max_unbiased = 8,
    .elem_exp_min_unbiased = -6,
    .elem_man_max = 0x6,  // mant 0b111 at max exp is reserved for NaN
    .elem_width_bits = 8,
    .elem_width_storage_bits = 8,
    .sat_supported = true,
    .elem_sat_pos_bits = 0x7E,
    .elem_sat_neg_bits = 0xFE,
    .nan_rep = InfNanRepresentation::ExpAllOnesManAllOnes,
};

struct NamedFormat {
    const char* name;
    const FormatParams* params;
};

const std::vector<NamedFormat>& all_formats() {
    static const std::vector<NamedFormat> formats = {
        {"MxFp4_E2M1", &kMxFp4Params},
        {"MxFp6R_E3M2", &kMxFp6RParams},
        {"MxFp6P_E2M3", &kMxFp6PParams},
        {"MxFp8_E5M2", &kMxFp8E5M2Params},
        {"MxFp8_E4M3", &kMxFp8E4M3Params},
    };
    return formats;
}

// Shared exponent that leaves the decoded element unscaled, so these tests
// exercise the element quantizer in isolation from block scaling.
constexpr std::uint8_t kUnityScale = 0x7F;

}  // namespace

// Every representable element encoding must survive decode -> encode. This is the
// property that catches whole-binade errors in the element quantizer: it fails if
// the subnormal branch rounds at the wrong bit position (every subnormal encoded
// a binade too small) or if the mantissa field is masked with elem_man_max rather
// than the full field width (E4M3 losing mantissa bit 0).
TEST(MxCommonTests, EveryRepresentableCodeRoundTrips) {
    for (const auto& [name, params] : all_formats()) {
        const std::uint32_t code_count = 1u << params->elem_width_bits;
        for (std::uint32_t code = 0; code < code_count; ++code) {
            const float value = convert_from_mxfp_elem_bits(code, kUnityScale, *params);
            if (std::isnan(value) || std::isinf(value)) {
                continue;  // Inf/NaN encodings are canonicalized, not round-tripped.
            }
            const std::uint32_t reencoded = convert_to_mxfp_elem_bits(value, *params);
            EXPECT_EQ(reencoded, code) << name << ": code 0x" << std::hex << code << " decoded to " << std::dec << value
                                       << " but re-encoded as 0x" << std::hex << reencoded;
        }
    }
}

struct MxEncodeCase {
    float input = 0.0f;
    std::uint32_t expected_bits = 0;
    float expected_decoded = 0.0f;
};

class MxFp4EncodeTests : public ::testing::TestWithParam<MxEncodeCase> {};

TEST_P(MxFp4EncodeTests, EncodesToExpectedBits) {
    const auto& p = GetParam();
    const std::uint32_t bits = convert_to_mxfp_elem_bits(p.input, kMxFp4Params);
    EXPECT_EQ(bits, p.expected_bits);
    EXPECT_EQ(convert_from_mxfp_elem_bits(bits, kUnityScale, kMxFp4Params), p.expected_decoded);
}

// MxFp4 (E2M1) lattice: 0, 0.5, 1, 1.5, 2, 3, 4, 6. Everything below 1.0 goes
// through the subnormal branch, so 0.5 is the case that regressed to zero.
INSTANTIATE_TEST_SUITE_P(
    MxCommonTests,
    MxFp4EncodeTests,
    ::testing::Values(
        MxEncodeCase{0.5f, 0x1u, 0.5f},    // smallest subnormal, must not collapse to 0
        MxEncodeCase{0.375f, 0x1u, 0.5f},  // above the 0.25 midpoint -> up
        MxEncodeCase{0.25f, 0x0u, 0.0f},   // exact midpoint, ties to even -> 0
        MxEncodeCase{0.125f, 0x0u, 0.0f},  // below midpoint -> 0
        MxEncodeCase{0.75f, 0x2u, 1.0f},   // subnormal rounding overflows into exp 1
        MxEncodeCase{1.0f, 0x2u, 1.0f},
        MxEncodeCase{1.5f, 0x3u, 1.5f},
        MxEncodeCase{2.0f, 0x4u, 2.0f},
        MxEncodeCase{3.0f, 0x5u, 3.0f},
        MxEncodeCase{4.0f, 0x6u, 4.0f},
        MxEncodeCase{6.0f, 0x7u, 6.0f}));

class MxFp6PEncodeTests : public ::testing::TestWithParam<MxEncodeCase> {};

TEST_P(MxFp6PEncodeTests, EncodesToExpectedBits) {
    const auto& p = GetParam();
    const std::uint32_t bits = convert_to_mxfp_elem_bits(p.input, kMxFp6PParams);
    EXPECT_EQ(bits, p.expected_bits);
    EXPECT_EQ(convert_from_mxfp_elem_bits(bits, kUnityScale, kMxFp6PParams), p.expected_decoded);
}

// MxFp6P (E2M3) subnormals step by 0.125 up to 1.0; these are the encodings that
// were each shifted one step down.
INSTANTIATE_TEST_SUITE_P(
    MxCommonTests,
    MxFp6PEncodeTests,
    ::testing::Values(
        MxEncodeCase{0.125f, 0x01u, 0.125f},
        MxEncodeCase{0.25f, 0x02u, 0.25f},
        MxEncodeCase{0.375f, 0x03u, 0.375f},
        MxEncodeCase{0.5f, 0x04u, 0.5f},
        MxEncodeCase{0.625f, 0x05u, 0.625f},
        MxEncodeCase{0.75f, 0x06u, 0.75f},
        MxEncodeCase{0.875f, 0x07u, 0.875f},
        MxEncodeCase{0.09375f, 0x01u, 0.125f},  // above the 0.0625 midpoint -> up
        MxEncodeCase{0.0625f, 0x00u, 0.0f}));   // exact midpoint, ties to even -> 0

// E4M3 reserves mantissa 0b111 at the top exponent for NaN, so its elem_man_max
// is 0x6 rather than the full 3-bit field. Masking the mantissa field with that
// value clears bit 0 of every element; these normal-range values all carry an odd
// mantissa and would each decode one step low.
TEST(MxCommonTests, MxFp8E4M3PreservesOddNormalMantissas) {
    const std::vector<MxEncodeCase> cases = {
        {9.0f, 0x51u, 9.0f},
        {1.875f, 0x3Fu, 1.875f},
        {240.0f, 0x77u, 240.0f},
        {0.140625f, 0x21u, 0.140625f},
        {0.001953125f, 0x01u, 0.001953125f},  // smallest subnormal
    };
    for (const auto& c : cases) {
        const std::uint32_t bits = convert_to_mxfp_elem_bits(c.input, kMxFp8E4M3Params);
        EXPECT_EQ(bits, c.expected_bits) << "input " << c.input;
        EXPECT_EQ(convert_from_mxfp_elem_bits(bits, kUnityScale, kMxFp8E4M3Params), c.expected_decoded)
            << "input " << c.input;
    }
}

// Widening the mantissa mask must not disturb the reserved encodings: NaN is
// S.1111.111 and overflow saturates to the largest normal (S.1111.110).
TEST(MxCommonTests, MxFp8E4M3ReservedEncodings) {
    EXPECT_EQ(convert_to_mxfp_elem_bits(std::numeric_limits<float>::quiet_NaN(), kMxFp8E4M3Params), 0x7Fu);
    EXPECT_EQ(convert_to_mxfp_elem_bits(std::numeric_limits<float>::infinity(), kMxFp8E4M3Params), 0x7Eu);
    EXPECT_EQ(convert_to_mxfp_elem_bits(-std::numeric_limits<float>::infinity(), kMxFp8E4M3Params), 0xFEu);
    EXPECT_EQ(convert_to_mxfp_elem_bits(448.0f, kMxFp8E4M3Params), 0x7Eu);  // largest normal
    EXPECT_EQ(convert_to_mxfp_elem_bits(500.0f, kMxFp8E4M3Params), 0x7Eu);  // saturates, never NaN
    EXPECT_TRUE(std::isnan(convert_from_mxfp_elem_bits(0x7Fu, kUnityScale, kMxFp8E4M3Params)));
    EXPECT_EQ(convert_from_mxfp_elem_bits(0x7Eu, kUnityScale, kMxFp8E4M3Params), 448.0f);
}

// The subnormal branch pre-shifts the significand with a truncating >> before
// rounding, so the dropped bits must be folded into a sticky bit. Without it, a
// value just above a rounding midpoint is indistinguishable from an exact tie and
// rounds down instead of up. Each input below is one fp32 ULP above the midpoint
// between zero and the format's smallest subnormal, so correct round-to-nearest
// (the OCP MX spec's rounding, which this host-side packer implements) must round
// away from zero rather than tie to even.
TEST(MxCommonTests, SubnormalRoundingKeepsStickyBitsAboveMidpoint) {
    // Stepped up with nextafter rather than written as a decimal literal, so the
    // one-ULP claim cannot drift: the nearest decimals land two ULPs up.
    constexpr float kUp = std::numeric_limits<float>::infinity();
    EXPECT_EQ(convert_to_mxfp_elem_bits(std::nextafter(0.25f, kUp), kMxFp4Params), 0x1u);          // -> 0.5
    EXPECT_EQ(convert_to_mxfp_elem_bits(std::nextafter(0.0625f, kUp), kMxFp6PParams), 0x01u);      // -> 0.125
    EXPECT_EQ(convert_to_mxfp_elem_bits(std::nextafter(0.03125f, kUp), kMxFp6RParams), 0x01u);     // -> 0.0625
    EXPECT_EQ(convert_to_mxfp_elem_bits(std::nextafter(0x1p-17f, kUp), kMxFp8E5M2Params), 0x01u);  // -> 2^-16
    EXPECT_EQ(convert_to_mxfp_elem_bits(std::nextafter(0x1p-10f, kUp), kMxFp8E4M3Params), 0x01u);  // -> 2^-9

    // Exactly on the midpoint (no dropped bits) still ties to even, i.e. to zero.
    EXPECT_EQ(convert_to_mxfp_elem_bits(0.25f, kMxFp4Params), 0x0u);
    EXPECT_EQ(convert_to_mxfp_elem_bits(0.0625f, kMxFp6PParams), 0x0u);
    EXPECT_EQ(convert_to_mxfp_elem_bits(0.03125f, kMxFp6RParams), 0x0u);
    EXPECT_EQ(convert_to_mxfp_elem_bits(0x1p-17f, kMxFp8E5M2Params), 0x0u);
    EXPECT_EQ(convert_to_mxfp_elem_bits(0x1p-10f, kMxFp8E4M3Params), 0x0u);
}
