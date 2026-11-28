// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#include "api/numeric/bfloat16.h"
#pragma GCC diagnostic pop

namespace {

bool bf16_helpers_test_is_nan(std::uint16_t v) { return (v & 0x7F80u) == 0x7F80u && (v & 0x007Fu) != 0; }

std::uint16_t bf16_helpers_test_reference_rne(std::uint32_t bits) {
    return static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

TEST(Bfloat16DeviceHelpers, EveryFp32NanStaysNan) {
    std::uint64_t rne_lost = 0;
    std::uint64_t trunc_lost = 0;
    for (std::uint32_t sign = 0; sign < 2; ++sign) {
        for (std::uint32_t mantissa = 1; mantissa < (1u << 23); ++mantissa) {
            const std::uint32_t bits = (sign << 31) | 0x7F800000u | mantissa;
            const float x = std::bit_cast<float>(bits);
            rne_lost += bf16_helpers_test_is_nan(fp32_to_bf16(x)) ? 0 : 1;
            trunc_lost += bf16_helpers_test_is_nan(fp32_to_bf16_truncate(x)) ? 0 : 1;
        }
    }
    EXPECT_EQ(rne_lost, 0u);
    EXPECT_EQ(trunc_lost, 0u);
}

TEST(Bfloat16DeviceHelpers, NanExamplesFromTheReport) {
    EXPECT_TRUE(bf16_helpers_test_is_nan(fp32_to_bf16(std::bit_cast<float>(std::uint32_t{0x7FFF8000}))));
    EXPECT_TRUE(bf16_helpers_test_is_nan(fp32_to_bf16(std::bit_cast<float>(std::uint32_t{0xFFFF8000}))));
    EXPECT_TRUE(bf16_helpers_test_is_nan(fp32_to_bf16(std::bit_cast<float>(std::uint32_t{0x7F800001}))));
    EXPECT_TRUE(bf16_helpers_test_is_nan(fp32_to_bf16_truncate(std::bit_cast<float>(std::uint32_t{0x7F800001}))));
}

TEST(Bfloat16DeviceHelpers, CommonNansAreUnchanged) {
    for (std::uint32_t bits : {0x7FC00000u, 0xFFC00000u, 0x7FFF0000u, 0xFFFF0000u, 0x7FC10000u}) {
        const float x = std::bit_cast<float>(bits);
        EXPECT_EQ(fp32_to_bf16(x), static_cast<std::uint16_t>(bits >> 16));
        EXPECT_EQ(fp32_to_bf16_truncate(x), static_cast<std::uint16_t>(bits >> 16));
    }
}

TEST(Bfloat16DeviceHelpers, NonNanInputsAreUnchanged) {
    for (std::uint64_t b = 0; b <= 0xFFFFFFFFull; b += 251) {
        const auto bits = static_cast<std::uint32_t>(b);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
            continue;
        }
        const float x = std::bit_cast<float>(bits);
        ASSERT_EQ(fp32_to_bf16(x), bf16_helpers_test_reference_rne(bits)) << std::hex << bits;
        ASSERT_EQ(fp32_to_bf16_truncate(x), static_cast<std::uint16_t>(bits >> 16)) << std::hex << bits;
    }
    for (std::uint32_t bits : {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7F7FFFFFu, 0x3F808000u}) {
        const float x = std::bit_cast<float>(bits);
        EXPECT_EQ(fp32_to_bf16(x), bf16_helpers_test_reference_rne(bits)) << std::hex << bits;
        EXPECT_EQ(fp32_to_bf16_truncate(x), static_cast<std::uint16_t>(bits >> 16)) << std::hex << bits;
    }
}

}  // namespace
