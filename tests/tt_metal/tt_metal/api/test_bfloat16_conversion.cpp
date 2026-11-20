// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

template <typename T>
uint16_t bf16_bits(T value) {
    return std::bit_cast<uint16_t>(bfloat16(value));
}

TEST(BFloat16Conversion, ExactMidpointsRoundToEven) {
    EXPECT_EQ(bf16_bits(0x1.01p0f), 0x3f80);
    EXPECT_EQ(bf16_bits(0x1.03p0), 0x3f82);
    EXPECT_EQ(bf16_bits(-0x1.03p0), 0xbf82);
}

TEST(BFloat16Conversion, IntegersOnEitherSideOfAMidpoint) {
    // The adjacent BF16 values are 2^24 and 2^24 + 2^17. Both midpoint neighbours round to the same FP32.
    constexpr int32_t midpoint = (1 << 24) + (1 << 16);
    EXPECT_EQ(bf16_bits(midpoint - 1), 0x4b80);
    EXPECT_EQ(bf16_bits(midpoint), 0x4b80);
    EXPECT_EQ(bf16_bits(midpoint + 1), 0x4b81);
    EXPECT_EQ(bf16_bits(-midpoint - 1), 0xcb81);
    EXPECT_EQ(bf16_bits(uint32_t{0x80800001}), 0x4f01);
    EXPECT_EQ(bf16_bits(uint64_t{0x8080000000000001}), 0x5f01);
    EXPECT_EQ(bf16_bits(std::numeric_limits<uint64_t>::max()), 0x5f80);
}

TEST(BFloat16Conversion, FloatingPointInputsOnEitherSideOfAMidpoint) {
    EXPECT_EQ(bf16_bits(std::nextafter(0x1.01p0, 2.0)), 0x3f81);
    EXPECT_EQ(bf16_bits(std::nextafter(0x1.03p0, 1.0)), 0x3f81);
    EXPECT_EQ(bf16_bits(std::nextafter(-0x1.01p0L, -2.0L)), 0xbf81);
}

TEST(BFloat16Conversion, RangeLimitsAndNaN) {
    EXPECT_EQ(bf16_bits(0x1p-134), 0x0000);
    EXPECT_EQ(bf16_bits(std::nextafter(0x1p-134, 1.0)), 0x0001);
    EXPECT_EQ(bf16_bits(std::nextafter(0x1.ffp127, 0.0)), 0x7f7f);
    EXPECT_EQ(bf16_bits(0x1.ffp127), 0x7f80);
    // A NaN payload can have the same low FP32 bits as a finite midpoint.
    const double nan = std::bit_cast<float>(uint32_t{0x7fc08000});
    EXPECT_EQ(bf16_bits(nan), 0x7fc0);
}

}  // namespace
