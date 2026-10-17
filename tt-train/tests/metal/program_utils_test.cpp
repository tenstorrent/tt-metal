// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal/common/program_utils.hpp"

#include <gtest/gtest.h>

#include <bit>

TEST(ProgramUtilsTest, PackTwoBfloat16UsesCanonicalConversion) {
    const float scaler = 1.0F / 384.0F;
    const auto rounded = bfloat16(scaler);
    const auto expected = pack_two_bfloat16_into_uint32({rounded, rounded});

    const uint32_t truncated_bits = std::bit_cast<uint32_t>(scaler) >> 16U;
    const uint32_t truncated = truncated_bits | (truncated_bits << 16U);

    EXPECT_EQ(pack_two_bfloat16_to_uint32(scaler), expected);
    EXPECT_NE(expected, truncated);
}
