// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>

#include <bit>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace {

// Regression for #60246. Comparison must be symmetric and must not treat
// different magnitudes as the same significant figures.

TEST(EqualWithinNSigFigs, RejectsDifferentMagnitudes) {
    EXPECT_FALSE(is_within_n_sig_figs(0.5f, 5.5f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(5.5f, 0.5f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(0.25f, 7.25f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(7.25f, 0.25f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(1e-7f, 9e-7f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(9e-7f, 1e-7f, 2));
}

TEST(EqualWithinNSigFigs, DeprecatedHelperPreservesLeadingDigitComparisons) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_FALSE(equal_within_n_sig_figs(0.5f, 5.5f, 2));
    EXPECT_FALSE(equal_within_n_sig_figs(5.5f, 0.5f, 2));
    EXPECT_FALSE(equal_within_n_sig_figs(1e-7f, 9e-7f, 2));
    EXPECT_FALSE(equal_within_n_sig_figs(9e-7f, 1e-7f, 2));
    EXPECT_TRUE(equal_within_n_sig_figs(2.71875f, 2.796875f, 2));
#pragma GCC diagnostic pop
}

TEST(EqualWithinNSigFigs, IsSymmetric) {
    const float values[] = {0.0f, -0.0f, 0.5f, 5.5f, 0.25f, 7.25f, 1e-7f, 9e-7f, -1.25f, 1000.0f};
    for (float a : values) {
        for (float b : values) {
            EXPECT_EQ(is_within_n_sig_figs(a, b, 2), is_within_n_sig_figs(b, a, 2)) << "a=" << a << " b=" << b;
        }
    }
}

TEST(EqualWithinNSigFigs, AcceptsValuesWithinTwoSignificantFigures) {
    EXPECT_TRUE(is_within_n_sig_figs(1.0f, 1.0f, 2));
    EXPECT_TRUE(is_within_n_sig_figs(1.00f, 1.04f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(1.00f, 1.06f, 2));

    EXPECT_TRUE(is_within_n_sig_figs(-2.50f, -2.54f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(-2.50f, 2.50f, 2));

    EXPECT_TRUE(is_within_n_sig_figs(100.0f, 104.0f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(100.0f, 106.0f, 2));

    // Half a unit in the second digit of 1e-7 is 5e-9.
    EXPECT_TRUE(is_within_n_sig_figs(1.00e-7f, 1.04e-7f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(1.00e-7f, 1.10e-7f, 2));
}

TEST(EqualWithinNSigFigs, RejectsApproxExpPairsThatShareATwoDigitPrefix) {
    // Approx-mode exp(1) is 2.796875 or 2.8125. bfloat16(e) is 2.71875.
    // Half a unit in the second digit there is 0.05, and both gaps are larger.
    EXPECT_FALSE(is_within_n_sig_figs(2.71875f, 2.796875f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(2.71875f, 2.8125f, 2));
}

TEST(EqualWithinNSigFigs, ZeroMatchesOnlyZero) {
    EXPECT_TRUE(is_within_n_sig_figs(0.0f, 0.0f, 2));
    EXPECT_TRUE(is_within_n_sig_figs(0.0f, -0.0f, 2));
    EXPECT_TRUE(is_within_n_sig_figs(-0.0f, 0.0f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(0.0f, 1e-7f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(-0.0f, 1e-7f, 2));
}

TEST(EqualWithinNSigFigs, DoesNotWidenTheDecadeJustBelowAPowerOfTen) {
    // 0x19416d9a is the float32 value of 1e-23. log10 of it is 7.82e-11 under -23.
    // 0x1940447f is about 6e-26 smaller. The half-unit band is 5e-26, so a nudge
    // of 1e-10 (tolerance 5e-25) would accept this pair.
    const auto just_under_1e_23 = std::bit_cast<float>(std::uint32_t{0x19416d9a});
    const auto six_e_26_below = std::bit_cast<float>(std::uint32_t{0x1940447f});
    EXPECT_FALSE(is_within_n_sig_figs(just_under_1e_23, six_e_26_below, 2));
    EXPECT_FALSE(is_within_n_sig_figs(six_e_26_below, just_under_1e_23, 2));

    // Scale is 1000, so the tolerance is 50.
    EXPECT_TRUE(is_within_n_sig_figs(1000.0f, 960.0f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(1000.0f, 940.0f, 2));
}

TEST(EqualWithinNSigFigs, MatchesOnlyIdenticalNonFiniteValues) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(is_within_n_sig_figs(inf, inf, 2));
    EXPECT_TRUE(is_within_n_sig_figs(-inf, -inf, 2));
    EXPECT_FALSE(is_within_n_sig_figs(inf, -inf, 2));
    EXPECT_FALSE(is_within_n_sig_figs(inf, 1.0f, 2));
    EXPECT_TRUE(is_within_n_sig_figs(nan, nan, 2));
    EXPECT_FALSE(is_within_n_sig_figs(nan, 1.0f, 2));
    EXPECT_FALSE(is_within_n_sig_figs(nan, inf, 2));
}

TEST(EqualWithinNSigFigs, RejectsInvalidSignificantFigureCounts) {
    EXPECT_ANY_THROW(is_within_n_sig_figs(1.0f, 2.0f, 0));
    EXPECT_ANY_THROW(is_within_n_sig_figs(1.0f, 2.0f, std::numeric_limits<float>::max_digits10 + 1));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_ANY_THROW(equal_within_n_sig_figs(1.0f, 2.0f, 0));
#pragma GCC diagnostic pop
}

TEST(IsClose, RejectsNonFiniteMismatch) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(is_close(inf, inf, 0.04f));
    EXPECT_TRUE(is_close(-inf, -inf, 0.04f));
    EXPECT_FALSE(is_close(1.0f, inf, 0.04f));
    EXPECT_FALSE(is_close(1.0f, -inf, 0.04f));
}

}  // namespace
