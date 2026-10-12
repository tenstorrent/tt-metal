// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/math.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

TEST(Math, DivUp) {
    EXPECT_EQ(tt::div_up(0, 3), 0);
    EXPECT_EQ(tt::div_up(6, 3), 2);
    EXPECT_EQ(tt::div_up(7, 3), 3);

    EXPECT_EQ(tt::div_up(-6, 3), -2);
    EXPECT_EQ(tt::div_up(-7, 3), -2);
    EXPECT_EQ(tt::div_up(7, -3), -2);
    EXPECT_EQ(tt::div_up(-7, -3), 3);
}

TEST(Math, DivUpNoOverflow) {
    constexpr auto int32_max = std::numeric_limits<std::int32_t>::max();
    constexpr auto int64_max = std::numeric_limits<std::int64_t>::max();
    constexpr auto uint32_max = std::numeric_limits<std::uint32_t>::max();
    constexpr auto uint64_max = std::numeric_limits<std::uint64_t>::max();
    constexpr auto size_max = std::numeric_limits<std::size_t>::max();

    static_assert(tt::div_up(int32_max, std::int32_t{2}) == int32_max / 2 + 1);
    static_assert(tt::div_up(int64_max, std::int64_t{2}) == int64_max / 2 + 1);
    static_assert(tt::div_up(uint32_max, std::uint32_t{2}) == uint32_max / 2 + 1);
    static_assert(tt::div_up(uint64_max, std::uint64_t{2}) == uint64_max / 2 + 1);
    static_assert(tt::div_up(size_max, size_max) == 1);

    EXPECT_EQ(tt::div_up(uint32_max - 1, uint32_max), 1U);
    EXPECT_EQ(tt::div_up(uint64_max - 1, uint64_max), 1U);
    EXPECT_EQ(tt::div_up(size_max - 1, size_max), 1U);
}

TEST(Math, DivUpCommonType) {
    static_assert(std::is_same_v<decltype(tt::div_up(std::uint32_t{1}, std::uint64_t{1})), std::uint64_t>);
    EXPECT_EQ(tt::div_up(std::uint32_t{7}, std::uint64_t{3}), 3U);
}

TEST(Math, RoundUp) {
    static_assert(tt::round_up(7, 3) == 9);
    static_assert(tt::round_up(-7, 3) == -6);
    static_assert(tt::round_up(7, -3) == 9);
    static_assert(tt::round_up(-7, -3) == -6);
    static_assert(tt::round_up(0, 3) == 0);
    static_assert(tt::round_up(6, 3) == 6);
    static_assert(tt::round_up(-6, -3) == -6);
    static_assert(tt::round_up(7u, 3u) == 9u);

    EXPECT_EQ(tt::round_up(7, 3), 9);
    EXPECT_EQ(tt::round_up(-7, 3), -6);
    EXPECT_EQ(tt::round_up(7, -3), 9);
    EXPECT_EQ(tt::round_up(-7, -3), -6);
    EXPECT_EQ(tt::round_up(0, 3), 0);
    EXPECT_EQ(tt::round_up(6, 3), 6);

    constexpr auto uint32_max = std::numeric_limits<std::uint32_t>::max();
    EXPECT_EQ(tt::round_up(uint32_max - 1, 2u), uint32_max - 1);
    static_assert(std::is_same_v<decltype(tt::round_up(std::uint8_t{250}, std::uint8_t{8})), int>);
    EXPECT_EQ(tt::round_up(std::uint8_t{250}, std::uint8_t{8}), 256);
}

TEST(Math, RoundDown) {
    static_assert(tt::round_down(7, 3) == 6);
    static_assert(tt::round_down(-7, 3) == -9);
    static_assert(tt::round_down(7, -3) == 6);
    static_assert(tt::round_down(-7, -3) == -9);
    static_assert(tt::round_down(0, 3) == 0);
    static_assert(tt::round_down(-6, 3) == -6);
    static_assert(tt::round_down(7u, 3u) == 6u);

    EXPECT_EQ(tt::round_down(7, 3), 6);
    EXPECT_EQ(tt::round_down(-7, 3), -9);
    EXPECT_EQ(tt::round_down(7, -3), 6);
    EXPECT_EQ(tt::round_down(-7, -3), -9);
    EXPECT_EQ(tt::round_down(0, 3), 0);
    EXPECT_EQ(tt::round_down(6, 3), 6);

    constexpr auto uint32_max = std::numeric_limits<std::uint32_t>::max();
    EXPECT_EQ(tt::round_down(uint32_max, 2u), uint32_max - 1);
}

TEST(Math, RoundUpRoundDownExhaustive) {
    for (int a = -50; a <= 50; ++a) {
        for (int b = -50; b <= 50; ++b) {
            if (b == 0) {
                continue;
            }
            const int abs_b = b < 0 ? -b : b;
            const int up = tt::round_up(a, b);
            const int down = tt::round_down(a, b);
            EXPECT_EQ(up % b, 0) << a << ", " << b;
            EXPECT_TRUE(up >= a && up < a + abs_b) << a << ", " << b;
            EXPECT_EQ(down % b, 0) << a << ", " << b;
            EXPECT_TRUE(down <= a && down > a - abs_b) << a << ", " << b;
        }
    }
}

}  // namespace
