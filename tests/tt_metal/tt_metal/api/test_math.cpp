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

}  // namespace
