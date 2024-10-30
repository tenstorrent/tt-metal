// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <deque>
#include <vector>

#include "tt_metal/tt_stl/any_range.hpp"

// convenience alias that guarantees at least vector and deque will fit within the capacity
using AnySizedRandomAccessIntRange = tt::stl::AnySizedRandomAccessRangeFor<int, std::vector<int>, std::deque<int>>;

TEST(AnyRangeTest, CanTypeEraseSizedRandomAccessRange) {
    AnySizedRandomAccessIntRange range{std::vector{1, 2, 3, 4, 5}};

    auto begin = range.begin();
    const auto end = range.end();
    begin[2] = 9;

    EXPECT_EQ(*begin, 1);
    ++begin;
    EXPECT_EQ(*begin, 2);
    ++begin;
    EXPECT_EQ(*begin, 9);
    ++begin;
    EXPECT_EQ(*begin, 4);
    ++begin;
    EXPECT_EQ(*begin, 5);
    ++begin;
    EXPECT_EQ(begin, end);

    {
        AnySizedRandomAccessIntRange another_range{std::deque{6, 7, 8}};

        using std::swap;
        swap(range, another_range);

        // another_range is now bound to std::vector from range
        const auto begin = another_range.begin();
        const auto end = another_range.end();

        EXPECT_EQ(end - begin, 5);
        EXPECT_EQ(end[-1], 5);
        EXPECT_EQ(end[-2], 4);
        EXPECT_EQ(end[-3], 9);
        EXPECT_EQ(end[-4], 2);
        EXPECT_EQ(end[-5], 1);
        EXPECT_EQ(end - 5, begin);
    }

    // range is now bound to std::deque from another_range
    auto rbegin = range.rbegin();
    const auto rend = range.rend();

    EXPECT_EQ(rbegin[0], 8);
    EXPECT_EQ(rbegin[1], 7);
    EXPECT_EQ(rbegin[2], 6);
    EXPECT_EQ(rbegin + 3, rend);
    EXPECT_EQ(rend - rbegin, 3);
    EXPECT_EQ(range.size(), 3);
    EXPECT_FALSE(range.empty());

    range[1] = 9;
    std::sort(range.rbegin(), range.rend());

    EXPECT_EQ(range[0], 9);
    EXPECT_EQ(range[1], 8);
    EXPECT_EQ(range[2], 6);

    {
        std::vector vector{10, 11, 12, 13, 14, 15};
        // a non-owning type-erased range, it stores vector as a reference
        range = vector;

        // modifying vector is observable through range
        vector[0] = 17;
        EXPECT_EQ(range[0], 17);

        auto begin = range.begin();
        auto end = range.end();

        // non-owning range writes back to vector
        begin += 3;
        *begin = 20;

        EXPECT_EQ(vector[3], 20);

        EXPECT_FALSE(begin == end);
        EXPECT_TRUE(begin != end);
        EXPECT_TRUE(begin < end);
        EXPECT_FALSE(begin > end);
        EXPECT_TRUE(begin <= end);
        EXPECT_FALSE(begin >= end);

        end -= 3;

        EXPECT_EQ(end - begin, 0);

        EXPECT_TRUE(end == begin);
        EXPECT_FALSE(end != begin);
        EXPECT_FALSE(end < begin);
        EXPECT_FALSE(end > begin);
        EXPECT_TRUE(end <= begin);
        EXPECT_TRUE(end >= begin);

        EXPECT_TRUE(vector.begin() == range.begin());
        EXPECT_TRUE(vector.end() == range.end());
        EXPECT_TRUE(vector.rbegin() == range.rbegin());
        EXPECT_TRUE(vector.rend() == range.rend());
    }

    // since range is non-owning, usage here would be UB after the vector has ended its lifetime

    {
        std::deque deque{10, 11, 12, 13, 14, 15};
        // an owning type-erased deque, it stores deque as a value
        range = std::move(deque);

        // modifying deque is not observable through range
        deque = {20, 21, 22, 23, 24, 25};
        EXPECT_EQ(range[0], 10);
        EXPECT_EQ(deque[0], 20);

        // owning range does not write back to deque
        range[3] = 30;
        EXPECT_EQ(deque[3], 23);
        EXPECT_EQ(range[3], 30);
    }

    // since range is owning, usage is still valid after the deque has ended its lifetime
    EXPECT_EQ(range[0], 10);
    EXPECT_EQ(range[1], 11);
    EXPECT_EQ(range[2], 12);
    EXPECT_EQ(range[3], 30);
    EXPECT_EQ(range[4], 14);
    EXPECT_EQ(range[5], 15);

    // an owning type-erased array
    range = std::array{30, 31, 32, 33, 34};

    EXPECT_EQ(range[0], 30);
    EXPECT_EQ(range[1], 31);
    EXPECT_EQ(range[2], 32);
    EXPECT_EQ(range[3], 33);
    EXPECT_EQ(range[4], 34);
}
