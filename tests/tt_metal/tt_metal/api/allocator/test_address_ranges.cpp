// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <vector>

#include "tt_metal/impl/allocator/algorithms/address_ranges.hpp"

namespace tt::tt_metal::allocator {
namespace {

using Ranges = std::vector<AddressRange>;

TEST(AddressRangesTest, Coalesce) {
    // Sorts, then merges overlapping and touching ranges
    EXPECT_EQ(coalesce({{40, 50}, {0, 10}, {5, 20}, {20, 30}}), (Ranges{{0, 30}, {40, 50}}));
    // Keeps a contained range inside its container, and a separate range separate
    EXPECT_EQ(coalesce({{0, 100}, {10, 20}, {101, 102}}), (Ranges{{0, 100}, {101, 102}}));
    // Empty
    EXPECT_EQ(coalesce({}), Ranges{});
}

TEST(AddressRangesTest, SetIntersection) {
    // Overlapping ranges
    EXPECT_EQ(
        set_intersection({{0, 10}, {20, 30}, {40, 50}}, {{5, 25}, {45, 60}}), (Ranges{{5, 10}, {20, 25}, {45, 50}}));
    // Touching ranges share no address
    EXPECT_EQ(set_intersection({{0, 10}}, {{10, 20}}), Ranges{});
    // One range spans several, from either side
    EXPECT_EQ(set_intersection({{0, 100}}, {{10, 20}, {30, 40}}), (Ranges{{10, 20}, {30, 40}}));
    EXPECT_EQ(set_intersection({{10, 20}, {30, 40}}, {{0, 100}}), (Ranges{{10, 20}, {30, 40}}));
    // Empty
    EXPECT_EQ(set_intersection({{0, 10}}, {}), Ranges{});
    EXPECT_EQ(set_intersection({}, {{0, 10}}), Ranges{});
}

TEST(AddressRangesTest, SetDifference) {
    // Splits a range
    EXPECT_EQ(set_difference({{10, 30}}, {{12, 15}, {18, 25}}), (Ranges{{10, 12}, {15, 18}, {25, 30}}));
    // A removed range spans several ranges
    EXPECT_EQ(set_difference({{0, 10}, {20, 30}, {40, 50}}, {{5, 45}}), (Ranges{{0, 5}, {45, 50}}));
    // Touching removed ranges remove nothing
    EXPECT_EQ(set_difference({{10, 20}}, {{0, 10}, {20, 30}}), (Ranges{{10, 20}}));
    // Removes a whole range
    EXPECT_EQ(set_difference({{10, 20}}, {{10, 20}}), Ranges{});
    EXPECT_EQ(set_difference({{10, 20}}, {{0, 30}}), Ranges{});
    // Accepts touching ranges, which only need to be sorted
    EXPECT_EQ(set_difference({{0, 10}, {10, 20}}, {{5, 15}}), (Ranges{{0, 5}, {15, 20}}));
    // Empty
    EXPECT_EQ(set_difference({{0, 10}}, {}), (Ranges{{0, 10}}));
    EXPECT_EQ(set_difference({}, {{0, 10}}), Ranges{});
}

TEST(AddressRangesTest, TrimBelow) {
    const auto trimmed_below = [](Ranges ranges, DeviceAddr limit) {
        trim_below(ranges, limit);
        return ranges;
    };
    // Trims the range that crosses the limit, and drops the ranges below it
    EXPECT_EQ(trimmed_below({{0, 10}, {15, 25}, {30, 40}}, 20), (Ranges{{20, 25}, {30, 40}}));
    // Drops a range that ends at the limit
    EXPECT_EQ(trimmed_below({{0, 20}, {20, 30}}, 20), (Ranges{{20, 30}}));
    // A limit of 0 changes nothing, not even dropping an empty range
    EXPECT_EQ(trimmed_below({{20, 30}, {5, 5}}, 0), (Ranges{{20, 30}, {5, 5}}));
    // Keeps the order of an unsorted list
    EXPECT_EQ(trimmed_below({{30, 40}, {0, 10}, {15, 25}}, 20), (Ranges{{30, 40}, {20, 25}}));
}

TEST(AddressRangesTest, FirstFit) {
    // Bottom-up takes the lowest fit
    const Ranges bottom_up_ranges = {{0, 8}, {16, 32}, {40, 100}};
    EXPECT_EQ(first_fit(bottom_up_ranges, 8, true), 0);
    EXPECT_EQ(first_fit(bottom_up_ranges, 16, true), 16);
    EXPECT_EQ(first_fit(bottom_up_ranges, 20, true), 40);
    // Top-down takes the highest fit
    const Ranges top_down_ranges = {{0, 100}, {200, 210}};
    EXPECT_EQ(first_fit(top_down_ranges, 10, false), 200);
    EXPECT_EQ(first_fit(top_down_ranges, 20, false), 80);
    // Does not combine touching ranges
    EXPECT_EQ(first_fit({{0, 8}, {8, 16}}, 16, true), std::nullopt);
    // No fit
    const Ranges small_ranges = {{0, 8}, {16, 24}};
    EXPECT_EQ(first_fit(small_ranges, 9, true), std::nullopt);
    EXPECT_EQ(first_fit(small_ranges, 9, false), std::nullopt);
    EXPECT_EQ(first_fit({}, 1, true), std::nullopt);
    EXPECT_EQ(first_fit({}, 1, false), std::nullopt);
    // Top-down with size > range.second: forming range.second - size first would wrap around and
    // pass a >= range.first check
    EXPECT_EQ(first_fit({{10, 20}}, 30, false), std::nullopt);
    // Bottom-up with a huge size: range.first + size wraps around to 0x800, which would pass a
    // <= range.second check
    constexpr DeviceAddr huge = std::numeric_limits<DeviceAddr>::max() - 0x800;
    EXPECT_EQ(first_fit({{0x1000, 0x2000}}, huge, true), std::nullopt);
}

}  // namespace
}  // namespace tt::tt_metal::allocator
