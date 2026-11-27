// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"

namespace basic_tests::CoreRangeSet {

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetValidConstruct) {
    EXPECT_NO_THROW(tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->cr2}));
    EXPECT_NO_THROW(tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->cr2}));

    tt::tt_metal::CoreRangeSet valid_ranges = tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->cr2});
    EXPECT_EQ(valid_ranges.ranges().size(), 2);
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetInvalidConstruct) {
    tt::tt_metal::CoreRange overlapping_range({1, 2}, {3, 3});
    EXPECT_ANY_THROW(tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->cr2, overlapping_range}));
    EXPECT_ANY_THROW(tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->cr1}));
}

using tt::tt_metal::CoreCoord;

namespace {
tt::tt_metal::CoreRangeSet from_coords(const std::vector<CoreCoord>& coords) {
    return tt::tt_metal::CoreRangeSet(ttsl::Span<const CoreCoord>(coords));
}
}  // namespace

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetFromCoordsEmptyAndSingleton) {
    EXPECT_TRUE(from_coords({}).empty());

    const auto single = from_coords({CoreCoord{3, 4}});
    EXPECT_EQ(single, tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({3, 4}, {3, 4})));
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetFromCoordsUnsortedInputMergesToSameCoverage) {
    // A 2x3 block listed out of order merges into one range.
    const auto block = from_coords({{1, 2}, {0, 0}, {1, 0}, {0, 2}, {1, 1}, {0, 1}});
    EXPECT_EQ(block, tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {1, 2})));

    // Sparse, unsorted cores keep exactly their coverage.
    const std::vector<CoreCoord> sparse{{5, 5}, {0, 7}, {2, 1}};
    const auto sparse_set = from_coords(sparse);
    EXPECT_EQ(sparse_set.num_cores(), sparse.size());
    for (const auto& core : sparse) {
        EXPECT_TRUE(sparse_set.contains(core));
    }
    EXPECT_FALSE(sparse_set.contains(CoreCoord{1, 1}));
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetFromCoordsDuplicateCoreThrows) {
    EXPECT_ANY_THROW(from_coords({{1, 1}, {1, 1}}));
    // Duplicates that are not adjacent in the input are still caught.
    EXPECT_ANY_THROW(from_coords({{2, 3}, {0, 0}, {4, 1}, {2, 3}}));
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetFromCoordsMatchesMergeRanges) {
    // The coordinate constructor must produce exactly the ranges, in the same order, that merge_ranges() produces
    // from the same cores as single-core ranges.
    std::mt19937 rng(0);
    for (int trial = 0; trial < 500; trial++) {
        const size_t width = 1 + (rng() % 12);
        const size_t height = 1 + (rng() % 10);
        const size_t x_offset = rng() % 40;
        const size_t y_offset = rng() % 40;
        const unsigned density = rng() % 101;
        std::vector<CoreCoord> coords;
        std::vector<tt::tt_metal::CoreRange> singles;
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                if (rng() % 100 < density) {
                    coords.push_back({x + x_offset, y + y_offset});
                }
            }
        }
        const bool add_duplicate = !coords.empty() && rng() % 10 == 0;
        if (add_duplicate) {
            coords.push_back(coords[rng() % coords.size()]);
        }
        std::shuffle(coords.begin(), coords.end(), rng);
        if (add_duplicate) {
            EXPECT_ANY_THROW(from_coords(coords)) << "trial " << trial;
            continue;
        }
        singles.reserve(coords.size());
        for (const auto& core : coords) {
            singles.emplace_back(core);
        }
        const auto expected = tt::tt_metal::CoreRangeSet(std::move(singles)).merge_ranges();
        ASSERT_EQ(from_coords(coords).ranges(), expected.ranges()) << "trial " << trial;
        // Row-major input skips the sort.
        std::sort(coords.begin(), coords.end(), [](const CoreCoord& a, const CoreCoord& b) {
            return std::tie(a.y, a.x) < std::tie(b.y, b.x);
        });
        ASSERT_EQ(from_coords(coords).ranges(), expected.ranges()) << "row-major trial " << trial;
    }
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetFromCoordsSameSpanAcrossEmptyRow) {
    const auto set = from_coords({{1, 0}, {2, 0}, {1, 2}, {2, 2}});
    const std::vector<tt::tt_metal::CoreRange> expected{
        tt::tt_metal::CoreRange({1, 0}, {2, 0}), tt::tt_metal::CoreRange({1, 2}, {2, 2})};
    EXPECT_EQ(set.ranges(), expected);
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetFromCoordsLargeCoordinates) {
    constexpr size_t max = std::numeric_limits<size_t>::max();
    const auto set = from_coords({{max, 0}, {0, 0}, {max, 1}});
    const std::vector<tt::tt_metal::CoreRange> expected{
        tt::tt_metal::CoreRange({0, 0}, {0, 0}), tt::tt_metal::CoreRange({max, 0}, {max, 1})};
    EXPECT_EQ(set.ranges(), expected);
    EXPECT_ANY_THROW(from_coords({{max, max}, {0, 0}, {max, max}}));
}

}  // namespace basic_tests::CoreRangeSet
