// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <string>
#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"

namespace basic_tests::CoreRangeSet {

TEST_F(CoreCoordFixture, TestCoreRangeSetValidConstruct) {
    EXPECT_NO_THROW(::CoreRangeSet(std::vector{this->sc1, this->cr2}));
    EXPECT_NO_THROW(::CoreRangeSet(std::vector{this->cr1, this->cr2}));

    ::CoreRangeSet valid_ranges = ::CoreRangeSet(std::vector{this->cr1, this->cr2});
    EXPECT_EQ(valid_ranges.ranges().size(), 2);
}

TEST_F(CoreCoordFixture, TestCoreRangeSetInvalidConstruct) {
    ::CoreRange overlapping_range({1, 2}, {3, 3});
    EXPECT_ANY_THROW(::CoreRangeSet(std::vector{this->cr1, this->cr2, overlapping_range}));
    EXPECT_ANY_THROW(::CoreRangeSet(std::vector{this->sc1, this->cr1}));
}

// Construction from a list of CoreCoords (CoreRangeSet(Span<const CoreCoord>)) — exercises the
// solid-rectangle fast path and its fall-back to merge_ranges.
TEST_F(CoreCoordFixture, TestCoreRangeSetFromCoreCoordsSolidRectangle) {
    // A filled 3-wide x 2-tall block of coords collapses to a single CoreRange.
    std::vector<CoreCoord> coords;
    for (size_t y = 1; y <= 2; y++) {
        for (size_t x = 3; x <= 5; x++) {
            coords.push_back(CoreCoord(x, y));
        }
    }
    ::CoreRangeSet crs(coords);
    EXPECT_EQ(crs.size(), 1u);
    EXPECT_EQ(*crs.ranges().begin(), ::CoreRange(CoreCoord(3, 1), CoreCoord(5, 2)));
}

TEST_F(CoreCoordFixture, TestCoreRangeSetFromCoreCoordsSingle) {
    std::vector<CoreCoord> coords = {CoreCoord(4, 2)};
    ::CoreRangeSet crs(coords);
    EXPECT_EQ(crs.size(), 1u);
    EXPECT_EQ(*crs.ranges().begin(), ::CoreRange(CoreCoord(4, 2), CoreCoord(4, 2)));
}

TEST_F(CoreCoordFixture, TestCoreRangeSetFromCoreCoordsLShapeNotCollapsed) {
    // 3 coords forming an L (a 2x2 with one corner missing): NOT a solid rectangle, must keep the hole.
    std::vector<CoreCoord> coords = {CoreCoord(0, 0), CoreCoord(1, 0), CoreCoord(0, 1)};
    ::CoreRangeSet crs(coords);
    EXPECT_TRUE(crs.contains(CoreCoord(0, 0)));
    EXPECT_TRUE(crs.contains(CoreCoord(1, 0)));
    EXPECT_TRUE(crs.contains(CoreCoord(0, 1)));
    EXPECT_FALSE(crs.contains(CoreCoord(1, 1)));  // the missing corner stays missing
}

TEST_F(CoreCoordFixture, TestCoreRangeSetFromCoreCoordsDuplicateThrows) {
    // Duplicate coords are rejected by validate_no_overlap (pre-existing behavior: the per-coord ranges
    // overlap). The fast path must NOT mask this. Here bbox area (2x2=4) == coord count (4), but (0,0)
    // is duplicated and (0,1) absent — a naive count==area check would wrongly accept this as a solid
    // 2x2 rectangle (no throw). The dup-detecting bitset forces the merge_ranges fall-back, which throws
    // exactly as the original code did.
    std::vector<CoreCoord> coords = {CoreCoord(0, 0), CoreCoord(0, 0), CoreCoord(1, 0), CoreCoord(1, 1)};
    EXPECT_ANY_THROW(::CoreRangeSet{coords});
}

}  // namespace basic_tests::CoreRangeSet
