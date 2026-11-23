// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <string>
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

}  // namespace basic_tests::CoreRangeSet
