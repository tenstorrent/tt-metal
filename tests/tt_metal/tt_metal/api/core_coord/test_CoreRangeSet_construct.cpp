// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

}  // namespace basic_tests::CoreRangeSet
