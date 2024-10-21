// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "core_coord_fixture.hpp"

namespace basic_tests::CoreRangeSet{

TEST_F(CoreCoordHarness, TestCoreRangeSetValidConstruct)
{
    EXPECT_NO_THROW(::CoreRangeSet(std::vector{this->sc1, this->cr2}));
    EXPECT_NO_THROW(::CoreRangeSet(std::vector{this->cr1, this->cr2}));

    ::CoreRangeSet valid_ranges = ::CoreRangeSet(std::vector{this->cr1, this->cr2});
    EXPECT_EQ(valid_ranges.ranges().size(), 2);
}

TEST_F(CoreCoordHarness, TestCoreRangeSetInvalidConstruct)
{
    ::CoreRange overlapping_range({1, 2}, {3, 3});
    EXPECT_ANY_THROW(::CoreRangeSet(std::vector{this->cr1, this->cr2, overlapping_range}));
    EXPECT_ANY_THROW(::CoreRangeSet(std::vector{this->sc1, this->cr1}));
}


}
