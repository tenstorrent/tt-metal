// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>

namespace basic_tests::CoreRangeSet {

TEST_F(CoreCoordFixture, TestCoreRangeSetIntersects) {
    // Intersects CoreCoord
    EXPECT_TRUE(::CoreRangeSet(this->cr1).intersects(this->cr5.start_coord));
    EXPECT_TRUE(::CoreRangeSet(this->cr5).intersects(this->cr1.end_coord));
    EXPECT_TRUE(::CoreRangeSet(std::vector{this->sc1, this->sc4}).intersects(this->cr3.start_coord));
    EXPECT_TRUE(::CoreRangeSet(std::vector{this->cr17, this->cr16}).intersects(this->sc4.start_coord));
    EXPECT_TRUE(::CoreRangeSet(this->cr11).intersects(this->cr12.end_coord));
    // Intersects CoreRange
    EXPECT_TRUE(::CoreRangeSet(this->cr1).intersects(this->cr5));
    EXPECT_TRUE(::CoreRangeSet(this->cr5).intersects(this->cr1));
    EXPECT_TRUE(::CoreRangeSet(std::vector{this->sc1, this->sc4}).intersects(this->cr3));
    EXPECT_TRUE(::CoreRangeSet(std::vector{this->cr17, this->cr16}).intersects(this->sc4));
    EXPECT_TRUE(::CoreRangeSet(this->cr11).intersects(this->cr12));
    // Intersects CoreRangeSet
    EXPECT_TRUE(::CoreRangeSet(this->cr1).intersects(::CoreRangeSet(this->cr5)));
    EXPECT_TRUE(::CoreRangeSet(this->cr5).intersects(::CoreRangeSet(this->cr1)));
    EXPECT_TRUE(::CoreRangeSet(std::vector{this->sc1, this->sc4}).intersects(::CoreRangeSet(this->cr3)));
    EXPECT_TRUE(::CoreRangeSet(std::vector{this->cr17, this->cr16})
                    .intersects(::CoreRangeSet(std::vector{this->sc2, this->sc4})));
    EXPECT_TRUE(::CoreRangeSet(this->sc2).intersects(::CoreRangeSet(std::vector{this->cr7, this->cr1})));
}

TEST_F(CoreCoordFixture, TestCoreRangeSetNotIntersects) {
    // Not Intersects CoreCoord
    EXPECT_FALSE(::CoreRangeSet(this->cr1).intersects(this->cr2.start_coord));
    EXPECT_FALSE(
        ::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc3, this->sc4}).intersects(this->cr17.start_coord));
    EXPECT_FALSE(::CoreRangeSet(std::vector{this->cr1, this->sc4}).intersects(this->cr7.start_coord));
    // Not Intersects CoreRange
    EXPECT_FALSE(::CoreRangeSet(this->cr1).intersects(this->cr2));
    EXPECT_FALSE(::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc3, this->sc4}).intersects(this->cr17));
    EXPECT_FALSE(::CoreRangeSet(std::vector{this->cr1, this->sc4}).intersects(this->cr7));
    // Not Intersects CoreRangeSet
    EXPECT_FALSE(::CoreRangeSet(this->cr1).intersects(::CoreRangeSet(std::vector{this->cr2, this->cr3})));
    EXPECT_FALSE(::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc3, this->sc4})
                     .intersects(::CoreRangeSet(std::vector{this->cr17, this->cr18})));
    EXPECT_FALSE(::CoreRangeSet(std::vector{this->cr1, this->sc4}).intersects(::CoreRangeSet(this->cr7)));
}

}  // namespace basic_tests::CoreRangeSet
