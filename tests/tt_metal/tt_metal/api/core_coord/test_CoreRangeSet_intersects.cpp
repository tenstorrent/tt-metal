// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>

namespace basic_tests::CoreRangeSet {

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetIntersects) {
    // Intersects CoreCoord
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr1).intersects(this->cr5.start_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr5).intersects(this->cr1.end_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc4}).intersects(this->cr3.start_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(std::vector{this->cr17, this->cr16}).intersects(this->sc4.start_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr11).intersects(this->cr12.end_coord));
    // Intersects CoreRange
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr1).intersects(this->cr5));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr5).intersects(this->cr1));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc4}).intersects(this->cr3));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(std::vector{this->cr17, this->cr16}).intersects(this->sc4));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr11).intersects(this->cr12));
    // Intersects CoreRangeSet
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr1).intersects(tt::tt_metal::CoreRangeSet(this->cr5)));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr5).intersects(tt::tt_metal::CoreRangeSet(this->cr1)));
    EXPECT_TRUE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc4})
            .intersects(tt::tt_metal::CoreRangeSet(this->cr3)));
    EXPECT_TRUE(
        tt::tt_metal::CoreRangeSet(std::vector{this->cr17, this->cr16})
            .intersects(tt::tt_metal::CoreRangeSet(std::vector{this->sc2, this->sc4})));
    EXPECT_TRUE(
        tt::tt_metal::CoreRangeSet(this->sc2).intersects(
            tt::tt_metal::CoreRangeSet(std::vector{this->cr7, this->cr1})));
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetNotIntersects) {
    // Not Intersects CoreCoord
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(this->cr1).intersects(this->cr2.start_coord));
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc3, this->sc4})
            .intersects(this->cr17.start_coord));
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->sc4}).intersects(this->cr7.start_coord));
    // Not Intersects CoreRange
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(this->cr1).intersects(this->cr2));
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc3, this->sc4}).intersects(this->cr17));
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->sc4}).intersects(this->cr7));
    // Not Intersects CoreRangeSet
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(this->cr1).intersects(
            tt::tt_metal::CoreRangeSet(std::vector{this->cr2, this->cr3})));
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc3, this->sc4})
            .intersects(tt::tt_metal::CoreRangeSet(std::vector{this->cr17, this->cr18})));
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->sc4})
            .intersects(tt::tt_metal::CoreRangeSet(this->cr7)));
}

}  // namespace basic_tests::CoreRangeSet
