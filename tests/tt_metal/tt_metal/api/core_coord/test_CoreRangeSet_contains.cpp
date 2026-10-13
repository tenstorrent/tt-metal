// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>

namespace basic_tests::CoreRangeSet {

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetContains) {
    // Contains CoreCoord
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr1).contains(this->cr5.start_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr5).contains(this->cr1.end_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc4}).contains(this->cr3.start_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(std::vector{this->cr17, this->cr16}).contains(this->sc2.start_coord));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr11).contains(this->cr12.end_coord));
    // Contains CoreRange
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr1).contains(this->sc1));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr8).contains(this->cr1));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr13).contains(this->cr16));
    EXPECT_TRUE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc5, this->sc6}).contains(this->cr1));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(std::vector{this->cr17, this->cr16}).contains(this->cr1));
    // Contains CoreRangeSet
    EXPECT_TRUE(
        tt::tt_metal::CoreRangeSet(this->cr1).contains(
            tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc5, this->sc6})));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet(this->cr5).contains(tt::tt_metal::CoreRangeSet()));
    EXPECT_TRUE(tt::tt_metal::CoreRangeSet().contains(tt::tt_metal::CoreRangeSet()));
    EXPECT_TRUE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc6, cr5, sc2})
            .contains(tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->cr2, this->cr3})));
    EXPECT_TRUE(
        tt::tt_metal::CoreRangeSet(this->cr12)
            .contains(tt::tt_metal::CoreRangeSet(std::vector{this->sc6, this->cr11})));
}

TEST_F(CoreCoordFixture, CPU_TestCoreRangeSetNotContains) {
    // Not Contains CoreCoord
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(this->cr1).contains(this->cr2.start_coord));
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc3, this->sc4})
            .contains(this->cr17.start_coord));
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->sc4}).contains(this->cr7.start_coord));
    // No Contains CoreRange
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(this->cr1).contains(this->sc3));
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc5, this->sc6}).contains(this->cr1));
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet(std::vector{this->cr7, this->cr1}).contains(this->cr16));
    // Not Contains CoreRangeSet
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc1, this->sc2, this->sc5})
            .contains(tt::tt_metal::CoreRangeSet(this->cr1)));
    EXPECT_FALSE(tt::tt_metal::CoreRangeSet().contains(tt::tt_metal::CoreRangeSet(this->cr5)));
    EXPECT_FALSE(
        tt::tt_metal::CoreRangeSet(std::vector{this->sc6, cr5})
            .contains(tt::tt_metal::CoreRangeSet(std::vector{this->cr1, this->cr2, this->cr3})));
}

}  // namespace basic_tests::CoreRangeSet
