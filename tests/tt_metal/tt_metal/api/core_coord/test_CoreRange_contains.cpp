// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>

namespace basic_tests::CoreRange {

TEST_F(CoreCoordFixture, TestCoreRangeContains) {
    // Contains CoreCoord
    EXPECT_TRUE(this->cr1.contains(this->sc1.start_coord));
    EXPECT_TRUE(this->cr1.contains(this->cr1.start_coord));
    EXPECT_TRUE(this->cr4.contains(this->cr2.start_coord));
    // Contains CoreRange
    EXPECT_TRUE(this->cr1.contains(this->sc1));
    EXPECT_TRUE(this->cr1.contains(this->cr1));
    EXPECT_TRUE(this->cr4.contains(this->cr2));
    // Contains CoreRangeSet
    EXPECT_TRUE(this->cr1.contains(::CoreRangeSet(this->sc1)));
    EXPECT_TRUE(this->cr1.contains(::CoreRangeSet(this->cr1)));
    EXPECT_TRUE(this->cr4.contains(::CoreRangeSet(std::vector{this->cr1, this->cr2, this->cr3})));
}

TEST_F(CoreCoordFixture, TestCoreRangeNotContains) {
    // Not Contains CoreCoord
    EXPECT_FALSE(this->sc1.contains(this->cr1.start_coord));
    EXPECT_FALSE(this->sc1.contains(this->sc2.start_coord));
    EXPECT_FALSE(this->cr1.contains(this->cr2.start_coord));
    EXPECT_FALSE(this->cr7.contains(this->sc1.start_coord));
    // Not Contains CoreRange
    EXPECT_FALSE(this->sc1.contains(this->cr1));
    EXPECT_FALSE(this->sc1.contains(this->sc2));
    EXPECT_FALSE(this->cr1.contains(this->cr2));
    EXPECT_FALSE(this->cr7.contains(this->sc1));
    // Not Contains CoreRangeSet
    EXPECT_FALSE(this->sc1.contains(::CoreRangeSet(this->cr1)));
    EXPECT_FALSE(this->sc1.contains(::CoreRangeSet(this->sc2)));
    EXPECT_FALSE(this->cr1.contains(::CoreRangeSet(std::vector{this->sc1, this->cr2})));
    EXPECT_FALSE(this->cr10.contains(::CoreRangeSet(std::vector{this->sc3, this->sc4, this->sc1})));
}

}  // namespace basic_tests::CoreRange
