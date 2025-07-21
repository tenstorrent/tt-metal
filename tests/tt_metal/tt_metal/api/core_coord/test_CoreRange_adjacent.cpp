// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <memory>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"

namespace basic_tests::CoreRange {

TEST_F(CoreCoordFixture, TestCoreRangeAdjacent) {
    EXPECT_TRUE(this->cr1.adjacent(this->cr9));
    EXPECT_TRUE(this->cr9.adjacent(this->cr1));
    EXPECT_TRUE(this->cr1.adjacent(this->cr10));
    EXPECT_TRUE(this->sc1.adjacent(this->sc2));
    EXPECT_TRUE(this->sc2.adjacent(this->sc3));
    EXPECT_TRUE(this->cr1.adjacent(this->cr3));
    EXPECT_TRUE(this->cr3.adjacent(this->cr1));
    EXPECT_TRUE(this->cr1.adjacent(this->cr7));
}

TEST_F(CoreCoordFixture, TestCoreRangeNotAdjacent) {
    EXPECT_FALSE(this->cr2.adjacent(this->cr3));
    EXPECT_FALSE(this->cr1.adjacent(this->cr6));
    EXPECT_FALSE(this->cr1.adjacent(this->cr11));
    EXPECT_FALSE(this->cr6.adjacent(this->sc1));
    EXPECT_FALSE(this->cr16.adjacent(this->sc4));
    EXPECT_FALSE(this->sc2.adjacent(this->sc4));
    EXPECT_FALSE(this->cr1.adjacent(this->cr2));
}

}  // namespace basic_tests::CoreRange
