// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>

namespace basic_tests::CoreRange {

TEST_F(CoreCoordFixture, TestCoreRangeIntersects) {
    EXPECT_TRUE(this->cr1.intersects(this->cr5));
    EXPECT_EQ(this->cr1.intersection(this->cr5).value(), ::CoreRange({1, 0}, {1, 1}));

    EXPECT_TRUE(this->sc1.intersects(this->cr6));
    EXPECT_EQ(this->sc1.intersection(this->cr6).value(), this->sc1);

    EXPECT_TRUE(this->cr4.intersects(this->cr5));
    EXPECT_EQ(this->cr4.intersection(this->cr5).value(), ::CoreRange({1, 0}, {5, 4}));

    EXPECT_TRUE(this->cr1.intersects(this->cr6));
    EXPECT_EQ(this->cr1.intersection(this->cr6).value(), this->cr1);

    EXPECT_TRUE(this->cr7.intersects(this->cr8));
    EXPECT_EQ(this->cr7.intersection(this->cr8).value(), this->cr7);
}

TEST_F(CoreCoordFixture, TestCoreRangeNotIntersects) {
    EXPECT_FALSE(this->cr1.intersects(this->cr2));
    EXPECT_FALSE(this->sc1.intersects(this->cr2));
    EXPECT_FALSE(this->cr1.intersects(this->cr7));
}

}  // namespace basic_tests::CoreRange
