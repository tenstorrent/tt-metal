// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.hpp"
#include "core_coord_fixture.hpp"

namespace basic_tests::CoreRange{

TEST_F(CoreCoordHarness, TestCoreRangeContains){
    EXPECT_TRUE(this->cr1.contains(this->sc1));
    EXPECT_TRUE(this->cr1.contains(this->cr1));
    EXPECT_TRUE(this->cr4.contains(this->cr2));
}

TEST_F(CoreCoordHarness, TestCoreRangeNotContains){
    EXPECT_FALSE( this->sc1.contains(this->cr1));
    EXPECT_FALSE( this->sc1.contains(this->sc2));
    EXPECT_FALSE( this->cr1.contains(this->cr2));
    EXPECT_FALSE( this->cr7.contains(this->sc1));
}

}
