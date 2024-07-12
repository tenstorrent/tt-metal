// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "core_coord_fixture.hpp"

namespace basic_tests::CoreRange {

TEST_F(CoreCoordHarness, TestCoreRangeIterator)
{

    uint32_t num_cores_iterated = 0;
    for (CoreCoord& core : this->cr1)
    {
        EXPECT_TRUE(this->cr1.contains(core));
        num_cores_iterated++;
    }
    EXPECT_EQ(num_cores_iterated, this->cr1.size());

    num_cores_iterated = 0;
    for (CoreCoord& core : this->cr2)
    {
        EXPECT_TRUE(this->cr2.contains(core));
        num_cores_iterated++;
    }
    EXPECT_EQ(num_cores_iterated, this->cr2.size());

    num_cores_iterated = 0;
    for (CoreCoord& core : this->cr3)
    {
        EXPECT_TRUE(this->cr3.contains(core));
        num_cores_iterated++;
    }
    EXPECT_EQ(num_cores_iterated, this->cr3.size());

    num_cores_iterated = 0;
    for (CoreCoord& core : this->cr5)
    {
        EXPECT_TRUE(this->cr5.contains(core));
        num_cores_iterated++;
    }
    EXPECT_EQ(num_cores_iterated, this->cr5.size());

    num_cores_iterated = 0;
    for (CoreCoord& core : this->cr17)
    {
        EXPECT_TRUE(this->cr17.contains(core));
        num_cores_iterated++;
    }
    EXPECT_EQ(num_cores_iterated, this->cr17.size());

    num_cores_iterated = 0;
    for (CoreCoord& core : this->cr18)
    {
        EXPECT_TRUE(this->cr18.contains(core));
        num_cores_iterated++;
    }
    EXPECT_EQ(num_cores_iterated, this->cr18.size());
}
}
