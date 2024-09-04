// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "core_coord_fixture.hpp"

namespace basic_tests::CoreRange {

TEST_F(CoreCoordHarness, TestCoreRangeIterator)
{
    vector<CoreCoord> cores_in_core_range;

    vector<CoreCoord> cores_iterated;
    for (CoreCoord& core : this->cr1)
    {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {CoreCoord(0, 0), CoreCoord(1, 0), CoreCoord(0, 1), CoreCoord(1, 1)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (CoreCoord& core : this->cr2)
    {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {CoreCoord(3, 3), CoreCoord(4, 3), CoreCoord(5, 3),
                            CoreCoord(3, 4), CoreCoord(4, 4), CoreCoord(5, 4)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (CoreCoord& core : this->cr3)
    {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {CoreCoord(1, 2), CoreCoord(2, 2)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (CoreCoord& core : this->cr15)
    {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {CoreCoord(0, 1), CoreCoord(0, 2)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (CoreCoord& core : this->cr17)
    {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {CoreCoord(2, 3)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (CoreCoord& core : this->cr18)
    {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {CoreCoord(3, 1), CoreCoord(3, 2), CoreCoord(3, 3)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);
}
}
