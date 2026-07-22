// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"

using std::vector;

namespace basic_tests::CoreRange {

TEST_F(CoreCoordFixture, TestCoreRangeIterator) {
    vector<tt::tt_metal::CoreCoord> cores_in_core_range;

    vector<tt::tt_metal::CoreCoord> cores_iterated;
    for (tt::tt_metal::CoreCoord& core : this->cr1) {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(1, 0), tt::tt_metal::CoreCoord(0, 1), tt::tt_metal::CoreCoord(1, 1)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (tt::tt_metal::CoreCoord& core : this->cr2) {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {
        tt::tt_metal::CoreCoord(3, 3), tt::tt_metal::CoreCoord(4, 3), tt::tt_metal::CoreCoord(5, 3), tt::tt_metal::CoreCoord(3, 4), tt::tt_metal::CoreCoord(4, 4), tt::tt_metal::CoreCoord(5, 4)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (tt::tt_metal::CoreCoord& core : this->cr3) {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {tt::tt_metal::CoreCoord(1, 2), tt::tt_metal::CoreCoord(2, 2)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (tt::tt_metal::CoreCoord& core : this->cr15) {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {tt::tt_metal::CoreCoord(0, 1), tt::tt_metal::CoreCoord(0, 2)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (tt::tt_metal::CoreCoord& core : this->cr17) {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {tt::tt_metal::CoreCoord(2, 3)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);

    cores_iterated.clear();
    for (tt::tt_metal::CoreCoord& core : this->cr18) {
        cores_iterated.push_back(core);
    }
    cores_in_core_range = {CoreCoord(3, 1), CoreCoord(3, 2), CoreCoord(3, 3)};
    EXPECT_EQ(cores_iterated, cores_in_core_range);
}
}  // namespace basic_tests::CoreRange
