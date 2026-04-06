// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>

class CoreCoordFixture : public ::testing::Test {
protected:
    CoreRange cr1 = CoreRange({0, 0}, {1, 1});
    CoreRange cr2 = CoreRange({3, 3}, {5, 4});
    CoreRange cr3 = CoreRange({1, 2}, {2, 2});
    CoreRange cr4 = CoreRange({0, 0}, {5, 4});
    CoreRange cr5 = CoreRange({1, 0}, {6, 4});
    CoreRange cr6 = CoreRange({0, 0}, {6, 4});
    CoreRange cr7 = CoreRange({2, 0}, {7, 4});
    CoreRange cr8 = CoreRange({0, 0}, {7, 4});
    CoreRange cr9 = CoreRange({2, 0}, {7, 1});
    CoreRange cr10 = CoreRange({0, 2}, {1, 2});
    CoreRange cr11 = CoreRange({1, 0}, {7, 1});
    CoreRange cr12 = CoreRange({0, 0}, {7, 1});
    CoreRange cr13 = CoreRange({0, 0}, {1, 2});
    CoreRange cr14 = CoreRange({0, 1}, {1, 1});
    CoreRange cr15 = CoreRange({0, 1}, {0, 2});
    CoreRange cr16 = CoreRange({0, 0}, {1, 2});
    CoreRange cr17 = CoreRange({2, 3}, {2, 3});
    CoreRange cr18 = CoreRange({3, 1}, {3, 3});

    CoreRange sc1 = CoreRange({1, 1}, {1, 1});
    CoreRange sc2 = CoreRange({0, 1}, {0, 1});
    CoreRange sc3 = CoreRange({0, 2}, {0, 2});
    CoreRange sc4 = CoreRange({1, 2}, {1, 2});
    CoreRange sc5 = CoreRange({1, 0}, {1, 0});
    CoreRange sc6 = CoreRange({0, 0}, {0, 0});
};
