// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>

class CoreCoordFixture : public ::testing::Test {
protected:
    tt::tt_metal::CoreRange cr1 = tt::tt_metal::CoreRange({0, 0}, {1, 1});
    tt::tt_metal::CoreRange cr2 = tt::tt_metal::CoreRange({3, 3}, {5, 4});
    tt::tt_metal::CoreRange cr3 = tt::tt_metal::CoreRange({1, 2}, {2, 2});
    tt::tt_metal::CoreRange cr4 = tt::tt_metal::CoreRange({0, 0}, {5, 4});
    tt::tt_metal::CoreRange cr5 = tt::tt_metal::CoreRange({1, 0}, {6, 4});
    tt::tt_metal::CoreRange cr6 = tt::tt_metal::CoreRange({0, 0}, {6, 4});
    tt::tt_metal::CoreRange cr7 = tt::tt_metal::CoreRange({2, 0}, {7, 4});
    tt::tt_metal::CoreRange cr8 = tt::tt_metal::CoreRange({0, 0}, {7, 4});
    tt::tt_metal::CoreRange cr9 = tt::tt_metal::CoreRange({2, 0}, {7, 1});
    tt::tt_metal::CoreRange cr10 = tt::tt_metal::CoreRange({0, 2}, {1, 2});
    tt::tt_metal::CoreRange cr11 = tt::tt_metal::CoreRange({1, 0}, {7, 1});
    tt::tt_metal::CoreRange cr12 = tt::tt_metal::CoreRange({0, 0}, {7, 1});
    tt::tt_metal::CoreRange cr13 = tt::tt_metal::CoreRange({0, 0}, {1, 2});
    tt::tt_metal::CoreRange cr14 = tt::tt_metal::CoreRange({0, 1}, {1, 1});
    tt::tt_metal::CoreRange cr15 = tt::tt_metal::CoreRange({0, 1}, {0, 2});
    tt::tt_metal::CoreRange cr16 = tt::tt_metal::CoreRange({0, 0}, {1, 2});
    tt::tt_metal::CoreRange cr17 = tt::tt_metal::CoreRange({2, 3}, {2, 3});
    tt::tt_metal::CoreRange cr18 = tt::tt_metal::CoreRange({3, 1}, {3, 3});

    tt::tt_metal::CoreRange sc1 = tt::tt_metal::CoreRange({1, 1}, {1, 1});
    tt::tt_metal::CoreRange sc2 = tt::tt_metal::CoreRange({0, 1}, {0, 1});
    tt::tt_metal::CoreRange sc3 = tt::tt_metal::CoreRange({0, 2}, {0, 2});
    tt::tt_metal::CoreRange sc4 = tt::tt_metal::CoreRange({1, 2}, {1, 2});
    tt::tt_metal::CoreRange sc5 = tt::tt_metal::CoreRange({1, 0}, {1, 0});
    tt::tt_metal::CoreRange sc6 = tt::tt_metal::CoreRange({0, 0}, {0, 0});
};
