// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

class CoreCoordHarness : public ::testing::Test {
   protected:
    CoreRange cr1 = {.start={0, 0}, .end={1, 1}};
    CoreRange cr2 = {.start={3, 3}, .end={5, 4}};
    CoreRange cr3 = {.start={1, 2}, .end={2, 2}};
    CoreRange cr4 = {.start={0, 0}, .end={5, 4}};
    CoreRange cr5 = {.start={1, 0}, .end={6, 4}};
    CoreRange cr6 = {.start={0, 0}, .end={6, 4}};
    CoreRange cr7 = {.start={2, 0}, .end={7, 4}};
    CoreRange cr8 = {.start={0, 0}, .end={7, 4}};
    CoreRange cr9 = {.start={2, 0}, .end={7, 1}};
    CoreRange cr10 = {.start={0, 2}, .end={1, 2}};
    CoreRange cr11 = {.start={1, 0}, .end={7, 1}};
    CoreRange cr12 = {.start={0, 0}, .end={7, 1}};
    CoreRange cr13 = {.start={0, 0}, .end={1, 2}};
    CoreRange cr14 = {.start={0, 1}, .end={1, 1}};
    CoreRange cr15 = {.start={0, 1}, .end={0, 2}};
    CoreRange cr16 = {.start={0, 0}, .end={1, 2}};

    CoreRange sc1 = {.start={1, 1}, .end={1, 1}};
    CoreRange sc2 = {.start={0, 1}, .end={0, 1}};
    CoreRange sc3 = {.start={0, 2}, .end={0, 2}};
    CoreRange sc4 = {.start={1, 2}, .end={1, 2}};
};
