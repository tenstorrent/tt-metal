/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

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
    CoreRange single_core = {.start={1, 1}, .end={1, 1}};

    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            tt::log_fatal("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
    }
};
