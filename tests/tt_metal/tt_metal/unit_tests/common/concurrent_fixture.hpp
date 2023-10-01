/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/common/concurrency_interface.hpp"

// Remove shared memory on setup and teardown to clean up after any potentially erroneous tests
class ConcurrentFixture : public ::testing::Test  {
   protected:
    void SetUp() override {
        tt::concurrent::remove_shared_memory();
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            tt::log_fatal("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

        // Debug tools are not supported in concurrent modes of execution
        if (getenv("TT_METAL_DPRINT_CORES") != nullptr) {
            tt::log_fatal("Concurrent test suite cannot run wih DPRINT enabled");
            GTEST_SKIP();
        }

        if (getenv("TT_METAL_WATCHER") != nullptr) {
            tt::log_fatal("Concurrent test suite cannot run wih Watcher enabled");
            GTEST_SKIP();
        }
    }
};
