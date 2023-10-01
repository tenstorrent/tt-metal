/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using namespace tt::tt_metal;
class ConcurrentCommandQueueFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        // tt::concurrent::remove_shared_memory();
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_fatal("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
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
