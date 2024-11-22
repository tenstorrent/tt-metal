// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

class ControlPlaneFixture : public ::testing::Test {
   protected:
    tt::ARCH arch_;
    tt::tt_metal::Device* device_;
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            tt::log_info(tt::LogTest, "Control plane test suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

    }

    void TearDown() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        }
    }
};
