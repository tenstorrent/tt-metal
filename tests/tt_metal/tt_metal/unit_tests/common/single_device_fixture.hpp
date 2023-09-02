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

class SingleDeviceFixture : public ::testing::Test  {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            tt::log_fatal("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int device_id = 0;

        device_ = tt::tt_metal::CreateDevice(device_id);
        tt::tt_metal::InitializeDevice(device_);
    }

    void TearDown() override {
        if (device_) {
            tt::tt_metal::CloseDevice(device_);
        }
    }

    tt::tt_metal::Device* device_ = nullptr;
    tt::ARCH arch_;
};
