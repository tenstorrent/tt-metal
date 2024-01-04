// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

using namespace tt::tt_metal;

class MultiCommandQueueSingleDeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        auto num_cqs = getenv("TT_METAL_NUM_HW_CQS");
        if (num_cqs == nullptr or strcmp(num_cqs, "2")) {
            TT_THROW("This suite must be run with TT_METAL_NUM_HW_CQS=2");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        device_ = tt::tt_metal::CreateDevice(0, std::stoi(num_cqs));
    }

    void TearDown() override {
        tt::tt_metal::CloseDevice(device_);
    }

    tt::tt_metal::Device* device_;
    tt::ARCH arch_;
};
