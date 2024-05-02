// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

namespace ttnn {

class MultiCommandQueueSingleDeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi CQ test suite, since it can only be run in Fast Dispatch Mode.";
        }

        if (arch_ == tt::ARCH::WORMHOLE_B0 and num_devices_ != 1) {
            device_ = tt::tt_metal::CreateDevice(0); // Create device here so teardown can gracefully run
            GTEST_SKIP() << "Skipping for Multi-Chip Wormhole, since not enough dispatch cores.";
        }
        device_ = tt::tt_metal::CreateDevice(0, 2);
    }

    void TearDown() override {
        tt::tt_metal::CloseDevice(device_);
    }

    tt::tt_metal::Device* device_;
    tt::ARCH arch_;
    size_t num_devices_;
};
}
