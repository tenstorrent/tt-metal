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
class CommandQueueFixture : public ::testing::Test {
   protected:
    tt::ARCH arch_;
    Device* device_;
    u32 pcie_id;

    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_fatal("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        if (this->arch_ != tt::ARCH::GRAYSKULL) {
            tt::log_info("Skipping since this fast dispatch suite can only be run with GS");
            GTEST_SKIP();
        }

        const int device_id = 0;
        this->device_ = tt::tt_metal::CreateDevice(device_id);
        tt::tt_metal::InitializeDevice(this->device_);

        this->pcie_id = 0;
    }

    void TearDown() override {
        if (this->arch_ != tt::ARCH::GRAYSKULL) {
            GTEST_SKIP();
        }
        tt::tt_metal::CloseDevice(this->device_);
    }
};
