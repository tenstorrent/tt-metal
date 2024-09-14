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
        auto num_cqs = tt::llrt::OptionsG.get_num_hw_cqs();
        if (num_cqs != 2) {
            TT_THROW("This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1) {
            tt::log_warning(tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
            dispatch_core_type = DispatchCoreType::ETH;
        }
        device_ = tt::tt_metal::CreateDevice(0, {.num_hw_cqs = num_cqs, .dispatch_core_type = dispatch_core_type});
    }

    void TearDown() override {
        tt::tt_metal::CloseDevice(device_);
    }

    tt::tt_metal::Device* device_;
    tt::ARCH arch_;
};

class SingleDeviceTraceFixture: public ::testing::Test {
protected:
    Device* device_;
    tt::ARCH arch_;

    void Setup(const size_t buffer_size, const uint8_t num_hw_cqs = 1) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        if (num_hw_cqs > 1) {
            // Running multi-CQ test. User must set this explicitly.
            auto num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
            if (num_cqs == nullptr or strcmp(num_cqs, "2")) {
                TT_THROW("This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
                GTEST_SKIP();
            }
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int device_id = 0;
        this->device_ = tt::tt_metal::CreateDevice(device_id, {.trace_region_size = buffer_size});
    }

    void TearDown() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

};
