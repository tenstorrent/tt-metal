// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include "impl/device/device.hpp"
#include "llrt/hal.hpp"
#include "tt_cluster_descriptor_types.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

class MultiCommandQueueSingleDeviceFixture : public DispatchFixture {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        this->slow_dispatch_ = false;
        auto num_cqs = tt::llrt::OptionsG.get_num_hw_cqs();
        if (num_cqs != 2) {
            TT_THROW("This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                tt::log_warning(tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
                dispatch_core_type = DispatchCoreType::ETH;
            }
        }
        device_ = tt::tt_metal::CreateDevice(0, num_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    }

    void TearDown() override {
        tt::tt_metal::CloseDevice(device_);
    }

    tt::tt_metal::Device* device_;
    tt::ARCH arch_;
};

class MultiCommandQueueSingleDeviceEventFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueSingleDeviceBufferFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueSingleDeviceProgramFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueMultiDeviceFixture : public DispatchFixture {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        this->slow_dispatch_ = false;
        auto num_cqs = tt::llrt::OptionsG.get_num_hw_cqs();
        if (num_cqs != 2) {
            TT_THROW("This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }
        const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());


        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (arch == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                tt::log_warning(tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
                dispatch_core_type = DispatchCoreType::ETH;
            }
        }

        const chip_id_t mmio_device_id = 0;
        reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id}, num_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        for (const auto &[id, device] : reserved_devices_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::Device*> devices_;
    std::map<chip_id_t, tt::tt_metal::Device*> reserved_devices_;
};

class MultiCommandQueueMultiDeviceBufferFixture : public MultiCommandQueueMultiDeviceFixture {};

class MultiCommandQueueMultiDeviceEventFixture : public MultiCommandQueueMultiDeviceFixture {};
