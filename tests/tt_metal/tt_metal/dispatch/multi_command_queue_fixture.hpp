// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

class MultiCommandQueueSingleDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (this->num_cqs_ != 2) {
            log_info(tt::LogTest, "This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        const chip_id_t device_id = 0;
        const DispatchCoreType dispatch_core_type = this->get_dispatch_core_type();
        this->create_device(device_id, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    }

    void TearDown() override {
        if (this->device_ != nullptr) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

    void validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            GTEST_SKIP();
        }
    }

    DispatchCoreType get_dispatch_core_type() {
        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                log_warning(
                    tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in SetUp()");
                dispatch_core_type = DispatchCoreType::ETH;
            }
        }
        return dispatch_core_type;
    }

    void create_device(
        const chip_id_t device_id,
        const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        const DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER) {
        this->device_ = tt::tt_metal::CreateDevice(
            device_id, this->num_cqs_, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_type);
    }

    tt::tt_metal::IDevice* device_ = nullptr;
    tt::ARCH arch_;
    uint8_t num_cqs_;
};

class MultiCommandQueueSingleDeviceEventFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueSingleDeviceBufferFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueSingleDeviceProgramFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueSingleDeviceTraceFixture : public MultiCommandQueueSingleDeviceFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (this->num_cqs_ != 2) {
            log_info(tt::LogTest, "This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevice(const size_t trace_region_size) {
        const chip_id_t device_id = 0;
        const DispatchCoreType dispatch_core_type = this->get_dispatch_core_type();
        this->create_device(device_id, trace_region_size, dispatch_core_type);
    }

    DispatchCoreType dispatch_core_type_;
};

class MultiCommandQueueMultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            GTEST_SKIP();
        }

        auto num_cqs = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (num_cqs != 2) {
            log_info(tt::LogTest, "This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }

        const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (arch == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                log_warning(
                    tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
                dispatch_core_type = DispatchCoreType::ETH;
            }
        }

        const chip_id_t mmio_device_id = 0;
        reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            {mmio_device_id}, num_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        for (const auto& [id, device] : reserved_devices_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
};

class MultiCommandQueueMultiDeviceBufferFixture : public MultiCommandQueueMultiDeviceFixture {};

class MultiCommandQueueMultiDeviceEventFixture : public MultiCommandQueueMultiDeviceFixture {};

}  // namespace tt::tt_metal
