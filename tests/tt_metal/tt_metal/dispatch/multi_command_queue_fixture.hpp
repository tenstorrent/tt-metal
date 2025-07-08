// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "command_queue_fixture.hpp"
#include "fabric_types.hpp"
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
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

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

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            return false;
        }
        return true;
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
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

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

class MultiCommandQueueMultiDeviceFixture : public CommandQueueMultiDeviceFixture {
protected:
    static bool ShouldSkip() {
        if (CommandQueueMultiDeviceFixture::ShouldSkip()) {
            return true;
        }

        if (tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs() != 2) {
            return true;
        }

        return false;
    }

    static std::string GetSkipMessage() {
        return "Requires fast dispatch, TT_METAL_GTEST_NUM_HW_CQS=2, at least 2 devices";
    }

    static void SetUpTestSuite() {
        if (ShouldSkip()) {
            return;
        }
        CommandQueueMultiDeviceFixture::DoSetUpTestSuite(2);
    }

    static void TearDownTestSuite() {
        if (ShouldSkip()) {
            return;
        }
        CommandQueueMultiDeviceFixture::DoTearDownTestSuite();
    }

    void SetUp() override {
        if (ShouldSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }
        CommandQueueMultiDeviceFixture::SetUp();
    }

    void TearDown() override {
        if (ShouldSkip()) {
            return;
        }
        CommandQueueMultiDeviceFixture::TearDown();
    }
};

class MultiCommandQueueMultiDeviceBufferFixture : public MultiCommandQueueMultiDeviceFixture {};

class MultiCommandQueueMultiDeviceEventFixture : public MultiCommandQueueMultiDeviceFixture {};

class MultiCommandQueueMultiDeviceOnFabricFixture : public MultiCommandQueueMultiDeviceFixture,
                                                    public ::testing::WithParamInterface<tt::tt_metal::FabricConfig> {
protected:
    static bool ShouldSkip() {
        if (MultiCommandQueueMultiDeviceFixture::ShouldSkip()) {
            return true;
        }
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            return true;
        }
        // Skip for TG as it's still being implemented
        if (tt::tt_metal::IsGalaxyCluster()) {
            return true;
        }
        return false;
    }

    static std::string GetSkipMessage() {
        return MultiCommandQueueMultiDeviceFixture::GetSkipMessage() + ", Wormhole B0, not Galaxy Cluster";
    }

    // Multiple fabric configs so need to reset the devices for each test
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (ShouldSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(true);
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_metal::detail::SetFabricConfig(GetParam(), FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        MultiCommandQueueMultiDeviceFixture::DoSetUpTestSuite(2);
        MultiCommandQueueMultiDeviceFixture::SetUp();

        if (::testing::Test::IsSkipped()) {
            tt::tt_metal::detail::SetFabricConfig(
                FabricConfig::DISABLED, FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        }
    }

    void TearDown() override {
        if (ShouldSkip()) {
            return;
        }
        MultiCommandQueueMultiDeviceFixture::TearDown();
        MultiCommandQueueMultiDeviceFixture::DoTearDownTestSuite();
        tt::tt_metal::detail::SetFabricConfig(FabricConfig::DISABLED);
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(false);
    }
};

}  // namespace tt::tt_metal
