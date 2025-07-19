// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_types.hpp"
#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// #22835: These Fixtures will be removed once tests are fully migrated, and replaced by
// UnitMeshMultiCQSingleDeviceFixtures
class MultiCommandQueueSingleDeviceFixture : public DispatchFixture {
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

class UnitMeshMultiCQSingleDeviceFixture : public DispatchFixture {
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

        this->create_devices();
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device.reset();
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

    void create_devices(std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const chip_id_t mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<chip_id_t> chip_ids;
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, trace_region_size, 2, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            this->devices_.push_back(device);
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    tt::ARCH arch_;
    uint8_t num_cqs_;
};

class UnitMeshMultiCQSingleDeviceProgramFixture : public UnitMeshMultiCQSingleDeviceFixture {};

class UnitMeshMultiCQSingleDeviceTraceFixture : public UnitMeshMultiCQSingleDeviceFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevices(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        this->create_devices(trace_region_size);
    }
};

class MultiCommandQueueSingleDeviceEventFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueSingleDeviceBufferFixture : public MultiCommandQueueSingleDeviceFixture {};

class MultiCommandQueueSingleDeviceProgramFixture : public MultiCommandQueueSingleDeviceFixture {};

// #22835: These Fixtures will be removed once tests are fully migrated, and replaced by
// UnitMeshMultiCQMultiDeviceFixtures
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

        std::vector<int> devices_to_open;
        devices_to_open.reserve(tt::tt_metal::GetNumAvailableDevices());
        for (int i = 0; i < tt::tt_metal::GetNumAvailableDevices(); ++i) {
            devices_to_open.push_back(i);
        }
        reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            devices_to_open, num_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        for (const auto& [id, device] : reserved_devices_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override {
        if (!reserved_devices_.empty()) {
            tt::tt_metal::detail::CloseDevices(reserved_devices_);
        }
    }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
};

class UnitMeshMultiCQMultiDeviceFixture : public DispatchFixture {
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

        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const chip_id_t mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<chip_id_t> chip_ids;
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 2, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            this->devices_.push_back(device);
        }
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device.reset();
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
};

class MultiCommandQueueMultiDeviceBufferFixture : public MultiCommandQueueMultiDeviceFixture {};

class MultiCommandQueueMultiDeviceEventFixture : public MultiCommandQueueMultiDeviceFixture {};

class DISABLED_MultiCQMultiDeviceOnFabricFixture : public UnitMeshMultiCQMultiDeviceFixture,
                                                   public ::testing::WithParamInterface<tt::tt_fabric::FabricConfig> {
private:
    // Save the result to reduce UMD calls
    inline static bool should_skip_ = false;
    bool original_fd_fabric_en_ = false;

protected:
    void SetUp() override {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Dispatch on Fabric tests only applicable on Wormhole B0";
        }
        // Skip for TG as it's still being implemented
        if (tt::tt_metal::IsGalaxyCluster()) {
            GTEST_SKIP();
        }
        original_fd_fabric_en_ = tt::tt_metal::MetalContext::instance().rtoptions().get_fd_fabric();
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(true);
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_fabric::SetFabricConfig(GetParam(), tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        UnitMeshMultiCQMultiDeviceFixture::SetUp();

        if (::testing::Test::IsSkipped()) {
            tt::tt_fabric::SetFabricConfig(
                tt::tt_fabric::FabricConfig::DISABLED, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        }
    }

    void TearDown() override {
        UnitMeshMultiCQMultiDeviceFixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(original_fd_fabric_en_);
    }
};

}  // namespace tt::tt_metal
