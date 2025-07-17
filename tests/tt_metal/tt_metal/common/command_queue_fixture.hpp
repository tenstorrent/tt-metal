// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.h>
#include <cstdint>
#include "fabric_types.hpp"
#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/device.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"

namespace tt::tt_metal {
// #22835: These Fixtures will be removed once tests are fully migrated, and replaced by UnitMeshCQFixtures
class CommandQueueFixture : public DispatchFixture {
protected:
    tt::tt_metal::IDevice* device_;
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_device();
    }

    void TearDown() override {
        if (!this->IsSlowDispatch()) {
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

    void create_device(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const chip_id_t device_id = *tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().begin();
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        this->device_ =
            tt::tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_config);
    }
};

class CommandQueueEventFixture : public CommandQueueFixture {};

class CommandQueueBufferFixture : public CommandQueueFixture {};

class CommandQueueProgramFixture : public CommandQueueFixture {};

class CommandQueueTraceFixture : public CommandQueueFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevice(const size_t trace_region_size) { this->create_device(trace_region_size); }
};

class UnitMeshCQFixture : public DispatchFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
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
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
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
            chip_ids, DEFAULT_L1_SMALL_SIZE, trace_region_size, 1, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            this->devices_.push_back(device);
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
};

class UnitMeshCQProgramFixture : public UnitMeshCQFixture {};

// #22835: These Fixtures will be removed once tests are fully migrated, and replaced by UnitMeshCQSingleCardFixture and
// UnitMeshCQSingleCardProgramFixture
class CommandQueueSingleCardFixture : virtual public DispatchFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override {
        if (!reserved_devices_.empty()) {
            tt::tt_metal::detail::CloseDevices(reserved_devices_);
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }

    void create_devices(const std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const chip_id_t mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<chip_id_t> chip_ids;
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        this->reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            chip_ids, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_config);
        this->devices_.clear();
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip) {
            const auto tunnels =
                tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
            for (const auto& tunnel : tunnels) {
                for (const auto& chip_id : tunnel) {
                    if (reserved_devices_.contains(chip_id)) {
                        this->devices_.push_back(this->reserved_devices_.at(chip_id));
                    }
                }
                break;
            }
        } else {
            for (const auto& chip_id : chip_ids) {
                this->devices_.push_back(this->reserved_devices_.at(chip_id));
            }
        }
    }

    // Devices to test
    std::vector<tt::tt_metal::IDevice*> devices_;
    // Devices that were initialized
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
};

class UnitMeshCQSingleCardFixture : virtual public DispatchFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
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
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
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
            chip_ids, DEFAULT_L1_SMALL_SIZE, trace_region_size, 1, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            this->devices_.push_back(device);
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
};

class UnitMeshCQSingleCardProgramFixture : virtual public UnitMeshCQSingleCardFixture {};

class CommandQueueSingleCardBufferFixture : public CommandQueueSingleCardFixture {};

class CommandQueueSingleCardTraceFixture : virtual public CommandQueueSingleCardFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices(90000000);
    }
};

class CommandQueueSingleCardProgramFixture : virtual public CommandQueueSingleCardFixture {};

class UnitMeshCQMultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            GTEST_SKIP();
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ < 2) {
            GTEST_SKIP();
        }

        std::vector<chip_id_t> chip_ids;
        for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            chip_ids.push_back(id);
        }

        auto dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            devices_.push_back(device);
        }
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device.reset();
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    size_t num_devices_;
};

// #22835: These Fixtures will be removed once tests are fully migrated, and replaced by UnitMeshCQMultiDeviceFixtures
class CommandQueueMultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            GTEST_SKIP();
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ < 2) {
            GTEST_SKIP();
        }

        std::vector<chip_id_t> chip_ids;
        for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            chip_ids.push_back(id);
        }

        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            chip_ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
    size_t num_devices_;
};

class CommandQueueMultiDeviceProgramFixture : public CommandQueueMultiDeviceFixture {};

class CommandQueueMultiDeviceBufferFixture : public CommandQueueMultiDeviceFixture {};

class DISABLED_CQMultiDeviceOnFabricFixture
    : public UnitMeshCQMultiDeviceFixture,
      public ::testing::WithParamInterface<tt::tt_fabric::FabricConfig> {
private:
    bool original_fd_fabric_en_ = false;
    inline static ARCH arch_ = tt::ARCH::Invalid;
    inline static bool is_galaxy_ = false;

protected:
    // Multiple fabric configs so need to reset the devices for each test
    static void SetUpTestSuite() {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        is_galaxy_ = tt::tt_metal::IsGalaxyCluster();
    }

    static void TearDownTestSuite() {}

    void SetUp() override {
        original_fd_fabric_en_ = tt::tt_metal::MetalContext::instance().rtoptions().get_fd_fabric();
        // Enable Fabric Dispatch
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(true);
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_fabric::SetFabricConfig(GetParam(), tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        UnitMeshCQMultiDeviceFixture::SetUp();

        if (::testing::Test::IsSkipped()) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        }
    }

    void TearDown() override {
        if (::testing::Test::IsSkipped()) {
            return;
        }
        UnitMeshCQMultiDeviceFixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(original_fd_fabric_en_);
    }
};

}  // namespace tt::tt_metal
