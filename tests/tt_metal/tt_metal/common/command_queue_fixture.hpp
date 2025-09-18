// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.hpp>
#include <cstdint>
#include "fabric_types.hpp"
#include "gtest/gtest.h"
#include "mesh_dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/device.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
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
class UnitMeshCQFixture : public MeshDispatchFixture {
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

    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
};

class UnitMeshCQEventFixture : public UnitMeshCQFixture {};

class UnitMeshCQProgramFixture : public UnitMeshCQFixture {};

class UnitMeshCQTraceFixture : public UnitMeshCQFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevices(const size_t trace_region_size) { this->create_devices(trace_region_size); }
};

class UnitMeshCQSingleCardFixture : virtual public MeshDispatchFixture {
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
        reserved_devices_ = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, trace_region_size, 1, dispatch_core_config);

        if (enable_remote_chip) {
            const auto tunnels =
                tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
            for (const auto& tunnel : tunnels) {
                for (const auto chip_id : tunnel) {
                    if (reserved_devices_.find(chip_id) != reserved_devices_.end()) {
                        devices_.push_back(reserved_devices_.at(chip_id));
                    }
                }
                break;
            }
        } else {
            devices_.push_back(reserved_devices_.at(mmio_device_id));
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    std::map<int, std::shared_ptr<distributed::MeshDevice>> reserved_devices_;
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshCQSingleCardProgramFixture : virtual public UnitMeshCQSingleCardFixture {};

class UnitMeshCQSingleCardTraceFixture : virtual public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices(90000000);
    }
};

using UnitMeshCQSingleCardBufferFixture = UnitMeshCQSingleCardFixture;

class UnitMeshCQMultiDeviceFixture : public MeshDispatchFixture {
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
    size_t num_devices_{};
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshCQMultiDeviceProgramFixture : public UnitMeshCQMultiDeviceFixture {};

class UnitMeshCQMultiDeviceBufferFixture : public UnitMeshCQMultiDeviceFixture {};

class DISABLED_CQMultiDeviceOnFabricFixture
    : public UnitMeshCQMultiDeviceFixture,
      public ::testing::WithParamInterface<tt::tt_fabric::FabricConfig> {
private:
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
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_fabric::SetFabricConfig(GetParam(), tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        UnitMeshCQMultiDeviceFixture::SetUp();
    }

    void TearDown() override {
        if (::testing::Test::IsSkipped()) {
            return;
        }
        UnitMeshCQMultiDeviceFixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

}  // namespace tt::tt_metal
