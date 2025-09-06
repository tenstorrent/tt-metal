// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_types.hpp"
#include "gtest/gtest.h"
#include "mesh_dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

class UnitMeshMultiCQSingleDeviceFixture : public MeshDispatchFixture {
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
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");

        // Check to deal with TG systems
        const chip_id_t device_id =
            (enable_remote_chip or tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster())
                ? *tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids().begin()
                : *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        this->create_device(device_id, DEFAULT_TRACE_REGION_SIZE);
    }

    void TearDown() override { device_.reset(); }

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

    void create_device(const chip_id_t device_id, const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        std::vector<chip_id_t> chip_id = {device_id};

        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_id, DEFAULT_L1_SMALL_SIZE, trace_region_size, 2, dispatch_core_config);
        this->device_ = reserved_devices[device_id];
    }

    std::shared_ptr<distributed::MeshDevice> device_;
    tt::ARCH arch_{tt::ARCH::Invalid};
    uint8_t num_cqs_{};
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshMultiCQSingleDeviceProgramFixture : public UnitMeshMultiCQSingleDeviceFixture {};

class UnitMeshMultiCQSingleDeviceBufferFixture : public UnitMeshMultiCQSingleDeviceFixture {};

class UnitMeshMultiCQSingleDeviceEventFixture : public UnitMeshMultiCQSingleDeviceFixture {};

class UnitMeshMultiCQSingleDeviceTraceFixture : public UnitMeshMultiCQSingleDeviceFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevice(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        this->create_device(0 /* device_id */, trace_region_size);
    }
};
class UnitMeshMultiCQMultiDeviceFixture : public MeshDispatchFixture {
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

        // Check to deal with TG systems
        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB or
            tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
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
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshMultiCQMultiDeviceBufferFixture : public UnitMeshMultiCQMultiDeviceFixture {};

class UnitMeshMultiCQMultiDeviceEventFixture : public UnitMeshMultiCQMultiDeviceFixture {};

class DISABLED_UnitMeshMultiCQMultiDeviceOnFabricFixture
    : public UnitMeshMultiCQMultiDeviceFixture,
      public ::testing::WithParamInterface<tt::tt_fabric::FabricConfig> {
private:
    // Save the result to reduce UMD calls
    inline static bool should_skip_ = false;

protected:
    void SetUp() override {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Dispatch on Fabric tests only applicable on Wormhole B0";
        }
        // Skip for TG as it's still being implemented
        if (tt::tt_metal::IsGalaxyCluster()) {
            GTEST_SKIP();
        }
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_fabric::SetFabricConfig(GetParam(), tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        UnitMeshMultiCQMultiDeviceFixture::SetUp();
    }

    void TearDown() override {
        UnitMeshMultiCQMultiDeviceFixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

}  // namespace tt::tt_metal
