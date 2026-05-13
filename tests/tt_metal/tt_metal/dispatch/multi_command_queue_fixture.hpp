// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "gtest/gtest.h"
#include "mesh_dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

class UnitMeshMultiCQSingleDeviceFixture : public MeshDispatchFixture {
protected:
    static UnitMeshDeviceConfig get_unit_mesh_config() {
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        const auto default_config = get_default_unit_mesh_config();
        const ChipId device_id = (enable_remote_chip or tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster())
                ? default_config.chip_ids.at(0)
                : *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        return UnitMeshDeviceConfig{
            .chip_ids = {device_id},
            .num_hw_cqs = 2,
        };
    }

    static void SetUpTestSuite() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch || tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs() != 2) {
            return;
        }
        MeshDispatchFixture::create_shared_devices(get_shared_devices(), get_unit_mesh_config());
    }

    static void TearDownTestSuite() { MeshDispatchFixture::destroy_shared_devices(get_shared_devices()); }

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (this->num_cqs_ != 2) {
            log_info(tt::LogTest, "This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }

        auto& shared_devices = get_shared_devices();
        if (shared_devices.needs_recovery) {
            MeshDispatchFixture::destroy_shared_devices(shared_devices);
        }
        if(!shared_devices.initialized) {
            MeshDispatchFixture::create_shared_devices(shared_devices, get_unit_mesh_config());
        }
        this->device_ = shared_devices.devices.at(0);
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->max_cbs_ = shared_devices.max_cbs;
    }

    void TearDown() override {
        device_.reset();
        if (HasFailure()) {
            get_shared_devices().needs_recovery = true;
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
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

    void create_device(const ChipId device_id, const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        SharedMeshDeviceState devices;
        MeshDispatchFixture::create_shared_devices(
            devices,
            UnitMeshDeviceConfig{
                .chip_ids = {device_id},
                .trace_region_size = trace_region_size,
                .num_hw_cqs = 2,
            });
        this->device_ = devices.id_to_device.at(device_id);
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
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        init_max_cbs();
    }

    void CreateDevice(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        this->create_device(0 /* device_id */, trace_region_size);
    }
};
class UnitMeshMultiCQMultiDeviceFixture : public MeshDispatchFixture {
protected:
    static UnitMeshDeviceConfig get_unit_mesh_config() {
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        UnitMeshDeviceConfig config = enable_remote_chip ||
                tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB ||
                tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()
            ? get_default_unit_mesh_config()
            : UnitMeshDeviceConfig{.chip_ids = {mmio_device_id}};
        config.num_hw_cqs = 2;

        return config;
    }

    static void SetUpTestSuite() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch || tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs() != 2) {
            return;
        }
        MeshDispatchFixture::create_shared_devices(get_shared_devices(), get_unit_mesh_config());
    }

    static void TearDownTestSuite() { MeshDispatchFixture::destroy_shared_devices(get_shared_devices()); }

    void SetUp() override {
        this->slow_dispatch_ = false;
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
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

        auto& shared_devices = get_shared_devices();
        if (shared_devices.needs_recovery) {
            MeshDispatchFixture::destroy_shared_devices(shared_devices);
        }
        if(!shared_devices.initialized) {
            MeshDispatchFixture::create_shared_devices(shared_devices, get_unit_mesh_config());
        }
        this->devices_ = shared_devices.devices;
        this->max_cbs_ = shared_devices.max_cbs;
    }

    void TearDown() override {
        devices_.clear();
        if (HasFailure()) {
            get_shared_devices().needs_recovery = true;
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshMultiCQMultiDeviceBufferFixture : public UnitMeshMultiCQMultiDeviceFixture {};

class UnitMeshMultiCQMultiDeviceEventFixture : public UnitMeshMultiCQMultiDeviceFixture {};

}  // namespace tt::tt_metal
