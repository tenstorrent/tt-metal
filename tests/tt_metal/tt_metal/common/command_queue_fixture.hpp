// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.hpp>
#include <cstdint>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "gtest/gtest.h"
#include "mesh_dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/device.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"

namespace tt::tt_metal {
// #22835: These Fixtures will be removed once tests are fully migrated, and replaced by UnitMeshCQFixtures
class UnitMeshCQFixture : public MeshDispatchFixture {
protected:
    static UnitMeshDeviceConfig get_unit_mesh_config(
        std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        std::size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE) {
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        UnitMeshDeviceConfig config = enable_remote_chip ||
                tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB
            ? get_default_unit_mesh_config()
            : UnitMeshDeviceConfig{.chip_ids = {mmio_device_id}};
        config.trace_region_size = trace_region_size;
        config.worker_l1_size = worker_l1_size;
        return config;
    }

    static void SetUpTestSuite() {
        if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
            return;
        }
        MeshDispatchFixture::create_shared_devices(get_shared_devices(), get_unit_mesh_config());
    }

    static void TearDownTestSuite() { MeshDispatchFixture::destroy_shared_devices(get_shared_devices()); }

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
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
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->max_cbs_ = shared_devices.max_cbs;
    }

    void TearDown() override {
        devices_.clear();
        if (HasFailure()) {
            get_shared_devices().needs_recovery = true;
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }
// TODO i dont like this still existing
    void create_devices(
        std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        std::size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE) {
        SharedMeshDeviceState devices;
        MeshDispatchFixture::create_shared_devices(devices, get_unit_mesh_config(trace_region_size, worker_l1_size));
        for (const auto& device : devices.devices) {
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
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        init_max_cbs();
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
        if (devices_.empty()) {
            GTEST_SKIP() << "No local devices available for testing (all devices are remote-only)";
        }
        init_max_cbs();
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device.reset();
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }

    void create_devices(std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        UnitMeshDeviceConfig config = enable_remote_chip ||
                tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB
            ? get_default_unit_mesh_config()
            : UnitMeshDeviceConfig{.chip_ids = {mmio_device_id}};
        config.trace_region_size = trace_region_size;
        SharedMeshDeviceState devices;
        MeshDispatchFixture::create_shared_devices(devices, config);
        reserved_devices_ = devices.id_to_device;

        if (enable_remote_chip) {
            const auto tunnels =
                tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
            for (const auto& tunnel : tunnels) {
                for (const auto chip_id : tunnel) {
                    if (reserved_devices_.contains(chip_id)) {
                        auto& device = reserved_devices_.at(chip_id);
                        // Only add devices that have local resources (skip remote-only devices)
                        if (!device->is_remote_only()) {
                            devices_.push_back(device);
                        }
                    }
                }
                break;
            }
            // In multi-rank Galaxy the MMIO chip's tunnel peers are on remote hosts,
            // so the loop above produces no local devices. Fall back to the always-local
            // MMIO device so tests continue to run on this rank.
            if (devices_.empty() && reserved_devices_.contains(mmio_device_id)) {
                auto& mmio_device = reserved_devices_.at(mmio_device_id);
                if (!mmio_device->is_remote_only()) {
                    devices_.push_back(mmio_device);
                }
            }
        } else {
            auto& mmio_device = reserved_devices_.at(mmio_device_id);
            if (!mmio_device->is_remote_only()) {
                devices_.push_back(mmio_device);
            }
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
        init_max_cbs();
    }
};

using UnitMeshCQSingleCardBufferFixture = UnitMeshCQSingleCardFixture;

// Suite-level shared fixture: creates devices once per test suite instead of per test.
// Tests using this fixture must NOT modify persistent device state (sub-device managers, etc.).
// On test failure, devices are automatically torn down and re-created before the next test.
class UnitMeshCQSingleCardSharedFixture : virtual public MeshDispatchFixture {
protected:
    static void SetUpTestSuite() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            return;
        }
        create_shared_devices();
    }

    static void TearDownTestSuite() { destroy_shared_devices(); }

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        auto& shared_devices = get_shared_devices();
        if (shared_devices.needs_recovery || !shared_devices.initialized) {
            destroy_shared_devices();
            create_shared_devices();
        }
        if (shared_devices.devices.empty()) {
            GTEST_SKIP() << "No local devices available for testing (all devices are remote-only)";
        }
        devices_ = shared_devices.devices;
        max_cbs_ = shared_devices.max_cbs;
    }

    void TearDown() override {
        if (HasFailure()) {
            get_shared_devices().needs_recovery = true;
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);

private:
    bool validate_dispatch_mode() {
        slow_dispatch_ = false;
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            return false;
        }
        return true;
    }

    static void create_shared_devices() {
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        UnitMeshDeviceConfig config = enable_remote_chip ||
                tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB
            ? get_default_unit_mesh_config()
            : UnitMeshDeviceConfig{.chip_ids = {mmio_device_id}};

        auto& shared_devices = get_shared_devices();
        MeshDispatchFixture::create_shared_devices(shared_devices, config);
        shared_devices.devices.clear();

        if (enable_remote_chip) {
            const auto tunnels =
                tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
            for (const auto& tunnel : tunnels) {
                for (const auto chip_id : tunnel) {
                    if (shared_devices.id_to_device.contains(chip_id)) {
                        auto& device = shared_devices.id_to_device.at(chip_id);
                        if (!device->is_remote_only()) {
                            shared_devices.devices.push_back(device);
                        }
                    }
                }
                break;
            }
            if (shared_devices.devices.empty() && shared_devices.id_to_device.contains(mmio_device_id)) {
                auto& mmio_device = shared_devices.id_to_device.at(mmio_device_id);
                if (!mmio_device->is_remote_only()) {
                    shared_devices.devices.push_back(mmio_device);
                }
            }
        } else {
            auto& mmio_device = shared_devices.id_to_device.at(mmio_device_id);
            if (!mmio_device->is_remote_only()) {
                shared_devices.devices.push_back(mmio_device);
            }
        }
    }

    static void destroy_shared_devices() { MeshDispatchFixture::destroy_shared_devices(get_shared_devices()); }
};

using UnitMeshCQSingleCardSharedBufferFixture = UnitMeshCQSingleCardSharedFixture;

class UnitMeshCQMultiDeviceFixture : public MeshDispatchFixture {
protected:
    static UnitMeshDeviceConfig get_unit_mesh_config() {
        UnitMeshDeviceConfig config;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            config.chip_ids.push_back(id);
        }
        return config;
    }

    static void SetUpTestSuite() {
        if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr || tt::tt_metal::GetNumAvailableDevices() < 2) {
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

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ < 2) {
            GTEST_SKIP();
        }

        auto& shared_devices = get_shared_devices();
        if (shared_devices.needs_recovery || !shared_devices.initialized) {
            MeshDispatchFixture::destroy_shared_devices(shared_devices);
            MeshDispatchFixture::create_shared_devices(shared_devices, get_unit_mesh_config());
        }
        devices_ = shared_devices.devices;
        max_cbs_ = shared_devices.max_cbs;
    }

    void TearDown() override {
        devices_.clear();
        if (HasFailure()) {
            get_shared_devices().needs_recovery = true;
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    size_t num_devices_{};
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshCQMultiDeviceProgramFixture : public UnitMeshCQMultiDeviceFixture {};

class UnitMeshCQMultiDeviceBufferFixture : public UnitMeshCQMultiDeviceFixture {};

}  // namespace tt::tt_metal
