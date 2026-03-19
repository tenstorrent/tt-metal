// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
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

    void create_devices(
        std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        std::size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<ChipId> chip_ids;
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, trace_region_size, 1, dispatch_core_config, {}, worker_l1_size);
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
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<ChipId> chip_ids;
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
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
        init_max_cbs();
    }
};

using UnitMeshCQSingleCardBufferFixture = UnitMeshCQSingleCardFixture;

// Suite-level shared fixture: creates devices once per test suite instead of per test.
// Tests using this fixture must NOT modify persistent device state (sub-device managers, etc.).
// On test failure, devices are automatically torn down and re-created before the next test.
class UnitMeshCQSingleCardSharedFixture : virtual public MeshDispatchFixture {
protected:
    inline static std::vector<std::shared_ptr<distributed::MeshDevice>> shared_devices_;
    inline static std::map<int, std::shared_ptr<distributed::MeshDevice>> shared_reserved_devices_;
    inline static bool devices_valid_ = false;
    inline static bool needs_recovery_ = false;
    inline static uint32_t shared_max_cbs_ = 0;

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

        if (needs_recovery_ || !devices_valid_) {
            destroy_shared_devices();
            create_shared_devices();
        }
        if (shared_devices_.empty()) {
            GTEST_SKIP() << "No local devices available for testing (all devices are remote-only)";
        }
        devices_ = shared_devices_;
        max_cbs_ = shared_max_cbs_;
    }

    void TearDown() override {
        if (HasFailure()) {
            needs_recovery_ = true;
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
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<ChipId> chip_ids;
        auto* enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        shared_reserved_devices_ = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);

        if (enable_remote_chip) {
            const auto tunnels =
                tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
            for (const auto& tunnel : tunnels) {
                for (const auto chip_id : tunnel) {
                    if (shared_reserved_devices_.contains(chip_id)) {
                        auto& device = shared_reserved_devices_.at(chip_id);
                        if (!device->is_remote_only()) {
                            shared_devices_.push_back(device);
                        }
                    }
                }
                break;
            }
            if (shared_devices_.empty() && shared_reserved_devices_.contains(mmio_device_id)) {
                auto& mmio_device = shared_reserved_devices_.at(mmio_device_id);
                if (!mmio_device->is_remote_only()) {
                    shared_devices_.push_back(mmio_device);
                }
            }
        } else {
            shared_devices_.push_back(shared_reserved_devices_.at(mmio_device_id));
        }

        shared_max_cbs_ = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
        devices_valid_ = true;
        needs_recovery_ = false;
    }

    static void destroy_shared_devices() {
        shared_devices_.clear();
        shared_reserved_devices_.clear();
        devices_valid_ = false;
    }
};

using UnitMeshCQSingleCardSharedBufferFixture = UnitMeshCQSingleCardSharedFixture;

class UnitMeshCQMultiDeviceFixture : public MeshDispatchFixture {
protected:
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

        std::vector<ChipId> chip_ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            chip_ids.push_back(id);
        }

        auto dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            devices_.push_back(device);
        }
        init_max_cbs();
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

}  // namespace tt::tt_metal
