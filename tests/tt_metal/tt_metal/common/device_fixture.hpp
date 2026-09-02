// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <limits>
#include <algorithm>

namespace tt::tt_metal {

class AnyDispatchMeshDeviceFixture : public MeshDispatchFixture {
private:
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> id_to_device_;

protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        this->DetectDispatchMode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            ids.push_back(id);
        }
        this->create_devices(ids);
        init_max_cbs();
    }

    void TearDown() override {
        // Device not initialized if skipped
        if (!id_to_device_.empty()) {
            for (auto [device_id, device] : id_to_device_) {
                device.reset();
            }
        }
    }

    void create_devices(const std::vector<ChipId>& device_ids) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        // TODO: Some CI machines have lots of cards, running all tests on all the cards is slow.
        // Coverage for multidevices should be decent if we just confirm 2 work.
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            device_ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
    }

    explicit AnyDispatchMeshDeviceFixture(
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) :
        MeshDispatchFixture(l1_small_size, trace_region_size) {}

public:
    std::pair<unsigned, unsigned> worker_grid_minimum_dims() {
        constexpr size_t UMAX = std::numeric_limits<unsigned>::max();
        std::pair<size_t, size_t> min_dims = {UMAX, UMAX};
        for (const auto& device : devices_) {
            auto coords = device->compute_with_storage_grid_size();
            min_dims.first = std::min(min_dims.first, coords.x);
            min_dims.second = std::min(min_dims.second, coords.y);
        }
        return min_dims;
    }
};

class MeshDeviceFixture : public AnyDispatchMeshDeviceFixture {
protected:
    void SetUp() override {
        // Save time. Don't do any setup if invalid dispatch mode
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        AnyDispatchMeshDeviceFixture::SetUp();
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = true;
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }

    explicit MeshDeviceFixture(
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) :
        AnyDispatchMeshDeviceFixture(l1_small_size, trace_region_size) {}
};

class AnyDispatchMeshDeviceSingleCardFixture : public MeshDispatchFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        this->DetectDispatchMode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
        init_max_cbs();
    }

    void TearDown() override {
        if (!id_to_device_.empty()) {
            for (auto [device_id, device] : id_to_device_) {
                device.reset();
            }
        }
    }

    virtual size_t num_command_queues() const { return 1; }

    virtual void create_devices() {
        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            ids.push_back(id);
        }
        create_devices(ids);
    }

    void create_devices(const std::vector<ChipId>& ids) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids, l1_small_size_, trace_region_size_, num_command_queues(), dispatch_core_config);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> id_to_device_;
};

// Same as MeshDeviceSingleCardFixture but remove the check for slow dispatch mode
class MeshDeviceSingleCardFixture : public AnyDispatchMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        AnyDispatchMeshDeviceSingleCardFixture::SetUp();
    }

    virtual bool validate_dispatch_mode() {
        this->slow_dispatch_ = true;
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }
};

class MeshDeviceSingleCardBufferFixture : public MeshDeviceSingleCardFixture {};

// Single unit-mesh fixture: always owns exactly one unit MeshDevice.
class UnitMeshAnyDispatchFixture : public AnyDispatchMeshDeviceSingleCardFixture {
public:
    distributed::MeshDevice& device() { return *devices_.front(); }

protected:
    void create_devices() override {
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        AnyDispatchMeshDeviceSingleCardFixture::create_devices({mmio_device_id});
    }
};

// Single unit-mesh fixture: always owns exactly one unit MeshDevice.
// Requires slow dispatch mode.
class UnitMeshFixture : public MeshDeviceSingleCardFixture {
public:
    distributed::MeshDevice& device() { return *devices_.front(); }

protected:
    void create_devices() override {
        const ChipId mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        AnyDispatchMeshDeviceSingleCardFixture::create_devices({mmio_device_id});
    }
};

class BlackholeSingleCardFixture : public MeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
        this->create_devices();
        init_max_cbs();
    }
};

class QuasarMeshDeviceSingleCardFixture : public UnitMeshFixture {
protected:
    void SetUp() override {
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Not a Quasar device";
        }
        this->create_devices();
        init_max_cbs();
    }
};

class QuasarMultiCQMeshDeviceSingleCardFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    size_t num_command_queues() const override { return 2; }
};

}  // namespace tt::tt_metal
