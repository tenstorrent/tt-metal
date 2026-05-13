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

class MeshDeviceFixture : public MeshDispatchFixture {
private:
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> id_to_device_;

protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        // Save time. Don't do any setup if invalid dispatch mode
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        // Some CI machines have lots of cards, running all tests on all cards is slow
        // Coverage for multidevices is decent if we just confirm 2 work
        this->num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ > 2) {
            this->num_devices_ = 2;
        }
        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            ids.push_back(id);
        }
        this->create_devices(ids);
        this->max_cbs_ = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
    }

    void TearDown() override {
        // Device not initialized if skipped
        if (!id_to_device_.empty()) {
            for (auto [device_id, device] : id_to_device_) {
                device.reset();
            }
        }
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

    void create_devices(const std::vector<ChipId>& device_ids) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            device_ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
        this->num_devices_ = this->devices_.size();
    }

    explicit MeshDeviceFixture(
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) :
        MeshDispatchFixture(l1_small_size, trace_region_size) {}

    size_t num_devices_{};

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

class MeshDeviceSingleCardFixture : public MeshDispatchFixture {
protected:
    static UnitMeshDeviceConfig get_unit_mesh_config() {
        UnitMeshDeviceConfig config;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            config.chip_ids.push_back(id);
        }
        return config;
    }

    static void SetUpTestSuite() {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            return;
        }
        MeshDispatchFixture::create_shared_devices(get_shared_devices(), get_unit_mesh_config());
    }

    static void TearDownTestSuite() { MeshDispatchFixture::destroy_shared_devices(get_shared_devices()); }

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        auto& shared_devices = get_shared_devices();
        if (shared_devices.needs_recovery){
            MeshDispatchFixture::destroy_shared_devices(shared_devices);
        }
        if(!shared_devices.initialized) {
            MeshDispatchFixture::create_shared_devices(shared_devices, get_unit_mesh_config());
        }
        this->devices_ = shared_devices.devices;

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->max_cbs_ = shared_devices.max_cbs;
        this->num_devices_ = devices_.size();
    }

    void TearDown() override {
        devices_.clear();
        if (HasFailure()) {
            get_shared_devices().needs_recovery = true;
        }
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

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    size_t num_devices_{};
};

class MeshDeviceSingleCardBufferFixture : public MeshDeviceSingleCardFixture {};

class BlackholeSingleCardFixture : public MeshDeviceSingleCardFixture {
protected:
    static void SetUpTestSuite() {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) {
            return;
        }
        MeshDeviceSingleCardFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() { MeshDeviceSingleCardFixture::TearDownTestSuite(); }

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
        MeshDeviceSingleCardFixture::SetUp();
    }
};

class QuasarMeshDeviceSingleCardFixture : public MeshDeviceSingleCardFixture {
protected:
    static void SetUpTestSuite() {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::QUASAR) {
            return;
        }
        MeshDeviceSingleCardFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() { MeshDeviceSingleCardFixture::TearDownTestSuite(); }

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Not a Quasar device";
        }
        MeshDeviceSingleCardFixture::SetUp();
    }
};

}  // namespace tt::tt_metal
