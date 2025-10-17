// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/device_pool.hpp>
#include <limits>
#include <algorithm>

namespace tt::tt_metal {

class MeshDeviceFixture : public MeshDispatchFixture {
private:
    std::map<chip_id_t, std::shared_ptr<distributed::MeshDevice>> id_to_device_;

protected:
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
        std::vector<chip_id_t> ids;
        for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            ids.push_back(id);
        }
        this->create_devices(ids);
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
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }

    void create_devices(const std::vector<chip_id_t>& device_ids) {
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
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override {
        if (!id_to_device_.empty()) {
            for (auto [device_id, device] : id_to_device_) {
                device.reset();
            }
        }
    }

    virtual bool validate_dispatch_mode() {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }

    void create_devices() {
        std::vector<chip_id_t> ids;
        for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
        this->num_devices_ = this->devices_.size();
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_{};
    std::map<chip_id_t, std::shared_ptr<distributed::MeshDevice>> id_to_device_;
    size_t num_devices_{};
};

class MeshDeviceSingleCardBufferFixture : public MeshDeviceSingleCardFixture {};

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
    }
};

}  // namespace tt::tt_metal
