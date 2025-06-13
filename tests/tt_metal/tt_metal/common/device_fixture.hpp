// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "dispatch_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/device_pool.hpp>
#include <limits>
#include <algorithm>

namespace tt::tt_metal {

class DeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        // Some CI machines have lots of cards, running all tests on all cards is slow
        // Coverage for multidevices is decent if we just confirm 2 work
        this->num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::GRAYSKULL && num_devices_ > 2) {
            this->num_devices_ = 2;
        }

        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            ids.push_back(id);
        }
        this->create_devices(ids);
    }

    void validate_dispatch_mode() {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }
    }

    void create_devices(const std::vector<chip_id_t>& device_ids) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        tt::DevicePool::initialize(device_ids, 1, l1_small_size_, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
        this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        this->num_devices_ = this->devices_.size();
    }

    DeviceFixture(size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) :
        DispatchFixture(l1_small_size, trace_region_size) {}

    size_t num_devices_;

public:
    std::pair<unsigned, unsigned> worker_grid_minimum_dims() {
        constexpr size_t UMAX = std::numeric_limits<unsigned>::max();
        std::pair<size_t, size_t> min_dims = {UMAX, UMAX};
        for (auto device : devices_) {
            auto coords = device->compute_with_storage_grid_size();
            min_dims.first = std::min(min_dims.first, coords.x);
            min_dims.second = std::min(min_dims.second, coords.y);
        }

        return min_dims;
    }
};

class DeviceFixtureWithL1Small : public DeviceFixture {
public:
    DeviceFixtureWithL1Small() : DeviceFixture(24 * 1024) {}
};

class DeviceSingleCardFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    virtual void validate_dispatch_mode() {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }
    }

    void create_devices() {
        const chip_id_t mmio_device_id = 0;
        this->reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id});
        this->device_ = this->reserved_devices_.at(mmio_device_id);
        this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        this->num_devices_ = this->reserved_devices_.size();
    }

    tt::tt_metal::IDevice* device_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
    size_t num_devices_;
};

class DeviceSingleCardBufferFixture : public DeviceSingleCardFixture {};

class BlackholeSingleCardFixture : public DeviceSingleCardFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
        this->create_devices();
    }
};

class DeviceSingleCardFastSlowDispatchFixture : public DeviceSingleCardFixture {
   protected:
    void validate_dispatch_mode() override {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            this->slow_dispatch_ = false;
        }
    }
};

}  // namespace tt::tt_metal
