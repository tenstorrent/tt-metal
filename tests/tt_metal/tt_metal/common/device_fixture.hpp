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
    // Static members shared across all tests in the suite
    static tt::ARCH arch_;
    static size_t num_devices_;
    static std::vector<tt::tt_metal::IDevice*> devices_;
    static std::map<chip_id_t, tt::tt_metal::IDevice*> device_map;

    // Suite-level setup: Initializes devices once for the entire test suite
    static void SetUpTestSuite() {
        validate_dispatch_mode();
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::GRAYSKULL && num_devices_ > 2) {
            num_devices_ = 2; // Limit to 2 devices for GRAYSKULL architecture
        }
        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            ids.push_back(id);
        }
        create_devices(ids);
    }

    // Suite-level cleanup: Closes all devices after the test suite completes
    static void TearDownTestSuite() {
        tt::tt_metal::detail::CloseDevices(device_map);
    }

    // Validates dispatch mode; skips the test if not in slow dispatch mode
    static void validate_dispatch_mode() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
    }

    // Creates devices and populates the static device map
    static void create_devices(const std::vector<chip_id_t>& device_ids) {
        const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
        tt::DevicePool::initialize(device_ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
        devices_ = tt::DevicePool::instance().get_all_active_devices();
        for (auto device : devices_) {
            device_map[device->id()] = device;
        }
    }

    // Constructor: No instance-specific setup, as it's handled at suite level
    DeviceFixture() : DispatchFixture(DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE) {}

public:
    // Utility method to compute minimum worker grid dimensions across devices
    std::pair<unsigned, unsigned> worker_grid_minimum_dims(void) {
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

// Static member definitions
tt::ARCH DeviceFixture::arch_;
size_t DeviceFixture::num_devices_;
std::vector<tt::tt_metal::IDevice*> DeviceFixture::devices_;
std::map<chip_id_t, tt::tt_metal::IDevice*> DeviceFixture::device_map;

// Derived fixture with custom L1 small size
class DeviceFixtureWithL1Small : public DeviceFixture {
protected:
    static void SetUpTestSuite() {
        validate_dispatch_mode();
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::GRAYSKULL && num_devices_ > 2) {
            num_devices_ = 2;
        }
        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            ids.push_back(id);
        }
        const size_t l1_small_size = 24 * 1024; // Custom L1 small size
        const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
        tt::DevicePool::initialize(ids, 1, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
        devices_ = tt::DevicePool::instance().get_all_active_devices();
        for (auto device : devices_) {
            device_map[device->id()] = device;
        }
    }

    static void TearDownTestSuite() {
        tt::tt_metal::detail::CloseDevices(device_map);
    }
};

// Fixture for single-card setups
class DeviceSingleCardFixture : public DispatchFixture {
protected:
    static tt::tt_metal::IDevice* device_;
    static std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
    static size_t num_devices_;

    static void SetUpTestSuite() {
        validate_dispatch_mode();
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        const chip_id_t mmio_device_id = 0;
        reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id});
        device_ = reserved_devices_.at(mmio_device_id);
        num_devices_ = reserved_devices_.size();
    }

    static void TearDownTestSuite() {
        tt::tt_metal::detail::CloseDevices(reserved_devices_);
    }

    static void validate_dispatch_mode() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
    }
};

// Static member definitions for DeviceSingleCardFixture
tt::tt_metal::IDevice* DeviceSingleCardFixture::device_;
std::map<chip_id_t, tt::tt_metal::IDevice*> DeviceSingleCardFixture::reserved_devices_;
size_t DeviceSingleCardFixture::num_devices_;

// Additional derived fixtures
class DeviceSingleCardBufferFixture : public DeviceSingleCardFixture {};

class BlackholeSingleCardFixture : public DeviceSingleCardFixture {
protected:
    static void SetUpTestSuite() {
        validate_dispatch_mode();
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP(); // Skip if not BLACKHOLE architecture
        }
        const chip_id_t mmio_device_id = 0;
        reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id});
        device_ = reserved_devices_.at(mmio_device_id);
        num_devices_ = reserved_devices_.size();
    }
};

class DeviceSingleCardFastSlowDispatchFixture : public DeviceSingleCardFixture {
protected:
    static void validate_dispatch_mode() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        // No skip; allow both fast and slow dispatch
    }
};

}  // namespace tt::tt_metal
