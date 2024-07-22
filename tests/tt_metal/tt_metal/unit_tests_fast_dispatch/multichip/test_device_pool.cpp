// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/unit_tests/common/basic_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

using namespace tt;
using namespace tt::test_utils;

TEST_F(FDBasicFixture, DevicePoolOpenClose) {
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
    std::vector<tt_metal::Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices again
    for (const auto& dev: devices) {
        dev->close();
    }
    devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    for (const auto& dev: devices) {
        dev->close();
    }
}

TEST_F(FDBasicFixture, DevicePoolReconfigDevices) {
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices with different configs
    for (const auto& dev: devices) {
        dev->close();
    }
    l1_small_size = 2048;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
    devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE(dev->is_initialized());
    }
    for (const auto& dev: devices) {
        dev->close();
    }
}

TEST_F(FDBasicFixture, DevicePoolAddDevices) {
    if (tt::tt_metal::GetNumAvailableDevices() != 8) {
        GTEST_SKIP();
    }
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get more devices
    for (const auto& dev: devices) {
        dev->close();
    }
    device_ids = {0, 1, 2, 3};
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
    devices = tt::DevicePool::instance().get_all_active_devices();
    ASSERT_TRUE(devices.size() >= 4);
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    for (const auto& dev: devices) {
        dev->close();
    }
}

TEST_F(FDBasicFixture, DevicePoolReduceDevices) {
    if (tt::tt_metal::GetNumAvailableDevices() != 8) {
        GTEST_SKIP();
    }
    std::vector<chip_id_t> device_ids{0, 1, 2, 3};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get less devices
    for (const auto& dev: devices) {
        dev->close();
    }
    device_ids = {0};
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
    auto dev = tt::DevicePool::instance().get_active_device(0);
    ASSERT_TRUE(dev->id() == 0);
    ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
    ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
    ASSERT_TRUE(dev->is_initialized());
    tt::DevicePool::instance().close_device(0);
}
