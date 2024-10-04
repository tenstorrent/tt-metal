// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/unit_tests/common/basic_fixture.hpp"
#include "tt_metal/host_api.hpp"
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
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
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
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
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
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
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
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
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
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
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
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
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
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    auto dev = tt::DevicePool::instance().get_active_device(0);
    ASSERT_TRUE(dev->id() == 0);
    ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
    ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
    ASSERT_TRUE(dev->is_initialized());
    tt::DevicePool::instance().close_device(0);
}

TEST_F(FDBasicFixture, DevicePoolShutdownSubmesh) {
    if (tt::tt_metal::GetNumAvailableDevices() != 32) {
        GTEST_SKIP();
    }
    chip_id_t mmio_device_id = 0;
    std::vector<chip_id_t> device_ids{mmio_device_id};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    std::vector<Device*> tunnel_0;
    std::vector<Device*> tunnel_1;
    auto mmio_dev_handle = tt::DevicePool::instance().get_active_device(mmio_device_id);
    auto tunnels_from_mmio = mmio_dev_handle->tunnels_from_mmio_;
    //iterate over all tunnels origination from this mmio device
    for (uint32_t ts = tunnels_from_mmio[0].size() - 1; ts > 0; ts--) {
        tunnel_0.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[0][ts]));
    }
    for (uint32_t ts = tunnels_from_mmio[1].size() - 1; ts > 0; ts--) {
        tunnel_1.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[1][ts]));
    }

    tt::DevicePool::instance().close_devices(tunnel_0);
    for (const auto& dev: tunnel_1) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    tt::DevicePool::instance().close_devices(tunnel_1);
}

TEST_F(FDBasicFixture, DevicePoolReopenSubmesh) {
     GTEST_SKIP();

    chip_id_t mmio_device_id = 0;
    std::vector<chip_id_t> device_ids{mmio_device_id};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    std::vector<Device*> tunnel_0;
    std::vector<Device*> tunnel_1;
    auto mmio_dev_handle = tt::DevicePool::instance().get_active_device(mmio_device_id);
    auto tunnels_from_mmio = mmio_dev_handle->tunnels_from_mmio_;
    //iterate over all tunnels origination from this mmio device
    for (uint32_t ts = tunnels_from_mmio[0].size() - 1; ts > 0; ts--) {
        tunnel_0.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[0][ts]));
    }
    for (uint32_t ts = tunnels_from_mmio[1].size() - 1; ts > 0; ts--) {
        tunnel_1.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[1][ts]));
    }

    tt::DevicePool::instance().close_devices(tunnel_0);
    for (const auto& dev: tunnel_1) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    tt::DevicePool::instance().close_devices(tunnel_1);
    tt::DevicePool::instance().close_devices(tunnel_0);
}
