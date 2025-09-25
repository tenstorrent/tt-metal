// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <memory>
#include <vector>

#include <tt-metalium/allocator_types.hpp>
#include <tt-metalium/device.hpp>
#include "hostdevcommon/common_values.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal.hpp"

namespace tt::tt_metal {

using namespace tt;

void CloseDevicesInPool() {
    auto devices = DevicePool::instance().get_all_active_devices();
    std::map<chip_id_t, IDevice*> chip_id_to_device;
    for (const auto& dev : devices) {
        chip_id_to_device[dev->id()] = dev;
    }
    detail::CloseDevices(chip_id_to_device);
}

TEST(DevicePool, DevicePoolOpenClose) {
    std::vector<chip_id_t> device_ids{*tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().begin()};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto& dispatch_core_config = tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
    auto devices = DevicePool::instance().get_all_active_devices();
    for (const auto& dev : devices) {
        ASSERT_EQ((int)(dev->allocator()->get_config().l1_small_size), l1_small_size);
        ASSERT_EQ((int)(dev->num_hw_cqs()), num_hw_cqs);
        ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices again
    for (const auto& dev : devices) {
        dev->close();
    }
    devices = DevicePool::instance().get_all_active_devices();
    for (const auto& dev : devices) {
        ASSERT_EQ((int)(dev->allocator()->get_config().l1_small_size), l1_small_size);
        ASSERT_EQ((int)(dev->num_hw_cqs()), num_hw_cqs);
        ASSERT_TRUE(dev->is_initialized());
    }
    CloseDevicesInPool();
}

TEST(DevicePool, DevicePoolReconfigDevices) {
    std::vector<chip_id_t> device_ids{*tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().begin()};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    int worker_l1_size = 0;
    const auto& dispatch_core_config = tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
    auto devices = DevicePool::instance().get_all_active_devices();
    for (const auto& dev : devices) {
        auto& config = dev->allocator()->get_config();
        ASSERT_TRUE((int)(dev->allocator()->get_config().l1_small_size) == l1_small_size);
        ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
        EXPECT_NE(config.l1_unreserved_base, config.worker_l1_size);
        ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices with different configs
    CloseDevicesInPool();
    l1_small_size = 2048;
    worker_l1_size = 4096;
    DevicePool::initialize(
        device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config, {}, worker_l1_size);
    devices = DevicePool::instance().get_all_active_devices();
    for (const auto& dev : devices) {
        auto& config = dev->allocator()->get_config();
        ASSERT_TRUE((int)(config.l1_small_size) == l1_small_size);
        EXPECT_EQ(config.worker_l1_size - config.l1_unreserved_base, worker_l1_size);
        ASSERT_TRUE(dev->is_initialized());
    }
    CloseDevicesInPool();
}

TEST(DevicePool, DevicePoolAddDevices) {
    if (tt_metal::GetNumAvailableDevices() != 8) {
        GTEST_SKIP();
    }
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto& dispatch_core_config = tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
    auto devices = DevicePool::instance().get_all_active_devices();
    for (const auto& dev : devices) {
        ASSERT_TRUE((int)(dev->allocator()->get_config().l1_small_size) == l1_small_size);
        ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
        ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get more devices
    CloseDevicesInPool();
    device_ids = {0, 1, 2, 3};
    DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
    devices = DevicePool::instance().get_all_active_devices();
    ASSERT_TRUE(devices.size() >= 4);
    for (const auto& dev : devices) {
        ASSERT_TRUE((int)(dev->allocator()->get_config().l1_small_size) == l1_small_size);
        ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
        ASSERT_TRUE(dev->is_initialized());
    }
    CloseDevicesInPool();
}

TEST(DevicePool, DevicePoolReduceDevices) {
    if (tt_metal::GetNumAvailableDevices() != 8) {
        GTEST_SKIP();
    }
    std::vector<chip_id_t> device_ids{0, 1, 2, 3};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto& dispatch_core_config = tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
    const auto devices = DevicePool::instance().get_all_active_devices();
    for (const auto& dev : devices) {
        ASSERT_TRUE((int)(dev->allocator()->get_config().l1_small_size) == l1_small_size);
        ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
        ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get less devices
    CloseDevicesInPool();
    device_ids = {0};
    DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
    auto dev = DevicePool::instance().get_active_device(0);
    ASSERT_TRUE(dev->id() == 0);
    ASSERT_TRUE((int)(dev->allocator()->get_config().l1_small_size) == l1_small_size);
    ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
    ASSERT_TRUE(dev->is_initialized());
    CloseDevicesInPool();
}

}  // namespace tt::tt_metal
