// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

using namespace tt::tt_metal;

namespace host_tests {

namespace multi_device_tests {
TEST_F(CommandQueueMultiDeviceFixture, DISABLED_TestAccessCommandQueue) {
    for (unsigned int device_id = 0; device_id < num_devices_; device_id++) {
        EXPECT_NO_THROW(devices_[device_id]->command_queue());
    }
}

TEST(FastDispatchHostSuite, TestCannotAccessCommandQueueForClosedDevice) {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
        GTEST_SKIP();
    }
    const unsigned int device_id = 0;
    Device* device = CreateDevice(device_id);
    EXPECT_NO_THROW(device->command_queue());
    CloseDevice(device);
    EXPECT_ANY_THROW(device->command_queue());
}

TEST_F(CommandQueueMultiDeviceFixture, DISABLED_TestDirectedLoopbackToUniqueHugepage) {
    std::unordered_map<chip_id_t, std::vector<uint32_t>> golden_data;

    const uint32_t byte_size = 2048 * 16;
    const uint64_t address = 0;

    for (chip_id_t device_id = 0; device_id < num_devices_; device_id++) {
        std::vector<uint32_t> data =
            tt::test_utils::generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));

        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        tt::Cluster::instance().write_sysmem(data.data(), data.size() * sizeof(uint32_t), address, mmio_device_id, channel);

        golden_data[device_id] = data;
    }

    std::vector<uint32_t> readback_data;
    readback_data.resize(byte_size / sizeof(uint32_t));
    for (chip_id_t device_id = 0; device_id < num_devices_; device_id++) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        tt::Cluster::instance().read_sysmem(readback_data.data(), byte_size, address, mmio_device_id, channel);
        EXPECT_EQ(readback_data, golden_data.at(device_id));
    }
}
}




}   // namespace host_tests
