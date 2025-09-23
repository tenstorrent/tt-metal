// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include "impl/dispatch/command_queue_common.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <tt-metalium/utils.hpp>

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

class SimulatorFixture : public MeshDeviceFixture {
protected:
    void SetUp() override {
        // Check if simulator mode is enabled
        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled()) {
            GTEST_SKIP()
                << "Simulator mode not enabled. Set TT_METAL_SIMULATOR environment variable to run simulator tests.";
        }

        // Call parent SetUp to initialize devices
        MeshDeviceFixture::SetUp();
    }
};

TEST_F(SimulatorFixture, SimulatorDeviceInitialization) {
    // Verify that all devices are properly initialized in simulator mode
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = devices_.at(id);

        // Check that device is valid
        EXPECT_NE(mesh_device, nullptr);

        // Verify device is accessible
        EXPECT_NO_THROW({});

        // Test that we can access the allocator
        EXPECT_NE(mesh_device->allocator(), nullptr);

        // Verify we can get base addresses
        EXPECT_GT(mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1), 0);
        EXPECT_GT(mesh_device->allocator()->get_base_allocator_addr(HalMemType::DRAM), 0);
    }
}
}  // namespace tt::tt_metal
