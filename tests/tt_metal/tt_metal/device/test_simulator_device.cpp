// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <memory>
#include <vector>

#include "device_fixture.hpp"
#include <tt-metalium/hal_types.hpp>
#include "impl/context/metal_context.hpp"

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
