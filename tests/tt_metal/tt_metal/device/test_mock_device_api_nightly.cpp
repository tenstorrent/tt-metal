// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/experimental/mock_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <umd/device/types/arch.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

class MockDeviceAPINightlyFixture : public ::testing::Test {
protected:
    void TearDown() override { experimental::disable_mock_mode(); }
};

TEST_F(MockDeviceAPINightlyFixture, NIGHTLY_SwitchFromMockToRealHardwareWithDeviceCreation) {
    // Comprehensive test: verify disable_mock_mode() properly reinitializes MetalContext
    // and cleans up device-specific data structures when switching from mock to real hardware
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());

    // Create and close a device to populate device-specific maps
    {
        auto device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
        device->close();
        device.reset();
    }

    // Disable mock mode - should reinitialize MetalContext for real hardware
    // This clears device-specific maps (dram_bank_offset_map_, l1_bank_offset_map_, etc.)
    // and reinitializes base objects (cluster_, hal_) with real hardware
    experimental::disable_mock_mode();
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    EXPECT_EQ(MetalContext::instance().get_cluster().get_target_device_type(), tt::TargetDevice::Silicon);

    // Create real hardware device - verifies full reinitialization worked correctly
    // including cleanup and reinitialization of device-specific data structures
    {
        auto real_device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
        EXPECT_EQ(MetalContext::instance().get_cluster().get_target_device_type(), tt::TargetDevice::Silicon);
        real_device->close();
        real_device.reset();
    }
}

TEST_F(MockDeviceAPINightlyFixture, NIGHTLY_SwitchFromRealToMockHardware) {
    // Skip if no real hardware available
    tt::ARCH detected_arch = get_physical_architecture();
    if (detected_arch == tt::ARCH::Invalid) {
        GTEST_SKIP() << "No TT hardware detected - skipping real-to-mock transition test";
    }

    // Phase 1: Create real hardware device
    {
        auto real_device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
        ASSERT_NE(real_device, nullptr);
        EXPECT_EQ(MetalContext::instance().get_cluster().get_target_device_type(), tt::TargetDevice::Silicon);
        real_device->close();
        real_device.reset();
    }

    // Phase 2: Switch to mock mode
    experimental::configure_mock_mode(detected_arch, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    EXPECT_EQ(MetalContext::instance().get_cluster().get_target_device_type(), tt::TargetDevice::Mock);

    // Phase 3: Create mock device - verifies full reinitialization worked
    {
        auto mock_device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
        ASSERT_NE(mock_device, nullptr);
        EXPECT_EQ(MetalContext::instance().get_cluster().get_target_device_type(), tt::TargetDevice::Mock);
        mock_device->close();
        mock_device.reset();
    }

    // Phase 4: Switch back to real hardware
    experimental::disable_mock_mode();
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    EXPECT_EQ(MetalContext::instance().get_cluster().get_target_device_type(), tt::TargetDevice::Silicon);

    // Phase 5: Verify real hardware works again
    {
        auto real_device2 =
            distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
        ASSERT_NE(real_device2, nullptr);
        EXPECT_EQ(MetalContext::instance().get_cluster().get_target_device_type(), tt::TargetDevice::Silicon);
        real_device2->close();
        real_device2.reset();
    }
}

}  // namespace tt::tt_metal
