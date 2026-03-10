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

class MockDeviceAPIFixture : public ::testing::Test {
protected:
    void TearDown() override { experimental::disable_mock_mode(); }
};

TEST_F(MockDeviceAPIFixture, ConfigureMockModeRegistersConfig) {
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    auto desc = experimental::get_mock_cluster_desc();
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(*desc, "blackhole_P150.yaml");
}

TEST_F(MockDeviceAPIFixture, ConfigureMockModeWormholeMultiChip) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 8);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    auto desc = experimental::get_mock_cluster_desc();
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(*desc, "t3k_cluster_desc.yaml");
}

TEST_F(MockDeviceAPIFixture, DisableMockModeClearsConfig) {
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    experimental::disable_mock_mode();
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    EXPECT_FALSE(experimental::get_mock_cluster_desc().has_value());
}

TEST_F(MockDeviceAPIFixture, WormholeConfigurationsAreValid) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "wormhole_N150.yaml");
    experimental::disable_mock_mode();

    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 2);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "wormhole_N300.yaml");
    experimental::disable_mock_mode();

    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 4);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "2x2_n300_cluster_desc.yaml");
    experimental::disable_mock_mode();

    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 8);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "t3k_cluster_desc.yaml");
    experimental::disable_mock_mode();

    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 32);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "tg_cluster_desc.yaml");
}

TEST_F(MockDeviceAPIFixture, BlackholeConfigurationsAreValid) {
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "blackhole_P150.yaml");
    experimental::disable_mock_mode();

    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 2);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "blackhole_P300_both_mmio.yaml");
}

TEST_F(MockDeviceAPIFixture, UnsupportedConfigurationThrows) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 99);
    EXPECT_THROW(experimental::get_mock_cluster_desc(), std::runtime_error);
}

TEST_F(MockDeviceAPIFixture, ConfigureMockModeFromHwDetectsArchitecture) {
    tt::ARCH detected_arch = get_physical_architecture();
    if (detected_arch == tt::ARCH::Invalid) {
        GTEST_SKIP() << "No TT hardware detected - skipping configure_mock_mode_from_hw test";
    }

    experimental::configure_mock_mode_from_hw();
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    auto desc = experimental::get_mock_cluster_desc();
    ASSERT_TRUE(desc.has_value());
}

TEST_F(MockDeviceAPIFixture, SwitchFromMockToRealHardware) {
    // Test API state transitions: configure, disable, reconfigure
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    EXPECT_TRUE(experimental::get_mock_cluster_desc().has_value());

    experimental::disable_mock_mode();
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    EXPECT_FALSE(experimental::get_mock_cluster_desc().has_value());

    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 2);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    auto desc = experimental::get_mock_cluster_desc();
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(*desc, "wormhole_N300.yaml");
}

TEST_F(MockDeviceAPIFixture, SwitchFromMockToRealHardwareWithDeviceCreation) {
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

}  // namespace tt::tt_metal
