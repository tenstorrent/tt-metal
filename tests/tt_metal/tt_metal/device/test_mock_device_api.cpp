// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test suite for programmatic mock device API
// This demonstrates how tt-mlir/tt-xla can enable mock mode without environment variables

#include <gtest/gtest.h>
#include <tt-metalium/experimental/mock_device.hpp>
#include <umd/device/types/arch.hpp>

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// Test fixture that cleans up mock mode after each test
class MockDeviceAPIFixture : public ::testing::Test {
protected:
    void TearDown() override {
        // Reset mock mode after each test
        experimental::disable_mock_mode();
    }
};

// Test: Verify configure_mock_mode registers config correctly
TEST_F(MockDeviceAPIFixture, ConfigureMockModeRegistersConfig) {
    // Initially, mock mode should not be registered
    EXPECT_FALSE(experimental::is_mock_mode_registered());

    // Configure mock mode for Blackhole with 1 chip
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);

    // Verify mock mode is now registered
    EXPECT_TRUE(experimental::is_mock_mode_registered());

    // Verify the config is correct
    auto config = experimental::get_registered_mock_config();
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->arch, tt::ARCH::BLACKHOLE);
    EXPECT_EQ(config->num_chips, 1);
}

// Test: Verify configure_mock_mode works for Wormhole with multiple chips
TEST_F(MockDeviceAPIFixture, ConfigureMockModeWormholeMultiChip) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 8);

    EXPECT_TRUE(experimental::is_mock_mode_registered());

    auto config = experimental::get_registered_mock_config();
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->arch, tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(config->num_chips, 8);
}

// Test: Verify disable_mock_mode clears the registration
TEST_F(MockDeviceAPIFixture, DisableMockModeClearsConfig) {
    // Enable mock mode
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());

    // Disable mock mode
    experimental::disable_mock_mode();

    // Verify it's cleared
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    EXPECT_FALSE(experimental::get_registered_mock_config().has_value());
}

}  // namespace tt::tt_metal
