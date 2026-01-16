// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/experimental/mock_device.hpp>
#include <umd/device/types/arch.hpp>

#include "llrt/get_platform_architecture.hpp"

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

}  // namespace tt::tt_metal
