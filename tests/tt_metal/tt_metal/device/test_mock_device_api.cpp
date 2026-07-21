// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/types/arch.hpp>

#include <cstdlib>
#include <optional>
#include <string>

#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_state.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
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

    // Note: 32-chip TG configuration removed as tg_cluster_desc.yaml doesn't exist in UMD
}

TEST_F(MockDeviceAPIFixture, BlackholeConfigurationsAreValid) {
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "blackhole_P150.yaml");
    experimental::disable_mock_mode();

    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 2);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "blackhole_P300_both_mmio.yaml");
}

TEST_F(MockDeviceAPIFixture, QuasarConfigurationsAreValid) {
    experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "quasar_Q1.yaml");
}

TEST_F(MockDeviceAPIFixture, UnsupportedConfigurationThrows) {
    bool threw_during_configure = false;
    try {
        experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 99);
    } catch (const std::runtime_error&) {
        threw_during_configure = true;
    }

    if (!threw_during_configure) {
        EXPECT_THROW(experimental::get_mock_cluster_desc(), std::runtime_error);
    }
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

class MockDeviceProfilerFixture : public ::testing::Test {
protected:
    void SetUp() override {
#if !defined(TRACY_ENABLE)
        GTEST_SKIP() << "Requires a Tracy-enabled build (ENABLE_TRACY=ON).";
#endif
        // The profiler-enabled flag is parsed from the environment when RunTimeOptions is
        // constructed (i.e. when the MetalContext is first created). Set it now and drop any
        // pre-existing context so the mock context created by the test picks the flag up.
        const char* prev = getenv("TT_METAL_DEVICE_PROFILER");
        prev_device_profiler_ = prev != nullptr ? std::optional<std::string>(prev) : std::nullopt;
        setenv("TT_METAL_DEVICE_PROFILER", "1", /*overwrite=*/1);
        if (MetalContext::instance_exists()) {
            detail::ReleaseOwnership();
        }
    }

    void TearDown() override {
#if !defined(TRACY_ENABLE)
        return;
#endif
        experimental::disable_mock_mode();
        // Restore the flag to whatever it was before the test rather than clobbering a value the
        // surrounding environment may have set.
        if (prev_device_profiler_.has_value()) {
            setenv("TT_METAL_DEVICE_PROFILER", prev_device_profiler_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv("TT_METAL_DEVICE_PROFILER");
        }
        // Drop the profiler-enabled context so later tests start from a clean state.
        if (MetalContext::instance_exists()) {
            detail::ReleaseOwnership();
        }
    }

    std::optional<std::string> prev_device_profiler_;
};

// Verify that the device profiler is not enabled on mock device.
TEST_F(MockDeviceProfilerFixture, DeviceProfilerIsNotStartedOnMockDevice) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);

    ASSERT_TRUE(MetalContext::instance().rtoptions().get_profiler_enabled())
        << "Test expects device profiler option to be enabled.";
    ASSERT_TRUE(MetalContext::instance().get_cluster().is_mock_or_emulated()) << "Test should run on mock device.";

    // Even though profiling was requested, getDeviceProfilerState() must report it as disabled for
    // a mock/emulated context.
    EXPECT_FALSE(getDeviceProfilerState(MetalContext::instance().get_context_id()))
        << "getDeviceProfilerState() must be false for a mock context even when profiling is "
           "requested";

    auto devices = detail::CreateDevices({0});
    ASSERT_FALSE(devices.empty());
    const ChipId mock_device_id = devices.begin()->first;

    // The device profiler must never register a mock device.
    const auto& profiler_state_manager = MetalContext::instance().profiler_state_manager();
    ASSERT_NE(profiler_state_manager, nullptr);
    EXPECT_FALSE(profiler_state_manager->device_profiler_map.contains(mock_device_id))
        << "Device profiler was started on mock device " << mock_device_id
        << " -- the profiler must be skipped for mock/emulated clusters";

    detail::CloseDevices(devices);
}

}  // namespace tt::tt_metal
