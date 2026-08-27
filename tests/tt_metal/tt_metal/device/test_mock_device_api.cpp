// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/types/arch.hpp>

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_state.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_metal {

class MockDeviceAPIFixture : public ::testing::Test {
protected:
    void TearDown() override { experimental::disable_mock_mode(); }
};

TEST_F(MockDeviceAPIFixture, CPU_ConfigureMockModeRegistersConfig) {
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    auto desc = experimental::get_mock_cluster_desc();
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(*desc, "blackhole_P150.yaml");
}

TEST_F(MockDeviceAPIFixture, CPU_ConfigureMockModeWormholeMultiChip) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 8);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    auto desc = experimental::get_mock_cluster_desc();
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(*desc, "t3k_cluster_desc.yaml");
}

TEST_F(MockDeviceAPIFixture, CPU_DisableMockModeClearsConfig) {
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_TRUE(experimental::is_mock_mode_registered());
    experimental::disable_mock_mode();
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    EXPECT_FALSE(experimental::get_mock_cluster_desc().has_value());
}

TEST_F(MockDeviceAPIFixture, CPU_WormholeConfigurationsAreValid) {
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

TEST_F(MockDeviceAPIFixture, CPU_BlackholeConfigurationsAreValid) {
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "blackhole_P150.yaml");
    experimental::disable_mock_mode();

    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 2);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "blackhole_P300_both_mmio.yaml");
}

TEST_F(MockDeviceAPIFixture, CPU_QuasarConfigurationsAreValid) {
    experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
    EXPECT_EQ(*experimental::get_mock_cluster_desc(), "quasar_Q1.yaml");
}

TEST_F(MockDeviceAPIFixture, CPU_UnsupportedConfigurationThrows) {
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

// NOT CPU_-prefixed: this test probes real silicon (get_physical_architecture())
// and skips when none is present, so it only provides coverage on device runners.
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

TEST_F(MockDeviceAPIFixture, CPU_SwitchFromMockToRealHardware) {
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

namespace {

// Debug tooling disables erisc IRAM, which makes the mock fabric compile a no-op; the tests below
// then have no routers to assert on. Read a fresh RunTimeOptions so this works before any context.
bool mock_fabric_compile_is_disabled() { return !llrt::RunTimeOptions{}.get_erisc_iram_enabled(); }

// get_num_fabric_initialized_routers() fatals for a device FabricBuilder never visited, so a
// router count is itself proof the compile ran.
void expect_mock_fabric_compiles_on_2_chips() {
    const std::vector<ChipId> device_ids{0, 1};
    auto devices = detail::CreateDevices(device_ids);
    ASSERT_EQ(devices.size(), device_ids.size());

    const auto& builder_context =
        MetalContext::instance().get_control_plane().get_fabric_context().get_builder_context();
    for (const auto& [device_id, _] : devices) {
        EXPECT_GT(builder_context.get_num_fabric_initialized_routers(device_id), 0u)
            << "no fabric routers were built on mock device " << device_id;
    }

    detail::CloseDevices(devices);
}

}  // namespace

// Mock compiles the fabric program even though it never programs or syncs a router; that compile
// is what warms the erisc kernel cache on a host with no silicon.
TEST_F(MockDeviceAPIFixture, FabricProgramIsCompiledOnMock) {
    if (mock_fabric_compile_is_disabled()) {
        GTEST_SKIP() << "erisc IRAM disabled; mock fabric compile is skipped";
    }
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 2);
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    expect_mock_fabric_compiles_on_2_chips();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
}

// Same with the tensix mux, which additionally needs the control plane's tensix datamover config.
TEST_F(MockDeviceAPIFixture, FabricProgramIsCompiledOnMockWithTensixMux) {
    if (mock_fabric_compile_is_disabled()) {
        GTEST_SKIP() << "erisc IRAM disabled; mock fabric compile is skipped";
    }
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 2);
    tt::tt_fabric::SetFabricConfig(
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        /*num_routing_planes=*/std::nullopt,
        tt::tt_fabric::FabricTensixConfig::MUX);
    expect_mock_fabric_compiles_on_2_chips();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
}

// Nothing on mock advances the bytes_acked counter H2D credit accounting polls, so without the
// self-ack the host wedges once it has written one FIFO's worth. Several FIFOs must go through.
TEST_F(MockDeviceAPIFixture, H2DSocketWritesDoNotDeadlockOnMock) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    constexpr uint32_t kFifoSize = 2048;
    constexpr uint32_t kPageSize = 512;
    constexpr uint32_t kNumPages = 32;  // 8x the FIFO, so it must wrap and reclaim credit repeatedly

    const distributed::MeshCoreCoord recv_core{distributed::MeshCoordinate(0, 0), CoreCoord(0, 0)};
    // Scoped: the socket owns device buffers, so it must be destroyed before the mesh it borrows.
    {
        distributed::H2DSocket socket(
            mesh_device, recv_core, BufferType::L1, kFifoSize, distributed::H2DMode::HOST_PUSH);
        socket.set_page_size(kPageSize);

        std::vector<uint32_t> page(kPageSize / sizeof(uint32_t), 0xa5a5a5a5);
        for (uint32_t i = 0; i < kNumPages; i++) {
            socket.write(page.data(), 1);
        }

        // The queries and the barrier (which the destructor also calls) must agree with reserve_bytes.
        EXPECT_TRUE(socket.has_space(kPageSize));
        socket.barrier(1000);
    }

    mesh_device->close();
}

// The D2H mirror: nothing produces data, so bytes_sent never advances and read() blocks forever.
// The payload is meaningless under mock; what matters is that the host path completes.
TEST_F(MockDeviceAPIFixture, D2HSocketReadsDoNotDeadlockOnMock) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    constexpr uint32_t kFifoSize = 2048;
    constexpr uint32_t kPageSize = 512;
    constexpr uint32_t kNumPages = 32;  // 8x the FIFO, so the ring must wrap repeatedly

    const distributed::MeshCoreCoord sender_core{distributed::MeshCoordinate(0, 0), CoreCoord(0, 0)};
    // Scoped: the socket owns device buffers, so it must be destroyed before the mesh it borrows.
    {
        distributed::D2HSocket socket(mesh_device, sender_core, kFifoSize);
        socket.set_page_size(kPageSize);

        std::vector<uint32_t> page(kPageSize / sizeof(uint32_t), 0);
        for (uint32_t i = 0; i < kNumPages; i++) {
            socket.read(page.data(), 1, /*notify_sender=*/true);
        }

        // Availability is scoped to the blocking read that asked for it, so between calls the socket
        // reports drained. A permanently full FIFO would instead livelock the drain loops below.
        EXPECT_FALSE(socket.has_data(kPageSize));
        EXPECT_EQ(socket.pages_available(), 0u);

        // The two idiomatic drain loops must terminate rather than spin.
        uint32_t drained = 0;
        while (socket.has_data(kPageSize)) {
            socket.read(page.data(), 1, /*notify_sender=*/true);
            ASSERT_LT(++drained, kNumPages) << "has_data() drain loop is not terminating on mock";
        }
        uint32_t discard_rounds = 0;
        while (socket.discard_pending_pages() > 0) {
            ASSERT_LT(++discard_rounds, kNumPages) << "discard_pending_pages() drain loop is not terminating on mock";
        }

        // A barrier must settle, and a read after it must still be satisfiable.
        socket.barrier(1000);
        socket.read(page.data(), 1, /*notify_sender=*/true);
        socket.barrier(1000);
    }

    mesh_device->close();
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
TEST_F(MockDeviceProfilerFixture, CPU_DeviceProfilerIsNotStartedOnMockDevice) {
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
