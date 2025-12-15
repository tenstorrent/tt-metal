// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// Test fixture for telemetry initialization tests
class TelemetryFixture : public BaseFabricFixture {
public:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt_fabric::FabricConfig::FABRIC_1D); }

    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

TEST_F(TelemetryFixture, TelemetryStaticInfoInitialized) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& devices = this->get_devices();

    if (devices.empty()) {
        GTEST_SKIP() << "No devices available for telemetry test";
    }

    // Test on first device
    auto device = devices[0];
    ASSERT_FALSE(device->get_devices().empty()) << "Device has no sub-devices";
    auto physical_chip_id = device->get_devices()[0]->id();
    auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);

    // Read fabric telemetry from all ethernet channels
    auto samples = tt::tt_fabric::read_fabric_telemetry(fabric_node_id);

    ASSERT_FALSE(samples.empty()) << "No telemetry samples read from device";

    // Verify each channel's telemetry static_info is initialized
    for (const auto& sample : samples) {
        const auto& static_info = sample.snapshot.static_info;

        // Check mesh_id is populated (should match fabric_node_id)
        EXPECT_EQ(static_info.mesh_id, fabric_node_id.mesh_id.get())
            << "mesh_id not correctly initialized for channel " << static_cast<int>(sample.channel_id);

        // Check device_id is populated (should match fabric_node_id)
        EXPECT_EQ(static_info.device_id, fabric_node_id.chip_id)
            << "device_id not correctly initialized for channel " << static_cast<int>(sample.channel_id);

        // Check direction is valid (0=EAST, 1=WEST, 2=NORTH, 3=SOUTH)
        EXPECT_LE(static_info.direction, 3)
            << "direction has invalid value for channel " << static_cast<int>(sample.channel_id);

        // Check supported_stats is non-zero (should have at least BANDWIDTH enabled)
        EXPECT_NE(static_info.supported_stats, 0)
            << "supported_stats is zero, telemetry disabled for channel " << static_cast<int>(sample.channel_id);

        // Verify BANDWIDTH bit is set (bit 1, value 0x02)
        constexpr uint8_t BANDWIDTH_BIT = 0x02;
        EXPECT_NE(static_info.supported_stats & BANDWIDTH_BIT, 0)
            << "BANDWIDTH telemetry not enabled for channel " << static_cast<int>(sample.channel_id)
            << " (supported_stats=0x" << std::hex << static_cast<int>(static_info.supported_stats) << std::dec << ")";

        // Verify dynamic_info counters are initialized (not garbage)
        const auto& tx_bw = sample.snapshot.dynamic_info.tx_bandwidth;
        const auto& rx_bw = sample.snapshot.dynamic_info.rx_bandwidth;

        // IMPORTANT: We check counters are NOT garbage, not that they're exactly zero.
        // Why? The router starts processing immediately after initialization. By the time
        // we read telemetry, it may have already processed packets, so counters could be
        // small non-zero values. Uninitialized memory contains random garbage (often ~10^18).
        // This threshold distinguishes initialized counters (0 to ~billions) from garbage.
        constexpr uint64_t GARBAGE_THRESHOLD = 1'000'000'000'000ULL;  // 10^12 cycles ≈ 14 min @ 1.2GHz

        EXPECT_LT(tx_bw.elapsed_active_cycles, GARBAGE_THRESHOLD)
            << "TX elapsed_active_cycles appears uninitialized (value=" << tx_bw.elapsed_active_cycles
            << ") for channel " << static_cast<int>(sample.channel_id);

        EXPECT_LT(tx_bw.elapsed_cycles, GARBAGE_THRESHOLD)
            << "TX elapsed_cycles appears uninitialized (value=" << tx_bw.elapsed_cycles << ") for channel "
            << static_cast<int>(sample.channel_id);

        EXPECT_LT(rx_bw.elapsed_active_cycles, GARBAGE_THRESHOLD)
            << "RX elapsed_active_cycles appears uninitialized (value=" << rx_bw.elapsed_active_cycles
            << ") for channel " << static_cast<int>(sample.channel_id);

        EXPECT_LT(rx_bw.elapsed_cycles, GARBAGE_THRESHOLD)
            << "RX elapsed_cycles appears uninitialized (value=" << rx_bw.elapsed_cycles << ") for channel "
            << static_cast<int>(sample.channel_id);

        // Print telemetry info for debugging
        log_info(
            tt::LogTest,
            "Channel {} telemetry: mesh_id={}, device_id={}, direction={}, supported_stats=0x{:02x}",
            static_cast<int>(sample.channel_id),
            static_info.mesh_id,
            static_info.device_id,
            static_info.direction,
            static_info.supported_stats);
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
