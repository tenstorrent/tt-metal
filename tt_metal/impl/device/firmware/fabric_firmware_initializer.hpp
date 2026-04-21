// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <mutex>
#include <set>
#include <unordered_set>
#include <utility>

#include "device/device_impl.hpp"
#include "firmware_initializer.hpp"

namespace tt::tt_fabric {
class ControlPlane;
class FabricBuilderContext;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

class FabricFirmwareInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key = InitializerKey::Fabric;

    FabricFirmwareInitializer(
        std::shared_ptr<const ContextDescriptor> descriptor, tt::tt_fabric::ControlPlane& control_plane);

    void init(const std::vector<Device*>& devices, [[maybe_unused]] const std::unordered_set<InitializerKey>& init_done)
        override;
    void configure() override;
    void teardown(std::unordered_set<InitializerKey>& init_done) override;
    void post_teardown() override;
    bool is_initialized() const override;

    // FIX E (#42429): Return the set of non-MMIO device IDs whose ETH relay path was confirmed
    // broken during fabric init (relay_broken=true in terminate_stale_erisc_routers).
    // DeviceManager uses this to skip dispatch kernel initialization for unreachable devices —
    // dispatch writes to non-MMIO devices go through the same dead ETH relay and hang.
    const std::unordered_set<ChipId>& get_dead_relay_devices() const { return dead_relay_devices_; }

private:
    // Compile fabric on all devices, parallelized via async.
    // Configure fabric sequentially (Galaxy hangs if parallelized).
    void compile_and_configure_fabric();

    // Wait for fabric router handshake on all devices.
    void wait_for_fabric_router_sync(uint32_t timeout_ms) const;

    // Verify ALL active ERISC channels are healthy after fabric init.
    // wait_for_fabric_router_sync only checks the master channel; this checks every channel
    // to detect persistent ERISC corruption that would cause dispatch hangs later.
    void verify_all_fabric_channels_healthy() const;

    // Compute the fabric router sync timeout from runtime options.
    uint32_t get_fabric_router_sync_timeout_ms() const;

    // Probe all active ERISC router channels for stale firmware and terminate any found.
    // Sends TERMINATE to each active ERISC channel, polls for EDMStatus::TERMINATED (100 ms).
    // On timeout, logs a warning and continues — does NOT assert RISC reset (that would
    // tear down the WH ETH PHY link and break non-MMIO L1 access for the rest of the mesh).
    // Called before configure_fabric_cores() clears L1 (Fix A).
    //
    // Returns {probe_dead_channels, relay_broken}:
    // - probe_dead_channels: ETH channel IDs whose probe read timed out (ERISC unresponsive).
    //   Passed to configure_fabric_cores() to skip assert_risc_reset_at_core() for dead channels.
    // - relay_broken: true when the relay queue reached saturation risk (kMaxRelayTimeouts
    //   consecutive read timeouts), indicating the non-MMIO device's ETH relay path is broken
    //   and the device is effectively unreachable for ANY L1 write (not just ETH core writes).
    //   Callers use this to skip dispatch kernel initialization for unreachable non-MMIO devices.
    std::pair<std::unordered_set<uint32_t>, bool> terminate_stale_erisc_routers(
        Device* dev, const tt_fabric::FabricBuilderContext& builder_context) const;

    tt::tt_fabric::ControlPlane& control_plane_;
    std::vector<Device*> devices_;
    std::atomic_flag initialized_;

    // FIX E: Non-MMIO devices whose ETH relay path was broken during fabric init.
    // These devices cannot receive ANY writes (including dispatch firmware) via the relay.
    std::unordered_set<ChipId> dead_relay_devices_;

    // GAP 5: Track channels that were force-reset during teardown.
    // On the next verify_all_fabric_channels_healthy() call, channels that were force-reset
    // in a previous session are expected to fail — log them as "degraded" rather than
    // "corrupt from prior crash" to aid diagnosis.
    // Key: (device_id, eth_chan_id).
    mutable std::mutex force_reset_channels_mutex_;
    std::set<std::pair<ChipId, uint32_t>> force_reset_channels_;
};

}  // namespace tt::tt_metal
