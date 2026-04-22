// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
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

    // ---------------------------------------------------------------------------
    // Test seam: compile-factory injection for Scenario Y
    //
    // When set (non-null), compile_and_configure_fabric() calls this function
    // instead of dev->compile_fabric() for every device in the async compile loop.
    // The function may return false (compile not performed), return true (success),
    // or throw std::exception (compile failure — join-before-rethrow path).
    //
    // Thread-local so parallel test workers on different threads don't interfere.
    // Set before MeshDevice::create(), cleared immediately after via clear_compile_fn_for_testing().
    //
    // PRODUCTION CODE: s_compile_fn_for_testing_ is default-constructed (empty std::function)
    // on every thread.  The branch in compile_and_configure_fabric() is a single bool check
    // on an inline-initialized thread_local — zero overhead in non-test builds.
    // ---------------------------------------------------------------------------
    using CompileFabricFn = std::function<bool(Device*)>;

    // Set/clear the per-thread compile seam.  NOT thread-safe with concurrent callers
    // on the same thread (tests call this single-threaded before MeshDevice::create()).
    static void set_compile_fn_for_testing(CompileFabricFn fn);
    static void clear_compile_fn_for_testing();

    // ---------------------------------------------------------------------------
    // Test seam: L1 status-read override for Scenario X
    //
    // When set (non-null), terminate_stale_erisc_routers() calls this function
    // instead of (and before) the real ReadFromDeviceL1() probe read for each
    // active ETH channel.  If the function returns a value, that value replaces
    // the L1 read result (status_buf[0]); if it returns std::nullopt, the real
    // L1 read is performed normally.
    //
    // Signature: std::optional<uint32_t>(Device*, uint32_t eth_chan_id)
    // A test can return 0xBAADF00D to simulate a corrupt EDMStatus value without
    // writing anything to hardware — exercising the is_known_edm_status() false
    // branch and the probe_dead_channels insertion path, safely.
    //
    // Thread-local — same rationale as CompileFabricFn above.
    // PRODUCTION: s_status_override_fn_ is default-constructed (empty std::function)
    // on every thread — zero overhead in non-test builds.
    // ---------------------------------------------------------------------------
    using StatusOverrideFn = std::function<std::optional<uint32_t>(Device*, uint32_t /*eth_chan_id*/)>;

    // Set/clear the per-thread status-override seam.
    static void set_status_override_fn_for_testing(StatusOverrideFn fn);
    static void clear_status_override_fn_for_testing();

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

    // FIX I (#42429): MMIO devices whose master router ETH channel connects to a dead-relay
    // non-MMIO device. Their fabric firmware was loaded but the startup handshake peer (the
    // non-MMIO device's ERISC) will never respond — relay path is broken.
    // These devices are excluded from fabric router sync and channel health checks.
    // Unlike dead_relay_devices_, they are NOT excluded from dispatch kernel init —
    // MMIO dispatch goes through PCIe, not ETH relay, and must proceed normally.
    std::unordered_set<ChipId> mmio_dead_peer_devices_;

    // GAP 5: Track channels that were force-reset during teardown.
    // On the next verify_all_fabric_channels_healthy() call, channels that were force-reset
    // in a previous session are expected to fail — log them as "degraded" rather than
    // "corrupt from prior crash" to aid diagnosis.
    // Key: (device_id, eth_chan_id).
    mutable std::mutex force_reset_channels_mutex_;
    std::set<std::pair<ChipId, uint32_t>> force_reset_channels_;

    // Thread-local compile seam (see set_compile_fn_for_testing / clear_compile_fn_for_testing).
    // Default-constructed (empty std::function) — not set in production builds.
    static thread_local CompileFabricFn s_compile_fn_for_testing_;

    // Thread-local status-override seam (see set_status_override_fn_for_testing).
    // Default-constructed (empty std::function) — not set in production builds.
    static thread_local StatusOverrideFn s_status_override_fn_;
};

}  // namespace tt::tt_metal
