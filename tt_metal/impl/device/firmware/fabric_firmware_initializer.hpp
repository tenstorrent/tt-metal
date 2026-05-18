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
    // Result fields:
    // - probe_dead_channels: ETH channel IDs whose probe read timed out (ERISC unresponsive).
    //   Passed to configure_fabric_cores() to skip assert_risc_reset_at_core() for dead channels.
    // - relay_broken: true when the relay queue reached saturation risk (kMaxRelayTimeouts
    //   consecutive read timeouts), indicating the non-MMIO device's ETH relay path is broken
    //   and the device is effectively unreachable for ANY L1 write (not just ETH core writes).
    //   Callers use this to skip dispatch kernel initialization for unreachable non-MMIO devices.
    // - base_umd_channels: ETH channel IDs where edm_status == 0x49706550 (base-UMD relay
    //   firmware is running — BRISC is alive).  FIX M (#42429): configure_fabric_cores() must
    //   SKIP assert_risc_reset_at_core() for these channels — halting their BRISC kills the ETH
    //   relay endpoint used by non-MMIO reads, cascading into a full hang.  write_launch_msg_to_core
    //   transitions this firmware to fabric firmware without needing a soft reset.
    // - external_umd_channels: ETH channel IDs where edm_status == 0x49706550 AND the channel
    //   has no in-cluster peer (not present in cluster_.get_ethernet_connections()).  These are
    //   out-of-mesh channels (e.g. T3K Device 4 chan 6 connecting to an external host).  FIX EXT
    //   (#42429): like base_umd_channels, soft-reset is skipped (preserve relay BRISC); UNLIKE
    //   base_umd_channels, write_launch_msg_to_core is also skipped — loading FABRIC_1D on an
    //   external channel whose peer can never respond causes ring-sync timeouts that cascade into
    //   fabric_channels_not_ready_for_traffic_ → GTEST_SKIP on 5 AllGather tests.
    struct TerminateStaleResult {
        std::unordered_set<uint32_t> probe_dead_channels;
        bool relay_broken;
        std::unordered_set<uint32_t> base_umd_channels;
        std::unordered_set<uint32_t> external_umd_channels;
        // FIX KL (#42429): MMIO device channels at base-UMD sentinel (0x49706550) with an
        // in-cluster peer.  Unlike non-MMIO base_umd_channels, these are NOT added to
        // skip_soft_reset_channels (FIX EE: MMIO ETH is PCIe-direct, soft-reset is safe).
        // However, configure_fabric_cores() still zeros fw_launch_addr (FIX EG) during the
        // soft-reset window, so FIX IJ must restore it AFTER write_launch_msg_to_core —
        // same as the non-MMIO path.  Tracked separately so configure_fabric() can extend
        // the FIX IJ condition without merging with skip_soft_reset_channels.
        std::unordered_set<uint32_t> mmio_base_umd_channels;
    };
    TerminateStaleResult terminate_stale_erisc_routers(
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
    // Declared mutable so wait_for_fabric_router_sync() (const) can populate it via FIX DX2.
    mutable std::unordered_set<ChipId> mmio_dead_peer_devices_;

    // FIX AN (#42429): MMIO devices whose own master router ETH channel was excluded
    // from configure_fabric_cores() (was in probe_dead_channels — L1 corrupt or
    // channel unresponsive). No firmware was loaded on the master chan, so the sync
    // value 0xa2b2c2d2 will never be written. Skip sync for these devices to avoid
    // the 10s-per-device timeout.
    // Declared mutable so wait_for_fabric_router_sync() (const) can populate it
    // for the FIX BG (0xdeadb07e pre-launch early-exit) path.
    mutable std::unordered_set<ChipId> mmio_dead_master_chan_devices_;

    // FIX TH2 (#42429): Set to true when any device has base-UMD channels this session.
    // Base-UMD ERISCs are transitioned via launch_msg (not soft reset), so they need extra
    // time to quiesce the relay and complete the ring handshake — the default 10s timeout
    // is insufficient. get_fabric_router_sync_timeout_ms() triples the timeout when set.
    bool has_base_umd_channels_ = false;

    // SINGLE-THREAD INVARIANT: wait_for_fabric_router_sync() and verify_all_fabric_channels_healthy()
    // are always called sequentially from the same thread. These mutable members are NOT thread-safe
    // (unordered_set::insert races under concurrent access). Do not call from parallel device threads.
    // (#42429 FIX BE audit Q5-D)

    // FIX TI (#42429): Set of devices whose ring barrier timed out in wait_for_fabric_router_sync
    // when base-UMD channels were present. Even with the extended 30s timeout (FIX TH2), the
    // inter-rank ring signal may not propagate if base-UMD ERISCs on a partner rank are still
    // quiescing. Channels on these devices will never reach READY_FOR_TRAFFIC, so
    // verify_all_fabric_channels_healthy() must skip them (otherwise the 150ms retry window
    // causes a false health-check failure).
    // Declared mutable so wait_for_fabric_router_sync() (const) can populate it.
    mutable std::unordered_set<ChipId> timeout_on_base_umd_devices_;

    // FIX TJ (#42429): Set to true after the first device times out in wait_for_fabric_router_sync
    // due to base-UMD ring-barrier failure.  When the ring barrier fails on one device it will
    // fail on ALL devices (barrier requires every ring member to complete before any can advance).
    // Subsequent devices in the polling loop are immediately added to timeout_on_base_umd_devices_
    // instead of waiting the full 30s timeout, reducing total wait from N×30s to 1×30s.
    // Declared mutable so wait_for_fabric_router_sync() (const) can set it.
    mutable bool ring_sync_already_timed_out_ = false;

    // FIX EXT (#42429): per-device ETH channels at 0x49706550 with no in-cluster peer.
    // Populated in compile_and_configure_fabric() from terminate_stale_erisc_routers() results.
    // These channels skip write_launch_msg_to_core (firmware not loaded), ring-sync validation,
    // and verify_all_fabric_channels_healthy checks.  Stored separately from base_umd_channels
    // so they do NOT trigger FIX RZ (fabric_stale_base_umd_channels_) or FIX TH2/TI timeout
    // extension — external channels are expected to never handshake.
    std::unordered_map<ChipId, std::unordered_set<uint32_t>> external_umd_channels_map_;

    // GAP 5: Track channels that were force-reset during teardown.
    // On the next verify_all_fabric_channels_healthy() call, channels that were force-reset
    // in a previous session are expected to fail — log them as "degraded" rather than
    // "corrupt from prior crash" to aid diagnosis.
    // Key: (device_id, eth_chan_id).
    mutable std::mutex force_reset_channels_mutex_;
    std::set<std::pair<ChipId, uint32_t>> force_reset_channels_;
    // FIX CL-2 (#42429): Non-MMIO relay-dead channels that skipped assert_risc_reset (FIX BU)
    // during the last teardown.  Direct L1 writes to these channels are impossible (relay dead),
    // so the critical state cannot be zeroed.  Injected into probe_dead_channels_map at the start
    // of the next compile_and_configure_fabric() so configure_fabric_cores() skips soft-reset on
    // them (avoids 5s relay timeout per channel).  Protected by force_reset_channels_mutex_.
    std::set<std::pair<ChipId, uint32_t>> pending_pre_dead_non_mmio_;

    // Thread-local compile seam (see set_compile_fn_for_testing / clear_compile_fn_for_testing).
    // Default-constructed (empty std::function) — not set in production builds.
    static thread_local CompileFabricFn s_compile_fn_for_testing_;

    // Thread-local status-override seam (see set_status_override_fn_for_testing).
    // Default-constructed (empty std::function) — not set in production builds.
    static thread_local StatusOverrideFn s_status_override_fn_;
};

}  // namespace tt::tt_metal
