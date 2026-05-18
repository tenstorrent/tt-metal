// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>
#include <unordered_set>

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_fabric {

// Thread-local configure-cores inject seam — default-constructed (empty std::function) on every
// thread.  Only set by tests via set_configure_cores_inject_fn() before MeshDevice::create().
thread_local ConfigureFabricCoresInjectFn s_configure_cores_inject_fn_;

void set_configure_cores_inject_fn(ConfigureFabricCoresInjectFn fn) {
    s_configure_cores_inject_fn_ = std::move(fn);
}

void clear_configure_cores_inject_fn() {
    s_configure_cores_inject_fn_ = {};
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program(tt::tt_metal::IDevice* device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Guard: if this device's physical chip is not in the fabric cluster mapping, skip
    // fabric program compilation rather than crashing with TT_FATAL inside FabricBuilder.
    // This can happen when hardware connectivity issues cause auto-discovery to downgrade
    // to a smaller mesh (e.g. 2x2 instead of 2x4) that excludes some physical chips.
    // The device will simply not participate in fabric routing for this session.
    if (!control_plane.is_physical_chip_in_fabric_cluster(device->id())) {
        log_warning(
            tt::LogFabric,
            "create_and_compile_tt_fabric_program: Physical chip {} is not in the fabric cluster "
            "chip mapping; skipping fabric program compilation. This typically indicates hardware "
            "connectivity issues that caused auto-discovery to select a smaller mesh topology. "
            "The device will not participate in fabric routing.",
            device->id());
        return nullptr;
    }

    auto fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();

    auto& fabric_context = control_plane.get_fabric_context();

    // Use FabricBuilder to coordinate the build phases
    FabricBuilder builder(device, *fabric_program_ptr, fabric_context);

    // Execute build phases
    builder.discover_channels();
    builder.create_routers();
    if (!builder.has_routers()) {
        return nullptr;
    }

    builder.connect_routers();
    builder.compile_ancillary_kernels();
    builder.create_kernels();

    // Compile the program
    tt::tt_metal::detail::CompileProgram(
        device, *fabric_program_ptr, tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch());

    return fabric_program_ptr;
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device) {
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        return create_and_compile_tt_fabric_program(device);
    }
    return nullptr;
}

FabricCoresHealth configure_fabric_cores(
    tt::tt_metal::IDevice* device,
    const std::unordered_set<uint32_t>& pre_known_dead_channels,
    const std::unordered_set<uint32_t>& skip_soft_reset_channels) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto soc_desc = cluster.get_soc_desc(device->id());
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto addresses_to_clear = builder_context.get_fabric_router_addresses_to_clear();
    const auto& router_config = builder_context.get_fabric_router_config();
    std::vector<uint32_t> router_zero_buf(router_config.router_buffer_clear_size_words, 0);

    // Fix #42429: After a cancelled CI run, SIGKILL, or normal AllGather CCL teardown, the
    // ERISC BRISC may be halted.  When an ERISC fabric router self-terminates (writes
    // EDMStatus::TERMINATED to its sync address and halts BRISC), subsequent L1 firmware
    // writes in this function have no effect because BRISC never executes the new code.
    // This causes the Phase 5 health check in quiesce_and_restart_fabric_workers() to fail
    // with TERMINATED ≠ READY_FOR_TRAFFIC on the second (post-AllGather) quiesce cycle.
    //
    // Fix: perform a BRISC-only soft reset (assert + deassert) before writing L1 to ensure
    // BRISC is live when the new fabric firmware is loaded.
    //
    // Safety: resetting only ERISC0/BRISC keeps the subordinate ERISC (NCRISC) running,
    // which maintains the ETH PHY link.  The reset window is brief (PCIe round-trip,
    // microseconds) and mirrors the pattern used in risc_firmware_initializer.cpp
    // reset_cores().  Although base UMD firmware is not expected to touch fabric-specific
    // state addresses, the L1 clear below (addresses_to_clear) zeroes edm_status_address,
    // termination_signal_address, edm_local_sync_address, and edm_local_tensix_sync_address
    // as a belt-and-suspenders measure — so even if base firmware or stale L1 leaves garbage,
    // the next session's terminate_stale_erisc_routers() sees clean zeros.
    // Track dead channels so the L1 clear loop below can skip them.
    // For non-MMIO chips, WriteToDeviceL1 routes writes through ethernet — the same path
    // that timed out during the soft reset — and can hang indefinitely (write_core has no
    // per-call timeout unlike read_non_mmio).  Skipping the L1 clear for dead channels is
    // safe: firmware won't start on them regardless of whether L1 was zeroed.
    //
    // Seed with pre_known_dead_channels: channels where the probe L1 read in
    // terminate_stale_erisc_routers() threw an exception (physically dead link — ERISC
    // completely unresponsive).  For these channels we skip assert_risc_reset_at_core()
    // entirely because the ETH relay path is non-functional.
    //
    // Note: channels with corrupt/garbage L1 status (probe read succeeded but value was
    // not a valid EDMStatus) are NOT in this set — their probe read succeeded, proving
    // the relay path works, so assert_risc_reset_at_core() should also succeed.  Those
    // channels proceed through normal soft reset here.
    std::unordered_set<uint32_t> dead_channels = pre_known_dead_channels;
    // Track channels that NEWLY failed in this call (not pre-known).
    // This lets the caller (configure_fabric) distinguish "expected degraded mode" from
    // "unexpected new failure" — the former warrants a warning, the latter a hard throw.
    std::unordered_set<uint32_t> newly_dead_channels;
    // FIX RR (#42429): Track pre_known channels that were successfully recovered by
    // PCIe-direct soft reset.  Returned to configure_fabric() so it can subtract them from
    // pre_dead_channels and load firmware on those channels instead of skipping them.
    std::unordered_set<uint32_t> recovered_channels;
    // FIX SA (#42429): Strategy A — deferred deassert channels.
    // MMIO channels where FIX S9 asserted ERISC reset and FIX EG zeroed fw_launch_addr,
    // but deassert was NOT performed here.  ERISC stays halted so all subsequent L1 writes
    // (L1 clear, ConfigureDeviceWithProgram, write_launch_msg_to_core) are atomic from
    // ERISC's perspective — no race with base-UMD execution.  device.cpp performs deassert
    // + FIX DW + FIX DU + go_msg for these channels after all L1 mutations are complete.
    std::unordered_set<uint32_t> deferred_deassert_channels;
    // Note: all_channels_healthy is computed at the END of this function (after FIX RR may
    // have recovered pre_known_dead channels from dead_channels).  Do not set it here.
    if (!pre_known_dead_channels.empty()) {
        if (device->is_mmio_capable()) {
            // FIX RR (#42429): MMIO devices have PCIe-direct access — the relay-queue-fill
            // concern that motivates skipping assert_risc_reset_at_core for non-MMIO channels
            // does NOT apply here.  ROM-postcode channels (0x49705180) that timed out in FIX RP
            // are still accessible via PCIe; a soft reset kicks them out of ROM phase so they
            // boot to base firmware instead of persisting as permanently degraded every session.
            // We attempt the reset and catch failures — if PCIe truly is broken for a channel
            // we mark it newly_dead; otherwise we clear it from dead_channels so the L1 write
            // loop below can zero its sync addresses and normal firmware launch can proceed.
            log_info(
                tt::LogMetal,
                "configure_fabric_cores: Device {} (MMIO) — attempting PCIe-direct soft reset "
                "for {} pre-confirmed problematic channels (FIX RR #42429). "
                "ROM-postcode channels that timed out in FIX RP can be recovered this way.",
                device->id(),
                pre_known_dead_channels.size());
        } else {
            log_warning(
                tt::LogMetal,
                "configure_fabric_cores: Device {} skipping assert_risc_reset_at_core for {} "
                "pre-confirmed problematic channels (probe timed out or L1 corrupt in "
                "terminate_stale_erisc_routers). This avoids the indefinite hang in read_non_mmio "
                "when the 4-slot relay ETH queue fills after dead channels exhaust UMD timeouts.",
                device->id(),
                pre_known_dead_channels.size());
        }
    }
    {
        const auto chip_id = device->id();
        for (const auto& [router_chan, _] : router_chans_and_direction) {
            // Skip channels already confirmed dead by terminate_stale_erisc_routers().
            // For non-MMIO devices, calling assert_risc_reset_at_core() on dead channels can
            // hang indefinitely because the UMD relay queue fills after the channel exhausts
            // its retry budget.  For MMIO devices, PCIe-direct access is available so we
            // attempt the reset below (FIX RR) and only skip on genuine PCIe failure.
            if (dead_channels.count(router_chan)) {
                if (!device->is_mmio_capable()) {
                    log_debug(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} is pre-confirmed dead "
                        "(non-MMIO) — skipping soft reset to avoid relay queue fill",
                        chip_id,
                        router_chan);
                    continue;
                }
                // FIX RR (#42429): MMIO device — attempt PCIe-direct soft reset.
                // If the channel is truly dead (PCIe error) the catch block marks it
                // newly_dead; otherwise remove it from dead_channels so L1 write proceeds.
                try {
                    auto virtual_core = cluster.get_virtual_eth_core_from_channel(chip_id, router_chan);
                    tt_cxy_pair core_loc(chip_id, virtual_core);
                    // GAP 7 / FIX OP RR (#42429): Time the FIX RR assert→deassert+BH window,
                    // matching the FIX OP timing log on the normal FIX S9 path.  Without this,
                    // CI logs show no timing data for the FIX RR recovery path — we can't tell
                    // if the channel spent 5ms or 5000ms in the reset window.
                    auto fix_rr_start = std::chrono::steady_clock::now();
                    cluster.assert_risc_reset_at_core(core_loc, tt::umd::RiscType::ERISC0);

                    // FIX EG RR (#42429): While ERISC0 is halted, zero fw_launch_addr (0x9004 =
                    // LAUNCH_ERISC_APP_FLAG) — same fix applied in the normal FIX S9 assert/deassert
                    // path, but MISSING from FIX RR for pre-known-dead channels.
                    //
                    // Root cause of CI run 26024194526 failure:
                    // Device 0 chan=8 had edm_status=0xDEADB07E (HOST_PRE_LAUNCH_CANARY) from the
                    // previous session → terminate_stale_erisc_routers added it to probe_dead_channels
                    // → configure_fabric_cores took the FIX RR path → assert/deassert WITHOUT zeroing
                    // fw_launch_addr → ERISC booted, saw stale fw_launch_addr=1 (set by
                    // active_erisc.cc fabric firmware in the prior session) → base-UMD firmware tried
                    // to execute the stale launch message from L1 BEFORE write_launch_msg_to_core ran
                    // → ERISC crashed → stuck at 0xDEADB07E → FIX EF 3000ms poll timed out →
                    // fabric_relay_path_broken_=true → non-MMIO devices (4,5,6,7) could not load
                    // firmware → ring sync timeout → test failure.
                    //
                    // FIX IJ fires AFTER write_launch_msg_to_core, but for the FIX RR path the ERISC
                    // had already crashed before FIX IJ ran — fw_launch_addr was still 1 from the
                    // FIX RR deassert.  FIX IJ's write was a no-op (1→1) and did not help.
                    //
                    // Fix: mirror FIX EG here — zero fw_launch_addr via PCIe while ERISC0 is halted.
                    // On deassert, ERISC0 boots from ROM cleanly, sees fw_launch_addr == 0, and does
                    // NOT prematurely launch any stale firmware.  write_launch_msg_to_core (and FIX IJ)
                    // then run after configure_fabric_cores returns — same as the normal channel path.
                    {
                        const auto& hal_egrr = tt::tt_metal::MetalContext::instance().hal();
                        const auto aeth_idx_egrr = hal_egrr.get_programmable_core_type_index(
                            tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
                        const uint32_t fw_launch_addr_egrr =
                            hal_egrr.get_jit_build_config(aeth_idx_egrr, 0, 0).fw_launch_addr;
                        // FIX MN RR (#42429): Capture fw_launch_addr BEFORE zeroing it.
                        std::vector<uint32_t> egrr_pre_val(1, 0xFFFFFFFF);
                        cluster.read_core(egrr_pre_val, sizeof(uint32_t),
                            core_loc,
                            static_cast<uint64_t>(fw_launch_addr_egrr));
                        log_info(
                            tt::LogMetal,
                            "FIX MN RR (#42429): FIX EG RR pre-zero snapshot — Device {} chan={} "
                            "fw_launch_addr=0x{:08X} pre_val=0x{:08X} (1=stale, 0=clean)",
                            chip_id, router_chan, fw_launch_addr_egrr, egrr_pre_val[0]);
                        cluster.write_core_immediate(
                            chip_id, virtual_core, std::vector<uint32_t>{0}, fw_launch_addr_egrr);
                        // FIX GI RR (#42429): Readback verify the zero write.
                        std::vector<uint32_t> egrr_verify(1, 0xFFFFFFFF);
                        cluster.read_core(egrr_verify, sizeof(uint32_t),
                            core_loc,
                            static_cast<uint64_t>(fw_launch_addr_egrr));
                        if (egrr_verify[0] != 0) {
                            log_warning(
                                tt::LogMetal,
                                "FIX GI RR (#42429): FIX EG RR readback MISMATCH — wrote 0 to "
                                "fw_launch_addr=0x{:08X} on Device {} chan={} but read back "
                                "0x{:08X}. ERISC may prematurely launch stale firmware.",
                                fw_launch_addr_egrr, chip_id, router_chan, egrr_verify[0]);
                        } else {
                            log_info(
                                tt::LogMetal,
                                "FIX EG RR (#42429): zeroed fw_launch_addr=0x{:08X} on Device {} "
                                "chan={} (readback verified=0x{:08X}) while ERISC0 halted "
                                "(pre-known-dead / FIX RR path)",
                                fw_launch_addr_egrr, chip_id, router_chan, egrr_verify[0]);
                        }
                    }

                    cluster.deassert_risc_reset_at_core(core_loc, tt::umd::RiscType::ERISC0);
                    // FIX BH (#42429): Wait for ERISC to boot from ROM phase (0x49705180)
                    // to base-UMD firmware before declaring recovery.  Without this wait,
                    // write_launch_msg_to_core races with ROM boot: the host writes the
                    // pre-launch canary (0xdeadb07e) to L1 while ERISC is still in ROM
                    // and cannot process the launch message.  0xdeadb07e then persists in L1;
                    // FIX BG marks the master channel as dead; AllGather is skipped (FIX QE);
                    // and the test fails — even though FIX RR's assert/deassert succeeded.
                    //
                    // We poll edm_status_address via PCIe-direct cluster.read_core until the
                    // value transitions away from the ROM postcode (kRomPostcode = 0x49705180).
                    // If it transitions within kFIX_BH_BootWaitMs (5000ms) we declare recovery.
                    // If it does not, the channel is irrecoverable this session and is moved to
                    // newly_dead_channels to prevent a downstream launch-message race.
                    {
                        constexpr uint32_t kRomPostcode_BH = 0x49705180u;
                        // FIX DP (#42429): After PCIe hard-reset of 20+ channels simultaneously,
                        // ERISC BRISC needs time to execute ROM before starting the application.
                        // 500ms was insufficient — all channels still at ROM postcode 0x49705180
                        // after 500ms → marked as newly_dead_channels → 24 channels dead → init
                        // failure.  3000ms was also insufficient when 24+ channels boot
                        // simultaneously (run 25964368272: ALL 24 MMIO channels still at
                        // 0x49705180 after 3s).  5000ms matches the FIX RP PARALLEL batch
                        // deadline and gives sufficient margin for WH ETH link training.
                        constexpr uint32_t kFIX_BH_BootWaitMs = 5000;
                        constexpr uint32_t kFIX_BH_PollIntervalMs = 5;
                        const uint64_t edm_addr =
                            static_cast<uint64_t>(router_config.edm_status_address);
                        uint32_t elapsed_bh_ms = 0;
                        bool fix_bh_ok = false;
                        while (elapsed_bh_ms < kFIX_BH_BootWaitMs) {
                            std::vector<uint32_t> status_buf(1, 0);
                            cluster.read_core(status_buf, sizeof(uint32_t), core_loc, edm_addr);
                            if (status_buf[0] != kRomPostcode_BH) {
                                fix_bh_ok = true;
                                log_info(
                                    tt::LogMetal,
                                    "configure_fabric_cores: device {} channel {} FIX BH — "
                                    "ERISC booted from ROM to 0x{:08x} after {}ms.",
                                    chip_id,
                                    router_chan,
                                    status_buf[0],
                                    elapsed_bh_ms);
                                break;
                            }
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(kFIX_BH_PollIntervalMs));
                            elapsed_bh_ms += kFIX_BH_PollIntervalMs;
                        }
                        if (fix_bh_ok) {
                            // Boot succeeded — clear from dead set so L1 init proceeds and
                            // configure_fabric() loads firmware on this channel.
                            dead_channels.erase(router_chan);
                            recovered_channels.insert(router_chan);
                            log_info(
                                tt::LogMetal,
                                "configure_fabric_cores: device {} channel {} FIX RR+BH — "
                                "PCIe-direct soft reset and ROM boot both succeeded. "
                                "Channel cleared from dead_channels; L1 init will proceed.",
                                chip_id,
                                router_chan);
                        } else {
                            // ROM boot did not complete within timeout — channel irrecoverable.
                            // Leave dead_channels intact so the L1 write loop below skips it
                            // (preventing a hang on the non-functional relay path).
                            // FIX DR (#42429): Read the actual value at timeout to distinguish
                            // "completely stuck at ROM postcode" from "partially booted".
                            uint32_t final_val = kRomPostcode_BH;
                            try {
                                std::vector<uint32_t> final_buf(1, 0);
                                cluster.read_core(final_buf, sizeof(uint32_t), core_loc, edm_addr);
                                final_val = final_buf[0];
                            } catch (...) {}
                            newly_dead_channels.insert(router_chan);
                            log_warning(
                                tt::LogMetal,
                                "configure_fabric_cores: device {} channel {} FIX BH — "
                                "ERISC did not exit ROM phase within {}ms "
                                "(edm_status=0x{:08x} after FIX RR deassert). "
                                "Channel remains dead; skipping L1 init.",
                                chip_id,
                                router_chan,
                                kFIX_BH_BootWaitMs,
                                final_val);
                        }
                    }
                    // GAP 7 / FIX OP RR (#42429): Log elapsed time for the entire FIX RR
                    // assert→deassert+BH window — mirrors FIX OP on the normal FIX S9 path.
                    {
                        auto fix_rr_end = std::chrono::steady_clock::now();
                        auto fix_rr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            fix_rr_end - fix_rr_start).count();
                        log_info(
                            tt::LogMetal,
                            "FIX OP RR (#42429): FIX RR assert→deassert+BH window — Device {} "
                            "chan={} total recovery took {}ms (recovered={})",
                            chip_id,
                            router_chan,
                            fix_rr_ms,
                            recovered_channels.count(router_chan) > 0);
                    }
                } catch (const std::exception& e) {
                    newly_dead_channels.insert(router_chan);
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} FIX RR — PCIe-direct "
                        "soft reset FAILED ({}). Channel remains dead.",
                        chip_id,
                        router_chan,
                        e.what());
                } catch (...) {
                    newly_dead_channels.insert(router_chan);
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} FIX RR — PCIe-direct "
                        "soft reset FAILED (unknown exception). Channel remains dead.",
                        chip_id,
                        router_chan);
                }
                continue;
            }

            // FIX M (#42429): Skip soft reset for base-UMD relay channels on NON-MMIO devices.
            // Their BRISC serves as the ETH relay endpoint for all host→non-MMIO reads.
            // Halting it (assert_risc_reset) kills the relay → subsequent reads hang forever.
            // write_launch_msg_to_core transitions this firmware to fabric firmware without a reset.
            // NOTE: the L1 clear loop below is ALSO skipped for non-MMIO base-UMD — see FIX TG.
            //
            // FIX S9 (#42429): On MMIO devices, base-UMD channels CAN be safely soft-reset.
            // PCIe-direct access means no ETH relay dependency for host→device reads.
            // The skip-soft-reset path (FIX M) leaves dirty L1/TXQ/MAC state from the prior
            // session, which is the root cause of channels sticking at STARTED (0xa0b0c0d0)
            // and REMOTE_HANDSHAKE_COMPLETE (0xa1b1c1d1) after the base-UMD transition.
            // Performing assert→deassert on MMIO base-UMD channels gives a clean slate before
            // the fabric firmware launch_msg.
            if (skip_soft_reset_channels.count(router_chan)) {
                if (!device->is_mmio_capable()) {
                    log_info(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} base-UMD relay firmware "
                        "(0x49706550) — skipping soft reset on non-MMIO (ETH relay needed) [FIX M #42429]",
                        chip_id,
                        router_chan);
                    continue;
                }
                // MMIO base-UMD: fall through to assert/deassert for clean state [FIX S9 #42429].
                log_info(
                    tt::LogMetal,
                    "configure_fabric_cores: device {} channel {} base-UMD relay firmware "
                    "(0x49706550) MMIO — performing soft reset to clear dirty L1/TXQ/MAC state "
                    "[FIX S9 #42429]",
                    chip_id,
                    router_chan);
            }

            try {
                // Test seam (Scenario W): if set, call the inject function instead of (or
                // before) the real assert_risc_reset_at_core().  If it throws, the catch
                // block below is exercised identically to a real UMD failure.
                if (s_configure_cores_inject_fn_) {
                    s_configure_cores_inject_fn_(device, router_chan);
                    // If the inject fn returned without throwing, skip the real hardware
                    // call — the seam fully replaces assert/deassert for this channel.
                    continue;
                }

                // get_virtual_eth_core_from_channel returns the virtual CoreCoord needed
                // for tt_cxy_pair, which is what assert/deassert_risc_reset_at_core expect.
                auto virtual_core = cluster.get_virtual_eth_core_from_channel(chip_id, router_chan);
                tt_cxy_pair core_loc(chip_id, virtual_core);

                // Assert ERISC0 (== BRISC) reset — halts only the main ERISC processor.
                // The subordinate ERISC continues running and maintains the ETH PHY link.
                // FIX OP (#42429): Time the full FIX S9 assert→deassert window.
                // The ERISC0 is halted for FIX EG (zero fw_launch_addr) + deassert + FIX DW
                // 50ms sleep + FIX DU poll.  If the window is unexpectedly long (> 200ms),
                // ETH PHY link retraining may occur — Strategy 11 should catch that, but
                // this timing log helps correlate link errors with reset duration.
                auto fix_s9_start = std::chrono::steady_clock::now();
                cluster.assert_risc_reset_at_core(core_loc, tt::umd::RiscType::ERISC0);

                // FIX EG (#42429): While ERISC0 is halted, zero fw_launch_addr (0x9004 =
                // LAUNCH_ERISC_APP_FLAG).
                //
                // Root cause of FIX EF timeout: after a fabric session ends, fw_launch_addr is
                // NOT cleared for fabric router channels (chan=8) — unlike dispatch channels
                // (FIX PF) and phase-2.5 force-reset channels (FIX PD).  On the next session,
                // FIX S9 asserts then deasserts ERISC0.  After deassert the ERISC boots from
                // ROM and sees the stale fw_launch_addr == 1.  The L1 clear only zeros 4 sync
                // addresses; the firmware CODE region from the previous session is still present
                // in L1.  ERISC dispatch firmware (active_erisc.cc) checks fw_launch_addr != 1
                // to decide whether to exit — with the stale value it instead begins executing
                // the old fabric binary still resident in L1.  That binary runs in a corrupt /
                // terminated state, crashes or hangs, and leaves ERISC0 unable to receive the
                // new write_launch_msg_to_core.  Result: edm_status_address stays at DEADB07E
                // indefinitely → FIX EF 500ms poll times out on all 4 MMIO devices
                // (CI run #26007922914).
                //
                // Fix: zero fw_launch_addr via PCIe while ERISC0 is halted (cannot self-modify
                // L1 while in reset).  On deassert, ERISC0 boots from ROM cleanly, sees
                // fw_launch_addr == 0, and does NOT prematurely launch any stale firmware.
                // Analogous to FIX PF (dispatch ETH channels) and FIX PD (phase-2.5 channels).
                {
                    const auto& hal_eg = tt::tt_metal::MetalContext::instance().hal();
                    const auto aeth_idx_eg = hal_eg.get_programmable_core_type_index(
                        tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
                    const uint32_t fw_launch_addr_eg =
                        hal_eg.get_jit_build_config(aeth_idx_eg, 0, 0).fw_launch_addr;
                    // FIX MN (#42429): Capture fw_launch_addr BEFORE zeroing it.
                    // If pre_val == 1, FIX EG caught genuinely stale state (prior session
                    // left launch flag set).  If pre_val == 0, the zero-write is harmless
                    // but confirms no stale state existed.  Any other value is anomalous.
                    std::vector<uint32_t> eg_pre_val(1, 0xFFFFFFFF);
                    cluster.read_core(eg_pre_val, sizeof(uint32_t),
                        tt_cxy_pair(chip_id, virtual_core),
                        static_cast<uint64_t>(fw_launch_addr_eg));
                    log_info(
                        tt::LogMetal,
                        "FIX MN (#42429): FIX EG pre-zero snapshot — Device {} chan={} "
                        "fw_launch_addr=0x{:08X} pre_val=0x{:08X} (1=stale, 0=clean)",
                        chip_id, router_chan, fw_launch_addr_eg, eg_pre_val[0]);
                    cluster.write_core_immediate(
                        chip_id, virtual_core, std::vector<uint32_t>{0}, fw_launch_addr_eg);
                    // FIX GI (#42429): Log + readback verify the FIX EG zero write.
                    // Without logging, a silent PCIe write failure leaves no trace in CI logs.
                    // Readback catches races where ERISC comes out of reset between write and
                    // deassert (should not happen since ERISC0 is still asserted, but belt+suspenders).
                    std::vector<uint32_t> eg_verify(1, 0xFFFFFFFF);
                    cluster.read_core(eg_verify, sizeof(uint32_t),
                        tt_cxy_pair(chip_id, virtual_core),
                        static_cast<uint64_t>(fw_launch_addr_eg));
                    if (eg_verify[0] != 0) {
                        log_warning(
                            tt::LogMetal,
                            "FIX GI (#42429): FIX EG readback MISMATCH — wrote 0 to "
                            "fw_launch_addr=0x{:08X} on Device {} chan={} but read back "
                            "0x{:08X}. ERISC may prematurely launch stale firmware.",
                            fw_launch_addr_eg, chip_id, router_chan, eg_verify[0]);
                    } else {
                        log_info(
                            tt::LogMetal,
                            "FIX EG (#42429): zeroed fw_launch_addr=0x{:08X} on Device {} "
                            "chan={} (readback verified=0x{:08X}) while ERISC0 halted",
                            fw_launch_addr_eg, chip_id, router_chan, eg_verify[0]);
                    }
                }

                // FIX SA (#42429): Strategy A — Deferred Deassert.
                // ERISC stays halted. All subsequent L1 writes (L1 clear loop,
                // ConfigureDeviceWithProgram, write_launch_msg_to_core) happen while ERISC
                // is in reset — no race with base-UMD execution.  device.cpp will:
                //   1. Write fw_launch_addr=1 while halted (FIX MM equivalent)
                //   2. Write launch_msg while halted (send_go=false)
                //   3. Write handshake_bypass=1 while halted (STRATEGY7 equivalent)
                //   4. Deassert ERISC reset
                //   5. FIX S8: Write BOOT_FENCE_READY to L1 (replaces FIX DW+DU+PQ)
                //   6. FIX S9: Write session_id to L1
                //   7. Write go_msg (RUN_MSG_GO)
                //
                // FIX OP: log how long assert has been held (deassert timing captured in device.cpp).
                {
                    auto fix_s9_end = std::chrono::steady_clock::now();
                    auto fix_s9_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        fix_s9_end - fix_s9_start).count();
                    log_info(
                        tt::LogMetal,
                        "FIX SA (#42429): Strategy A — Device {} chan={} ERISC0 asserted for "
                        "{}ms (FIX EG done). Deferring deassert to device.cpp after all L1 writes.",
                        chip_id, router_chan, fix_s9_ms);
                }
                deferred_deassert_channels.insert(router_chan);
                // Skip FIX DW, FIX DU, FIX MM — all done in device.cpp after L1 writes.
                // L1 clear loop below still runs (PCIe writes to halted ERISC L1 work fine).
            } catch (const std::exception& e) {
                // Fatal for this channel: remote chip is unreachable.
                // Skip L1 writes for this channel — WriteToDeviceL1 on a non-MMIO chip routes
                // through ethernet and will hang indefinitely on a dead channel.
                dead_channels.insert(router_chan);
                newly_dead_channels.insert(router_chan);
                log_warning(
                    tt::LogMetal,
                    "configure_fabric_cores: Failed ERISC0 soft reset on device {} channel {}: {}. "
                    "Skipping L1 clear for this channel — firmware will not start on it.",
                    chip_id,
                    router_chan,
                    e.what());
            } catch (...) {
                dead_channels.insert(router_chan);
                newly_dead_channels.insert(router_chan);
                log_warning(
                    tt::LogMetal,
                    "configure_fabric_cores: Failed ERISC0 soft reset on device {} channel {} "
                    "(unknown exception). Skipping L1 clear for this channel.",
                    chip_id,
                    router_chan);
            }
        }
    }

    // STRATEGY 11 (#42429): ETH link status check before firmware launch.
    // For MMIO devices, read the ETH link error status register (Wormhole 0x1440) for each
    // channel.  A non-zero value signals that the PHY link is not trained — fabric firmware
    // handshakes will never complete on that channel (eth_send_packet silently drops).
    // This can happen when a prior force-reset disrupted the PHY and the 100ms post-deassert
    // delay was insufficient for link retraining.
    // Non-MMIO devices: skip — reads would route through the ETH relay we just killed.
    // Wormhole ETH_LINK_ERR_STATUS_ADDR = 0x1440; error codes >= 11 mean "not connected".
    static constexpr uint32_t kEthLinkErrStatusAddr = 0x1440;  // tt::umd::wormhole::ETH_LINK_ERR_STATUS_ADDR
    static constexpr uint32_t kEthLinkErrCodeNotConnected = 11;  // tt::umd::wormhole::ETH_LINK_UNUSED_ERROR_CODE_RANGE_START
    if (device->is_mmio_capable()) {
        const auto chip_id = device->id();  // FIX CG: chip_id scope for Strategy 11 log statements
        for (const auto& [router_chan, _] : router_chans_and_direction) {
            if (dead_channels.count(router_chan)) {
                continue;  // Already dead — no point checking link status.
            }
            auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
            std::vector<uint32_t> eth_link_buf(1, 0);
            try {
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device, router_logical_core, kEthLinkErrStatusAddr, 4, eth_link_buf, CoreType::ETH);
                if (eth_link_buf[0] != 0) {
                    const bool not_connected = eth_link_buf[0] >= kEthLinkErrCodeNotConnected;
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} ETH link error status=0x{:08x} "
                        "({}) before firmware launch. Handshake will likely fail. [Strategy 11 #42429]",
                        chip_id,
                        router_chan,
                        eth_link_buf[0],
                        not_connected ? "link not connected/trained" : "link config error");
                } else {
                    log_debug(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} ETH link status OK (0x0) "
                        "[Strategy 11 #42429]",
                        chip_id,
                        router_chan);
                }
            } catch (const std::exception& e) {
                log_debug(
                    tt::LogMetal,
                    "configure_fabric_cores: device {} channel {} ETH link status read failed: {} "
                    "[Strategy 11 #42429]",
                    chip_id,
                    router_chan,
                    e.what());
            }
        }
    }

    for (const auto& [router_chan, _] : router_chans_and_direction) {
        if (dead_channels.count(router_chan)) {
            // Skip L1 clear for dead channels — WriteToDeviceL1 on non-MMIO chips routes
            // writes through ethernet and will hang if the channel is unreachable.
            //
            // Cascade prevention (#42429): for corrupt-but-reachable channels (probe read
            // succeeded in terminate_stale_erisc_routers, but status was garbage),
            // terminate_stale_erisc_routers now zeroes edm_status_address before adding
            // the channel to probe_dead_channels.  This means the NEXT session sees
            // edm_status=0 ("clean") and does not add the channel to pre_known_dead_channels,
            // so this L1 clear loop will process it normally — breaking the infinite cascade
            // where corrupt status persisted across container restarts on bare metal.
            continue;
        }
        // FIX TG (#42429): For NON-MMIO base-UMD relay channels, preserve edm_status_address
        // (0x49706550) so the next session's terminate_stale_erisc_routers() can identify
        // base-UMD state and fire FIX M (launch_msg transition).
        //
        // FIX TG2 (#42429): PARTIAL L1 clear for non-MMIO — zero all sync-critical addresses
        // EXCEPT edm_status_address.  Original FIX TG skipped ALL clears, but that left stale
        // edm_local_sync_address / edm_local_tensix_sync_address / termination_signal_address
        // from a previous failed ring-sync session (stuck at REMOTE_HANDSHAKE_COMPLETE
        // 0xa1b1c1d1).  After tt-smi -r the ERISC restarts into base-UMD (writes 0x49706550
        // back to edm_status_address) but does NOT reset the sync addresses.  The new
        // session's fabric firmware then boots, encounters the stale handshake state, and
        // stalls at REMOTE_HANDSHAKE_COMPLETE again — causing the same 120s ring-sync
        // timeout across multiple smi-reset cycles (FIX UP2 INFRA_ERROR pattern observed
        // on runs 25293661493 + 25294660215 on t3k-08/t3k-05 respectively).
        //
        // FIX S9 (#42429): For MMIO base-UMD channels, we already performed a full soft-reset
        // above (assert→deassert).  That gives a clean slate — do a FULL L1 clear (including
        // edm_status_address) to ensure no stale state survives into the fabric FW launch.
        // edm_status_address will be 0x49706550 briefly (base-UMD writes it back after
        // deassert), but the fabric firmware launch_msg transitions it to fabric state before
        // the next session reads it.
        if (skip_soft_reset_channels.count(router_chan)) {
            auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
            if (!device->is_mmio_capable()) {
                // Non-MMIO: partial clear (preserve 0x49706550 for base-UMD detection).
                for (const auto& address : addresses_to_clear) {
                    if (address == router_config.edm_status_address) {
                        log_debug(
                            tt::LogMetal,
                            "configure_fabric_cores: device {} channel {} base-UMD non-MMIO — "
                            "preserving edm_status_address (0x49706550 sentinel) [FIX TG #42429]",
                            device->id(),
                            router_chan);
                        continue;  // Preserve 0x49706550 sentinel for next-session base-UMD detection
                    }
                    log_debug(
                        tt::LogMetal,
                        "configure_fabric_cores: device {} channel {} base-UMD non-MMIO — "
                        "clearing sync address 0x{:08x} [FIX TG2 #42429]",
                        device->id(),
                        router_chan,
                        address);
                    tt::tt_metal::detail::WriteToDeviceL1(
                        device, router_logical_core, address, router_zero_buf, CoreType::ETH);
                }
                continue;
            }
            // MMIO base-UMD (FIX S9): soft-reset already performed above — full L1 clear.
            log_debug(
                tt::LogMetal,
                "configure_fabric_cores: device {} channel {} base-UMD MMIO — full L1 clear "
                "after soft reset [FIX S9 #42429]",
                device->id(),
                router_chan);
            // Fall through to the full L1 clear below.
        }
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }

        // FIX GH (#42429): fw_launch_addr restore moved to configure_fabric() in device.cpp.
        //
        // PREVIOUS PLACEMENT BUG: FIX GH was written here (immediately after L1 clear) which
        // fires BEFORE the new fabric firmware binary is loaded into L1 by ConfigureDeviceWithProgram
        // and BEFORE write_launch_msg_to_core writes the launch message.  When base-UMD sees
        // fw_launch_addr=1 at this point, it tries to launch from the LAUNCH address in L1,
        // but that region still contains stale or zeroed content — ERISC crashes or hangs,
        // leaving it stuck at 0xdeadb07e forever.
        //
        // FIX IJ: fw_launch_addr_value is now written in configure_fabric() in device.cpp,
        // after write_launch_msg_to_core, for MMIO skip_soft_reset ETH channels.  At that
        // point the firmware binary AND launch message are both in L1, so base-UMD can safely
        // act on the flag.
        //
        // FIX MM (#42429): fw_launch_addr restore is now unconditionally applied below
        // (after this loop) for ALL surviving MMIO channels that had FIX EG/EG RR run.
        // FIX IJ/KL in device.cpp are disabled (#ifdef FIXIJ_REDUNDANT_AFTER_FIX_MM).
    }

    // STRATEGY7 (#42429): Write handshake_bypass=1 to each ERISC's L1 handshake_info.
    // FIX S7-MOVE (#42429): For deferred_deassert channels (Strategy A), the bypass write
    // moves to device.cpp — AFTER ConfigureDeviceWithProgram, while ERISC is still halted.
    // This prevents ConfigureDeviceWithProgram's BSS zeroing from overwriting the flag.
    // Non-deferred channels (non-MMIO) still get the write here.
    {
        const uint32_t handshake_bypass_offset = 32;  // offsetof(handshake_info_t, handshake_bypass)
        const uint32_t handshake_bypass_l1_addr =
            static_cast<uint32_t>(router_config.handshake_addr) + handshake_bypass_offset;
        const uint32_t chip_id_s7 = device->id();

        for (const auto& [router_chan_s7, _] : router_chans_and_direction) {
            if (dead_channels.count(router_chan_s7)) {
                continue;  // Dead channels — no firmware will run.
            }
            if (deferred_deassert_channels.count(router_chan_s7)) {
                // FIX S7-MOVE: bypass write handled in device.cpp after ConfigureDeviceWithProgram.
                log_debug(
                    tt::LogMetal,
                    "STRATEGY7 (#42429): Device {} chan={} — deferred (Strategy A), "
                    "handshake_bypass write deferred to device.cpp [FIX S7-MOVE]",
                    chip_id_s7, router_chan_s7);
                continue;
            }
            auto router_logical_core_s7 = soc_desc.get_eth_core_for_channel(router_chan_s7, CoordSystem::LOGICAL);
            std::vector<uint32_t> bypass_buf = {1};
            tt::tt_metal::detail::WriteToDeviceL1(
                device, router_logical_core_s7, handshake_bypass_l1_addr, bypass_buf, CoreType::ETH);

            // FIX NO (#42429): Readback verify the handshake_bypass write.
            std::vector<uint32_t> bypass_verify(1, 0xFFFFFFFF);
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, router_logical_core_s7, handshake_bypass_l1_addr, sizeof(uint32_t), bypass_verify, CoreType::ETH);
            if (bypass_verify[0] != 1) {
                log_warning(
                    tt::LogMetal,
                    "FIX NO (#42429): STRATEGY7 handshake_bypass readback MISMATCH — wrote 1 to "
                    "L1[0x{:08X}] on Device {} chan={} but read back 0x{:08X}. "
                    "Firmware may enter full handshake loop — race conditions NOT eliminated.",
                    handshake_bypass_l1_addr, chip_id_s7, router_chan_s7, bypass_verify[0]);
            } else {
                log_info(
                    tt::LogMetal,
                    "STRATEGY7 (#42429): wrote handshake_bypass=1 at L1[0x{:08X}] for Device {} chan={} "
                    "(handshake_addr=0x{:08X} + offset={}) — FIX NO readback verified",
                    handshake_bypass_l1_addr, chip_id_s7, router_chan_s7,
                    static_cast<uint32_t>(router_config.handshake_addr), handshake_bypass_offset);
            }
        }
    }

    // FIX MM (#42429): fw_launch_addr restore — MOVED to device.cpp for deferred channels.
    // Strategy A (FIX SA) keeps ERISC halted through all L1 writes. For deferred channels,
    // fw_launch_addr restore happens in device.cpp while ERISC is still halted, which is
    // the correct sequence: FIX EG zero → L1 clear → ConfigureDeviceWithProgram → FIX MM
    // restore → write_launch_msg → handshake_bypass → deassert → FIX DW → FIX DU → go_msg.
    //
    // On MMIO devices, ALL channels go through FIX S9 and are deferred — FIX MM is a no-op here.
    // On non-MMIO devices, FIX EG was never run, so FIX MM is also not needed.
    // Kept as a comment block for historical context; the actual restore is in device.cpp.

    // FIX RR (#42429): re-evaluate all_channels_healthy after FIX RR may have recovered
    // pre_known_dead channels.  If all pre-known dead channels were recovered (dead_channels
    // is now empty) and no new failures occurred, all channels are healthy.
    const bool all_channels_healthy = dead_channels.empty() && newly_dead_channels.empty();

    // FIX MN (#42429): Summary log — captures total channel count, dead count, recovered count,
    // and skip-soft-reset count for post-mortem analysis.  Without this, CI logs only show
    // individual channel decisions; there is no single line that summarizes the outcome.
    //
    // FIX OP (#42429): Added mmio flag and path breakdown so post-mortem analysis can
    // immediately tell which path each channel class took (FIX M / FIX S9 / normal).
    // Without this, you must grep per-channel logs to reconstruct the path — error-prone
    // when analyzing logs from 8+ devices with 2+ channels each.
    const bool is_mmio = device->is_mmio_capable();
    // Count channels that went through each path:
    // - fix_m_count: non-MMIO base-UMD channels that skipped soft reset (FIX M)
    // - fix_s9_count: MMIO base-UMD channels that did assert/deassert (FIX S9)
    // - normal_reset_count: channels that aren't in skip_soft_reset_channels
    uint32_t fix_m_count = 0, fix_s9_count = 0, normal_reset_count = 0;
    for (const auto& [rc, _] : router_chans_and_direction) {
        if (dead_channels.count(rc)) continue;  // don't double-count dead
        if (skip_soft_reset_channels.count(rc)) {
            if (is_mmio) {
                ++fix_s9_count;
            } else {
                ++fix_m_count;
            }
        } else {
            ++normal_reset_count;
        }
    }
    log_info(
        tt::LogMetal,
        "FIX MN (#42429): configure_fabric_cores SUMMARY — Device {} mmio={} "
        "total_active={} dead={} newly_dead={} recovered={} "
        "skip_soft_reset={} pre_known_dead={} healthy={} "
        "paths: fix_m={} fix_s9={} normal={}",
        device->id(),
        is_mmio,
        router_chans_and_direction.size(),
        dead_channels.size(),
        newly_dead_channels.size(),
        recovered_channels.size(),
        skip_soft_reset_channels.size(),
        pre_known_dead_channels.size(),
        all_channels_healthy ? "true" : "false",
        fix_m_count, fix_s9_count, normal_reset_count);

    return FabricCoresHealth{all_channels_healthy, std::move(newly_dead_channels), std::move(recovered_channels), std::move(deferred_deassert_channels)};
}

}  // namespace tt::tt_fabric
