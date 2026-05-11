// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_firmware_initializer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <set>
#include <thread>

#include <enchantum/enchantum.hpp>
#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "impl/context/context_descriptor.hpp"
#include "core_coord.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "allocator/l1_banking_allocator.hpp"
#include "debug/noc_logging.hpp"
#include "dispatch/dispatch_core_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/topology.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/llrt.hpp"
#include "common/executor.hpp"
#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include "fabric/fabric_builder_context.hpp"
#include "fabric/fabric_context.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_align.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

RiscFirmwareInitializer::RiscFirmwareInitializer(
    std::shared_ptr<const ContextDescriptor> descriptor,
    const GetControlPlaneFn& get_control_plane,
    dispatch_core_manager& dispatch_core_manager) :
    FirmwareInitializer(std::move(descriptor)),
    get_control_plane_(get_control_plane),
    dispatch_core_manager_(dispatch_core_manager),
    num_hw_cqs_(static_cast<uint8_t>(descriptor_->num_cqs())) {
    const Hal& hal = descriptor_->hal();
    size_t worker_l1_size = descriptor_->worker_l1_size();
    std::uint32_t max_alignment = std::max(hal.get_alignment(HalMemType::DRAM), hal.get_alignment(HalMemType::L1));
    worker_l1_unreserved_start_ = tt::align(
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size,
        max_alignment);
}

RiscFirmwareInitializer::~RiscFirmwareInitializer() = default;

void RiscFirmwareInitializer::init(
    const std::vector<Device*>& /*devices*/, const std::unordered_set<InitializerKey>& /*init_done*/) {
    TT_THROW(
        "RiscFirmwareInitializer::init is not implemented. Use run_async_build_phase and run_launch_phase instead.");
}

void RiscFirmwareInitializer::run_async_build_phase(const std::set<tt::ChipId>& device_ids) {
    ZoneScopedN("FW builds and Device Inits");

    std::vector<std::shared_future<void>> futures;
    futures.reserve(device_ids.size());

    // Reserve tables per device ID (single-threaded)
    dram_bank_offset_map_.reserve(device_ids.size());
    l1_bank_offset_map_.reserve(device_ids.size());
    dram_bank_to_noc_xy_.reserve(device_ids.size());
    l1_bank_to_noc_xy_.reserve(device_ids.size());
    worker_logical_col_to_virtual_col_.reserve(device_ids.size());
    worker_logical_row_to_virtual_row_.reserve(device_ids.size());
    for (tt::ChipId device_id : device_ids) {
        dram_bank_offset_map_[device_id].reserve(num_hw_cqs_);
        l1_bank_offset_map_[device_id].reserve(num_hw_cqs_);
        dram_bank_to_noc_xy_[device_id].reserve(num_hw_cqs_);
        l1_bank_to_noc_xy_[device_id].reserve(num_hw_cqs_);
        worker_logical_col_to_virtual_col_[device_id].reserve(num_hw_cqs_);
        worker_logical_row_to_virtual_row_[device_id].reserve(num_hw_cqs_);
    }

    // Set pointers to directly access the tables for each device as
    // multi threaded unordered_map access is not thread-safe
    struct PerDeviceTableRefs {
        tt::ChipId device_id;
        std::vector<int32_t>* dram_bank_offset_map;
        std::vector<int32_t>* l1_bank_offset_map;
        std::vector<uint16_t>* dram_bank_to_noc_xy;
        std::vector<uint16_t>* l1_bank_to_noc_xy;
        std::vector<uint8_t>* worker_logical_col_to_virtual_col;
        std::vector<uint8_t>* worker_logical_row_to_virtual_row;
    };
    std::vector<PerDeviceTableRefs> table_refs;
    table_refs.reserve(device_ids.size());
    for (tt::ChipId device_id : device_ids) {
        table_refs.push_back(
            {device_id,
             &dram_bank_offset_map_[device_id],
             &l1_bank_offset_map_[device_id],
             &dram_bank_to_noc_xy_[device_id],
             &l1_bank_to_noc_xy_[device_id],
             &worker_logical_col_to_virtual_col_[device_id],
             &worker_logical_row_to_virtual_row_[device_id]});
    }

    // FIX NV (#42429): Capture MMIO device IDs before spawning async tasks.
    // get_device_aiclk() on non-MMIO (remote) chips calls RemoteChip::get_clock() →
    // WormholeTTDevice::get_clock() → WormholeArcMessenger::send_message() →
    // wait_for_non_mmio_flush(). When the MMIO relay is dead (stale firmware from a
    // prior session), this blocks for 5 seconds per chip, multiplying across all remote
    // chips in the cluster. The aiclk value is debug-only ([[maybe_unused]]) — skip it
    // for non-MMIO chips to avoid relay-dependent hangs during init.
    const std::set<tt::ChipId> mmio_ids_set = cluster_.mmio_chip_ids();

    for (auto refs : table_refs) {
        futures.emplace_back(detail::async([this, refs, mmio_ids_set]() {
            tt::ChipId device_id = refs.device_id;
            // Clear L1/DRAM if requested - skip for mock devices (no memory), but do for emulated (memory-backed)
            if (cluster_.get_target_device_type() != tt::TargetDevice::Mock) {
                if (rtoptions_.get_clear_l1()) {
                    clear_l1_state(device_id);
                }
                if (rtoptions_.get_clear_dram()) {
                    clear_dram_state(device_id);
                }
            }
            // FIX NV (#42429): Skip get_device_aiclk for non-MMIO (remote) chips.
            // The ARC messenger path requires wait_for_non_mmio_flush() which hangs 5s
            // per chip on a dead relay. The aiclk value is debug-only — not worth the
            // risk of relay-timeout delays multiplying across all remote chips.
            if (mmio_ids_set.count(device_id)) {
                [[maybe_unused]] int ai_clk = cluster_.get_device_aiclk(device_id);
                log_debug(tt::LogMetal, "AI CLK for device {} is:   {} MHz", device_id, ai_clk);
            }
            generate_device_bank_to_noc_tables(
                device_id,
                *refs.dram_bank_offset_map,
                *refs.l1_bank_offset_map,
                *refs.dram_bank_to_noc_xy,
                *refs.l1_bank_to_noc_xy);
            generate_worker_logical_to_virtual_map(
                device_id, *refs.worker_logical_col_to_virtual_col, *refs.worker_logical_row_to_virtual_row);

            // Skip build env registration and firmware building for mock/emulated devices
            if (!cluster_.is_mock_or_emulated()) {
                BuildEnvManager::get_instance().add_build_env(device_id, num_hw_cqs_);
                // build_firmware ensures that the FW is built only once for a given build key
                // (which captures the fw_compile_hash).
                BuildEnvManager::get_instance().build_firmware(device_id);
                // Clear the entire launch message ring buffer on ethernet cores before application firmware is
                // activated. This is required since ethernet cores context switch between application and routing
                // firmware. If ERISC application firmware is activated before the launch messages are cleared, it
                // can enter an undefined state by reading a corrupted launch message. Routing firmware will never
                // run in this case, causing UMD issued transactions to hang.
                //
                // FIX NW (#42429): Skip clear_launch_messages_on_eth_cores for non-MMIO (remote) chips.
                // clear_launch_messages_on_eth_cores calls cluster_.write_core() and cluster_.l1_barrier()
                // which both go through RemoteProtocol → write_to_non_mmio / wait_for_non_mmio_flush.
                // When the MMIO relay is dead (stale FABRIC firmware left by a prior session), those
                // calls time out (5s each) and throw, propagating through the async future up to SetUp()
                // and failing the test fixture — same root cause as FIX NV for get_device_aiclk.
                // Skipping non-MMIO chips here is safe: run_launch_phase calls
                // terminate_active_ethernet_cores_on_all_chips() which unconditionally resets all ETH
                // cores on every chip (MMIO and non-MMIO) before firmware is launched. Any stale launch
                // messages on non-MMIO ETH cores are cleared by that full ETH-core reset.
                if (mmio_ids_set.count(device_id)) {
                    clear_launch_messages_on_eth_cores(device_id);
                }
            }
        }));
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

void RiscFirmwareInitializer::run_launch_phase(const std::set<tt::ChipId>& device_ids) {
    // Launch FW on each device sequentially, since a multithreaded launch leads to initialization hangs.
    // See https://github.com/tenstorrent/tt-metal/issues/35701
    ZoneScopedN("Resets and FW Launch");
    if (!cluster_.is_mock_or_emulated()) {
        terminate_active_ethernet_cores_on_all_chips();

        const auto mmio_ids_set = cluster_.mmio_chip_ids();
        for (tt::ChipId device_id : device_ids) {
            ClearNocData(descriptor_->env_impl(), device_id);
            reset_cores(device_id);
            // FIX NZ (#42429): skip initialize_and_launch_firmware for non-MMIO devices with a
            // known-broken relay.  initialize_and_launch_firmware calls write_core (guarded by
            // FIX NY) and wait_until_cores_done, which polls worker cores via read_core.  For a
            // remote chip with a dead relay, each read_core blocks for 5 s; a full 64-core tensix
            // grid = 64 × 5 s = 320 s before the GHA 5-minute action timeout fires.  FIX NZ
            // (read_core throw) is a belt-and-suspenders guard, but skipping here is cleaner:
            // the fabric firmware initializer will handle the dead relay cleanly on the next
            // run (terminate_stale_erisc_routers rebuilds state from scratch).
            if (!mmio_ids_set.count(device_id) && cluster_.is_relay_broken(device_id)) {
                log_warning(
                    tt::LogAlways,
                    "run_launch_phase: FIX NZ — skipping initialize_and_launch_firmware for "
                    "non-MMIO device {} (relay broken). Firmware will be loaded on next clean init. (#42429)",
                    device_id);
                continue;
            }
            // FIX BX (#42429): belt-and-suspenders over FIX NZ. If the relay was NOT yet broken
            // at the FIX NZ check above but becomes broken DURING initialize_and_launch_firmware
            // (e.g. write_to_non_mmio times out → FIX BW marks relay_broken_chips_ and re-throws),
            // catch that exception here for non-MMIO devices and continue without crashing.
            // Root cause: write_core_immediate's FIX AE previously only marked relay broken at
            // the UMD driver level, not in relay_broken_chips_, so FIX NZ never saw the broken
            // state. FIX BW fixes the gap; FIX BX is the outer safety net for any remaining
            // window between the FIX NZ pre-check and the first write in initialize_and_launch_firmware.
            if (!mmio_ids_set.count(device_id)) {
                try {
                    initialize_and_launch_firmware(device_id);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogAlways,
                        "run_launch_phase: FIX BX — initialize_and_launch_firmware threw for "
                        "non-MMIO device {} (relay likely broken mid-init): {}. "
                        "Skipping — fabric init will handle on next session. (#42429)",
                        device_id,
                        e.what());
                }
            } else {
                initialize_and_launch_firmware(device_id);
            }
        }

        // FIX TV (#42429): Wait for MMIO ETH channels to complete rebooting to base-UMD
        // firmware after reset_cores() may have PCIe-force-reset them.
        //
        // Root cause of probe_dead regression (run 25295860739):
        //   Session N left MMIO ETH channels running FABRIC firmware (degraded-cluster
        //   fast teardown didn't send TERMINATE).  Session N+1's reset_cores() detected
        //   them as still-running, sent exit signals, got no response, and PCIe-force-reset
        //   them (assert+deassert risc_reset).  The channels enter hardware reset and begin
        //   rebooting (~1-2s to base-UMD).  Session N+1's fabric init immediately called
        //   terminate_stale_erisc_routers(), which probes MMIO channels via ReadFromDeviceL1
        //   (ETH command-queue protocol, requires ERISC to be running to service reads).
        //   Since the ERISCs are still mid-reboot, the command-queue read times out →
        //   probe_dead on MMIO devices.  probe_dead on MMIO cascades to relay_timeout on
        //   non-MMIO (relay path not yet established) → all ETH channels dead.
        //
        // Fix: poll MMIO ETH heartbeat after the reset_cores() loop.  If channels are
        // already running base-UMD (no force-reset happened), they respond in <1 poll
        // interval (~10ms, negligible overhead).  If they were force-reset, we wait up to
        // kMmioEthRebootMs for the heartbeat to become non-zero and start incrementing.
        if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC) &&
            hal_.get_eth_fw_is_cooperative() && get_control_plane_) {
            const uint32_t hb_addr = hal_.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
            if (hb_addr == 0u) {
                log_debug(
                    tt::LogAlways,
                    "run_launch_phase: FIX TV — ETH heartbeat address not wired for this arch; "
                    "skipping MMIO ETH reboot wait. (#42429)");
            } else {
                struct EthRebootPollState {
                    tt_cxy_pair target;
                    uint32_t prev_hb = 0;
                    bool nonzero_seen = false;
                    bool ready = false;
                };
                std::vector<EthRebootPollState> poll_states;
                for (const tt::ChipId mmio_id : mmio_ids_set) {
                    for (const auto& logical_core :
                         this->get_control_plane_().get_active_ethernet_cores(mmio_id)) {
                        CoreCoord virt = cluster_.get_virtual_coordinate_from_logical_coordinates(
                            mmio_id, logical_core, CoreType::ETH);
                        poll_states.push_back({tt_cxy_pair(mmio_id, virt), 0, false, false});
                    }
                }
                if (!poll_states.empty()) {
                    constexpr int kMmioEthRebootMs = 3000;
                    constexpr auto kPollInterval = std::chrono::milliseconds(10);
                    const auto tv_start = std::chrono::steady_clock::now();
                    while (true) {
                        bool all_done = true;
                        for (auto& ps : poll_states) {
                            if (ps.ready) continue;
                            uint32_t hb_val = 0;
                            try {
                                cluster_.read_reg(&hb_val, ps.target, hb_addr);
                            } catch (...) {
                                ps.ready = true;  // PCIe read failed — count as done
                                continue;
                            }
                            if (!ps.nonzero_seen) {
                                if (hb_val != 0) {
                                    ps.prev_hb = hb_val;
                                    ps.nonzero_seen = true;
                                    // FIX TW (#42429): UMD base firmware writes a static 0xABCDxxxx
                                    // marker to the heartbeat register — it never increments.
                                    // Detect it immediately rather than waiting for a value change.
                                    if ((hb_val >> 16) == 0xABCDu) {
                                        ps.ready = true;
                                    }
                                }
                            } else if ((hb_val >> 16) == 0xABCDu || hb_val != ps.prev_hb) {
                                // Ready if UMD static marker OR incrementing counter detected.
                                ps.ready = true;
                            }
                            if (!ps.ready) all_done = false;
                        }
                        if (all_done) break;
                        const auto elapsed_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - tv_start)
                                .count();
                        if (elapsed_ms >= kMmioEthRebootMs) {
                            const auto not_ready = static_cast<int>(std::count_if(
                                poll_states.begin(),
                                poll_states.end(),
                                [](const EthRebootPollState& ps) { return !ps.ready; }));
                            log_warning(
                                tt::LogAlways,
                                "run_launch_phase: FIX TV — MMIO ETH heartbeat poll timed out after {}ms; "
                                "{}/{} channel(s) not yet reporting base firmware. "
                                "terminate_stale_erisc_routers may see probe_dead on these channels. (#42429)",
                                elapsed_ms,
                                not_ready,
                                static_cast<int>(poll_states.size()));
                            break;
                        }
                        std::this_thread::sleep_for(kPollInterval);
                    }
                    const auto tv_elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - tv_start)
                            .count();
                    const auto ready_count = static_cast<int>(std::count_if(
                        poll_states.begin(),
                        poll_states.end(),
                        [](const EthRebootPollState& ps) { return ps.ready; }));
                    if (ready_count == static_cast<int>(poll_states.size())) {
                        log_info(
                            tt::LogAlways,
                            "run_launch_phase: FIX TV — all {} MMIO ETH channel(s) confirmed base "
                            "firmware heartbeat in {}ms. (#42429)",
                            poll_states.size(),
                            tv_elapsed_ms);
                    }
                }
            }
        }
    }
    initialized_ = true;
}

void RiscFirmwareInitializer::configure() {}

void RiscFirmwareInitializer::teardown_simulator_ethernet_cores() {
    // If simulator is enabled, force a teardown of active ethernet cores for WH
    if (rtoptions_.get_simulator_enabled()) {
        if (hal_.get_eth_fw_is_cooperative()) {
            auto all_devices = cluster_.all_chip_ids();
            for (tt::ChipId device_id : all_devices) {
                for (const auto& logical_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
                    CoreCoord virtual_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, CoreType::ETH);
                    erisc_send_exit_signal(device_id, virtual_core, false);
                    while (erisc_app_still_running(device_id, virtual_core)) {
                    }
                }
            }
        }
    }
}

void RiscFirmwareInitializer::teardown(std::unordered_set<InitializerKey>& /*init_done*/) {
    auto all_devices = cluster_.all_chip_ids();

    teardown_simulator_ethernet_cores();

    if (!cluster_.is_mock_or_emulated()) {
        // FIX AC (#42429 follow-up): Two-phase ETH reset to fix "next binary hangs in
        // init_tt_device" after a quiesce failure.
        //
        // Root cause of prior bug (FIX AB): teardown first ran assert_cores/l1_barrier
        // on ALL devices — each timed out for 5s per relay-broken non-MMIO device (~20s
        // total waste).  Then FIX AB hard-reset MMIO ETH channels, but only 2ms before
        // ~Cluster destroyed the driver — not enough time for ERISCs to reboot.
        // Non-MMIO ETH channels were never reset.  The next binary's init_tt_device()
        // tried to read non-MMIO chip memory through MMIO relay channels still running
        // fabric firmware, causing indefinite hangs.
        //
        // New teardown order:
        //   Step 1: Detect relay_broken_non_mmio and any_teardown_timed_out.
        //   Step 2: If relay broken — hard-reset MMIO ETH channels via PCIe (no relay
        //           needed) then poll each channel's heartbeat address until (val >> 16) == 0xABCD
        //           (UMD base firmware running), with 1s timeout per core.
        //   Step 3: assert_cores/l1_barrier loop — skip ALL non-MMIO devices when any
        //           relay is broken (all share the same MMIO relay path; even devices
        //           not in relay_broken_non_mmio are unreachable and would timeout).
        //   Step 4: set_internal_routing_info_for_ethernet_cores — skip when relay
        //           broken (relay write, would timeout).
        //   Step 5: Hard-reset MMIO ETH channels via PCIe when teardown timed out but
        //           relay wasn't broken (no step 2 in that case).
        //           Non-MMIO ETH channels with stale firmware are NOT reset here —
        //           relay-based resets are unreliable (UMD relay protocol state is not
        //           restored simply by rebooting MMIO ERISCs).  Stale non-MMIO ERISCs
        //           are safely cleaned up by terminate_stale_erisc_routers on next init.

        // Step 1: scan device flags.
        std::unordered_set<tt::ChipId> relay_broken_non_mmio;
        bool any_teardown_timed_out = false;
        const auto mmio_ids_set = cluster_.mmio_chip_ids();

        if (descriptor_->metal_context().is_device_manager_initialized() && get_control_plane_) {
            auto& dm = descriptor_->metal_context().device_manager();
            for (tt::ChipId device_id : cluster_.all_chip_ids()) {
                // Use get_device() (not get_active_device()): close() has already been
                // called by DeviceManager::close_devices(), but fabric_relay_path_broken_
                // and fabric_teardown_timed_out_ are still valid in memory.
                const Device* dev = dm->get_device(device_id);
                if (!dev) {
                    continue;
                }
                if (!mmio_ids_set.count(device_id) && dev->is_fabric_relay_path_broken()) {
                    relay_broken_non_mmio.insert(device_id);
                }
                // FIX BA (#42429): Also include non-MMIO devices where FIX AM fired
                // (fabric_channels_not_ready_for_traffic_ set, fabric_relay_path_broken_ NOT set).
                //
                // Root cause of run 25066686656 failure:
                //   Phase 5 for non-MMIO devices 4 and 5 saw master chan stuck at STARTED after
                //   3001ms (FIX AL early-exit).  FIX AM fired: set
                //   fabric_channels_not_ready_for_traffic_=true but did NOT set
                //   fabric_relay_path_broken_ (STARTED means ERISC firmware is running — just the
                //   out-of-mesh ETH handshake partner isn't responding).
                //
                //   Because fabric_relay_path_broken_ was false, Step 1 here did NOT add devices
                //   4/5 to relay_broken_non_mmio.  Consequently FIX AC (MMIO ETH PCIe reset) and
                //   FIX AY (deferred non-MMIO ERISC reset via restored relay) did NOT run for
                //   them.  The non-MMIO ERISCs remained running FABRIC firmware in STARTED state.
                //
                //   When the next process started (t3k_ttnn_tests), TopologyDiscovery::
                //   discover_remote_devices() called create_remote_device() → init_tt_device() →
                //   read_from_arc_apb() → read_non_mmio().  The FABRIC-mode ERISC on the non-MMIO
                //   device does not respond to UMD relay reads → 5s timeout per device → every
                //   test fixture constructor threw → all 359 tests failed.
                //
                // Fix: treat non-MMIO devices with fabric_channels_not_ready_for_traffic_=true as
                // relay-broken for teardown purposes.  This forces FIX AC (MMIO ETH PCIe reset)
                // and FIX AY (deferred non-MMIO ETH reset via restored relay), cleaning up the
                // STARTED-state ERISCs before the process exits.  FIX AV and FIX AW handle the
                // case where relay re-sync fails within the current process.
                //
                // FIX TK (#42429): do NOT trigger relay_broken_non_mmio when channels_not_ready was
                // set due to ring sync timeout (FIX TI path, detected by is_fabric_ring_sync_timed_out()).
                // In the FIX TI case the ETH channels are mid-transition from base-UMD firmware via
                // launch_msg (FIX M) — they are stuck at REMOTE_HANDSHAKE_COMPLETE.  Triggering FIX AC
                // (PCIe reset of MMIO ETH channels) in this state causes ALL MMIO ETH heartbeats to
                // time out (5s × 24 cores), resulting in the machine having only 4/8 chips visible
                // when the topology check runs.  Skip this path; the runner's per-job tt-smi reset
                // will recover the hardware before the next run.
                if (!mmio_ids_set.count(device_id) && dev->is_fabric_channels_not_ready_for_traffic() &&
                    !relay_broken_non_mmio.count(device_id) && !dev->is_fabric_ring_sync_timed_out()) {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX BA — non-MMIO device {} has fabric_channels_not_ready_for_traffic "
                        "(FIX AM STARTED early-exit) but relay not marked broken. Adding to "
                        "relay_broken_non_mmio to trigger FIX AC + FIX AY cleanup. (#42429)",
                        device_id);
                    relay_broken_non_mmio.insert(device_id);
                } else if (
                    !mmio_ids_set.count(device_id) && dev->is_fabric_channels_not_ready_for_traffic() &&
                    !relay_broken_non_mmio.count(device_id) && dev->is_fabric_ring_sync_timed_out()) {
                    // FIX TK guard: log that FIX BA was skipped for this device because the
                    // channels_not_ready state came from a ring sync timeout (FIX TI path),
                    // not the FIX AM STARTED-state path.  Without this log, FIX BA being skipped
                    // is invisible — the device has channels_not_ready but no FIX BA entry.
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX TK — non-MMIO device {} has fabric_channels_not_ready_for_traffic "
                        "but fabric_ring_sync_timed_out is set (FIX TI path). Skipping FIX BA "
                        "relay_broken_non_mmio — channels are mid-transition from base-UMD, not "
                        "STARTED-state. Runner tt-smi reset will recover. (#42429)",
                        device_id);
                }
                if (dev->is_fabric_teardown_timed_out()) {
                    any_teardown_timed_out = true;
                }
            }
        }

        // Step 2: Early MMIO ETH reset when relay is broken.
        // Must run BEFORE assert_cores/l1_barrier; we poll for ERISC reboot completion.
        // FIX TG (#42429): guard all get_control_plane_() calls in teardown against lazy
        // re-initialization.  GetControlPlaneFn is bound to MetalEnvImpl::get_control_plane()
        // which initializes control_plane_ on first call.  When set_default_fabric_topology()
        // resets control_plane_ to null (e.g. in CustomMeshGraphFabric2DFixture::TearDown),
        // calling get_control_plane_() here would re-run topology discovery on degraded hardware
        // and throw unordered_map::at — crashing the process during global GTest teardown.
        // is_control_plane_initialized() checks control_plane_ != nullptr without initializing.
        const bool cp_ready = get_control_plane_ && descriptor_->env_impl().is_control_plane_initialized();

        if (!relay_broken_non_mmio.empty() && cp_ready) {
            log_warning(
                tt::LogAlways,
                "teardown: FIX AC — {} non-MMIO device(s) have fabric_relay_path_broken. "
                "Hard-resetting MMIO ETH channels via PCIe BEFORE assert_cores loop "
                "to restore UMD relay firmware and avoid 5s-per-device relay timeouts.",
                relay_broken_non_mmio.size());
            for (const tt::ChipId mmio_id : mmio_ids_set) {
                for (const auto& logical_core :
                     this->get_control_plane_().get_active_ethernet_cores(mmio_id)) {
                    CoreCoord virtual_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        mmio_id, logical_core, CoreType::ETH);
                    try {
                        // PCIe-direct for MMIO — safe even with broken relay.
                        cluster_.assert_risc_reset_at_core(
                            tt_cxy_pair(mmio_id, virtual_core), tt::umd::RiscType::ALL);
                        cluster_.deassert_risc_reset_at_core(
                            tt_cxy_pair(mmio_id, virtual_core), tt::umd::RiscType::ALL);
                        log_info(
                            tt::LogAlways,
                            "teardown: FIX AC — reset MMIO ETH core {} on device {}",
                            virtual_core.str(),
                            mmio_id);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogAlways,
                            "teardown: FIX AC early MMIO reset of ETH core {} on device {} failed: {}",
                            virtual_core.str(),
                            mmio_id,
                            e.what());
                    } catch (...) {
                        // FIX BQ (#42429): log non-std exceptions from FIX AC early MMIO reset
                        log_debug(
                            tt::LogMetal,
                            "teardown: FIX AC early MMIO reset of ETH core {} on device {} threw non-std exception",
                            virtual_core.str(),
                            mmio_id);
                    }
                }
            }
            // FIX AR2 (#42429): Wait 100ms after FIX AC deassert before polling heartbeats.
            // ROM boot zeros L1 (including the heartbeat address) within ~10ms of deassert.
            // Without this delay, the first heartbeat read can see the STALE pre-reset
            // 0xABCDxxxx value from the previous firmware instance.  FIX TW then immediately
            // marks the channel "ready" (false positive), and FIX AQ proceeds under the
            // assumption that the relay is restored — but the ERISC is still in ROM boot.
            // 100ms is conservative: ROM zeros L1 in <10ms, so even 20ms would suffice.
            // This delay is paid once per FIX AC event (not per channel).
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            log_info(
                tt::LogAlways,
                "teardown: FIX AR2 (#42429) — 100ms post-deassert delay complete; "
                "starting heartbeat poll (avoids stale 0xABCDxxxx false positive).");

            // FIX AR: poll ALL reset ETH cores in a single shared time window instead of
            // sequentially (1000ms per core × N cores).  WH ETH link training takes ~1–3s;
            // with per-core sequential polling, every core times out because its individual
            // 1s window starts long after the simultaneous PCIe reset.  Parallel polling
            // gives all cores the full kBulkPollMs to converge.
            //
            // Two-phase heartbeat check:
            //   Phase 1: wait for heartbeat != 0  (ROM zeroes L1 on boot; firmware writes first value)
            //   Phase 2: wait for value to CHANGE, or detect static 0xABCDxxxx UMD marker (FIX TW)
            // WH heartbeat at 0x1F80 (test_results[48]) — UMD base firmware writes a STATIC 0xABCDxxxx marker.
            // cluster_.read_reg() on MMIO cores goes through PCIe — safe even with broken relay.
            //
            // Heartbeat addresses (populated in each arch's HAL active_eth file):
            //   WH: 0x1F80  (test_results[48], written by base UMD relay firmware)
            //   BH: MEM_SYSENG_ETH_HEARTBEAT = 0x7CC70  (eth_status_t.heartbeat[0])
            //   QA: MEM_SYSENG_ETH_HEARTBEAT = 0x7CC70  (same struct layout as BH)
            //
            // hal_.get_eth_fw_mailbox_val(HEARTBEAT) == 0 means arch not yet wired up —
            // fall back to 500ms sleep so we don't silently skip the wait.
            // FIX PG (#42429): track whether ANY MMIO ETH core confirmed its heartbeat.
            // If none do, the relay is NOT restored and FIX AY must be skipped.
            // Declared outside the heartbeat block so it is visible at the FIX AY gate below.
            bool ac_heartbeat_any_ready = false;
            {
                const uint32_t hb_addr = hal_.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
                if (hb_addr == 0u) {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AC — ETH heartbeat address not wired for this arch; "
                        "using 500ms sleep as fallback.");
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                } else {
                    struct CorePollState {
                        tt_cxy_pair target;
                        uint32_t prev_hb = 0;
                        bool nonzero_seen = false;
                        bool ready = false;
                    };
                    std::vector<CorePollState> poll_states;
                    for (const tt::ChipId poll_mmio_id : mmio_ids_set) {
                        for (const auto& poll_logical_core :
                             this->get_control_plane_().get_active_ethernet_cores(poll_mmio_id)) {
                            CoreCoord poll_virt = cluster_.get_virtual_coordinate_from_logical_coordinates(
                                poll_mmio_id, poll_logical_core, CoreType::ETH);
                            poll_states.push_back({tt_cxy_pair(poll_mmio_id, poll_virt), 0, false, false});
                        }
                    }
                    constexpr int kBulkPollMs = 5000;
                    constexpr auto kPollInterval = std::chrono::milliseconds(10);
                    const auto bulk_start = std::chrono::steady_clock::now();
                    while (true) {
                        bool all_done = true;
                        for (auto& ps : poll_states) {
                            if (ps.ready) continue;
                            uint32_t hb_val = 0;
                            try {
                                cluster_.read_reg(&hb_val, ps.target, hb_addr);
                            } catch (...) {
                                ps.ready = true;  // PCIe read failed — count as done
                                continue;
                            }
                            if (!ps.nonzero_seen) {
                                if (hb_val != 0) {
                                    ps.prev_hb = hb_val;
                                    ps.nonzero_seen = true;
                                    // FIX TW (#42429): UMD base firmware writes a static 0xABCDxxxx
                                    // marker — detect it immediately, no increment to wait for.
                                    if ((hb_val >> 16) == 0xABCDu) {
                                        ps.ready = true;
                                    }
                                }
                            } else if ((hb_val >> 16) == 0xABCDu || hb_val != ps.prev_hb) {
                                // Ready if UMD static marker OR incrementing counter detected.
                                ps.ready = true;
                            }
                            if (!ps.ready) all_done = false;
                        }
                        if (all_done) break;
                        const auto elapsed_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - bulk_start)
                                .count();
                        if (elapsed_ms >= kBulkPollMs) break;
                        std::this_thread::sleep_for(kPollInterval);
                    }
                    for (const auto& ps : poll_states) {
                        if (ps.ready) {
                            ac_heartbeat_any_ready = true;
                        } else {
                            log_warning(
                                tt::LogAlways,
                                "teardown: FIX AC — ETH core {} did not report base firmware "
                                "heartbeat within {}ms; proceeding.",
                                ps.target.str(),
                                kBulkPollMs);
                        }
                    }
                }
            }
            if (ac_heartbeat_any_ready) {
                log_info(tt::LogAlways, "teardown: FIX AC — MMIO ETH channels rebooted; relay should be restored.");
            } else {
                log_warning(
                    tt::LogAlways,
                    "teardown: FIX PG (#42429): ALL MMIO ETH heartbeats timed out — relay NOT restored; "
                    "skipping FIX AY to avoid N x 5s wasted timeouts.");
            }

            // FIX AQ (#42429): Secondary poll of edm_status_address after the FIX AR heartbeat
            // poll — closes the race between "heartbeat incrementing" and "UMD relay has written
            // its sentinel (0x49706550) to edm_status_address (0x18070)".
            //
            // Root cause: the BRISC hardware reset ROM writes 0x49705180 to L1 address 0x18070
            // (edm_status_address) as a postcode during power-on init.  The FIX AR heartbeat
            // poll (at address 0x1F80) exits as soon as the heartbeat counter changes — which
            // can happen a few milliseconds BEFORE the UMD relay firmware overwrites 0x18070
            // with its "ready" sentinel 0x49706550.  If the NEXT process's
            // terminate_stale_erisc_routers() reads 0x18070 during this narrow window it sees
            // 0x49705180, classifies ALL channels as corrupt, and initializes fabric in degraded
            // mode — cascading into a 45-second test timeout.
            //
            // Fix: poll each MMIO active ETH channel's edm_status_address until the value is
            // no longer 0x49705180 (ROM postcode).  0x49706550 (UMD relay sentinel) and
            // 0x00000000 (clean) are both safe values for the next session.  Any other non-zero
            // value is also fine (e.g. EDMStatus enum values from a prior session).  The
            // After PCIe hard reset (FIX AC), UMD base firmware raises its heartbeat (FIX AR, 5s
            // window) but then takes additional time to write 0x49706550 to edm_status_address.
            // Poll for up to 10s so the next session sees 0x49706550, not the ROM postcode.
            // FIX AY-C (#42429): edm_status address needed both by FIX AQ (poll loop) and
            // FIX AY (per-pair stagger check).  Declare here so FIX AY can use it.
            std::optional<uint32_t> ay_edm_status_addr;
            if (get_control_plane_) {
                try {
                    const auto& fabric_ctx = this->get_control_plane_().get_fabric_context();
                    const auto& builder_ctx = fabric_ctx.get_builder_context();
                    const auto edm_status_addr_aq =
                        builder_ctx.get_fabric_router_sync_address_and_status().first;
                    ay_edm_status_addr = edm_status_addr_aq;  // hoist for FIX AY stagger
                    // FIX AQ-3 (#42429): The ERISC BRISC ROM writes a family of intermediate
                    // postcodes during boot: 0x49705180 → 0x49705530 → ... → 0x49706550
                    // (base-UMD sentinel).  The original check (edm_val != kRomPostcode) only
                    // treated 0x49705180 as "still in ROM boot."  If the MMIO ERISC was at
                    // 0x49705530 (a later intermediate state), it looked "ready" — causing
                    // FIX AY to fire write-only resets on non-MMIO channels prematurely,
                    // triggering a simultaneous-boot race on both sides of the inter-chip ETH
                    // link.  Fix: wait until the ERISC has fully exited the 0x4970xxxx family
                    // (i.e. reached the 0x49706550 base-UMD sentinel or any non-0x4970xxxx value).
                    constexpr uint32_t kBaseUmdFirmwareSentinel = 0x49706550u;
                    constexpr int kEdmStatusPollMs = 10000;  // FIX AQ: ROM boot to base-UMD sentinel can take >1s after PCIe hard reset; 10s matches FIX AR heartbeat window
                    constexpr auto kEdmStatusPollInterval = std::chrono::milliseconds(5);
                    struct EdmPollState {
                        tt_cxy_pair target;
                        bool ready = false;
                    };
                    std::vector<EdmPollState> edm_states;
                    for (const tt::ChipId aq_mmio_id : mmio_ids_set) {
                        for (const auto& aq_logical_core :
                             this->get_control_plane_().get_active_ethernet_cores(aq_mmio_id)) {
                            CoreCoord aq_virt = cluster_.get_virtual_coordinate_from_logical_coordinates(
                                aq_mmio_id, aq_logical_core, CoreType::ETH);
                            edm_states.push_back({tt_cxy_pair(aq_mmio_id, aq_virt), false});
                        }
                    }
                    const auto aq_start = std::chrono::steady_clock::now();
                    while (true) {
                        bool all_clear = true;
                        for (auto& es : edm_states) {
                            if (es.ready) continue;
                            uint32_t edm_val = 0;
                            try {
                                cluster_.read_reg(&edm_val, es.target, edm_status_addr_aq);
                            } catch (...) {
                                es.ready = true;  // read failed — treat as clear
                                continue;
                            }
                            // FIX AQ-3: treat the entire 0x4970xxxx family as "still in ROM
                            // boot" — only declare ready when the ERISC has written the
                            // base-UMD sentinel (0x49706550) or any value outside this family.
                            const bool still_in_rom_boot =
                                (edm_val & 0xFFFF0000u) == 0x49700000u &&
                                edm_val != kBaseUmdFirmwareSentinel;
                            if (!still_in_rom_boot) {
                                es.ready = true;
                            } else {
                                all_clear = false;
                            }
                        }
                        if (all_clear) break;
                        const auto aq_elapsed_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - aq_start)
                                .count();
                        if (aq_elapsed_ms >= kEdmStatusPollMs) break;
                        std::this_thread::sleep_for(kEdmStatusPollInterval);
                    }
                    for (const auto& es : edm_states) {
                        if (!es.ready) {
                            uint32_t final_val = 0;
                            try { cluster_.read_reg(&final_val, es.target, edm_status_addr_aq); } catch (...) {}
                            log_warning(
                                tt::LogAlways,
                                "teardown: FIX AQ — ETH core {} edm_status_address still 0x{:08x} "
                                "(ROM boot postcode 0x4970xxxx family) after {}ms; next session may see corrupt L1. "
                                "(#42429 FIX AQ-3)",
                                es.target.str(),
                                final_val,
                                kEdmStatusPollMs);
                        }
                    }
                    log_info(
                        tt::LogAlways,
                        "teardown: FIX AQ — edm_status_address sentinel poll complete (Step 2 path).");
                } catch (const tt::tt_fabric::FabricContextNullException&) {
                    // FIX BQ (#42429): Typed catch replaces FIX BP's fragile e.what() string match.
                    // fabric_context_ is already null (teardown/atexit path) — nothing to poll.
                    // The previous code would sleep 10s here on every test teardown when
                    // fabric_context_ was null: 20+ GTest teardowns × 10s = 13+ minutes of hang.
                    log_debug(
                        tt::LogMetal,
                        "teardown: FIX AQ — fabric_context already torn down (FIX BQ typed catch), "
                        "skipping 10s fallback wait. ROM-postcode channels left for next init. (#42429)");
                } catch (const std::exception& e) {
                    // FIX AQ-fallback (#42429): fabric_context_ exists but poll failed
                    // for another reason.  Fall back to a time-based wait long enough for
                    // FIX AC-reset ETH channels to complete ROM boot and write the UMD
                    // relay sentinel (0x49706550) to edm_status_address.
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AQ — edm_status_address poll threw: {}; "
                        "adding 10s fallback wait for ROM-postcode channels to clear. (#42429)",
                        e.what());
                    std::this_thread::sleep_for(std::chrono::milliseconds(10000));  // FIX AQ-fallback
                } catch (...) {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AQ — edm_status_address poll threw unknown; "
                        "adding 10s fallback wait. (#42429)");
                    std::this_thread::sleep_for(std::chrono::milliseconds(10000));  // FIX AQ-fallback
                }
            }

            // FIX AY (#42429): Deferred non-MMIO ETH ERISC reset via restored relay.
            //
            // Root cause of second-session hang (run 25040706453):
            //   FIX AX skipped assert_risc_reset_at_core for non-MMIO channels whose
            //   relay was dead (to avoid 5s-per-channel UMD timeout).  Those ERISCs
            //   remained running FABRIC firmware after the first session ended.
            //   When the next process started (unit_tests_ttnn), TopologyDiscovery::
            //   discover_remote_devices() → init_tt_device hit the 5s timeout for each
            //   non-MMIO device (FABRIC firmware ignores UMD relay reads).  After the
            //   timeout, MetalContext initialization was partially complete / in a broken
            //   state; subsequent tests that tried to open MeshDevice triggered
            //   configure_fabric() → write_non_mmio → UMD relay queue fill →
            //   while(full) spin → 15-min SIGALRM.
            //
            // Fix: Now that MMIO ETH relay is hardware-restored (FIX AC heartbeat poll
            //   passed), use write-only assert+deassert for each non-MMIO ETH ERISC.
            //   The reset write goes: PCIe → MMIO relay ERISC (BASE fw) → NOC →
            //   non-MMIO chip hardware SOFT_RESET register.  The non-MMIO ERISC does
            //   not need to respond — the hardware register write fires regardless of
            //   its firmware state.
            //
            //   We use write-only variants (assert_risc_reset_at_core_write_only /
            //   deassert_risc_reset_at_core_write_only) rather than the normal read-
            //   modify-write path because the read step (get_risc_reset_state via relay)
            //   times out when the non-MMIO ERISC is running FABRIC firmware — it does
            //   not service UMD relay reads.  The write-only path skips the read and
            //   writes the full reset value directly, then waits for the MMIO relay to
            //   drain (best-effort; FIX AE marks relay broken on flush timeout).
            //
            //   FIX AV (#42429): break out of the per-core loop on first failure for a
            //   device (all cores share the same relay path; one failure predicts all).
            if (get_control_plane_ && ac_heartbeat_any_ready) {
                log_info(
                    tt::LogAlways,
                    "teardown: FIX AY — attempting deferred ETH ERISC write-only reset for {} "
                    "relay-broken non-MMIO device(s) via restored MMIO relay.",
                    relay_broken_non_mmio.size());
                uint32_t ay_succeeded = 0;
                uint32_t ay_failed = 0;
                uint32_t ay_skipped_relay_not_ready = 0;
                // FIX AY-C (#42429): per-pair MMIO relay readiness guard (Mitigation C).
                // Before sending a write-only reset to a non-MMIO ETH channel, verify that
                // its MMIO peer channel has fully exited ROM boot (edm_status == 0x49706550).
                // This prevents the simultaneous-boot race where both sides of the inter-chip
                // ETH link start ROM boot simultaneously, stalling PHY link training.
                //
                // Approach: for each non-MMIO (non_mmio_id, eth_chan_id) pair, look up its
                // peer via get_ethernet_connections() → (peer_chip_id, peer_eth_chan).  If
                // the peer is an MMIO device, read its edm_status via PCIe-direct reg read.
                // If still in 0x4970xxxx ROM boot family, skip the reset for this channel.
                // Channels skipped here are handled by FIX RR in the next session's init.
                const auto& eth_connections = cluster_.get_ethernet_connections();
                constexpr uint32_t kAyBaseUmdSentinel = 0x49706550u;
                for (const tt::ChipId non_mmio_id : relay_broken_non_mmio) {
                    bool device_relay_dead = false;
                    for (const auto& eth_logical_core :
                         this->get_control_plane_().get_active_ethernet_cores(non_mmio_id)) {
                        if (device_relay_dead) {
                            ++ay_failed;
                            continue;
                        }
                        CoreCoord eth_virt;
                        try {
                            eth_virt = cluster_.get_virtual_coordinate_from_logical_coordinates(
                                non_mmio_id, eth_logical_core, CoreType::ETH);
                        } catch (...) {
                            ++ay_failed;
                            continue;
                        }
                        // FIX AY-C: stagger per MMIO↔non-MMIO channel pair.
                        // Resolve logical core → channel id → MMIO peer → edm_status check.
                        if (ay_edm_status_addr.has_value()) {
                            try {
                                const auto& non_mmio_soc = cluster_.get_soc_desc(non_mmio_id);
                                const int eth_chan_id = non_mmio_soc.get_eth_channel_for_core(
                                    tt::umd::CoreCoord(
                                        eth_logical_core.x, eth_logical_core.y,
                                        CoreType::ETH, CoordSystem::LOGICAL),
                                    CoordSystem::LOGICAL);
                                auto dev_it = eth_connections.find(non_mmio_id);
                                if (dev_it != eth_connections.end()) {
                                    auto chan_it = dev_it->second.find(eth_chan_id);
                                    if (chan_it != dev_it->second.end()) {
                                        const ChipId peer_chip = std::get<0>(chan_it->second);
                                        const auto peer_chan = std::get<1>(chan_it->second);
                                        // Only stagger when peer is the MMIO relay chip.
                                        if (mmio_ids_set.count(peer_chip)) {
                                            const CoreCoord peer_logical =
                                                cluster_.get_soc_desc(peer_chip)
                                                    .get_eth_core_for_channel(peer_chan, CoordSystem::LOGICAL);
                                            const CoreCoord peer_virt =
                                                cluster_.get_virtual_coordinate_from_logical_coordinates(
                                                    peer_chip, peer_logical, CoreType::ETH);
                                            uint32_t peer_status = 0;
                                            try {
                                                cluster_.read_reg(
                                                    &peer_status,
                                                    tt_cxy_pair(peer_chip, peer_virt),
                                                    *ay_edm_status_addr);
                                            } catch (...) {
                                                // Read failed — relay may not yet be up.
                                                // Skip this channel; next session handles it.
                                                log_warning(
                                                    tt::LogAlways,
                                                    "teardown: FIX AY-C — could not read peer "
                                                    "edm_status for non-MMIO dev={} chan={} "
                                                    "(peer dev={} chan={}) — skipping reset. "
                                                    "Next session FIX RR will recover. (#42429)",
                                                    non_mmio_id, eth_chan_id, peer_chip, peer_chan);
                                                ++ay_skipped_relay_not_ready;
                                                continue;
                                            }
                                            const bool peer_still_in_rom_boot =
                                                (peer_status & 0xFFFF0000u) == 0x49700000u &&
                                                peer_status != kAyBaseUmdSentinel;
                                            if (peer_still_in_rom_boot) {
                                                log_warning(
                                                    tt::LogAlways,
                                                    "teardown: FIX AY-C — MMIO peer dev={} chan={} "
                                                    "edm_status=0x{:08x} (ROM boot family) — "
                                                    "skipping write-only reset of non-MMIO dev={} "
                                                    "chan={} to prevent simultaneous-boot race. "
                                                    "Next session FIX RR will recover. (#42429)",
                                                    peer_chip, peer_chan, peer_status,
                                                    non_mmio_id, eth_chan_id);
                                                ++ay_skipped_relay_not_ready;
                                                continue;
                                            }
                                        }
                                    }
                                }
                            } catch (...) {
                                // Could not resolve channel/peer — proceed with the reset
                                // (safe: worst case we might race, but the A fix covers that).
                            }
                        }
                        try {
                            // Write-only reset: skips relay read that times out when
                            // non-MMIO ERISC is running FABRIC firmware (#42429).
                            cluster_.assert_risc_reset_at_core_write_only(
                                tt_cxy_pair(non_mmio_id, eth_virt), tt::umd::RiscType::ALL);
                            cluster_.deassert_risc_reset_at_core_write_only(
                                tt_cxy_pair(non_mmio_id, eth_virt));
                            ++ay_succeeded;
                        } catch (const std::exception& e) {
                            log_warning(
                                tt::LogAlways,
                                "teardown: FIX AY/AV — write-only reset of ETH {} on non-MMIO "
                                "device {} failed: {}. "
                                "Skipping all remaining ETH cores on this device. (FIX AV #42429)",
                                eth_virt.str(),
                                non_mmio_id,
                                e.what());
                            ++ay_failed;
                            device_relay_dead = true;
                        } catch (...) {
                            log_warning(
                                tt::LogAlways,
                                "teardown: FIX AY/AV — write-only reset of ETH {} on non-MMIO "
                                "device {} threw non-std exception. "
                                "Skipping all remaining ETH cores on this device. (FIX AV #42429)",
                                eth_virt.str(),
                                non_mmio_id);
                            ++ay_failed;
                            device_relay_dead = true;
                        }
                    }
                }
                if (ay_failed == 0 && ay_skipped_relay_not_ready == 0) {
                    log_info(
                        tt::LogAlways,
                        "teardown: FIX AY — all {} non-MMIO ETH ERISCs reset to base firmware "
                        "via write-only relay path. Next session should not encounter FABRIC fw.",
                        ay_succeeded);
                } else {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AY/AV/C — {}/{} non-MMIO ETH ERISCs reset successfully "
                        "({} failed, {} skipped by FIX AY-C relay-not-ready guard). "
                        "Skipped channels will be recovered by FIX RR in next session. (#42429)",
                        ay_succeeded,
                        ay_succeeded + ay_failed + ay_skipped_relay_not_ready,
                        ay_failed,
                        ay_skipped_relay_not_ready);
                }
            } else if (get_control_plane_ && !relay_broken_non_mmio.empty()) {
                log_warning(
                    tt::LogAlways,
                    "teardown: FIX PG: skipping FIX AY — relay not restored, {} non-MMIO "
                    "device(s) will be handled by FIX BC on next SetUp.",
                    relay_broken_non_mmio.size());
            }

            // FIX AE supersedes FIX AW: ~Cluster() now marks ALL remote chips relay-broken
            // before close_device() so wait_for_non_mmio_flush() returns instantly.
            // No per-chip notification needed here.
        }

        // Step 3: assert_cores / l1_barrier.  Skip ALL non-MMIO devices when any relay
        // is broken — all non-MMIO devices share the same MMIO relay path.  If the
        // MMIO ERISCs are running fabric firmware (stuck handshake state), every
        // non-MMIO relay read/write will timeout for 5s.  We can't reach any non-MMIO
        // device, not just the ones explicitly in relay_broken_non_mmio.
        //
        // FIX AZ (#42429): relay may be dead even when relay_broken_non_mmio was empty
        // after Steps 1/2 (e.g. RiscFirmwareInitializer teardown runs without a prior
        // FabricFirmwareInitializer session).  Use a local flag so that once assert_cores
        // throws for any non-MMIO device we know the relay path is dead and skip all
        // subsequent non-MMIO devices.  We do NOT insert into relay_broken_non_mmio here
        // to avoid inadvertently suppressing Step 5's MMIO ETH reset (which guards on
        // relay_broken_non_mmio.empty()).
        //
        // FIX AZ+ (#42429 follow-up): extend the l1_barrier skip to MMIO devices as well.
        // On some runners (e.g. t3k-05) device 4 is classified as MMIO by mmio_chip_ids()
        // even though configure_fabric logged mmio=false for it.  When assert_cores times
        // out for such a device, calling l1_barrier unconditionally causes a second 5s
        // timeout.  Skipping l1_barrier whenever assert_cores throws — regardless of
        // MMIO/non-MMIO classification — eliminates this wasted time.  The
        // relay_dead_detected_step3 flag (and relay_broken_non_mmio skip logic) is
        // intentionally kept non-MMIO-only, as it drives relay-path reasoning for Step 4/5.
        bool relay_dead_detected_step3 = false;
        for (tt::ChipId device_id : all_devices) {
            const bool is_non_mmio = !mmio_ids_set.count(device_id);
            if (is_non_mmio && (!relay_broken_non_mmio.empty() || relay_dead_detected_step3)) {
                log_info(
                    tt::LogAlways,
                    "teardown: FIX AC/AZ — relay broken; skipping assert_cores/l1_barrier for non-MMIO device {}",
                    device_id);
                continue;
            }
            // FIX AZ/AZ+: track whether assert_cores threw for this device (any device,
            // MMIO or non-MMIO).  If it did, skip l1_barrier to avoid a secondary timeout.
            // Additionally, for non-MMIO devices set relay_dead_detected_step3 so all
            // subsequent non-MMIO devices (which share the MMIO relay path) are skipped.
            bool assert_cores_threw = false;
            try {
                assert_cores(device_id);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogAlways,
                    "teardown: assert_cores failed for device {} (likely dead ERISC relay): {}",
                    device_id,
                    e.what());
                assert_cores_threw = true;
                if (is_non_mmio) {
                    relay_dead_detected_step3 = true;
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AZ — assert_cores threw for non-MMIO device {}; "
                        "skipping l1_barrier and all subsequent non-MMIO devices",
                        device_id);
                } else {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AZ+ — assert_cores threw for MMIO device {}; "
                        "skipping l1_barrier to avoid secondary timeout",
                        device_id);
                }
            } catch (...) {
                assert_cores_threw = true;
                if (is_non_mmio) {
                    relay_dead_detected_step3 = true;
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AZ — assert_cores threw (unknown exception) for non-MMIO device {}; "
                        "skipping l1_barrier and all subsequent non-MMIO devices",
                        device_id);
                } else {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AZ+ — assert_cores threw (unknown exception) for MMIO device {}; "
                        "skipping l1_barrier to avoid secondary timeout",
                        device_id);
                }
            }
            if (assert_cores_threw) {
                continue;
            }
            try {
                cluster_.l1_barrier(device_id);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogAlways,
                    "teardown: l1_barrier failed for device {}: {}",
                    device_id,
                    e.what());
            } catch (...) {
                // FIX BQ (#42429): log non-std exceptions from l1_barrier
                log_debug(
                    tt::LogMetal,
                    "teardown: l1_barrier threw non-std exception for device {}",
                    device_id);
            }
        }

        // Step 4: set internal routing to false to exit active ethernet FW.
        // Skip when relay is broken — this call writes to non-MMIO devices via relay
        // and will timeout for each unreachable chip.  Non-MMIO ERISCs with stale
        // firmware are cleaned up by terminate_stale_erisc_routers on next init.
        // Wrapping in try/catch: wait_for_non_mmio_flush() can throw UmdException
        // which does not inherit from std::exception.
        // FIX AZ: also skip when relay_dead_detected_step3 (dead relay found mid Step 3).
        const bool any_relay_broken = !relay_broken_non_mmio.empty() || relay_dead_detected_step3;
        if (cp_ready && !any_relay_broken) {
            try {
                cluster_.set_internal_routing_info_for_ethernet_cores(this->get_control_plane_(), false);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogAlways,
                    "teardown: set_internal_routing_info_for_ethernet_cores failed: {}",
                    e.what());
            } catch (...) {
                log_warning(
                    tt::LogAlways,
                    "teardown: set_internal_routing_info_for_ethernet_cores failed with unknown exception type "
                    "(likely UmdException<RuntimeError> from dead ERISC relay on remote chip)");
            }
        } else if (any_relay_broken) {
            log_warning(
                tt::LogAlways,
                "teardown: FIX AC/AZ — skipping set_internal_routing_info_for_ethernet_cores (relay broken; "
                "would timeout per non-MMIO device). Non-MMIO ERISCs handled by terminate_stale_erisc_routers "
                "on next init.");
        }

        // Step 5: Hard-reset MMIO ETH channels when teardown timed out but relay wasn't
        // broken.  (When relay was broken, MMIO ERISCs were already reset via PCIe in
        // Step 2.)
        //
        // We no longer attempt relay-based resets of non-MMIO ETH channels here.
        // After Step 2 resets the MMIO ERISCs, UMD hasn't re-initialized the relay
        // protocol state — empirically, the relay is still unreachable even after the
        // 500ms sleep, causing 5-second timeouts per non-MMIO ETH core (and for each
        // device that wasn't explicitly in relay_broken_non_mmio but shares the relay
        // path).  Non-MMIO ERISCs with stale firmware are safely cleaned up by
        // terminate_stale_erisc_routers on the next process's init.
        if (any_teardown_timed_out && relay_broken_non_mmio.empty() && cp_ready) {
            // Relay path is intact (no non-MMIO relay broken), but teardown timed
            // out — some ETH channels may have stale firmware state.  Reset MMIO
            // channels now (relay_broken_non_mmio was empty so step 2 was skipped).
            log_warning(
                tt::LogAlways,
                "teardown: FIX AC (timeout-only) — fabric_teardown_timed_out set, no relay broken. "
                "Hard-resetting active ETH channels on MMIO devices via PCIe.");
            for (const tt::ChipId mmio_id : mmio_ids_set) {
                for (const auto& logical_core :
                     this->get_control_plane_().get_active_ethernet_cores(mmio_id)) {
                    CoreCoord virtual_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        mmio_id, logical_core, CoreType::ETH);
                    try {
                        cluster_.assert_risc_reset_at_core(
                            tt_cxy_pair(mmio_id, virtual_core), tt::umd::RiscType::ALL);
                        cluster_.deassert_risc_reset_at_core(
                            tt_cxy_pair(mmio_id, virtual_core), tt::umd::RiscType::ALL);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogAlways,
                            "teardown: FIX AC (timeout) reset of ETH core {} on MMIO device {} failed: {}",
                            virtual_core.str(),
                            mmio_id,
                            e.what());
                    } catch (...) {
                        // FIX BQ (#42429): log non-std exceptions from FIX AC timeout reset
                        log_debug(
                            tt::LogMetal,
                            "teardown: FIX AC (timeout) reset of ETH core {} on MMIO device {} threw non-std exception",
                            virtual_core.str(),
                            mmio_id);
                    }
                }
            }
            // FIX AR: same parallel bulk poll as Step 2 — see comment there for rationale.
            // Relay is intact in this path (relay_broken_non_mmio was empty),
            // but cluster_.read_reg() via PCIe is still the right read path for MMIO cores.
            {
                const uint32_t hb_addr = hal_.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
                if (hb_addr == 0u) {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AC (timeout) — ETH heartbeat address not wired for this arch; "
                        "using 200ms sleep as fallback.");
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                } else {
                    struct CorePollState {
                        tt_cxy_pair target;
                        uint32_t prev_hb = 0;
                        bool nonzero_seen = false;
                        bool ready = false;
                    };
                    std::vector<CorePollState> poll_states;
                    for (const tt::ChipId poll_mmio_id : mmio_ids_set) {
                        for (const auto& poll_logical_core :
                             this->get_control_plane_().get_active_ethernet_cores(poll_mmio_id)) {
                            CoreCoord poll_virt = cluster_.get_virtual_coordinate_from_logical_coordinates(
                                poll_mmio_id, poll_logical_core, CoreType::ETH);
                            poll_states.push_back({tt_cxy_pair(poll_mmio_id, poll_virt), 0, false, false});
                        }
                    }
                    constexpr int kBulkPollMs = 5000;
                    constexpr auto kPollInterval = std::chrono::milliseconds(10);
                    const auto bulk_start = std::chrono::steady_clock::now();
                    while (true) {
                        bool all_done = true;
                        for (auto& ps : poll_states) {
                            if (ps.ready) continue;
                            uint32_t hb_val = 0;
                            try {
                                cluster_.read_reg(&hb_val, ps.target, hb_addr);
                            } catch (...) {
                                ps.ready = true;  // PCIe read failed — count as done
                                continue;
                            }
                            if (!ps.nonzero_seen) {
                                if (hb_val != 0) {
                                    ps.prev_hb = hb_val;
                                    ps.nonzero_seen = true;
                                    // FIX TW (#42429): UMD base firmware writes a static 0xABCDxxxx
                                    // marker — detect it immediately, no increment to wait for.
                                    if ((hb_val >> 16) == 0xABCDu) {
                                        ps.ready = true;
                                    }
                                }
                            } else if ((hb_val >> 16) == 0xABCDu || hb_val != ps.prev_hb) {
                                // Ready if UMD static marker OR incrementing counter detected.
                                ps.ready = true;
                            }
                            if (!ps.ready) all_done = false;
                        }
                        if (all_done) break;
                        const auto elapsed_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - bulk_start)
                                .count();
                        if (elapsed_ms >= kBulkPollMs) break;
                        std::this_thread::sleep_for(kPollInterval);
                    }
                    for (const auto& ps : poll_states) {
                        if (!ps.ready) {
                            log_warning(
                                tt::LogAlways,
                                "teardown: FIX AC (timeout) — ETH core {} did not report base firmware "
                                "heartbeat within {}ms; proceeding.",
                                ps.target.str(),
                                kBulkPollMs);
                        }
                    }
                }
            }
            log_info(tt::LogAlways, "teardown: FIX AC — MMIO ETH channel reset complete.");

            // FIX AQ (#42429): Secondary edm_status_address sentinel poll — same race as in
            // Step 2 (see comment above), applied to the Step 5 (timeout-only) path.
            if (get_control_plane_) {
                try {
                    const auto& fabric_ctx5 = this->get_control_plane_().get_fabric_context();
                    const auto& builder_ctx5 = fabric_ctx5.get_builder_context();
                    const auto edm_status_addr_aq5 =
                        builder_ctx5.get_fabric_router_sync_address_and_status().first;
                    constexpr uint32_t kRomPostcode5 = 0x49705180u;
                    constexpr int kEdmStatusPollMs5 = 10000;  // FIX AQ Step 5: same boot timing as Step 2 — increase to match
                    constexpr auto kEdmStatusPollInterval5 = std::chrono::milliseconds(5);
                    struct EdmPollState5 {
                        tt_cxy_pair target;
                        bool ready = false;
                    };
                    std::vector<EdmPollState5> edm_states5;
                    for (const tt::ChipId aq5_mmio_id : mmio_ids_set) {
                        for (const auto& aq5_logical_core :
                             this->get_control_plane_().get_active_ethernet_cores(aq5_mmio_id)) {
                            CoreCoord aq5_virt = cluster_.get_virtual_coordinate_from_logical_coordinates(
                                aq5_mmio_id, aq5_logical_core, CoreType::ETH);
                            edm_states5.push_back({tt_cxy_pair(aq5_mmio_id, aq5_virt), false});
                        }
                    }
                    const auto aq5_start = std::chrono::steady_clock::now();
                    while (true) {
                        bool all_clear5 = true;
                        for (auto& es5 : edm_states5) {
                            if (es5.ready) continue;
                            uint32_t edm_val5 = 0;
                            try {
                                cluster_.read_reg(&edm_val5, es5.target, edm_status_addr_aq5);
                            } catch (...) {
                                es5.ready = true;
                                continue;
                            }
                            if (edm_val5 != kRomPostcode5) {
                                es5.ready = true;
                            } else {
                                all_clear5 = false;
                            }
                        }
                        if (all_clear5) break;
                        const auto aq5_elapsed_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - aq5_start)
                                .count();
                        if (aq5_elapsed_ms >= kEdmStatusPollMs5) break;
                        std::this_thread::sleep_for(kEdmStatusPollInterval5);
                    }
                    for (const auto& es5 : edm_states5) {
                        if (!es5.ready) {
                            uint32_t final_val5 = 0;
                            try { cluster_.read_reg(&final_val5, es5.target, edm_status_addr_aq5); } catch (...) {}
                            log_warning(
                                tt::LogAlways,
                                "teardown: FIX AQ — ETH core {} edm_status_address still 0x{:08x} "
                                "(ROM postcode 0x49705180) after {}ms; next session may see corrupt L1. "
                                "(#42429 FIX AQ)",
                                es5.target.str(),
                                final_val5,
                                kEdmStatusPollMs5);
                        }
                    }
                    log_info(
                        tt::LogAlways,
                        "teardown: FIX AQ — edm_status_address sentinel poll complete (Step 5 path).");
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogAlways,
                        "teardown: FIX AQ — edm_status_address poll (Step 5) threw: {}; proceeding.",
                        e.what());
                } catch (...) {
                    log_warning(tt::LogAlways, "teardown: FIX AQ — edm_status_address poll (Step 5) threw unknown; proceeding.");
                }
            }
        }
    }

    initialized_ = false;
}

bool RiscFirmwareInitializer::is_initialized() const { return initialized_; }

void RiscFirmwareInitializer::clear_l1_state(tt::ChipId device_id) {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", device_id);
    CoreCoord logical_grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t l1_size_per_core = cluster_.get_soc_desc(device_id).worker_l1_size;
    TT_ASSERT(l1_size_per_core % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(l1_size_per_core / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            auto virtual_core =
                cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            cluster_.write_core(device_id, virtual_core, zero_vec, start_address);
        }
    }

    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        static uint32_t zero_vec_size = hal::get_erisc_l1_unreserved_size();
        auto zero_vec_addr = hal::get_erisc_l1_unreserved_base();
        static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        cluster_.write_core(device_id, virtual_core, zero_vec, zero_vec_addr);
    }

    bool has_dram_fw = hal_.has_programmable_core_type(HalProgrammableCoreType::DRAM);
    if (has_dram_fw) {
        uint32_t dram_l1_size = hal_.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE);
        std::vector<uint32_t> dram_zero_vec(dram_l1_size / sizeof(uint32_t), 0);
        const auto& soc_d = cluster_.get_soc_desc(device_id);
        for (const auto& dram_core : soc_d.get_cores(CoreType::DRAM, CoordSystem::TRANSLATED)) {
            CoreCoord virtual_core{dram_core.x, dram_core.y};
            cluster_.write_core(
                dram_zero_vec.data(),
                dram_l1_size,
                tt_cxy_pair(device_id, virtual_core),
                hal_.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE));
        }
    }

    // FIX PL (#42429): l1_barrier() routes through the ERISC relay on non-MMIO chips.
    // If the relay is dead (stale from a prior process) and FIX AF has added the
    // read_non_mmio() timeout, the barrier will throw instead of hanging forever.
    // This call is in run_async_build_phase(), before reset_cores() has established
    // relay health — catch and warn so init can continue.
    const bool is_mmio_dev = cluster_.mmio_chip_ids().count(device_id);
    if (is_mmio_dev) {
        cluster_.l1_barrier(device_id);
    } else {
        try {
            cluster_.l1_barrier(device_id);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogAlways,
                "clear_l1_state: l1_barrier timed out on non-MMIO device {} (dead ERISC relay): {}. "
                "L1 clear writes may not have flushed — proceeding with init.",
                device_id,
                e.what());
        } catch (...) {
            log_warning(
                tt::LogAlways,
                "clear_l1_state: l1_barrier timed out on non-MMIO device {} (dead ERISC relay, unknown exception). "
                "L1 clear writes may not have flushed — proceeding with init.",
                device_id);
        }
    }
}

void RiscFirmwareInitializer::clear_dram_state(tt::ChipId device_id) {
    log_debug(tt::LogMetal, "Clearing DRAM for device {}", device_id);
    auto dram_size_per_channel = cluster_.get_soc_desc(device_id).dram_view_size;
    auto num_dram_channels = cluster_.get_soc_desc(device_id).get_num_dram_views();
    constexpr uint32_t start_address = 0;
    std::vector<uint8_t> zero_vec(dram_size_per_channel, 0);
    const bool is_mmio_dev = cluster_.mmio_chip_ids().count(device_id);
    for (int channel = 0; channel < num_dram_channels; ++channel) {
        cluster_.write_dram_vec(zero_vec.data(), zero_vec.size(), device_id, channel, start_address);
        // FIX PL (#42429): dram_barrier() routes through the ERISC relay on non-MMIO chips.
        // Same race as l1_barrier in clear_l1_state() — guard for dead relay.
        if (is_mmio_dev) {
            cluster_.dram_barrier(device_id);
        } else {
            try {
                cluster_.dram_barrier(device_id);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogAlways,
                    "clear_dram_state: dram_barrier timed out on non-MMIO device {} channel {} "
                    "(dead ERISC relay): {}. DRAM clear writes may not have flushed — proceeding with init.",
                    device_id,
                    channel,
                    e.what());
                break;  // relay is dead; subsequent channels will also fail
            } catch (...) {
                log_warning(
                    tt::LogAlways,
                    "clear_dram_state: dram_barrier timed out on non-MMIO device {} channel {} "
                    "(dead ERISC relay, unknown exception). Proceeding with init.",
                    device_id,
                    channel);
                break;
            }
        }
    }
}

void RiscFirmwareInitializer::clear_launch_messages_on_eth_cores(tt::ChipId device_id) {
    auto clear_ethernet_core = [&](const CoreCoord& logical_eth_core, HalProgrammableCoreType programmable_core_type) {
        auto factory = hal_.get_dev_msgs_factory(programmable_core_type);
        std::vector<std::byte> init_launch_msg_data(
            dev_msgs::launch_msg_buffer_num_entries * factory.size_of<dev_msgs::launch_msg_t>(), std::byte{0});
        dev_msgs::go_msg_t go_msg = factory.create<dev_msgs::go_msg_t>();
        go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

        CoreCoord virtual_eth_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
        cluster_.write_core(
            init_launch_msg_data.data(),
            init_launch_msg_data.size(),
            tt_cxy_pair(device_id, virtual_eth_core),
            hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
        cluster_.write_core(
            go_msg.data(),
            go_msg.size(),
            {static_cast<size_t>(device_id), virtual_eth_core},
            hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG));
    };

    if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        return;
    }
    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core, HalProgrammableCoreType::ACTIVE_ETH);
    }
    for (const auto& eth_core : this->get_control_plane_().get_inactive_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core, HalProgrammableCoreType::IDLE_ETH);
    }
    cluster_.l1_barrier(device_id);
}

void RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset(tt::ChipId device_id) {
    for (const auto& logical_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
        if (rtoptions_.get_enable_2_erisc_mode()) {
            llrt::internal_::return_to_base_firmware_and_wait_for_heartbeat(device_id, virtual_core);
        }
        tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
        cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
    }
}

void RiscFirmwareInitializer::assert_tensix_workers_impl(tt::ChipId device_id) {
    CoreCoord grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
        }
    }
}

void RiscFirmwareInitializer::assert_inactive_ethernet_cores(tt::ChipId device_id) {
    for (const auto& logical_core : this->get_control_plane_().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
        cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), tt::umd::RiscType::ALL);
    }
}

void RiscFirmwareInitializer::assert_dram_cores(tt::ChipId device_id) {
    bool has_dram_fw = hal_.has_programmable_core_type(HalProgrammableCoreType::DRAM);
    if (has_dram_fw) {
        const auto& soc_d = cluster_.get_soc_desc(device_id);
        for (const auto& dram_core : soc_d.get_cores(CoreType::DRAM, CoordSystem::TRANSLATED)) {
            CoreCoord virtual_core{dram_core.x, dram_core.y};
            cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), tt::umd::RiscType::BRISC);
        }
    }
}

void RiscFirmwareInitializer::terminate_active_ethernet_cores_on_all_chips() {
    if (cluster_.arch() != ARCH::BLACKHOLE ||
        !has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC) ||
        hal_.get_eth_fw_is_cooperative()) {
        return;
    }

    constexpr auto k_ActiveEthCoreType = HalProgrammableCoreType::ACTIVE_ETH;
    auto dev_msgs_factory = hal_.get_dev_msgs_factory(k_ActiveEthCoreType);
    DeviceAddr launch_base_addr = hal_.get_dev_addr(k_ActiveEthCoreType, HalL1MemAddrType::LAUNCH);
    DeviceAddr rd_ptr_addr = hal_.get_dev_addr(k_ActiveEthCoreType, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    auto launch_msg_size = dev_msgs_factory.size_of<dev_msgs::launch_msg_t>();
    auto launch_msg_buf = dev_msgs_factory.create<dev_msgs::launch_msg_t>();

    for (tt::ChipId chip_id : cluster_.all_chip_ids()) {
        for (const auto& logical_core : this->get_control_plane_().get_active_ethernet_cores(chip_id)) {
            CoreCoord virtual_core =
                cluster_.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, CoreType::ETH);
            uint32_t rd_ptr = 0;
            cluster_.read_core(&rd_ptr, sizeof(rd_ptr), tt_cxy_pair(chip_id, virtual_core), rd_ptr_addr);
            rd_ptr &= (dev_msgs::launch_msg_buffer_num_entries - 1);
            DeviceAddr launch_slot_addr = launch_base_addr + (rd_ptr * launch_msg_size);
            cluster_.read_core(
                launch_msg_buf.data(), launch_msg_buf.size(), tt_cxy_pair(chip_id, virtual_core), launch_slot_addr);
            launch_msg_buf.view().kernel_config().exit_erisc_kernel() = 1;
            cluster_.write_core(
                launch_msg_buf.data(), launch_msg_buf.size(), tt_cxy_pair(chip_id, virtual_core), launch_slot_addr);
        }
        // FIX PL (#42429): l1_barrier() routes through the ERISC relay on non-MMIO BH chips.
        // terminate_active_ethernet_cores_on_all_chips() runs in run_launch_phase() before
        // reset_cores() has confirmed relay health. Guard for dead relay.
        const bool chip_is_mmio = cluster_.mmio_chip_ids().count(chip_id);
        if (chip_is_mmio) {
            cluster_.l1_barrier(chip_id);
        } else {
            try {
                cluster_.l1_barrier(chip_id);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogAlways,
                    "terminate_active_ethernet_cores_on_all_chips: l1_barrier timed out on non-MMIO device {} "
                    "(dead ERISC relay): {}. ETH terminate writes may not have flushed.",
                    chip_id,
                    e.what());
            } catch (...) {
                log_warning(
                    tt::LogAlways,
                    "terminate_active_ethernet_cores_on_all_chips: l1_barrier timed out on non-MMIO device {} "
                    "(dead ERISC relay, unknown exception).",
                    chip_id);
            }
        }
    }
}

void RiscFirmwareInitializer::propagate_dead_mmio_peers() {
    // Fixed-point transitive closure: if MMIO device M has an Ethernet-connected
    // peer that is already in mmio_dead_peer_devices_, add M too.  A single-hop
    // dead relay on one MMIO chip can make multi-hop paths through that chip
    // unreachable from neighbouring MMIO chips.  Repeat until stable.
    const auto& all_mmio = cluster_.mmio_chip_ids();
    bool changed = true;
    while (changed) {
        changed = false;
        for (const tt::ChipId mmio_id : all_mmio) {
            if (mmio_dead_peer_devices_.count(mmio_id)) {
                continue;  // already in the set
            }
            for (const tt::ChipId peer : cluster_.get_ethernet_connected_device_ids(mmio_id)) {
                if (mmio_dead_peer_devices_.count(peer)) {
                    mmio_dead_peer_devices_.insert(mmio_id);
                    log_warning(
                        tt::LogAlways,
                        "propagate_dead_mmio_peers: MMIO device {} added — Ethernet peer {} already dead.",
                        mmio_id,
                        peer);
                    changed = true;
                    break;
                }
            }
        }
    }
}

void RiscFirmwareInitializer::reset_cores(tt::ChipId device_id) {
    ZoneScoped;
    std::unordered_map<tt::ChipId, std::unordered_set<CoreCoord>> device_to_early_exit_cores;

    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        if (hal_.get_eth_fw_is_cooperative()) {
            // If the first read on any ETH core for this device throws (dead ERISC relay), all
            // subsequent ETH cores on the same device will also fail — the relay is shared.
            // Track this per-device so we skip the 5-second read_non_mmio timeout for each
            // additional core and go straight to force-reset.
            //
            // Pre-check: if this device's MMIO host is already in mmio_dead_peer_devices_
            // (populated by prior reset_cores() calls and transitively closed), mark the relay
            // dead immediately — no point attempting reads that will each block for 5 seconds.
            const tt::ChipId mmio_host = cluster_.get_associated_mmio_device(device_id);
            bool relay_dead = mmio_dead_peer_devices_.count(mmio_host) > 0;
            if (relay_dead) {
                log_warning(
                    tt::LogAlways,
                    "reset_cores: device {} MMIO host {} in dead-peer set — "
                    "all ETH cores treated as stale without relay reads.",
                    device_id,
                    mmio_host);
            }
            for (const auto& logical_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
                CoreCoord virtual_core =
                    cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
                bool still_running = false;
                if (relay_dead) {
                    // Already confirmed relay is dead for this device — no point attempting
                    // reads that will each block for 5 seconds before throwing. All remaining
                    // ETH cores on this device are treated as stale and force-reset.
                    // Skipping per-core log here — the initial erisc_app_still_running() failure
                    // already reports relay dead and all subsequent cores as stale. Per-core
                    // repetition adds noise without new information.
                    still_running = true;
                } else {
                    // erisc_app_still_running() reads from the ETH core via Cluster::read_core().
                    // For remote (non-MMIO) chips this routes through the UMD legacy ERISC relay.
                    // If that relay is itself stale (left by a killed predecessor process), the
                    // read times out and throws.  In that case the core is definitely stale and
                    // must be included in the force-reset set — treat "can't read" as "still running".
                    try {
                        still_running = erisc_app_still_running(device_id, virtual_core);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogAlways,
                            "reset_cores: erisc_app_still_running() failed for device {} core {} "
                            "(dead ERISC relay on remote chip): {}. Treating all ETH cores on this device as stale.",
                            device_id,
                            virtual_core.str(),
                            e.what());
                        relay_dead = true;
                        still_running = true;
                    } catch (...) {
                        // UMD throws std::runtime_error (caught above); catch(...) is a safety net only.
                        relay_dead = true;
                        still_running = true;
                    }
                    // First detection of a dead relay for this device: mark the MMIO host
                    // and transitively propagate so subsequent devices benefit immediately.
                    if (relay_dead && !mmio_dead_peer_devices_.count(mmio_host)) {
                        mmio_dead_peer_devices_.insert(mmio_host);
                        log_warning(
                            tt::LogAlways,
                            "reset_cores: marking MMIO host {} as dead-peer (dead relay detected on device {}).",
                            mmio_host,
                            device_id);
                        propagate_dead_mmio_peers();
                    }
                }
                // FIX PF (GAP-51): When erisc_app_still_running() returns true on an MMIO device,
                // the non-zero fw_launch_addr may be a stale artifact from initialize_firmware()
                // that was never cleared after the prior process exited. If UMD base firmware is
                // actively running (heartbeat in 0xABCDxxxx format on BH/QA, or incrementing on WH),
                // Metal ETH dispatch is NOT running — sending it a Metal exit signal is pointless
                // (UMD doesn't understand the protocol) and causes a 500ms wait_until_cores_done()
                // timeout per channel × many channels × many tests = cascade of hangs.
                // For MMIO devices: read_reg() goes through PCIe — safe even with a broken relay.
                if (still_running && !relay_dead && device_id == mmio_host) {
                    const uint32_t hb_addr = hal_.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
                    if (hb_addr != 0u) {
                        uint32_t hb_val = 0;
                        try {
                            cluster_.read_reg(&hb_val, tt_cxy_pair(device_id, virtual_core), hb_addr);
                        } catch (...) { /* PCIe read failed — fall through to normal exit path */ }
                        if ((hb_val >> 16) == 0xABCDu) {
                            // UMD base firmware confirmed running. Clear stale fw_launch_addr
                            // and bypass exit signal + 500ms wait entirely.
                            std::vector<uint32_t> clear_flag{0};
                            try {
                                cluster_.write_core_immediate(
                                    device_id,
                                    virtual_core,
                                    clear_flag,
                                    get_active_erisc_launch_flag_addr());
                            } catch (...) {}
                            log_info(
                                tt::LogAlways,
                                "FIX PF: device {} core {} — UMD base fw heartbeat=0x{:08x}; "
                                "skipping Metal exit signal (stale fw_launch_addr cleared).",
                                device_id,
                                virtual_core.str(),
                                hb_val);
                            still_running = false;
                        }
                    }
                }
                if (still_running) {
                    if (relay_dead) {
                        // Relay is dead — erisc_send_exit_signal() would also block for 5s
                        // before throwing. Skip it; the core goes straight to force-reset.
                        log_warning(
                            tt::LogAlways,
                            "reset_cores: skipping erisc_send_exit_signal() for device {} core {} "
                            "(relay dead). Core will be force-reset.",
                            device_id,
                            virtual_core.str());
                    } else {
                        // erisc_send_exit_signal() also reads/writes via the cluster, which may
                        // throw for remote devices with a dead ERISC relay. Catch and continue —
                        // the core will still be added to the force-reset set below.
                        try {
                            erisc_send_exit_signal(device_id, virtual_core, false);
                        } catch (const std::exception& e) {
                            log_warning(
                                tt::LogAlways,
                                "reset_cores: erisc_send_exit_signal() failed for device {} core {} "
                                "(dead ERISC relay): {}. Core will be force-reset.",
                                device_id,
                                virtual_core.str(),
                                e.what());
                            relay_dead = true;
                        } catch (...) {
                            // UMD throws std::runtime_error (caught above); catch(...) is a safety net only.
                            relay_dead = true;
                        }
                    }
                    device_to_early_exit_cores[device_id].insert(virtual_core);
                }
            }
        } else {
            assert_active_ethernet_cores_to_reset(device_id);
        }
    }

    // Track whether any ETH cores on this device were unresponsive. If so, subsequent
    // assert calls that route through the UMD ERISC relay may also time out (the relay
    // is the same stale ERISC). We catch those rather than crashing.
    bool had_unresponsive_eth_cores = false;

    for (auto& id_and_cores : device_to_early_exit_cores) {
        // Use a short timeout: healthy GO-state cores respond to exit signals in <1ms.
        // Cores stuck in RUN_MSG_INIT (killed mid-init) will never respond regardless of
        // how long we wait. 500ms gives healthy cores plenty of margin while avoiding a
        // 10-second hang for the stale-INIT case.
        // skip_dispatch_alert=true: this is an internal reset path — if cores don't
        // respond it's expected and handled by force-reset below. Do NOT trigger
        // on_dispatch_timeout_detected() / tt-triage here.
        const int timeout_ms = 500;
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(
                    id_and_cores.first,
                    dev_msgs::RUN_MSG_GO,
                    id_and_cores.second,
                    timeout_ms,
                    /*skip_dispatch_alert=*/true);
            } catch (std::runtime_error&) {
                had_unresponsive_eth_cores = true;
                log_warning(
                    tt::LogAlways,
                    "Detected dispatch kernels still running but failed to complete an early exit. "
                    "Force-resetting stale ETH cores on device {} to prevent worker L1 corruption by stale ERISC NOC "
                    "traffic.",
                    id_and_cores.first);
                // Force-halt any ETH cores that did not exit cleanly. For local (MMIO-capable) chips this
                // succeeds via the PCIe register path. For remote chips the reset write must route through
                // the UMD legacy ERISC firmware; if that ERISC is itself stale the write will time out, so
                // we catch the resulting exception and continue — at minimum, local-chip stale ERISCs are
                // halted, which prevents them from issuing NOC writes that would corrupt the fresh worker L1
                // firmware written below.
                for (const CoreCoord& virtual_core : id_and_cores.second) {
                    try {
                        cluster_.assert_risc_reset_at_core(
                            tt_cxy_pair(id_and_cores.first, virtual_core), tt::umd::RiscType::ALL);
                        // De-assert reset so the ERISC returns to running base/legacy UMD firmware.
                        // initialize_firmware() for WH cooperative active ETH sends a go message and
                        // requires the ERISC to be running (not halted) to pick up the new firmware.
                        // Without this, the ERISC stays in hardware reset, ETH PHY links go down,
                        // and concurrent topology discovery / dispatch fabric operations hang or crash.
                        cluster_.deassert_risc_reset_at_core(
                            tt_cxy_pair(id_and_cores.first, virtual_core), tt::umd::RiscType::ALL);
                        // FIX PA: clear the ERISC dispatch launch flag after force-reset so that
                        // erisc_app_still_running() correctly sees this core as idle on the next open.
                        // Hardware reset (assert + deassert) halts the ERISC and restarts base UMD
                        // firmware, but does NOT zero L1. The fw_launch_addr flag retains its non-zero
                        // value from the previous dispatch session, causing every subsequent test open
                        // to hit a spurious 500ms wait_until_cores_done timeout, re-trigger the
                        // force-reset, and loop — leaving the test suite unable to make progress.
                        // (Observed: run 25094103200, devices 0-3 stuck in perpetual 500ms stall.)
                        // For MMIO devices: write goes directly via PCIe — no relay required.
                        // For non-MMIO devices: caught below; best-effort (FIX AE handles relay).
                        try {
                            const std::vector<uint32_t> zero = {0};
                            cluster_.write_core_immediate(
                                id_and_cores.first,
                                virtual_core,
                                zero,
                                get_active_erisc_launch_flag_addr());
                        } catch (...) {
                            // Best-effort: MMIO chips always succeed via PCIe;
                            // non-MMIO chips with a dead relay may throw — acceptable.
                        }
                    } catch (const std::exception& reset_err) {
                        log_warning(
                            tt::LogAlways,
                            "Failed to force-reset stale ETH core {} on device {}: {}. "
                            "Worker L1 may be corrupted by stale ERISC traffic.",
                            virtual_core.str(),
                            id_and_cores.first,
                            reset_err.what());
                        // FIX QB (#42429): For non-MMIO devices the assert_risc_reset_at_core
                        // command routes through the UMD legacy ERISC relay. When that relay is
                        // dead, every core in the loop times out at ~5 seconds. Break after the
                        // first failure to avoid an N×5s serial hang (4 ETH cores × 3 non-MMIO
                        // devices = 60s wasted before any real work starts on the next test).
                        // MMIO devices use the PCIe path so each core is fast — no early exit.
                        if (cluster_.get_associated_mmio_device(id_and_cores.first) !=
                            id_and_cores.first) {
                            break;
                        }
                    } catch (...) {
                        // UMD throws std::runtime_error (caught above); catch(...) is a safety net only.
                        if (cluster_.get_associated_mmio_device(id_and_cores.first) !=
                            id_and_cores.first) {
                            break;
                        }
                    }
                }
            }
        }
    }

    // When stale ETH cores were unresponsive, assert calls for remote devices route
    // through the dead ERISC relay and will time out. Catch and log rather than crash.
    //
    // FIX QC: For non-MMIO devices the ERISC relay is the only path to the chip.
    // safe_assert still pays the full 5-second relay timeout before the exception fires,
    // multiplied by every core iterated (120 tensix + inactive ETH). Skip the call
    // entirely for non-MMIO devices with a known-dead relay — we cannot reset those
    // cores anyway, and avoiding the timeouts is critical for staying within CI budgets.
    const bool is_non_mmio = cluster_.get_associated_mmio_device(device_id) != device_id;

    auto safe_assert = [&](auto fn, const char* label) {
        if (had_unresponsive_eth_cores) {
            if (is_non_mmio) {
                log_warning(
                    tt::LogAlways,
                    "reset_cores: skipping {} for non-MMIO device {} (dead ERISC relay — cannot reach chip, avoiding relay timeout)",
                    label,
                    device_id);
                return;
            }
            try {
                fn();
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogAlways,
                    "reset_cores: {} failed for device {} (dead ERISC relay after stale ETH force-reset): {}",
                    label,
                    device_id,
                    e.what());
            } catch (...) {
                log_warning(
                    tt::LogAlways,
                    "reset_cores: {} failed for device {} with unknown exception type (dead ERISC relay after stale ETH force-reset)",
                    label,
                    device_id);
            }
        } else {
            fn();
        }
    };

    safe_assert([&] { assert_tensix_workers_impl(device_id); }, "assert_tensix_workers_impl");
    safe_assert([&] { assert_dram_cores(device_id); }, "assert_dram_cores");
    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        safe_assert([&] { assert_inactive_ethernet_cores(device_id); }, "assert_inactive_ethernet_cores");
    }
    safe_assert([&] { cluster_.l1_barrier(device_id); }, "l1_barrier");
}

void RiscFirmwareInitializer::assert_cores(tt::ChipId device_id) {
    assert_tensix_workers_impl(device_id);
    if (!hal_.get_eth_fw_is_cooperative()) {
        assert_active_ethernet_cores_to_reset(device_id);
    }
    assert_inactive_ethernet_cores(device_id);
    assert_dram_cores(device_id);
}

CoreCoord RiscFirmwareInitializer::virtual_noc0_coordinate(tt::ChipId device_id, uint8_t noc_index, CoreCoord coord) {
    const auto& grid_size = cluster_.get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y || cluster_.arch() == ARCH::BLACKHOLE) {
        return coord;
    }
    coord = cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, coord);
    CoreCoord virtual_coord = {
        hal_.noc_coordinate(noc_index, grid_size.x, coord.x), hal_.noc_coordinate(noc_index, grid_size.y, coord.y)};
    return virtual_coord;
}

void RiscFirmwareInitializer::generate_device_bank_to_noc_tables(tt::ChipId device_id) {
    generate_device_bank_to_noc_tables(
        device_id,
        dram_bank_offset_map_[device_id],
        l1_bank_offset_map_[device_id],
        dram_bank_to_noc_xy_[device_id],
        l1_bank_to_noc_xy_[device_id]);
}

void RiscFirmwareInitializer::generate_device_bank_to_noc_tables(
    tt::ChipId device_id,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<int32_t>& l1_bank_offset_map,
    std::vector<uint16_t>& dram_bank_to_noc_xy,
    std::vector<uint16_t>& l1_bank_to_noc_xy) {
    BankMapping l1_bank_remap(descriptor_->l1_bank_remap().begin(), descriptor_->l1_bank_remap().end());
    auto config = L1BankingAllocator::generate_config(
        descriptor_->metal_context().get_dispatch_core_manager(),
        descriptor_->env_impl(),
        device_id,
        num_hw_cqs_,
        DEFAULT_L1_SMALL_SIZE,      // Not required for noc table gen
        DEFAULT_TRACE_REGION_SIZE,  // Not required for noc table gen
        worker_l1_unreserved_start_,
        l1_bank_remap);
    const auto allocator = L1BankingAllocator(config);
    const auto& soc_d = cluster_.get_soc_desc(device_id);
    const size_t num_dram_banks = allocator.get_num_banks(BufferType::DRAM);
    dram_bank_offset_map.clear();
    dram_bank_offset_map.resize(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        dram_bank_offset_map[bank_id] = allocator.get_bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = allocator.get_num_banks(BufferType::L1);
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    l1_bank_offset_map.clear();
    l1_bank_offset_map.resize(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] = cluster_.get_virtual_coordinate_from_logical_coordinates(
            device_id, allocator.get_logical_core_from_bank_id(bank_id), CoreType::WORKER);
        l1_bank_offset_map[bank_id] = allocator.get_bank_offset(BufferType::L1, bank_id);
    }

    dram_bank_to_noc_xy.clear();
    dram_bank_to_noc_xy.reserve(hal_.get_num_nocs() * num_dram_banks);
    bool noc_translation_enabled =
        !cluster_.is_mock_or_emulated() && cluster_.get_cluster_desc()->get_noc_translation_table_en().at(device_id);
    bool dram_is_virtualized =
        noc_translation_enabled && (hal_.get_virtualized_core_types().contains(dev_msgs::AddressableCoreType::DRAM));
    for (unsigned int noc = 0; noc < hal_.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < num_dram_banks; bank_id++) {
            CoreCoord dram_noc_coord =
                soc_d.get_preferred_worker_core_for_dram_view(allocator.get_dram_channel_from_bank_id(bank_id), noc);
            uint16_t noc_x, noc_y;
            if (dram_is_virtualized) {
                noc_x = dram_noc_coord.x;
                noc_y = dram_noc_coord.y;
            } else {
                noc_x = hal_.noc_coordinate(noc, soc_d.grid_size.x, dram_noc_coord.x);
                noc_y = hal_.noc_coordinate(noc, soc_d.grid_size.y, dram_noc_coord.y);
            }
            uint16_t xy = ((noc_y << hal_.get_noc_addr_node_id_bits()) | noc_x) << hal_.get_noc_coord_reg_offset();
            dram_bank_to_noc_xy.push_back(xy);
        }
    }

    l1_bank_to_noc_xy.clear();
    l1_bank_to_noc_xy.reserve(hal_.get_num_nocs() * l1_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < hal_.get_num_nocs(); noc++) {
        for (const auto& noc_coord : l1_noc_coord_per_bank) {
            auto l1_noc_coords = virtual_noc0_coordinate(device_id, noc, noc_coord);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << hal_.get_noc_addr_node_id_bits()) | noc_x) << hal_.get_noc_coord_reg_offset();
            l1_bank_to_noc_xy.push_back(xy);
        }
    }
}

void RiscFirmwareInitializer::generate_worker_logical_to_virtual_map(tt::ChipId device_id) {
    generate_worker_logical_to_virtual_map(
        device_id, worker_logical_col_to_virtual_col_[device_id], worker_logical_row_to_virtual_row_[device_id]);
}

void RiscFirmwareInitializer::generate_worker_logical_to_virtual_map(
    tt::ChipId device_id,
    std::vector<uint8_t>& worker_logical_col_to_virtual_col,
    std::vector<uint8_t>& worker_logical_row_to_virtual_row) {
    const auto& soc_desc = cluster_.get_soc_desc(device_id);
    auto tensix_grid_size = soc_desc.get_grid_size(CoreType::TENSIX);

    worker_logical_col_to_virtual_col.clear();
    worker_logical_row_to_virtual_row.clear();
    worker_logical_col_to_virtual_col.reserve(tensix_grid_size.x);
    worker_logical_row_to_virtual_row.reserve(tensix_grid_size.y);

    for (size_t x = 0; x < tensix_grid_size.x; x++) {
        worker_logical_col_to_virtual_col.push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{x, 0}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .x);
    }
    for (size_t y = 0; y < tensix_grid_size.y; y++) {
        worker_logical_row_to_virtual_row.push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{0, y}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .y);
    }
}

void RiscFirmwareInitializer::initialize_device_bank_to_noc_tables(
    tt::ChipId device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    std::optional<CoreCoord> end_core) {
    // Firmware uses bank_noc_xy_t (uint32_t for Quasar configs with odd bank counts, uint16_t otherwise)
    // to ensure table sizes are always 4-byte aligned for l1_to_local_mem_copy. Match that here.
    const uint32_t dram_noc_xy_size = dram_bank_to_noc_xy_[device_id].size();
    const uint32_t l1_noc_xy_size = l1_bank_to_noc_xy_[device_id].size();
    const uint32_t num_nocs = hal_.get_num_nocs();
    const bool use_u32_entries = (cluster_.arch() == tt::ARCH::QUASAR) &&
                                 ((dram_noc_xy_size / num_nocs) % 2 != 0 || (l1_noc_xy_size / num_nocs) % 2 != 0);

    std::vector<uint32_t> dram_noc_xy_padded, l1_noc_xy_padded;
    void* dram_noc_data;
    void* l1_noc_data;
    uint32_t dram_to_noc_sz_in_bytes;
    uint32_t l1_to_noc_sz_in_bytes;

    if (use_u32_entries) {
        dram_noc_xy_padded.assign(dram_bank_to_noc_xy_[device_id].begin(), dram_bank_to_noc_xy_[device_id].end());
        l1_noc_xy_padded.assign(l1_bank_to_noc_xy_[device_id].begin(), l1_bank_to_noc_xy_[device_id].end());
        dram_noc_data = dram_noc_xy_padded.data();
        l1_noc_data = l1_noc_xy_padded.data();
        dram_to_noc_sz_in_bytes = dram_noc_xy_size * sizeof(uint32_t);
        l1_to_noc_sz_in_bytes = l1_noc_xy_size * sizeof(uint32_t);
    } else {
        dram_noc_data = dram_bank_to_noc_xy_[device_id].data();
        l1_noc_data = l1_bank_to_noc_xy_[device_id].data();
        dram_to_noc_sz_in_bytes = dram_noc_xy_size * sizeof(uint16_t);
        l1_to_noc_sz_in_bytes = l1_noc_xy_size * sizeof(uint16_t);
    }

    const uint32_t dram_offset_sz_in_bytes = dram_bank_offset_map_[device_id].size() * sizeof(int32_t);
    const uint32_t l1_offset_sz_in_bytes = l1_bank_offset_map_[device_id].size() * sizeof(int32_t);

    const uint64_t mem_bank_to_noc_addr = hal_.get_dev_noc_addr(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t mem_bank_to_noc_size = hal_.get_dev_size(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    TT_ASSERT(
        (dram_to_noc_sz_in_bytes + l1_to_noc_sz_in_bytes + dram_offset_sz_in_bytes + l1_offset_sz_in_bytes) <=
            mem_bank_to_noc_size,
        "Size of bank_to_noc table is greater than available space");

    if (end_core.has_value()) {
        auto start_core = virtual_core;
        cluster_.noc_multicast_write(
            dram_noc_data, dram_to_noc_sz_in_bytes, device_id, start_core, end_core.value(), mem_bank_to_noc_addr);

        uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
        cluster_.noc_multicast_write(
            l1_noc_data, l1_to_noc_sz_in_bytes, device_id, start_core, end_core.value(), l1_noc_addr);

        uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
        cluster_.noc_multicast_write(
            dram_bank_offset_map_[device_id].data(),
            dram_offset_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            dram_offset_addr);

        uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
        cluster_.noc_multicast_write(
            l1_bank_offset_map_[device_id].data(),
            l1_offset_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            l1_offset_addr);
    } else {
        cluster_.write_core(
            dram_noc_data, dram_to_noc_sz_in_bytes, tt_cxy_pair(device_id, virtual_core), mem_bank_to_noc_addr);

        uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
        cluster_.write_core(l1_noc_data, l1_to_noc_sz_in_bytes, tt_cxy_pair(device_id, virtual_core), l1_noc_addr);

        uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
        cluster_.write_core(
            dram_bank_offset_map_[device_id].data(),
            dram_offset_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            dram_offset_addr);

        uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
        cluster_.write_core(
            l1_bank_offset_map_[device_id].data(),
            l1_offset_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            l1_offset_addr);
    }
}

void RiscFirmwareInitializer::initialize_worker_logical_to_virtual_tables(
    tt::ChipId device_id, const HalProgrammableCoreType& core_type, CoreCoord start_core, CoreCoord end_core) {
    const auto& soc_desc = cluster_.get_soc_desc(device_id);
    const uint32_t logical_col_to_virtual_col_sz_in_bytes =
        worker_logical_col_to_virtual_col_[device_id].size() * sizeof(uint8_t);
    const uint8_t firmware_grid_size_x = tt::round_up(soc_desc.grid_size.x, 4);
    const uint32_t logical_row_to_virtual_row_sz_in_bytes =
        worker_logical_row_to_virtual_row_[device_id].size() * sizeof(uint8_t);
    const uint64_t logical_to_virtual_map_addr =
        hal_.get_dev_addr(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    const uint32_t logical_to_virtual_map_size =
        hal_.get_dev_size(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);

    TT_ASSERT(
        (firmware_grid_size_x + logical_row_to_virtual_row_sz_in_bytes) <= logical_to_virtual_map_size,
        "Size of logical to virtual map is greater than available space");

    uint64_t logical_col_to_virtual_col_addr = logical_to_virtual_map_addr;
    cluster_.noc_multicast_write(
        worker_logical_col_to_virtual_col_[device_id].data(),
        logical_col_to_virtual_col_sz_in_bytes,
        device_id,
        start_core,
        end_core,
        logical_col_to_virtual_col_addr);

    uint64_t logical_row_to_virtual_row_addr = logical_to_virtual_map_addr + (firmware_grid_size_x * sizeof(uint8_t));
    cluster_.noc_multicast_write(
        worker_logical_row_to_virtual_row_[device_id].data(),
        logical_row_to_virtual_row_sz_in_bytes,
        device_id,
        start_core,
        end_core,
        logical_row_to_virtual_row_addr);
}

uint32_t RiscFirmwareInitializer::get_active_erisc_launch_flag_addr() {
    auto core_type_idx = hal_.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    return hal_.get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr;
}

bool RiscFirmwareInitializer::erisc_app_still_running(tt::ChipId device_id, CoreCoord virtual_core) {
    if (cluster_.arch() != ARCH::WORMHOLE_B0) {
        return false;
    }
    TT_ASSERT(
        cluster_.is_ethernet_core(virtual_core, device_id),
        "Invalid core {} for context switch check",
        virtual_core.str());
    std::uint32_t launch_erisc_addr = get_active_erisc_launch_flag_addr();
    auto data = cluster_.read_core(device_id, virtual_core, launch_erisc_addr, sizeof(std::uint32_t));
    return (data[0] != 0);
}

void RiscFirmwareInitializer::erisc_send_exit_signal(tt::ChipId device_id, CoreCoord virtual_core, bool is_idle_eth) {
    HalProgrammableCoreType programmable_core_type =
        is_idle_eth ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    auto dev_msgs_factory = hal_.get_dev_msgs_factory(programmable_core_type);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    DeviceAddr launch_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH);

    cluster_.read_core(
        launch_msg.data(), launch_msg.size(), {static_cast<size_t>(device_id), virtual_core}, launch_addr);

    launch_msg.view().kernel_config().exit_erisc_kernel() = 1;
    llrt::write_launch_msg_to_core(device_id, virtual_core, launch_msg.view(), go_msg.view(), false);

    if (!is_idle_eth) {
        std::vector<uint32_t> clear_flag_data = {0};
        cluster_.write_core_immediate(device_id, virtual_core, clear_flag_data, get_active_erisc_launch_flag_addr());
    }
}

dev_msgs::core_info_msg_t RiscFirmwareInitializer::populate_core_info_msg(
    tt::ChipId device_id, HalProgrammableCoreType programmable_core_type) const {
    const metal_SocDescriptor& soc_d = cluster_.get_soc_desc(device_id);
    auto factory = hal_.get_dev_msgs_factory(programmable_core_type);
    dev_msgs::core_info_msg_t buffer = factory.create<dev_msgs::core_info_msg_t>();
    auto core_info = buffer.view();
    core_info.noc_pcie_addr_base() = hal_.get_pcie_addr_lower_bound();
    core_info.noc_pcie_addr_end() = hal_.get_pcie_addr_upper_bound();
    core_info.noc_dram_addr_base() = 0;
    core_info.noc_dram_addr_end() = soc_d.dram_core_size;
    core_info.l1_unreserved_start() = align(worker_l1_unreserved_start_, hal_.get_alignment(HalMemType::DRAM));
    if (programmable_core_type == HalProgrammableCoreType::TENSIX) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::WORKER;
    } else if (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::ACTIVE_ETH;
    } else if (programmable_core_type == HalProgrammableCoreType::DRAM) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::DRAM;
    } else {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::IDLE_ETH;
    }
    const std::vector<tt::umd::CoreCoord>& pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    std::unordered_set<tt::umd::CoreCoord> dram_cores;
    auto num_dram_channels = cluster_.get_soc_desc(device_id).get_num_dram_views();
    for (uint32_t dram_channel = 0; dram_channel < num_dram_channels; dram_channel++) {
        for (uint32_t noc = 0; noc < hal_.get_num_nocs(); noc++) {
            auto worker_dram_ep = soc_d.get_preferred_worker_core_for_dram_view(dram_channel, noc);
            auto eth_dram_ep = soc_d.get_preferred_eth_core_for_dram_view(dram_channel, noc);
            auto physical_worker_dram_ep =
                soc_d.translate_coord_to(worker_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            auto physical_eth_dram_ep =
                soc_d.translate_coord_to(eth_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            dram_cores.insert(physical_worker_dram_ep);
            dram_cores.insert(physical_eth_dram_ep);
        }
    }

    const std::vector<tt::umd::CoreCoord>& eth_cores = soc_d.get_cores(CoreType::ETH, CoordSystem::NOC0);

    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= core_info.non_worker_cores().size(),
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= core_info.virtual_non_worker_cores().size(),
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    auto set_addressable_core =
        [](dev_msgs::addressable_core_t::View core, const CoreCoord& core_coord, dev_msgs::AddressableCoreType type) {
            core.x() = core_coord.x;
            core.y() = core_coord.y;
            core.type() = type;
        };
    for (auto non_worker_core : core_info.non_worker_cores()) {
        set_addressable_core(
            non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    for (auto virtual_non_worker_core : core_info.virtual_non_worker_cores()) {
        set_addressable_core(
            virtual_non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    int non_worker_cores_idx = 0;
    bool skip_physical = cluster_.arch() == ARCH::BLACKHOLE and hal_.is_coordinate_virtualization_enabled();
    if (not skip_physical) {
        for (tt::umd::CoreCoord core : pcie_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::PCIE);
        }
        for (tt::umd::CoreCoord core : dram_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::DRAM);
        }
        for (tt::umd::CoreCoord core : eth_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::ETH);
        }
    }

    if (hal_.is_coordinate_virtualization_enabled()) {
        uint32_t virtual_non_worker_cores_idx = 0;
        for (tt::umd::CoreCoord core : eth_cores) {
            auto virtual_core = cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
            set_addressable_core(
                core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                virtual_core,
                dev_msgs::AddressableCoreType::ETH);
        }

        if (cluster_.arch() == ARCH::BLACKHOLE) {
            for (const CoreCoord& core : pcie_cores) {
                auto virtual_core =
                    cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::PCIE);
            }

            for (const CoreCoord& core : dram_cores) {
                auto virtual_core =
                    cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::DRAM);
            }
        }
    }

    std::vector<uint32_t> harvested_axis_coord;
    CoreCoord logical_grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t harvested_noc_coords = umd::CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        cluster_.get_soc_desc(device_id).arch, cluster_.get_harvesting_mask(device_id));
    uint32_t max_along_axis =
        hal_.get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW ? soc_d.grid_size.y : soc_d.grid_size.x;
    for (uint32_t idx = 0; idx < max_along_axis; idx++) {
        bool harvested_axis = (harvested_noc_coords >> idx) & 0x1;
        if (harvested_axis) {
            harvested_axis_coord.push_back(idx);
        }
    }
    TT_ASSERT(
        harvested_axis_coord.size() <= core_info.harvested_coords().size(),
        "Detected more harvested rows than fit in mailbox.");
    for (size_t idx = 0; idx < core_info.harvested_coords().size(); idx++) {
        core_info.harvested_coords()[idx] =
            (idx < harvested_axis_coord.size()) ? harvested_axis_coord[idx] : dev_msgs::CORE_COORD_INVALID;
        if (hal_.is_coordinate_virtualization_enabled() and idx < harvested_axis_coord.size()) {
            uint32_t end_virtual_grid;
            if (hal_.get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW) {
                end_virtual_grid = hal_.get_virtual_worker_start_y() + logical_grid_size.y;
            } else if (cluster_.arch() == ARCH::BLACKHOLE) {
                end_virtual_grid = max_along_axis - 1;
            } else {
                end_virtual_grid = hal_.get_virtual_worker_start_x() + logical_grid_size.x;
            }
            core_info.virtual_harvested_coords()[idx] = end_virtual_grid + harvested_axis_coord.size() - (idx + 1);
        } else {
            core_info.virtual_harvested_coords()[idx] = dev_msgs::CORE_COORD_INVALID;
        }
    }

    core_info.noc_size_x() = soc_d.grid_size.x;
    core_info.noc_size_y() = soc_d.grid_size.y;
    core_info.worker_grid_size_x() = logical_grid_size.x;
    core_info.worker_grid_size_y() = logical_grid_size.y;

    return buffer;
}

void RiscFirmwareInitializer::initialize_firmware(
    tt::ChipId device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    dev_msgs::launch_msg_t::View launch_msg,
    dev_msgs::go_msg_t::ConstView go_msg,
    std::optional<CoreCoord> end_core) {
    ZoneScoped;

    TT_FATAL(
        core_type != HalProgrammableCoreType::TENSIX or end_core.has_value(),
        "Tensix cores require end_core to be specified for bank to noc table initialization.");

    initialize_device_bank_to_noc_tables(device_id, core_type, virtual_core, end_core);
    if (core_type == HalProgrammableCoreType::TENSIX) {
        initialize_worker_logical_to_virtual_tables(device_id, core_type, virtual_core, end_core.value());
    }

    uint32_t core_type_idx = hal_.get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal_.get_processor_classes_count(core_type);
    auto jit_build_config = hal_.get_jit_build_config(core_type_idx, 0, 0);

    const auto start_core = virtual_core;

    size_t launch_msg_size = launch_msg.size();
    std::vector<std::byte> init_launch_msg_data(
        dev_msgs::launch_msg_buffer_num_entries * launch_msg_size, std::byte{0});
    auto prepare_initial_launch_msg = [&]() {
        for (size_t i = 0; i < dev_msgs::launch_msg_buffer_num_entries; ++i) {
            std::copy(
                launch_msg.data(),
                launch_msg.data() + launch_msg_size,
                init_launch_msg_data.data() + (i * launch_msg_size));
        }
    };
    const auto write_initial_go_launch_msg = [&]() {
        auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
        uint32_t launch_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH);
        uint32_t go_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG);
        uint64_t launch_msg_buffer_read_ptr_addr =
            hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
        uint32_t go_message_index_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG_INDEX);
        if (core_type != HalProgrammableCoreType::TENSIX) {
            cluster_.write_core(
                init_launch_msg_data.data(),
                init_launch_msg_data.size(),
                tt_cxy_pair(device_id, virtual_core),
                launch_addr);
            cluster_.write_core(go_msg.data(), go_msg.size(), tt_cxy_pair(device_id, virtual_core), go_addr);
            uint32_t zero = 0;
            cluster_.write_core(
                &zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), launch_msg_buffer_read_ptr_addr);
            cluster_.write_core(&zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), go_message_index_addr);
        } else {
            cluster_.noc_multicast_write(
                init_launch_msg_data.data(),
                init_launch_msg_data.size(),
                device_id,
                start_core,
                end_core.value(),
                launch_addr);
            cluster_.noc_multicast_write(
                go_msg.data(), go_msg.size(), device_id, start_core, end_core.value(), go_addr);
            uint32_t zero = 0;
            cluster_.noc_multicast_write(
                &zero, sizeof(uint32_t), device_id, start_core, end_core.value(), launch_msg_buffer_read_ptr_addr);
            cluster_.noc_multicast_write(
                &zero, sizeof(uint32_t), device_id, start_core, end_core.value(), go_message_index_addr);
        }
    };

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [_, num_build_states] = BuildEnvManager::get_instance().get_build_index_and_state_count(
                    core_type_idx, processor_class, true);
                for (uint32_t riscv_id = 0; riscv_id < num_build_states; riscv_id++) {
                    auto fw_path = BuildEnvManager::get_instance().get_firmware_binary_path(
                        device_id, core_type_idx, processor_class, riscv_id);
                    const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                    uint32_t fw_size = binary_mem.get_text_size();
                    hal_.set_iram_text_size(
                        launch_msg, core_type, static_cast<HalProcessorClassType>(processor_class), riscv_id, fw_size);

                    if (not rtoptions_.get_skip_loading_fw()) {
                        llrt::test_load_multicast_write_risc_binary(
                            binary_mem,
                            device_id,
                            start_core,
                            end_core.value(),
                            core_type_idx,
                            processor_class,
                            riscv_id);
                    }
                }
            }

            if (!rtoptions_.get_fast_dispatch()) {
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
            } else {
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_DEV;
            }
            prepare_initial_launch_msg();
            write_initial_go_launch_msg();
            if (rtoptions_.get_fast_dispatch() && dispatch_core_manager_.get_dispatch_core_type() == CoreType::WORKER) {
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
                prepare_initial_launch_msg();
                for (const auto& logical_core : dispatch_core_manager_.get_all_logical_dispatch_cores(device_id)) {
                    auto virtual_dispatch_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, CoreType::WORKER);
                    auto programmable_core_type = llrt::get_core_type(device_id, virtual_dispatch_core);
                    cluster_.write_core(
                        init_launch_msg_data.data(),
                        init_launch_msg_data.size(),
                        tt_cxy_pair(device_id, virtual_dispatch_core),
                        hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
                }
            }

            cluster_.noc_multicast_write(
                &jit_build_config.fw_launch_addr_value,
                sizeof(uint32_t),
                device_id,
                start_core,
                end_core.value(),
                jit_build_config.fw_launch_addr);

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
                break;
            }
            const bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            const bool is_active_eth = !is_idle_eth;
            tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX;
            if (is_active_eth) {
                reset_val &= ~tt::umd::RiscType::ERISC0;
            }
            if (is_idle_eth or !hal_.get_eth_fw_is_cooperative()) {
                cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
            }
            if (not rtoptions_.get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto num_build_states = hal_.get_processor_types_count(core_type_idx, processor_class);
                    for (uint32_t eriscv_id = 0; eriscv_id < num_build_states; eriscv_id++) {
                        auto fw_path = BuildEnvManager::get_instance().get_firmware_binary_path(
                            device_id, core_type_idx, processor_class, eriscv_id);
                        const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, eriscv_id);
                    }
                }
            }
            launch_msg.kernel_config().mode() = (!rtoptions_.get_fast_dispatch() or is_idle_eth)
                                                    ? dev_msgs::DISPATCH_MODE_HOST
                                                    : dev_msgs::DISPATCH_MODE_DEV;
            prepare_initial_launch_msg();
            write_initial_go_launch_msg();
            if (core_type == HalProgrammableCoreType::ACTIVE_ETH) {
                DeviceAddr mailbox_addr = hal_.get_dev_addr(core_type, HalL1MemAddrType::MAILBOX);
                auto factory = hal_.get_dev_msgs_factory(core_type);
                DeviceAddr ncrisc_halt_addr =
                    mailbox_addr + factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::ncrisc_halt);
                std::vector<uint8_t> data(factory.size_of<dev_msgs::ncrisc_halt_msg_t>(), 0);
                cluster_.write_core(data.data(), data.size(), tt_cxy_pair(device_id, virtual_core), ncrisc_halt_addr);
            }

            if (hal_.get_eth_fw_is_cooperative() || core_type != HalProgrammableCoreType::ACTIVE_ETH ||
                !rtoptions_.get_enable_2_erisc_mode()) {
                cluster_.write_core(
                    &jit_build_config.fw_launch_addr_value,
                    sizeof(uint32_t),
                    tt_cxy_pair(device_id, virtual_core),
                    jit_build_config.fw_launch_addr);
            } else {
                constexpr uint32_t mailbox_index = 0;
                tt::llrt::internal_::send_msg_to_eth_mailbox(
                    device_id,
                    virtual_core,
                    tt_metal::FWMailboxMsg::ETH_MSG_RELEASE_CORE,
                    mailbox_index,
                    {/*l1 addr to exec*/ jit_build_config.fw_launch_addr_value},
                    false);
            }

            break;
        }
        case HalProgrammableCoreType::DRAM: {
            cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), tt::umd::RiscType::BRISC);
            if (not rtoptions_.get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto num_build_states = hal_.get_processor_types_count(core_type_idx, processor_class);
                    for (uint32_t drisc_id = 0; drisc_id < num_build_states; drisc_id++) {
                        auto fw_path = BuildEnvManager::get_instance().get_firmware_binary_path(
                            device_id, core_type_idx, processor_class, drisc_id);
                        const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, drisc_id);
                    }
                }
            }
            launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
            prepare_initial_launch_msg();

            uint64_t launch_addr = hal_.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::LAUNCH);
            uint64_t go_addr = hal_.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG);
            uint64_t launch_msg_rd_ptr_addr =
                hal_.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
            uint64_t go_message_index_addr =
                hal_.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG_INDEX);
            cluster_.write_core(
                init_launch_msg_data.data(),
                init_launch_msg_data.size(),
                tt_cxy_pair(device_id, virtual_core),
                launch_addr);
            cluster_.write_core(go_msg.data(), go_msg.size(), tt_cxy_pair(device_id, virtual_core), go_addr);
            uint32_t zero = 0;
            cluster_.write_core(&zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), launch_msg_rd_ptr_addr);
            cluster_.write_core(&zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), go_message_index_addr);

            // Write reset PC (register address, no L1 NOC offset needed)
            cluster_.write_core(
                &jit_build_config.fw_launch_addr_value,
                sizeof(uint32_t),
                tt_cxy_pair(device_id, virtual_core),
                jit_build_config.fw_launch_addr);
            break;
        }
        default:
            TT_THROW(
                "Unsupported programable core type {} to initialize build states", enchantum::to_string(core_type));
    }
}

void RiscFirmwareInitializer::initialize_and_launch_firmware(tt::ChipId device_id) {
    ZoneScoped;

    log_debug(tt::LogMetal, "Initializing worker cores");
    std::unordered_set<CoreCoord> not_done_cores;
    CoreCoord logical_grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);

    auto dev_msgs_factory = hal_.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::TENSIX);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    for (uint32_t y = 0; y < logical_grid_size.y; y++) {
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            core_info.view().absolute_logical_x() = logical_core.x;
            core_info.view().absolute_logical_y() = logical_core.y;
            cluster_.write_core_immediate(
                core_info.data(),
                core_info.size(),
                {static_cast<size_t>(device_id), worker_core},
                hal_.get_dev_addr(llrt::get_core_type(device_id, worker_core), HalL1MemAddrType::CORE_INFO));
            not_done_cores.insert(worker_core);
        }
    }
    CoreCoord start_core =
        cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord(0, 0), CoreType::WORKER);
    CoreCoord end_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
        device_id, CoreCoord(logical_grid_size.x - 1, logical_grid_size.y - 1), CoreType::WORKER);
    initialize_firmware(
        device_id, HalProgrammableCoreType::TENSIX, start_core, launch_msg.view(), go_msg.view(), end_core);

    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        static std::vector<uint32_t> zero_vec_erisc_init(
            hal_.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t),
            0);

        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);

        cluster_.write_core_immediate(
            device_id,
            virtual_core,
            zero_vec_erisc_init,
            hal_.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
    }

    log_debug(tt::LogMetal, "Initializing active ethernet cores");
    dev_msgs_factory = hal_.get_dev_msgs_factory(HalProgrammableCoreType::ACTIVE_ETH);
    core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::ACTIVE_ETH);
    launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    std::unordered_set<CoreCoord> multi_risc_active_eth_cores;
    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info.view().absolute_logical_x() = eth_core.x;
        core_info.view().absolute_logical_y() = eth_core.y;
        cluster_.write_core_immediate(
            core_info.data(),
            core_info.size(),
            {static_cast<size_t>(device_id), virtual_core},
            hal_.get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(
            device_id, HalProgrammableCoreType::ACTIVE_ETH, virtual_core, launch_msg.view(), go_msg.view());
        if (!hal_.get_eth_fw_is_cooperative()) {
            multi_risc_active_eth_cores.insert(virtual_core);
            not_done_cores.insert(virtual_core);
        }
    }

    log_debug(tt::LogMetal, "Initializing idle ethernet cores");
    dev_msgs_factory = hal_.get_dev_msgs_factory(HalProgrammableCoreType::IDLE_ETH);
    core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::IDLE_ETH);
    launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;
    for (const auto& eth_core : this->get_control_plane_().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info.view().absolute_logical_x() = eth_core.x;
        core_info.view().absolute_logical_y() = eth_core.y;
        cluster_.write_core_immediate(
            core_info.data(),
            core_info.size(),
            {static_cast<size_t>(device_id), virtual_core},
            hal_.get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(
            device_id, HalProgrammableCoreType::IDLE_ETH, virtual_core, launch_msg.view(), go_msg.view());
        // FIX SB (GAP-76): initialize_firmware() for IDLE_ETH returns early (breaks) when
        // INIT_FABRIC is not set — it writes no go_msg and does not assert risc reset.
        // If we add the core to not_done_cores unconditionally, deassert_risc_reset_at_core
        // is later called for it, the core starts from stale L1 firmware that writes 0x55
        // to run_mailbox, and wait_until_cores_done TT_FATALs.
        // Guard matches the early-break condition inside initialize_firmware() itself.
        if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            not_done_cores.insert(virtual_core);
        }
    }

    std::unordered_set<CoreCoord> dram_not_done_cores;
    bool has_dram_fw =
        hal_.get_programmable_core_type_index(HalProgrammableCoreType::DRAM) < hal_.get_programmable_core_type_count();
    if (has_dram_fw) {
        log_debug(tt::LogMetal, "Initializing DRAM cores");
        auto dram_dev_msgs_factory = hal_.get_dev_msgs_factory(HalProgrammableCoreType::DRAM);
        auto dram_core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::DRAM);
        auto dram_launch_msg = dram_dev_msgs_factory.create<dev_msgs::launch_msg_t>();
        auto dram_go_msg = dram_dev_msgs_factory.create<dev_msgs::go_msg_t>();
        dram_go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;
        const metal_SocDescriptor& soc_d = cluster_.get_soc_desc(device_id);
        for (const auto& dram_noc : soc_d.get_cores(CoreType::DRAM, CoordSystem::TRANSLATED)) {
            CoreCoord virtual_dram_core{dram_noc.x, dram_noc.y};
            dram_core_info.view().absolute_logical_x() = dram_noc.x;
            dram_core_info.view().absolute_logical_y() = dram_noc.y;
            uint64_t core_info_addr = hal_.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::CORE_INFO);
            cluster_.write_core(
                dram_core_info.data(),
                dram_core_info.size(),
                {static_cast<size_t>(device_id), virtual_dram_core},
                core_info_addr);
            initialize_firmware(
                device_id,
                HalProgrammableCoreType::DRAM,
                virtual_dram_core,
                dram_launch_msg.view(),
                dram_go_msg.view());
            dram_not_done_cores.insert(virtual_dram_core);
        }
    }

    cluster_.l1_barrier(device_id);

    // FIX XX (#42429): Guard deassert_risc_reset for non-MMIO devices whose relay broke
    // DURING the multicast phase (FIX AE marks relay broken mid-init).  FIX NZ skips the
    // entire initialize_and_launch_firmware call when relay is broken BEFORE entry, but if
    // the relay dies mid-flight the multicast silently fails to deliver firmware/go_msg to
    // Tensix cores.  Deasserting reset on cores that never received firmware causes them to
    // boot from stale SRAM (e.g. go_msg=0x02) → FIX SC fires → cores dead but no hang.
    // Skip deassert entirely when the relay is known-broken to avoid starting stale cores.
    {
        const bool is_non_mmio = cluster_.get_associated_mmio_device(device_id) != device_id;
        if (is_non_mmio && cluster_.is_relay_broken(device_id)) {
            log_warning(
                tt::LogAlways,
                "FIX XX (#42429): skipping deassert_risc_reset + wait for non-MMIO device {} "
                "(relay broken mid-init — firmware multicast likely failed). "
                "Cores left in reset; board reset may be required.",
                device_id);
            return;
        }
    }

    for (const auto& worker_core : not_done_cores) {
        if (multi_risc_active_eth_cores.contains(worker_core) && rtoptions_.get_enable_2_erisc_mode()) {
            continue;
        }

        tt::umd::RiscType reset_val;
        if (cluster_.arch() == ARCH::QUASAR) {
            reset_val = tt::umd::RiscType::ALL_NEO_DMS;
        } else {
            reset_val = tt::umd::RiscType::BRISC;
            if (multi_risc_active_eth_cores.contains(worker_core)) {
                reset_val |= tt::umd::RiscType::ERISC1;
            }
        }
        cluster_.deassert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), reset_val);
    }
    for (const auto& dram_core : dram_not_done_cores) {
        cluster_.deassert_risc_reset_at_core(tt_cxy_pair(device_id, dram_core), tt::umd::RiscType::BRISC);
    }

    // FIX SB (GAP-76): Pre-scan for stale go_msg signal values (e.g. 0x55) that survive a
    // tt-smi reset because Tensix SRAM is not cleared.  After the NOC multicast write +
    // l1_barrier + deassert_risc_reset, any core that still has an unknown signal value is
    // running stale firmware that will never write RUN_MSG_DONE.  Writing RUN_MSG_DONE here
    // via PCIe prevents wait_until_cores_done from spinning for 10 s and crashing with
    // TT_THROW.  The board is still in a degraded state — conftest will detect the FIX SA
    // WARNING and trigger tt-smi -r — but at least we fail fast with a clear diagnostic
    // rather than a 10 s freeze per affected core.
    {
        static constexpr std::array<uint8_t, 6> kKnownRunMsgValues = {
            static_cast<uint8_t>(dev_msgs::RUN_MSG_DONE),
            static_cast<uint8_t>(dev_msgs::RUN_MSG_INIT),
            static_cast<uint8_t>(dev_msgs::RUN_MSG_GO),
            static_cast<uint8_t>(dev_msgs::RUN_MSG_RESET_READ_PTR),
            static_cast<uint8_t>(dev_msgs::RUN_MSG_RESET_READ_PTR_FROM_HOST),
            static_cast<uint8_t>(dev_msgs::RUN_MSG_REPLAY_TRACE),
        };
        auto done_go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
        done_go_msg.view().signal() = dev_msgs::RUN_MSG_DONE;
        for (const auto& worker_core : not_done_cores) {
            // FIX SC-ADDR (GAP-76): use the correct go_msg address per core type.
            // ETH cores (ACTIVE_ETH / IDLE_ETH) keep go_msg at GET_ETH_MAILBOX_ADDRESS_HOST(go_messages),
            // which differs from the Tensix go_msg address.  Using the Tensix address for an ETH core
            // reads garbage from the wrong L1 offset → signal 0x02 (unknown) → FIX SC fires →
            // RUN_MSG_DONE written to wrong ETH L1 address → potential ETH dispatch FW corruption.
            const HalProgrammableCoreType worker_core_type = llrt::get_core_type(device_id, worker_core);
            // Build a short type string for diagnostic logs — TENSIX / ACTIVE_ETH / IDLE_ETH.
            const char* core_type_str = [&]() -> const char* {
                switch (worker_core_type) {
                    case HalProgrammableCoreType::TENSIX:     return "TENSIX";
                    case HalProgrammableCoreType::ACTIVE_ETH: return "ACTIVE_ETH";
                    case HalProgrammableCoreType::IDLE_ETH:   return "IDLE_ETH";
                    default:                                   return "UNKNOWN";
                }
            }();
            const uint32_t go_msg_addr =
                hal_.get_dev_addr(worker_core_type, HalL1MemAddrType::GO_MSG);
            auto cur_go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
            cluster_.read_core(
                cur_go_msg.data(), cur_go_msg.size(), tt_cxy_pair(device_id, worker_core), go_msg_addr);
            const uint8_t signal = cur_go_msg.view().signal();
            const bool is_known =
                std::find(kKnownRunMsgValues.begin(), kKnownRunMsgValues.end(), signal) != kKnownRunMsgValues.end();
            // FIX SC-ADDR diagnostic: log that the per-core-type address was used (debug level).
            // For ETH cores this confirms the correct ETH mailbox address is being read, not the
            // Tensix go_msg address (which would produce a garbage signal and falsely trigger FIX SC).
            log_debug(
                LogDevice,
                "FIX SC-ADDR (GAP-76): Device {} core {} ({}) go_msg_addr=0x{:08x} signal=0x{:02x} — {}",
                device_id,
                worker_core.str(),
                core_type_str,
                go_msg_addr,
                signal,
                is_known ? "valid (no FIX SC)" : "STALE → FIX SC will fire");
            if (!is_known) {
                log_warning(
                    tt::LogAlways,
                    "FIX SC (GAP-76): Device {} core {} ({}) has stale go_msg=0x{:02x} after firmware "
                    "multicast write — asserting BRISC reset to halt stale firmware then writing "
                    "RUN_MSG_DONE; board reset will be required",
                    device_id,
                    worker_core.str(),
                    core_type_str,
                    signal);
                // FIX SC: Assert BRISC reset BEFORE writing RUN_MSG_DONE.  Without this the stale
                // firmware is still running and immediately overwrites our write back to 0x55,
                // causing wait_until_cores_done to spin for 10 s.
                try {
                    cluster_.assert_risc_reset_at_core(
                        tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogAlways,
                        "FIX SC (GAP-76): assert_risc_reset failed for Device {} core {}: {}",
                        device_id,
                        worker_core.str(),
                        e.what());
                }
                cluster_.write_core(
                    done_go_msg.data(),
                    done_go_msg.size(),
                    tt_cxy_pair(device_id, worker_core),
                    go_msg_addr);
            }
        }
    }

    log_debug(LogDevice, "Waiting for firmware init complete");
    const int timeout_ms = 10000;
    try {
        llrt::internal_::wait_until_cores_done(device_id, dev_msgs::RUN_MSG_INIT, not_done_cores, timeout_ms);
    } catch (std::runtime_error&) {
        TT_THROW("Device {} init: failed to initialize FW! Try resetting the board.", device_id);
    }
    log_debug(LogDevice, "Firmware init complete");

    if (!dram_not_done_cores.empty()) {
        log_debug(LogDevice, "Waiting for DRAM firmware init complete");
        try {
            llrt::internal_::wait_until_cores_done(device_id, dev_msgs::RUN_MSG_INIT, dram_not_done_cores, timeout_ms);
        } catch (std::runtime_error&) {
            TT_THROW("Device {} init: failed to initialize DRAM FW!", device_id);
        }
        log_debug(LogDevice, "DRAM firmware init complete");
    }
}

}  // namespace tt::tt_metal
