// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "fabric_firmware_initializer.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <tt_metal.hpp>
#include "device/device_impl.hpp"
#include "common/executor.hpp"
#include "impl/context/context_descriptor.hpp"

#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include "fabric/fabric_host_utils.hpp"
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_builder_context.hpp"

// Timeout hierarchy (all host-side wall-clock):
//   5000ms — teardown/init per-phase: covers full firmware startup/shutdown cycle
//    150ms — quiesce Phase 2.5 ERISC terminate: cooperative shutdown, faster path
//    100ms — stale ERISC probe: base firmware never responds, fail fast
//    150ms — verify_all_fabric_channels_healthy: 3×50ms retry window
// Device-side poll bounds: 1M iterations + PAUSE ≈ seconds of wall-clock coverage
//
// These values are spread across fabric_firmware_initializer.cpp, device.cpp, and
// metal_env.cpp. If you adjust one timeout, review the others for consistency.

namespace tt::tt_metal {

namespace {

// Returns a human-readable name for a known EDMStatus value, or "(unknown)" if the value
// does not match any enumerator. This is the single source of truth for EDMStatus → string
// mapping; is_known_edm_status() delegates to this function.
//
// UPDATE THIS FUNCTION WHEN ADDING NEW EDMStatus VALUES in
// tt_metal/fabric/fabric_edm_packet_header.hpp (currently 15 enumerators).
static const char* edm_status_name(tt::tt_fabric::EDMStatus s) {
    switch (s) {
        case tt::tt_fabric::EDMStatus::STARTED:                     return "STARTED";
        case tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE:   return "REMOTE_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE:    return "LOCAL_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC:           return "READY_FOR_TRAFFIC";
        case tt::tt_fabric::EDMStatus::TERMINATED:                  return "TERMINATED";
        case tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED:      return "INITIALIZATION_STARTED";
        case tt::tt_fabric::EDMStatus::TXQ_INITIALIZED:             return "TXQ_INITIALIZED";
        case tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED:      return "STREAM_REG_INITIALIZED";
        case tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED: return "DOWNSTREAM_EDM_SETUP_STARTED";
        case tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE:      return "EDM_VCS_SETUP_COMPLETE";
        case tt::tt_fabric::EDMStatus::WORKER_INTERFACES_INITIALIZED: return "WORKER_INTERFACES_INITIALIZED";
        case tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE: return "ETHERNET_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::VCS_OPENED:                  return "VCS_OPENED";
        case tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED:   return "ROUTING_TABLE_INITIALIZED";
        case tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE:     return "INITIALIZATION_COMPLETE";
        default: return "(unknown)";
    }
}

// Returns true iff `status` is one of the well-known EDMStatus sentinel values written by
// a live fabric ERISC router at some point in its lifecycle. Any other nonzero value at
// the router_sync_address indicates the L1 slot is corrupt or has been overwritten by
// unrelated NOC traffic — the ERISC is NOT running recognizable firmware and the
// TERMINATE handshake will not complete.
//
// Delegates to edm_status_name() so there is only one switch over EDMStatus values.
bool is_known_edm_status(uint32_t status) {
    // Cast raw uint32_t to the enum and check if edm_status_name recognises it.
    // "(unknown)" is the sentinel returned for unrecognised values.
    const char* name = edm_status_name(static_cast<tt::tt_fabric::EDMStatus>(status));
    return name[0] != '(';
}

}  // namespace

// Thread-local compile seam — default-constructed (empty std::function) in all threads.
// Only set by tests via set_compile_fn_for_testing() before MeshDevice::create().
thread_local FabricFirmwareInitializer::CompileFabricFn
    FabricFirmwareInitializer::s_compile_fn_for_testing_;

void FabricFirmwareInitializer::set_compile_fn_for_testing(CompileFabricFn fn) {
    s_compile_fn_for_testing_ = std::move(fn);
}

void FabricFirmwareInitializer::clear_compile_fn_for_testing() {
    s_compile_fn_for_testing_ = {};
}

// Thread-local status-override seam — default-constructed (empty std::function) in all threads.
// Only set by tests via set_status_override_fn_for_testing() before MeshDevice::create().
thread_local FabricFirmwareInitializer::StatusOverrideFn
    FabricFirmwareInitializer::s_status_override_fn_;

void FabricFirmwareInitializer::set_status_override_fn_for_testing(StatusOverrideFn fn) {
    s_status_override_fn_ = std::move(fn);
}

void FabricFirmwareInitializer::clear_status_override_fn_for_testing() {
    s_status_override_fn_ = {};
}

FabricFirmwareInitializer::FabricFirmwareInitializer(
    std::shared_ptr<const ContextDescriptor> descriptor, tt::tt_fabric::ControlPlane& control_plane) :
    FirmwareInitializer(std::move(descriptor)), control_plane_(control_plane) {}

void FabricFirmwareInitializer::init(
    const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& /*init_done*/) {
    devices_ = devices;

    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    if (descriptor_->is_mock_device()) {
        log_info(tt::LogMetal, "Skipping fabric initialization for mock devices");
        return;
    }

    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        log_info(tt::LogMetal, "Initializing Fabric");

        // GAP 1 (#42429): Pre-init L1 health scan — canary check BEFORE routing table writes.
        //
        // write_routing_tables_to_all_chips() writes to Tensix L1 on non-MMIO devices through the
        // ETH relay.  If a prior process left stale ERISC firmware running (e.g. crashed mid-
        // handshake, leaving edm_status = 0x49706550 "iPeP"), those relay ERISCs are unresponsive
        // and the routing table write either hangs or silently corrupts.
        //
        // This scan emits a WARNING if any active ETH channel shows a non-zero / non-TERMINATED
        // edm_status.  It does NOT gate the init path — terminate_stale_erisc_routers() (called
        // inside compile_and_configure_fabric()) is the authoritative recovery step.  The purpose
        // here is early-warning telemetry: if a routing table write later hangs, the log will show
        // the pre-existing stale state that caused it.
        //
        // If any channel fails to read (ETH relay completely unresponsive), the warning includes
        // the exception message and skips that device — terminate_stale_erisc_routers() will deal
        // with it properly via the relay_broken / probe_dead_channels machinery.
        try {
            // router_config_ is populated in FabricBuilderContext's constructor, so
            // get_fabric_router_sync_address_and_status() is safe to call here even though
            // write_routing_tables_to_all_chips() has not yet run.  What is NOT safe is
            // get_num_fabric_initialized_routers(), which TT_FATALs until per-device router
            // counts are registered by write_routing_tables_to_all_chips().  Use
            // is_physical_chip_in_fabric_cluster() instead to skip non-fabric devices.
            const auto router_sync_address =
                control_plane_.get_fabric_context().get_builder_context()
                    .get_fabric_router_sync_address_and_status().first;
            constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);

            for (auto* dev : devices_) {
                if (!control_plane_.is_physical_chip_in_fabric_cluster(dev->id())) {
                    continue;
                }
                const auto fabric_node_id =
                    control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
                const auto& active_channels =
                    control_plane_.get_active_fabric_eth_channels(fabric_node_id);

                uint32_t stale_count = 0;
                uint32_t corrupt_count = 0;
                for (const auto& [eth_chan_id, direction] : active_channels) {
                    const auto eth_logical_core =
                        cluster_.get_soc_desc(dev->id())
                            .get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
                    std::vector<uint32_t> status_buf(1, 0);
                    try {
                        detail::ReadFromDeviceL1(
                            dev, eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
                    } catch (const std::exception& read_ex) {
                        log_warning(
                            tt::LogMetal,
                            "pre-init L1 scan: Device {} chan={} read FAILED ({}) — ETH relay may "
                            "already be unresponsive before routing table write",
                            dev->id(),
                            eth_chan_id,
                            read_ex.what());
                        continue;
                    }
                    const uint32_t status = status_buf[0];
                    if (status == 0 || status == terminated_val) {
                        continue;  // clean
                    }
                    if (is_known_edm_status(status)) {
                        stale_count++;
                    } else {
                        corrupt_count++;
                    }
                }

                if (stale_count > 0 || corrupt_count > 0) {
                    log_warning(
                        tt::LogMetal,
                        "pre-init L1 scan: Device {} has {} stale-running and {} corrupt ETH "
                        "channel(s) BEFORE routing table write. terminate_stale_erisc_routers() "
                        "will attempt recovery — but routing table writes to non-MMIO devices "
                        "that use these relay ERISCs may race with stale firmware (#42429).",
                        dev->id(),
                        stale_count,
                        corrupt_count);
                }
            }
        } catch (const std::exception& scan_ex) {
            // Non-fatal: terminate_stale_erisc_routers() is the recovery path.
            // Log and proceed — the scan is informational only.
            log_warning(
                tt::LogMetal,
                "pre-init L1 scan failed ({}): proceeding with routing table write. "
                "If init hangs, check for stale ERISC firmware from a prior process.",
                scan_ex.what());
        }

        control_plane_.write_routing_tables_to_all_chips();
        compile_and_configure_fabric();
        log_info(tt::LogMetal, "Fabric Initialized with config {}", fabric_config);
    } else if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::TERMINATE_FABRIC)) {
        log_info(tt::LogMetal, "Compiling fabric to setup fabric context for fabric termination");
        for (auto* dev : devices_) {
            dev->compile_fabric();
        }
    } else {
        log_info(tt::LogMetal, "Fabric initialized through Fabric Manager");
    }
}

void FabricFirmwareInitializer::configure() {
    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        wait_for_fabric_router_sync(get_fabric_router_sync_timeout_ms());
        // After the master-channel handshake passes, verify ALL active ERISC channels are
        // healthy.  Persistent corruption (e.g. 0x49705180 from a prior process crash) on
        // non-master channels is invisible to wait_for_fabric_router_sync but will cause
        // dispatch hangs when the test tries to use those fabric paths.  Fail-fast here
        // instead of letting the test hang for minutes.
        verify_all_fabric_channels_healthy();
    }
    initialized_.test_and_set();
}

void FabricFirmwareInitializer::teardown(std::unordered_set<InitializerKey>& init_done) {
    // Clear force_reset_channels_ at the start of each teardown cycle to prevent stale entries
    // from prior cycles causing misclassification in verify_all_fabric_channels_healthy().
    {
        std::lock_guard<std::mutex> lock(force_reset_channels_mutex_);
        force_reset_channels_.clear();
    }

    if (init_done.contains(InitializerKey::Dispatch)) {
        log_error(
            tt::LogMetal,
            "FabricFirmwareInitializer::teardown: DispatchKernelInitializer is still active — "
            "teardown ordering violation. Proceeding with best-effort fabric teardown to avoid "
            "leaving ERISCs running.");
    }
    if (descriptor_->is_mock_device()) {
        log_info(tt::LogMetal, "Skipping fabric teardown for mock devices");
        init_done.erase(key);
        return;
    }
    if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::TERMINATE_FABRIC)) {
        devices_.clear();
        initialized_.clear();
        init_done.erase(key);
        return;
    }

    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        devices_.clear();
        initialized_.clear();
        init_done.erase(key);
        return;
    }

    const auto& fabric_context = control_plane_.get_fabric_context();
    const auto& builder_ctx = fabric_context.get_builder_context();
    auto [termination_signal_address, signal] = builder_ctx.get_fabric_router_termination_address_and_signal();
    std::vector<uint32_t> termination_signal(1, signal);

    // Terminate fabric tensix mux cores if enabled
    // TODO: issue #26855, move the termination process to device
    if (descriptor_->fabric_tensix_config() != tt::tt_fabric::FabricTensixConfig::DISABLED) {
        const auto& tensix_config = builder_ctx.get_tensix_config();

        for (auto* dev : devices_) {
            if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
                continue;
            }

            const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
            const auto& active_fabric_eth_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);

            for (const auto& [eth_chan_id, direction] : active_fabric_eth_channels) {
                auto core_id = tensix_config.get_core_id_for_channel(dev->id(), eth_chan_id);
                auto [tensix_termination_address, tensix_signal] =
                    tensix_config.get_termination_address_and_signal(core_id);
                std::vector<uint32_t> tensix_termination_signal(1, tensix_signal);
                auto mux_core = tensix_config.get_core_for_channel(dev->id(), eth_chan_id);

                detail::WriteToDeviceL1(
                    dev, mux_core, tensix_termination_address, tensix_termination_signal, CoreType::WORKER);
            }

            try {
                cluster_.l1_barrier(dev->id());
            } catch (const std::exception& barrier_ex) {
                log_warning(
                    tt::LogMetal,
                    "FabricFirmwareInitializer::teardown: l1_barrier threw on Device {}: {} — "
                    "likely dead ERISC relay; continuing teardown.",
                    dev->id(),
                    barrier_ex.what());
            } catch (...) {
                log_warning(
                    tt::LogMetal,
                    "FabricFirmwareInitializer::teardown: l1_barrier threw non-std exception on Device {} "
                    "(likely UmdException<RuntimeError> from dead ERISC relay). Continuing teardown.",
                    dev->id());
            }

            // Poll each MUX core until TERMINATED before proceeding.
            // Without this, configure_fabric() in the next test can write new launch messages
            // to the same Tensix cores before the old MUX workers finish terminating, causing
            // two kernels to compete for the same cores and hang dispatch on remote devices.
            //
            // On timeout we force-halt the Tensix MUX core (assert RISC reset) — mirroring the
            // ETH router path in MetalEnvImpl::teardown_fabric_config — so a stuck mux cannot
            // continue emitting NOC traffic into worker L1 that the next bring-up reprograms.
            // The core is re-initialized on the next fabric bring-up.
            for (const auto& [eth_chan_id, direction] : active_fabric_eth_channels) {
                auto core_id = tensix_config.get_core_id_for_channel(dev->id(), eth_chan_id);
                auto config = tensix_config.get_config(core_id);
                uint32_t status_addr = static_cast<uint32_t>(config->get_status_address());
                auto mux_core = tensix_config.get_core_for_channel(dev->id(), eth_chan_id);

                std::vector<uint32_t> status_buf(1, 0);
                const auto start = std::chrono::steady_clock::now();
                constexpr uint32_t timeout_ms = 5000;
                constexpr uint32_t kSpinsBetweenSleeps = 64;
                uint32_t spin_counter = 0;
                bool terminated = false;
                while (true) {
                    // Wrapped in try/catch: ReadFromDeviceL1 can throw for non-MMIO (remote)
                    // devices if the tunnel is down or the device is unresponsive.  Without
                    // this guard a single throw exits the loop via exception, bypassing the
                    // timeout path and aborting all Phase 2 (ETH router) teardown for this
                    // device.  On read failure we treat the MUX as "not yet terminated" and
                    // let the loop continue until the timeout naturally expires, then fall
                    // into the force-reset path.
                    try {
                        detail::ReadFromDeviceL1(dev, mux_core, status_addr, 4, status_buf, CoreType::WORKER);
                    } catch (const std::exception& read_ex) {
                        log_warning(
                            tt::LogMetal,
                            "FabricFirmwareInitializer::teardown: ReadFromDeviceL1 threw on "
                            "Device {} Tensix MUX core ({},{}): {} — treating as not-terminated",
                            dev->id(),
                            mux_core.x,
                            mux_core.y,
                            read_ex.what());
                        // Don't break — fall through to timeout check so we eventually
                        // force-reset rather than spin indefinitely.
                    }
                    if (status_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED)) {
                        terminated = true;
                        break;
                    }
                    const auto elapsed =
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                            .count();
                    if (elapsed > timeout_ms) {
                        break;
                    }
                    // Back off: ReadFromDeviceL1 already round-trips to MMIO, so a tight loop
                    // hammers the device. Pause every iteration; yield once per spin window.
                    if (++spin_counter >= kSpinsBetweenSleeps) {
                        spin_counter = 0;
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    } else {
                        ttsl::pause();
                    }
                }

                if (!terminated) {
                    log_warning(
                        tt::LogMetal,
                        "FabricFirmwareInitializer::teardown: Timeout waiting for Tensix MUX TERMINATED on "
                        "Device {} eth_chan {} (status=0x{:08x}), force-resetting Tensix MUX to prevent "
                        "stale NOC traffic into worker L1",
                        dev->id(),
                        eth_chan_id,
                        status_buf[0]);
                    // Translate the logical worker core to a virtual coordinate and assert reset
                    // on the Tensix RISCs. The MUX kernel will be fully re-initialized on the
                    // next fabric bring-up. Catch so one stuck core cannot prevent reset on the
                    // remaining cores on this device.
                    try {
                        const auto virtual_mux_coord = cluster_.get_virtual_coordinate_from_logical_coordinates(
                            dev->id(), mux_core, CoreType::WORKER);
                        cluster_.assert_risc_reset_at_core(
                            tt_cxy_pair(dev->id(), virtual_mux_coord), tt::umd::RiscType::ALL);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogMetal,
                            "FabricFirmwareInitializer::teardown: assert_risc_reset_at_core failed on Device {} "
                            "eth_chan {}: {}",
                            dev->id(),
                            eth_chan_id,
                            e.what());
                    }
                }
            }
        }
    }

    // Terminate fabric routers via master router on each device
    for (auto* dev : devices_) {
        if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
            continue;
        }

        auto master_router_logical_core = cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(
            builder_ctx.get_fabric_master_router_chan(dev->id()), CoordSystem::LOGICAL);
        detail::WriteToDeviceL1(
            dev, master_router_logical_core, termination_signal_address, termination_signal, CoreType::ETH);

        // Ensure the TERMINATE write has landed in L1 before we begin polling for
        // EDMStatus::TERMINATED.  The Tensix MUX path has an equivalent l1_barrier
        // after its TERMINATE writes (see above).  While the NOC protocol guarantees
        // read-after-write ordering for the *same* core, the barrier also flushes any
        // in-flight NOC writes to *other* cores on this chip, preventing stale reads
        // during the round-robin poll across all active channels.
        cluster_.l1_barrier(dev->id());
    }

    // Fix B: Poll ALL active ETH router channels per device for EDMStatus::TERMINATED before
    // returning. Without this poll, the next init's configure_fabric_cores() can clear L1 and
    // load new firmware while old firmware is still running — leaving ERISCs in a zombie state
    // (edm_status = RUNNING) that causes the next session's AllGather to hang.
    //
    // Previously this only polled the *master* router per device, but slave routers on
    // non-master channels (e.g. channels 1-3 on each device in FABRIC_1D) could still be
    // running real EDM firmware when teardown returned.  That left edm_status=0x49705180
    // (ACTIVE) on all slave channels, which terminate_stale_erisc_routers then detected
    // at the next init — but 50ms was too short to drain the still-live firmware, so new
    // firmware was loaded over actively-running ERISC code, corrupting L1 on iteration 2.
    //
    // IMPORTANT: Use a single global deadline shared across ALL devices and channels.
    // On large meshes (e.g. Galaxy 6U with 30+ active ETH channels), a per-channel 5s
    // timeout would produce 150+ seconds of serial waits, blowing the CI step budget.
    // With a global deadline, all channels that respond quickly are polled immediately;
    // any channels that don't respond by the deadline are force-reset in a second pass.
    {
        const auto router_sync_address = builder_ctx.get_fabric_router_sync_address_and_status().first;
        constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
        constexpr uint32_t teardown_timeout_ms = 5000;
        constexpr uint32_t kSpinsBetweenSleeps = 64;

        // Record a single global deadline for the entire ETH poll phase.
        const auto global_deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(teardown_timeout_ms);

        // Collect (dev, chan_id, eth_logical_core) for all active ETH channels.
        struct PendingChannel {
            Device* dev;
            uint32_t eth_chan_id;
            CoreCoord eth_logical_core;
        };
        std::vector<PendingChannel> pending;
        for (auto* dev : devices_) {
            if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
                continue;
            }
            const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
            const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);
            for (const auto& [eth_chan_id, direction] : active_channels) {
                pending.push_back(
                    {dev,
                     eth_chan_id,
                     cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL)});
            }
        }

        // Poll all pending channels until each terminates or the global deadline expires.
        // Channels that terminate early are removed from the list; remaining channels after
        // the deadline get force-reset in a second pass below.
        uint32_t spin_counter = 0;
        while (!pending.empty() && std::chrono::steady_clock::now() < global_deadline) {
            pending.erase(
                std::remove_if(
                    pending.begin(),
                    pending.end(),
                    [&](const PendingChannel& ch) {
                        try {
                            std::vector<uint32_t> status_buf(1, 0);
                            detail::ReadFromDeviceL1(
                                ch.dev, ch.eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
                            return status_buf[0] == terminated_val;
                        } catch (const std::exception& e) {
                            // Read failed (e.g. ERISC completely unresponsive).
                            // Keep this channel in pending so it gets force-reset in the
                            // second pass. Do not let one bad read abort the entire poll.
                            log_warning(
                                tt::LogMetal,
                                "FabricFirmwareInitializer::teardown: ReadFromDeviceL1 threw on "
                                "Device {} chan={}: {} — keeping channel pending for force-reset",
                                ch.dev->id(),
                                ch.eth_chan_id,
                                e.what());
                            return false;
                        }
                    }),
                pending.end());
            if (!pending.empty()) {
                if (++spin_counter >= kSpinsBetweenSleeps) {
                    spin_counter = 0;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    ttsl::pause();
                }
            }
        }

        // Log which channels missed the deadline — critical for diagnosing partial teardown.
        if (!pending.empty()) {
            std::string missed_list;
            for (const auto& ch : pending) {
                if (!missed_list.empty()) missed_list += ", ";
                missed_list += fmt::format("dev={}/chan={}", ch.dev->id(), ch.eth_chan_id);
            }
            log_warning(
                tt::LogMetal,
                "FabricFirmwareInitializer::teardown: Global deadline expired with {} channel(s) "
                "still pending TERMINATE: [{}]. Force-resetting these channels.",
                pending.size(),
                missed_list);
        }

        // Force-reset any channels that did not terminate within the global deadline.
        // GAP 5: Record force-reset channels so the next verify_all_fabric_channels_healthy()
        // can distinguish "was force-reset" from "corrupt from prior crash".
        std::vector<std::string> reset_failed_channels;
        for (const auto& ch : pending) {
            // Diagnostic read: log the last-seen status before asserting reset.
            // Wrapped in try/catch — if the read itself throws (e.g. device unresponsive),
            // we still want to proceed with force_reset_channels_ registration and
            // assert_risc_reset_at_core for ALL remaining channels.  Without this guard,
            // a single bad read would abort the loop, leaving other channels un-reset and
            // skipping devices_.clear() / init_done.erase(key) at the end of teardown.
            std::vector<uint32_t> status_buf(1, 0xDEAD'DEAD);
            try {
                detail::ReadFromDeviceL1(
                    ch.dev, ch.eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
            } catch (const std::exception& read_ex) {
                log_warning(
                    tt::LogMetal,
                    "FabricFirmwareInitializer::teardown: diagnostic ReadFromDeviceL1 threw on "
                    "Device {} chan={}: {} — using sentinel status for log, proceeding with reset",
                    ch.dev->id(),
                    ch.eth_chan_id,
                    read_ex.what());
            }
            log_warning(
                tt::LogMetal,
                "FabricFirmwareInitializer::teardown: Device {} ETH chan={} did not "
                "terminate within {}ms (status=0x{:08x}) — asserting ERISC reset to prevent "
                "stale firmware racing with next init's L1 clear",
                ch.dev->id(),
                ch.eth_chan_id,
                teardown_timeout_ms,
                status_buf[0]);
            {
                std::lock_guard<std::mutex> lock(force_reset_channels_mutex_);
                force_reset_channels_.emplace(ch.dev->id(), ch.eth_chan_id);
            }
            // Intentional asymmetry: terminate_stale_erisc_routers() and metal_env.cpp's
            // teardown_fabric_config() both AVOID assert_risc_reset on WH because it tears
            // down the ETH PHY link and corrupts partner ERISC L1 on adjacent chips.
            // Here in session teardown, however, the benefit of preventing a zombie ERISC
            // from racing with the next init's L1 clear outweighs the PHY link disruption.
            // The partner chip's next init will handle the resulting corrupt L1 via
            // terminate_stale_erisc_routers()'s CORRUPT path (send TERMINATE best-effort,
            // skip poll). force_reset_channels_ is populated so verify_all_fabric_channels_healthy()
            // classifies these as "DEGRADED" rather than "corrupt from prior crash".
            try {
                const auto virtual_eth_coord = cluster_.get_virtual_coordinate_from_logical_coordinates(
                    ch.dev->id(), ch.eth_logical_core, CoreType::ETH);
                cluster_.assert_risc_reset_at_core(
                    tt_cxy_pair(ch.dev->id(), virtual_eth_coord), tt::umd::RiscType::ALL);
            } catch (const std::exception& e) {
                log_error(
                    tt::LogMetal,
                    "FabricFirmwareInitializer::teardown: assert_risc_reset_at_core failed on "
                    "Device {} chan={}: {} — ERISC may still be running! "
                    "Next fabric init should expect corrupt state on this channel.",
                    ch.dev->id(),
                    ch.eth_chan_id,
                    e.what());
                reset_failed_channels.push_back(
                    fmt::format("dev={}/chan={}", ch.dev->id(), ch.eth_chan_id));
            }
        }

        if (!reset_failed_channels.empty()) {
            std::string failed_list;
            for (const auto& s : reset_failed_channels) {
                if (!failed_list.empty()) failed_list += ", ";
                failed_list += s;
            }
            log_error(
                tt::LogMetal,
                "FabricFirmwareInitializer::teardown: {} channel(s) could NOT be force-reset and "
                "may still be running: [{}]. Next fabric init should expect corrupt state on these channels.",
                reset_failed_channels.size(),
                failed_list);
        }
    }

    // FIX A (#42429): Drain UMD ETH relay queue for non-MMIO devices before returning.
    //
    // Root cause: each timed-out probe read in terminate_stale_erisc_routers() (previous or
    // current session) leaves one stuck command in the 4-slot UMD ETH relay queue
    // (remote_communication_legacy_firmware.cpp read_non_mmio).  The relay_timeout_count guard
    // in terminate_stale_erisc_routers() prevents the queue from *filling* during a single call,
    // but the stuck commands persist in UMD's internal cmd_buf across teardown → next-session
    // init.  On the next call to terminate_stale_erisc_routers(), relay_timeout_count starts at
    // 0 (local variable), but if the queue already has ≥ 3 slots occupied from the prior
    // session the very first new probe read fills the queue → indefinite while(full) hang.
    //
    // Fix: call l1_barrier() on each non-MMIO device after the ETH router poll/force-reset
    // loop.  l1_barrier() calls driver_->l1_membar() → wait_for_non_mmio_flush(), which spins
    // until the relay ERISC drains all pending commands from the queue.  This resets the relay
    // to an empty state so the next session's terminate_stale_erisc_routers() starts with a
    // clean queue regardless of how many probe-read timeouts occurred in this session.
    //
    // Wrapped in try/catch: if the relay ERISC itself is dead (completely unresponsive),
    // wait_for_non_mmio_flush() may throw or block indefinitely.  We tolerate this failure —
    // the relay drain is best-effort.  The next session's relay_broken guard still protects
    // against queue saturation; the drain just prevents *cumulative* degradation across
    // back-to-back sessions.
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) == dev->id()) {
            // MMIO devices: l1_barrier is always safe, but there is no relay queue to drain.
            continue;
        }
        try {
            cluster_.l1_barrier(dev->id());
            log_debug(
                tt::LogMetal,
                "FabricFirmwareInitializer::teardown: relay drain l1_barrier completed for "
                "non-MMIO Device {} (UMD ETH relay queue flushed)",
                dev->id());
        } catch (const std::exception& drain_ex) {
            log_warning(
                tt::LogMetal,
                "FabricFirmwareInitializer::teardown: relay drain l1_barrier threw for "
                "non-MMIO Device {}: {} — relay queue may still have stuck commands; "
                "next session's relay_broken guard will protect against queue saturation",
                dev->id(),
                drain_ex.what());
        } catch (...) {
            log_warning(
                tt::LogMetal,
                "FabricFirmwareInitializer::teardown: relay drain l1_barrier threw (unknown "
                "exception type, likely UmdException<RuntimeError>) for non-MMIO Device {} — "
                "relay queue may still have stuck commands",
                dev->id());
        }
    }

    devices_.clear();
    initialized_.clear();
    // force_reset_channels_ was populated during this teardown and will be consumed by the
    // next session's verify_all_fabric_channels_healthy() for degraded-channel diagnostics.
    // It is cleared at the start of the next teardown() call.
    init_done.erase(key);
}

void FabricFirmwareInitializer::post_teardown() {
    // Reset fabric config
    descriptor_->metal_context().set_fabric_config(tt::tt_fabric::FabricConfig::DISABLED);
}

bool FabricFirmwareInitializer::is_initialized() const { return initialized_.test(); }

// Fix A/C/D + F4 (#42429): Detect stale ERISC router firmware on ALL active channels for a
// device and terminate/reset them before configure_fabric_cores() clears L1 and loads new
// firmware.
//
// For each active fabric ETH channel, read router_sync_address and classify:
//   - 0 or TERMINATED         -> clean, nothing to do
//   - CORRUPT (F4, #42429)    -> status is not a valid EDMStatus value. Typically caused by
//                                a prior process whose worker (mux) spun in close_finish()
//                                waiting for an EDM ACK that never arrived, then was
//                                BRISC-halted by Device::close(), leaving L1 mid-handshake.
//                                Action: send TERMINATE best-effort (one shot, no poll), log
//                                loudly, continue. The 50ms poll would always time out here
//                                (no firmware is running that will ever write TERMINATED) so
//                                skipping it saves ~42s on a T3K where 800+ channels can be
//                                corrupt.
//   - STALE RUNNING           -> status is a valid non-terminal EDMStatus (e.g.
//                                READY_FOR_TRAFFIC). Send TERMINATE, poll up to 50ms for
//                                TERMINATED. On timeout, log and continue — do NOT assert
//                                RISC reset. Rationale: a non-responsive channel in base
//                                firmware can be safely L1-overwritten by
//                                configure_fabric_cores(); resetting a WH ERISC tears down
//                                the ETH PHY link and breaks non-MMIO L1 access for the
//                                rest of the mesh.
//                                TODO(F2, #42429): replace with surgical per-channel reset
//                                once single-ERISC reset is verified to not drop the PHY.
std::pair<std::unordered_set<uint32_t>, bool> FabricFirmwareInitializer::terminate_stale_erisc_routers(
    Device* dev, const tt_fabric::FabricBuilderContext& builder_context) const {
    // Channels whose probe L1 read threw (physically dead link — remote ERISC completely
    // unresponsive).  Returned to the caller so configure_fabric_cores() can skip
    // assert_risc_reset_at_core() for them, avoiding the ch7-style indefinite hang.
    std::unordered_set<uint32_t> probe_dead_channels;

    if (builder_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
        return {probe_dead_channels, false};
    }

    const auto router_sync_address = builder_context.get_fabric_router_sync_address_and_status().first;
    const auto [term_addr, term_signal] = builder_context.get_fabric_router_termination_address_and_signal();
    constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
    constexpr uint32_t stale_timeout_ms = 100;  // EDM firmware responds in <5ms; base firmware never responds
    constexpr uint32_t kSpinsBetweenSleeps = 64;

    const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
    const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);

    uint32_t corrupt_count = 0;
    uint32_t stale_running_count = 0;
    uint32_t stale_timeout_count = 0;

    // Fix N300 indefinite hang: on non-MMIO devices (e.g. N300 Device 1), every probe
    // read that times out leaves one stuck command in the 4-slot ETH relay queue (UMD
    // remote_communication_legacy_firmware.cpp read_non_mmio).  Once the queue fills,
    // the next read_non_mmio enters a no-timeout while(full) spin → indefinite hang.
    // Track relay timeouts and stop issuing reads once the queue is one slot from full.
    // kMaxRelayTimeouts = cmd_buf_size - 1 = 3 (WH/BH ETH relay queue has 4 slots).
    uint32_t relay_timeout_count = 0;
    constexpr uint32_t kMaxRelayTimeouts = 3;
    bool relay_broken = false;

    for (const auto& [eth_chan_id, direction] : active_channels) {
        const auto eth_logical_core =
            cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);

        // If prior probe-read relay timeouts have brought the relay queue near capacity,
        // skip reads for remaining channels to prevent the no-timeout while(full) hang.
        if (relay_broken) {
            probe_dead_channels.insert(eth_chan_id);
            corrupt_count++;
            log_warning(
                tt::LogMetal,
                "terminate_stale_erisc_routers: Device {} chan={} skipped (relay broken after "
                "{} timeouts) — added to probe_dead_channels to prevent relay queue saturation",
                dev->id(),
                eth_chan_id,
                relay_timeout_count);
            continue;
        }

        std::vector<uint32_t> status_buf(1, 0);

        // Test seam (Scenario X): if set, call the status-override function before the real
        // L1 probe read.  If it returns a value, use that value as status_buf[0] and skip
        // the real ReadFromDeviceL1 entirely — no hardware access, safe for CI.
        // A test can return 0xBAADF00D to exercise the is_known_edm_status() false branch.
        bool seam_provided_status = false;
        if (s_status_override_fn_) {
            auto override_val = s_status_override_fn_(dev, eth_chan_id);
            if (override_val.has_value()) {
                status_buf[0] = *override_val;
                seam_provided_status = true;
            }
        }

        // Fix F5 (#42429): wrap the initial probe read in a try-catch.  On a T3K after an
        // abrupt prior-process crash, some ERISC channels are in a state where even the probe
        // L1 read hangs indefinitely (the Ethernet core service doesn't respond at all).
        // In that case, UMD throws "Timeout waiting for Ethernet core service remote IO
        // request" from read_non_mmio.  Treat a read timeout identically to the "corrupt
        // L1 word" case: send TERMINATE best-effort and continue — configure_fabric_cores()
        // will wipe L1 and load fresh firmware.
        //
        // Also record the channel in probe_dead_channels so configure_fabric_cores() can skip
        // assert_risc_reset_at_core() entirely.  On T3K Device 4 with 4 physically dead ETH
        // links, channels 0/1/6 correctly timeout (5 s UMD timeout fires), but channel 7
        // hangs for >10 minutes without triggering the timeout — suspected lock contention
        // or kernel blocking state accumulated from the prior three timeouts.  Skipping the
        // call entirely for pre-confirmed dead channels avoids this indefinite hang.
        if (!seam_provided_status) try {
            detail::ReadFromDeviceL1(dev, eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
        } catch (const std::exception& read_ex) {
            log_error(
                tt::LogMetal,
                "terminate_stale_erisc_routers: Device {} chan={} probe read TIMED OUT ({}). "
                "ERISC is completely unresponsive; sending TERMINATE best-effort, skipping poll. "
                "configure_fabric_cores() will skip soft reset for this channel.",
                dev->id(),
                eth_chan_id,
                read_ex.what());
            probe_dead_channels.insert(eth_chan_id);
            try {
                std::vector<uint32_t> term_buf(1, static_cast<uint32_t>(term_signal));
                detail::WriteToDeviceL1(dev, eth_logical_core, term_addr, term_buf, CoreType::ETH);
            } catch (...) {
                // write-side also unresponsive — best effort only, ignore
            }
            // Best-effort: zero edm_status_address even on probe-dead channels.
            // If the read threw but the write succeeds (asymmetric failure), this prevents
            // the next session from seeing stale garbage.  If the write also throws, no harm.
            try {
                std::vector<uint32_t> zero_buf(1, 0);
                detail::WriteToDeviceL1(dev, eth_logical_core, router_sync_address, zero_buf, CoreType::ETH);
            } catch (...) {
                // write-side also unresponsive — best effort only
            }
            // Track relay timeouts.  Once we reach kMaxRelayTimeouts, the ETH relay queue
            // for this device has (cmd_buf_size - 1) stuck commands and one slot remains.
            // One more timed-out read would fill the queue; the FOLLOWING read would enter
            // read_non_mmio's no-timeout while(full) loop.  Set relay_broken so the next
            // channel iteration skips the read entirely.
            if (++relay_timeout_count >= kMaxRelayTimeouts) {
                log_warning(
                    tt::LogMetal,
                    "terminate_stale_erisc_routers: Device {} relay timeout count {} >= {} — "
                    "ETH relay path appears broken (crashed relay ERISCs on non-MMIO device). "
                    "Remaining channels will skip probe reads to prevent relay queue fill "
                    "and indefinite hang in read_non_mmio while(full) loop.",
                    dev->id(),
                    relay_timeout_count,
                    kMaxRelayTimeouts);
                relay_broken = true;
            }
            corrupt_count++;
            continue;
        }

        if (status_buf[0] == 0 || status_buf[0] == terminated_val) {
            continue;  // clean — nothing to do
        }

        // Fix F4 / FIX K (#42429): if the status word is not a valid EDMStatus value, the L1
        // slot is corrupt OR shows the base-UMD-firmware sentinel (0x49706550 = "iPeP").
        // Skip the 50ms wait per channel (saves ~42s on a T3K with ~840 such channels).
        // Do NOT send TERMINATE — see FIX K comment below for why that kills the ETH relay.
        // configure_fabric_cores() will clear the L1 via soft-reset and load new firmware;
        // whether that recovers cleanly is observed downstream at wait_for_fabric_endpoint_ready.
        const bool known_status = is_known_edm_status(status_buf[0]);
        if (!known_status) {
            // FIX K (#42429): Do NOT send TERMINATE to corrupt/unknown-status channels.
            //
            // 0x49706550 ("iPeP") is the base-UMD-firmware sentinel — it means the ERISC is
            // running the stock UMD relay firmware, NOT stale fabric firmware.  Base UMD firmware
            // actively polls the termination_signal_address; writing TERMINATE causes it to exit
            // gracefully.  On MMIO devices (chips 0-3) this kills the ETH relay that non-MMIO
            // devices (chips 4-7) depend on for ALL L1 / register / reset operations.
            //
            // The original rationale was "give unknown firmware a best-effort chance to stop".
            // In practice this was harmful: sending TERMINATE to a live relay ERISC kills it
            // ~9 ms into PHASE 1, and by the time PHASE 2 tries to soft-reset non-MMIO Device 4
            // (which goes through that same relay), the relay is already dead → 5 s UMD timeout.
            //
            // configure_fabric_cores() issues a hard BRISC soft-reset (assert+deassert) which
            // reliably restarts the ERISC regardless of its prior firmware state, so no graceful
            // TERMINATE handshake is needed here.
            log_error(
                tt::LogMetal,
                "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} is NOT a "
                "valid EDMStatus value — ERISC L1 appears CORRUPT or shows base-UMD-firmware "
                "sentinel (see #42429). NOT sending TERMINATE (would kill ETH relay). "
                "configure_fabric_cores() will issue soft-reset to recover.",
                dev->id(),
                eth_chan_id,
                status_buf[0]);
            // Fix #42429 (cascade prevention): zero edm_status_address so the NEXT session's
            // terminate_stale_erisc_routers() sees a clean 0 instead of the same garbage value.
            // Without this, the corrupt status persists in L1 across container restarts (bare
            // metal L1 is NOT cleared on process exit), causing every subsequent session to
            // re-classify this channel as corrupt → add to probe_dead_channels → skip L1 clear
            // → garbage persists → cascade forever until hardware reset.
            //
            // The probe read above SUCCEEDED (we're in the !known_status path, not the catch
            // path), so the write path to this channel works.  Zeroing edm_status_address lets
            // the next session treat this channel as clean and attempt normal initialization,
            // which will either succeed (the ETH link recovered) or fail at soft reset (caught
            // by configure_fabric_cores) — either way, no infinite cascade.
            try {
                std::vector<uint32_t> zero_buf(1, 0);
                detail::WriteToDeviceL1(dev, eth_logical_core, router_sync_address, zero_buf, CoreType::ETH);
                log_warning(
                    tt::LogMetal,
                    "terminate_stale_erisc_routers: Device {} chan={} zeroed edm_status_address "
                    "(was 0x{:08x}) to break corruption cascade for next session",
                    dev->id(),
                    eth_chan_id,
                    status_buf[0]);
            } catch (...) {
                // write failed — best effort; the channel may truly be unreachable
            }
            // Fix #42429 (corrupt path): do NOT add to probe_dead_channels here.
            // The probe read above SUCCEEDED (we're in the !known_status path, not the catch
            // path), proving the ETH relay to this channel is functional.  assert_risc_reset_at_core()
            // in configure_fabric_cores() uses the same relay path (UMD register write through ETH),
            // so it should also succeed.  Let it attempt normal soft reset — if it times out, THAT
            // timeout catches the dead channel at configure_fabric_cores() level.
            //
            // Previously, we added corrupt channels to probe_dead_channels, which caused
            // configure_fabric_cores() to skip soft reset entirely.  Without the soft reset,
            // BRISC stayed halted and new firmware couldn't start → configure_fabric_cores()
            // returned false → session failed, even though the channel was actually reachable.
            corrupt_count++;
            continue;
        }

        log_warning(
            tt::LogMetal,
            "terminate_stale_erisc_routers: Device {} ETH chan={} edm_status=0x{:08x} "
            "(expected 0 or TERMINATED=0x{:08x}) — stale firmware running; sending TERMINATE",
            dev->id(),
            eth_chan_id,
            status_buf[0],
            terminated_val);
        stale_running_count++;

        std::vector<uint32_t> term_buf(1, static_cast<uint32_t>(term_signal));
        try {
            detail::WriteToDeviceL1(dev, eth_logical_core, term_addr, term_buf, CoreType::ETH);
        } catch (const std::exception& write_ex) {
            log_warning(
                tt::LogMetal,
                "terminate_stale_erisc_routers: Device {} chan={} TERMINATE write failed ({}). "
                "Skipping poll — configure_fabric_cores() will reset L1.",
                dev->id(),
                eth_chan_id,
                write_ex.what());
            stale_timeout_count++;
            continue;
        }

        // Poll for TERMINATED
        const auto stale_start = std::chrono::steady_clock::now();
        uint32_t spin_counter = 0;
        bool terminated_ok = false;
        while (true) {
            // Wrapped in try/catch: on non-MMIO devices, ReadFromDeviceL1 can throw
            // "Timeout waiting for Ethernet core service" if the tunnel is disrupted
            // during the 100ms window.  Without this guard, a single throw escapes the
            // loop and aborts cleanup for ALL remaining channels on this device.
            // On exception, treat as "not yet terminated" and continue the timeout poll.
            try {
                detail::ReadFromDeviceL1(
                    dev, eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
            } catch (const std::exception& read_ex) {
                log_warning(
                    tt::LogMetal,
                    "terminate_stale_erisc_routers: Device {} chan={} poll read threw: {} — "
                    "treating as not-terminated, continuing timeout",
                    dev->id(),
                    eth_chan_id,
                    read_ex.what());
                // Fall through to timeout check rather than breaking out.
            }
            if (status_buf[0] == terminated_val) {
                terminated_ok = true;
                break;
            }
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - stale_start)
                    .count();
            if (elapsed > stale_timeout_ms) {
                break;
            }
            if (++spin_counter >= kSpinsBetweenSleeps) {
                spin_counter = 0;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else {
                ttsl::pause();
            }
        }

        const auto stale_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - stale_start)
                .count();
        if (terminated_ok) {
            log_info(
                tt::LogMetal,
                "terminate_stale_erisc_routers: Device {} chan={} responded to TERMINATE in {}ms",
                dev->id(),
                eth_chan_id,
                stale_elapsed);
        } else {
            // Do NOT assert_risc_reset_at_core here: on WH, resetting an ERISC takes the
            // ETH PHY link down and breaks non-MMIO L1 access for the rest of the mesh.
            // The ERISC is almost certainly running base firmware (stale L1 value from a
            // halted predecessor); configure_fabric_cores() will safely overwrite L1 and
            // the new fabric firmware will boot normally.
            // TODO(F2, #42429): once single-ERISC reset is verified not to drop the PHY link,
            // promote this path to a surgical per-channel reset.
            log_warning(
                tt::LogMetal,
                "terminate_stale_erisc_routers: Device {} chan={} did not respond to TERMINATE "
                "within {}ms (elapsed {}ms) — likely base firmware with stale L1; continuing without reset",
                dev->id(),
                eth_chan_id,
                stale_timeout_ms,
                stale_elapsed);
            stale_timeout_count++;
        }
    }

    if (corrupt_count > 0 || stale_running_count > 0) {
        log_info(
            tt::LogMetal,
            "terminate_stale_erisc_routers: Device {} summary: corrupt={} stale_running={} "
            "stale_term_timeout={} (of stale_running) probe_dead={}",
            dev->id(),
            corrupt_count,
            stale_running_count,
            stale_timeout_count,
            probe_dead_channels.size());
    }

    return {probe_dead_channels, relay_broken};
}

void FabricFirmwareInitializer::compile_and_configure_fabric() {
    // Snapshot the compile seam once at function entry.  In production (no test override),
    // s_compile_fn_for_testing_ is an empty std::function and we fall back to
    // dev->compile_fabric().  In test code, set_compile_fn_for_testing() installs a stub
    // that can throw, return false, or add delays to exercise the join-before-rethrow path.
    //
    // Snapshotting here (rather than re-reading s_compile_fn_for_testing_ inside each lambda)
    // is important for two reasons:
    //   1. Thread-safety: the test thread may clear the seam via clear_compile_fn_for_testing()
    //      while async tasks are in flight; lambdas hold a captured copy and are unaffected.
    //   2. Correctness: all tasks for this init round use the same compile function,
    //      even if the test seam is changed by another test on the same thread afterward.
    const CompileFabricFn compile_fn = s_compile_fn_for_testing_
        ? s_compile_fn_for_testing_
        : CompileFabricFn([](Device* d) { return d->compile_fabric(); });

    std::vector<std::shared_future<Device*>> events;
    events.reserve(devices_.size());
    for (auto* dev : devices_) {
        events.emplace_back(detail::async([dev, compile_fn]() {
            if (compile_fn(dev)) {
                return dev;
            }
            // Compile failure mostly comes from Nebula (TG)
            log_trace(tt::LogMetal, "Did not build fabric on Device {}", dev->id());
            return static_cast<Device*>(nullptr);
        }));
    }

    if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        return;
    }

    const auto& builder_context = control_plane_.get_fabric_context().get_builder_context();

    // Join ALL futures before acting on any exception.  If we rethrow immediately on
    // the first error, the remaining taskflow tasks are abandoned as orphans — they
    // continue running and access BuildEnvManager::device_id_to_build_env_ via the
    // non-const operator[] while the main thread's exception-cleanup path corrupts the
    // map's internal storage, causing SIGSEGV ("Address not mapped" at
    // unordered_map<int,DeviceBuildEnv>::operator[]).
    std::exception_ptr first_ex;
    std::vector<Device*> compiled_devices;
    compiled_devices.reserve(events.size());
    for (const auto& event : events) {
        try {
            compiled_devices.push_back(event.get());
        } catch (...) {
            if (!first_ex) {
                first_ex = std::current_exception();
            }
        }
    }
    // Rethrow now that all tasks have completed — no orphaned threads remain.
    if (first_ex) {
        std::rethrow_exception(first_ex);
    }

    // FIX J (#42429): Two-phase probe-then-configure to prevent non-MMIO relay read races.
    //
    // PROBLEM: The original single loop interleaved terminate_stale_erisc_routers() (probe reads)
    // with configure_fabric() (switches ETH channels to fabric firmware).  On T3K, Devices 0-3
    // are MMIO and Devices 4-7 are non-MMIO, relaying through Devices 0-3's ETH ERISCs.
    //
    // Timeline of the race:
    //   1. Device 0-3 are probed and configured (ETH channels switch to fabric firmware).
    //   2. Device 4's probe read routes through Device 0-3's ETH relay.
    //   3. Fabric firmware on the relay does NOT service UMD relay-read protocol.
    //   4. UMD relay read times out after 5s → relay_broken=true → Device 4-7 added to
    //      dead_relay_devices_ → no fabric firmware loaded on them.
    //   5. Device 0's subordinate ERISCs connecting to Device 4-7 wait forever for peer
    //      ETH handshake → edm_local_sync_ptr never reaches num_local_edms-1 → master
    //      never writes LOCAL_HANDSHAKE_COMPLETE → wait_for_fabric_router_sync() sees
    //      0x00000000 and throws "Timeout after 10000 ms".
    //
    // FIX: Run ALL probe reads FIRST (while relay is still base UMD firmware), then run
    // ALL configure_fabric() calls.  This eliminates the race entirely.
    //
    // FIX H (#42429): Once any non-MMIO device confirms its relay is broken, all subsequent
    // non-MMIO devices share the same fate — on T3K the relay path to all remote chips routes
    // through the same MMIO-device ETH relay ERISCs.  Rather than paying 3×15s probe timeouts
    // per additional non-MMIO device, track whether any relay broke and fast-path the rest.

    // PHASE 1: Probe ALL devices FIRST before any configure_fabric() call.
    // At this point all ETH relay ERISCs are still running base UMD firmware and can service
    // the relay read protocol used by terminate_stale_erisc_routers().
    std::unordered_map<ChipId, std::unordered_set<uint32_t>> probe_dead_channels_map;
    bool any_relay_broken = false;
    for (auto* dev : compiled_devices) {
        if (dev) {
            const bool is_non_mmio = (cluster_.get_associated_mmio_device(dev->id()) != dev->id());

            // Fix A: probe for stale ERISC firmware on all active channels BEFORE
            // configure_fabric_cores() clears L1 and loads the new firmware image.
            // This gives old firmware a chance to terminate cleanly rather than being
            // interrupted mid-execution by an L1 overwrite.
            //
            // The returned pair contains:
            // - probe_dead_channels: channels whose probe read timed out (ERISC unresponsive).
            //   Passed to configure_fabric() so configure_fabric_cores() can skip
            //   assert_risc_reset_at_core() for those channels (#42429).
            // - relay_broken: true when kMaxRelayTimeouts consecutive read timeouts occurred,
            //   indicating the non-MMIO device's ETH relay path is fully broken.
            //   FIX E: track these devices so DeviceManager can skip dispatch kernel init
            //   for them (dispatch writes also route through the dead ETH relay and hang).
            // FIX E2 (#42429): also mark as dead-relay when probe_dead_channels is non-empty
            //   even if relay_broken is false (relay_timeout_count < kMaxRelayTimeouts=3).
            //   With N probe-read timeouts, the relay has N stuck commands in its 4-slot queue.
            //   Dispatch firmware initialization writes to non-ETH cores via the same relay and
            //   can fill the remaining (4-N) slots → hang.  Any probe timeout means the relay
            //   is compromised for subsequent writes.
            std::unordered_set<uint32_t> probe_dead_channels;
            bool relay_broken = false;

            if (any_relay_broken && is_non_mmio) {
                // FIX H: relay already confirmed broken by a prior non-MMIO device — skip the
                // 3×15s probe timeout sequence entirely.  Mark ALL active ETH channels dead so
                // configure_fabric() skips every relay-routed write (ETH reset, runtime args,
                // ConfigureDeviceWithProgram, l1_barrier).  This cuts ~135s off the hang path
                // (3 skipped devices × 3 timeouts × 15s) when the relay ERISCs were left in a
                // corrupt mid-handshake state by a prior abrupt process termination (#42429).
                relay_broken = true;
                const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
                const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);
                for (const auto& [chan_id, dir] : active_channels) {
                    probe_dead_channels.insert(chan_id);
                }
                log_warning(
                    tt::LogMetal,
                    "compile_and_configure_fabric: Device {} non-MMIO ETH relay already confirmed "
                    "broken by prior device — skipping terminate_stale_erisc_routers() probe "
                    "(FIX H #42429). Marking all {} ETH channel(s) as dead.",
                    dev->id(),
                    probe_dead_channels.size());
            } else {
                std::tie(probe_dead_channels, relay_broken) =
                    terminate_stale_erisc_routers(dev, builder_context);
            }

            if (relay_broken || !probe_dead_channels.empty()) {
                dead_relay_devices_.insert(dev->id());
                log_warning(
                    tt::LogMetal,
                    "compile_and_configure_fabric: Device {} ETH relay compromised (relay_broken={}, "
                    "probe_dead_channels={}) — marking as dead-relay device. "
                    "Dispatch kernel initialization will be skipped (#42429 FIX E2).",
                    dev->id(),
                    relay_broken,
                    probe_dead_channels.size());
                if (relay_broken && is_non_mmio) {
                    any_relay_broken = true;
                }
            }

            probe_dead_channels_map[dev->id()] = std::move(probe_dead_channels);
        }
    }

    // PHASE 2: Configure ALL devices now that probing is complete.
    // configure_fabric() switches ETH channels from base UMD firmware to fabric firmware.
    // configure_fabric_cores() performs ERISC0 soft reset (assert_risc_reset_at_core /
    // deassert_risc_reset_at_core) which is a relay read for non-MMIO devices.  If MMIO
    // relay ERISCs are already running fabric firmware, those relay reads time out.
    //
    // FIX J2: configure non-MMIO devices FIRST (while the MMIO ETH relay is still base
    // UMD firmware and can service relay reads), then configure MMIO devices.
    // This is the same ordering discipline as PHASE 1 probe reads.
    size_t configured_count = 0;
    // Pass 1: non-MMIO devices (relay-dependent — must run before MMIO ETH switches fw)
    for (auto* dev : compiled_devices) {
        if (dev && cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            dev->configure_fabric(probe_dead_channels_map[dev->id()]);
            configured_count++;
        }
    }
    // Pass 2: MMIO devices (PCIe-direct — safe to configure after non-MMIO relay ops complete)
    for (auto* dev : compiled_devices) {
        if (dev && cluster_.get_associated_mmio_device(dev->id()) == dev->id()) {
            dev->configure_fabric(probe_dead_channels_map[dev->id()]);
            configured_count++;
        }
    }
    log_info(tt::LogMetal, "Fabric initialized on {} devices", configured_count);

    // FIX I (#42429): Identify MMIO devices whose master router ETH channel connects to a
    // dead-relay non-MMIO device.  Those channels have fabric firmware loaded but their ETH
    // peer (on the non-MMIO device) will never complete the startup handshake — the relay
    // path to it is broken, so the peer ERISC never writes the sync value back.
    // Track them in mmio_dead_peer_devices_ so wait_for_fabric_router_sync() and
    // verify_all_fabric_channels_healthy() can skip them.
    // Cannot add to dead_relay_devices_: that would cause DeviceManager to skip dispatch
    // kernel init for the MMIO device, which is wrong (MMIO dispatch uses PCIe, not ETH relay).
    if (!dead_relay_devices_.empty()) {
        const auto& eth_connections = cluster_.get_ethernet_connections();
        for (auto* dev : compiled_devices) {
            if (!dev) {
                continue;
            }
            // Only MMIO devices are relevant here (non-MMIO are already in dead_relay_devices_).
            if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
                continue;
            }
            // Already tracked as dead-relay — nothing further needed.
            if (dead_relay_devices_.count(dev->id()) > 0) {
                continue;
            }
            const auto master_chan = builder_context.get_fabric_master_router_chan(dev->id());
            auto dev_conn_it = eth_connections.find(dev->id());
            if (dev_conn_it == eth_connections.end()) {
                continue;
            }
            for (const auto& [eth_chan, peer_info] : dev_conn_it->second) {
                if (static_cast<int>(eth_chan) != static_cast<int>(master_chan)) {
                    continue;
                }
                const ChipId peer_chip_id = std::get<0>(peer_info);
                if (dead_relay_devices_.count(peer_chip_id) > 0) {
                    mmio_dead_peer_devices_.insert(dev->id());
                    log_warning(
                        tt::LogMetal,
                        "compile_and_configure_fabric: Device {} MMIO master router chan={} connects "
                        "to dead-relay Device {} — marking for sync skip. Firmware on chan={} loaded "
                        "but peer handshake will never complete (peer ETH relay broken). "
                        "Dispatch init not affected. (#42429 FIX I)",
                        dev->id(),
                        master_chan,
                        peer_chip_id,
                        master_chan);
                }
                break;  // master_chan found; no need to iterate further channels for this device
            }
        }
    }
}

void FabricFirmwareInitializer::wait_for_fabric_router_sync(uint32_t timeout_ms) const {
    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    const auto& fabric_context = control_plane_.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();

    auto wait_for_handshake = [&](Device* dev) {
        if (!dev) {
            TT_THROW("Fabric router sync on null device. All devices must be opened for Fabric.");
        }
        if (builder_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
            return;
        }
        // FIX G (#42429): skip dead-relay devices — their ETH relay path is broken, so any
        // read through it (ReadFromDeviceL1 → l1_barrier → wait_for_non_mmio_flush) will
        // block until UMD's relay timeout fires and then throw.  These devices had fabric
        // firmware skipped in compile_and_configure_fabric() (FIX E2); there is no router
        // to sync with.  Attempting the sync here is always wrong.
        if (dead_relay_devices_.count(dev->id()) > 0) {
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_router_sync: Device {} is a dead-relay device — skipping router "
                "sync (no fabric firmware loaded, relay path broken). (#42429 FIX G)",
                dev->id());
            return;
        }
        // FIX I (#42429): skip MMIO devices whose master router channel connects to a dead-relay
        // non-MMIO peer.  Firmware was loaded on the master channel but the peer will never write
        // the sync value back — waiting here would always time out and throw.
        if (mmio_dead_peer_devices_.count(dev->id()) > 0) {
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_router_sync: Device {} MMIO master router connects to dead-relay "
                "peer — skipping router sync (firmware loaded but peer handshake will never "
                "complete). (#42429 FIX I)",
                dev->id());
            return;
        }

        const auto master_router_chan = builder_context.get_fabric_master_router_chan(dev->id());
        const auto master_router_logical_core =
            cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(master_router_chan, CoordSystem::LOGICAL);

        const auto [router_sync_address, expected_status] = builder_context.get_fabric_router_sync_address_and_status();
        std::vector<std::uint32_t> master_router_status{0};

        auto start_time = std::chrono::steady_clock::now();
        while (master_router_status[0] != expected_status) {
            // Wrap in try/catch: on a T3K after an abrupt prior-process crash, the master
            // router channel may be completely unresponsive — l1_barrier or read_non_mmio
            // will block for the UMD timeout or throw instead of returning cleanly.
            // Treat a read exception as an immediate timeout so we fail with a diagnostic
            // rather than hanging indefinitely (seen as exit=124 after 10+ minutes).
            try {
                detail::ReadFromDeviceL1(
                    dev, master_router_logical_core, router_sync_address, 4, master_router_status, CoreType::ETH);
            } catch (const std::exception& read_ex) {
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Device {} master chan={} read TIMED OUT ({}). "
                    "Treating as router sync failure.",
                    dev->id(),
                    master_router_chan,
                    read_ex.what());
                TT_THROW(
                    "Fabric Router Sync: Device {} master chan={} read timed out: {}",
                    dev->id(),
                    master_router_chan,
                    read_ex.what());
            } catch (...) {
                TT_THROW(
                    "Fabric Router Sync: Device {} master chan={} read failed (unknown exception).",
                    dev->id(),
                    master_router_chan);
            }
            if (master_router_status[0] == expected_status) {
                break;
            }
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            if (elapsed_ms > timeout_ms) {
                log_info(
                    tt::LogMetal,
                    "Fabric Router Sync: master chan={}, logical core={}, sync address=0x{:08x}",
                    master_router_chan,
                    master_router_logical_core.str(),
                    router_sync_address);
                TT_THROW(
                    "Fabric Router Sync: Timeout after {} ms. Device {}: Expected status 0x{:08x}, got 0x{:08x}",
                    timeout_ms,
                    dev->id(),
                    expected_status,
                    master_router_status[0]);
            }
        }

        auto ready_address_and_signal = builder_context.get_fabric_router_ready_address_and_signal();
        if (ready_address_and_signal) {
            std::vector<uint32_t> ready_signal(1, ready_address_and_signal->second);
            detail::WriteToDeviceL1(
                dev, master_router_logical_core, ready_address_and_signal->first, ready_signal, CoreType::ETH);
        }
    };

    // Poll devices in tunnel order: farthest-to-closest, then MMIO device itself
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }

        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        for (const auto& tunnel : tunnels_from_mmio) {
            for (auto j = tunnel.size() - 1; j > 0; j--) {
                // Find the device in our device list by chip ID
                auto it =
                    std::find_if(devices_.begin(), devices_.end(), [&](Device* d) { return d->id() == tunnel[j]; });
                if (it != devices_.end()) {
                    wait_for_handshake(*it);
                }
            }
        }

        wait_for_handshake(dev);
    }
}

void FabricFirmwareInitializer::verify_all_fabric_channels_healthy() const {
    // Same early-return guard as wait_for_fabric_router_sync(): on single-chip and TTSim
    // configs fabric_config is DISABLED so fabric_context_ is null — skip the health check.
    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    const auto& fabric_context = control_plane_.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto [router_sync_address, sync_status] = builder_context.get_fabric_router_sync_address_and_status();
    // By the time this check runs, the master ERISC has already propagated READY_FOR_TRAFFIC
    // (0xa3b3c3d3) to all subordinate channels.  LOCAL_HANDSHAKE_COMPLETE (0xa2b2c2d2) is only
    // the intermediate state wait_for_fabric_router_sync() polls on; using it here would be a
    // false positive on every healthy run.
    const uint32_t expected_status = static_cast<uint32_t>(tt_fabric::EDMStatus::READY_FOR_TRAFFIC);

    // Check ALL active ERISC channels on every device — not just the master channel.
    // wait_for_fabric_router_sync() only polls the master channel, which can pass even when
    // non-master channels are still in a corrupt state (e.g. 0x49705180 from a prior process
    // crash).  If we let those through, dispatch cores will hang later when they try to push
    // commands through the broken fabric path, causing 5s+ timeouts followed by 16+ minutes
    // of triage scripts — all wasted time.
    //
    // Retry with backoff: a channel that's 1ms away from READY_FOR_TRAFFIC should not cause a
    // false failure.  We retry up to kMaxRetries times with kRetryDelayMs between attempts.
    // Only channels that still fail after all retries are reported.
    //
    // Distinguish corrupt vs. still-initializing: use is_known_edm_status() to classify failing
    // channels.  A channel at a valid-but-not-yet-ready EDMStatus (e.g. DOWNSTREAM_EDM_SETUP_STARTED)
    // is "still initializing" — may recover with more time.  A channel at an unrecognized value
    // (e.g. 0x49705530) has corrupt L1 and will never recover without a chip reset.
    //
    // Fail-fast: if any channel is NOT at expected_status after retries, throw now so the test
    // fails immediately with a clear diagnostic instead of hanging.
    constexpr uint32_t kMaxRetries = 3;
    constexpr uint32_t kRetryDelayMs = 50;  // 3 * 50ms = 150ms total; sufficient for slow subordinate init

    // Collect (device_id, eth_chan_id, eth_logical_core) for all channels to check.
    struct ChannelInfo {
        Device* dev;
        uint32_t eth_chan_id;
        CoreCoord eth_logical_core;
    };
    std::vector<ChannelInfo> channels_to_check;

    for (auto* dev : devices_) {
        if (builder_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
            continue;
        }
        // FIX G (#42429): skip dead-relay devices — no fabric firmware was loaded on them
        // (FIX E2 skipped dispatch kernel init), so there is no router to verify.  Any
        // read attempt through their broken relay path will timeout or throw.
        if (dead_relay_devices_.count(dev->id()) > 0) {
            log_warning(
                tt::LogMetal,
                "verify_all_fabric_channels_healthy: Device {} is a dead-relay device — "
                "skipping channel health verification (no fabric firmware loaded). (#42429 FIX G)",
                dev->id());
            continue;
        }
        // FIX I (#42429): skip MMIO devices whose master router connects to a dead-relay peer.
        // Firmware was loaded but peer never started — channel will not be at READY_FOR_TRAFFIC.
        if (mmio_dead_peer_devices_.count(dev->id()) > 0) {
            log_warning(
                tt::LogMetal,
                "verify_all_fabric_channels_healthy: Device {} MMIO master router connects to "
                "dead-relay peer — skipping health check (peer firmware never started). "
                "(#42429 FIX I)",
                dev->id());
            continue;
        }
        const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
        const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);
        for (const auto& [eth_chan_id, direction] : active_channels) {
            const auto eth_logical_core =
                cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
            channels_to_check.push_back({dev, eth_chan_id, eth_logical_core});
        }
    }

    // Track which channels have NOT yet reached expected_status.
    // Start with all channels as "pending"; remove them as they pass.
    struct FailedChannel {
        ChipId device_id;
        uint32_t eth_chan_id;
        uint32_t actual_status;
        bool is_corrupt;  // true = unrecognized status (corrupt L1), false = valid but not ready
    };
    std::vector<FailedChannel> failed_channels;

    // Indices into channels_to_check that still need checking.
    std::vector<size_t> pending_indices;
    pending_indices.reserve(channels_to_check.size());
    for (size_t i = 0; i < channels_to_check.size(); i++) {
        pending_indices.push_back(i);
    }

    for (uint32_t attempt = 0; attempt < kMaxRetries && !pending_indices.empty(); attempt++) {
        if (attempt > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs));
        }

        std::vector<size_t> still_pending;
        for (size_t idx : pending_indices) {
            const auto& ch = channels_to_check[idx];
            std::vector<uint32_t> status_buf(1, 0);
            // Wrap in try/catch: on a T3K after an abrupt prior-process crash, some ERISC
            // channels are completely unresponsive — l1_barrier or read_non_mmio will throw
            // (or in rare cases block for the UMD timeout) rather than returning cleanly.
            // Treat a read exception as "corrupt L1" so we fail-fast with a diagnostic
            // instead of hanging here indefinitely (exit=124 after 10+ minutes).
            try {
                detail::ReadFromDeviceL1(
                    ch.dev, ch.eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
            } catch (const std::exception& read_ex) {
                log_warning(
                    tt::LogMetal,
                    "verify_all_fabric_channels_healthy: Device {} chan={} read TIMED OUT ({}). "
                    "Treating channel as CORRUPT.",
                    ch.dev->id(),
                    ch.eth_chan_id,
                    read_ex.what());
                // Mark as unrecoverable — no point retrying a completely unresponsive channel.
                if (attempt == kMaxRetries - 1) {
                    failed_channels.push_back({ch.dev->id(), ch.eth_chan_id, 0xDEAD'DEAD, /*is_corrupt=*/true});
                }
                // Skip retry for this channel.
                continue;
            } catch (...) {
                log_warning(
                    tt::LogMetal,
                    "verify_all_fabric_channels_healthy: Device {} chan={} read failed (unknown exception). "
                    "Treating channel as CORRUPT.",
                    ch.dev->id(),
                    ch.eth_chan_id);
                if (attempt == kMaxRetries - 1) {
                    failed_channels.push_back({ch.dev->id(), ch.eth_chan_id, 0xDEAD'DEAD, /*is_corrupt=*/true});
                }
                continue;
            }
            if (status_buf[0] != expected_status) {
                still_pending.push_back(idx);
                // On last attempt, record the failure.
                if (attempt == kMaxRetries - 1) {
                    const bool corrupt = !is_known_edm_status(status_buf[0]);
                    failed_channels.push_back({ch.dev->id(), ch.eth_chan_id, status_buf[0], corrupt});
                }
            }
        }
        pending_indices = std::move(still_pending);
    }

    if (failed_channels.empty()) {
        return;
    }

    // Log each failure with appropriate classification.
    // GAP 5: Check if the channel was force-reset in a prior teardown session.
    std::set<std::pair<ChipId, uint32_t>> force_reset_snapshot;
    {
        std::lock_guard<std::mutex> lock(force_reset_channels_mutex_);
        force_reset_snapshot = force_reset_channels_;
    }

    std::string failure_details;
    uint32_t corrupt_count = 0;
    uint32_t initializing_count = 0;
    uint32_t degraded_count = 0;

    for (const auto& fc : failed_channels) {
        const bool was_force_reset = force_reset_snapshot.count({fc.device_id, fc.eth_chan_id}) > 0;

        std::string classification;
        if (was_force_reset) {
            classification = "DEGRADED (force-reset in prior teardown)";
            degraded_count++;
        } else if (fc.is_corrupt) {
            classification = "CORRUPT (unrecognized status — L1 garbage)";
            corrupt_count++;
        } else {
            // Valid EDMStatus but not READY_FOR_TRAFFIC — name it.
            auto status_enum = static_cast<tt::tt_fabric::EDMStatus>(fc.actual_status);
            classification = fmt::format(
                "STILL_INITIALIZING (status={})", edm_status_name(status_enum));
            initializing_count++;
        }

        log_error(
            tt::LogMetal,
            "verify_all_fabric_channels_healthy: Device {} chan={} actual=0x{:08x} expected=0x{:08x} — {}",
            fc.device_id,
            fc.eth_chan_id,
            fc.actual_status,
            expected_status,
            classification);

        failure_details += fmt::format(
            "  dev={} chan={} status=0x{:08x} ({})\n",
            fc.device_id,
            fc.eth_chan_id,
            fc.actual_status,
            classification);
    }

    TT_THROW(
        "Fabric health check failed after {} retries: {} ERISC channel(s) did not reach "
        "READY_FOR_TRAFFIC (0x{:08x}). Breakdown: {} corrupt, {} still-initializing, {} degraded "
        "(force-reset in prior session).\n{}"
        "Corrupt channels require a tt-smi chip reset to recover. "
        "Still-initializing channels may need a longer fabric_router_sync_timeout. "
        "See #42429.",
        kMaxRetries,
        failed_channels.size(),
        expected_status,
        corrupt_count,
        initializing_count,
        degraded_count,
        failure_details);
}

uint32_t FabricFirmwareInitializer::get_fabric_router_sync_timeout_ms() const {
    if (rtoptions_.get_simulator_enabled()) {
        return 15000;
    }
    auto timeout = rtoptions_.get_fabric_router_sync_timeout_ms();
    return timeout.value_or(10000);
}

}  // namespace tt::tt_metal
