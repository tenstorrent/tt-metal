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
#include "device/device_manager.hpp"
#include "common/executor.hpp"
#include "impl/context/context_descriptor.hpp"

#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include "fabric/fabric_host_utils.hpp"
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_builder_context.hpp"
#include "device/edm_status_utils.hpp"
#include "hal_types.hpp"
#include "hal.hpp"  // FIX XZ: FWMailboxMsg for MMIO ETH heartbeat poll

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

// edm_status_name(), edm_status_str(), and is_known_edm_status() are defined in
// device/edm_status_utils.hpp (shared with device.cpp).

// Thread-local compile seam — default-constructed (empty std::function) in all threads.
// Only set by tests via set_compile_fn_for_testing() before MeshDevice::create().
thread_local FabricFirmwareInitializer::CompileFabricFn FabricFirmwareInitializer::s_compile_fn_for_testing_;

void FabricFirmwareInitializer::set_compile_fn_for_testing(CompileFabricFn fn) {
    s_compile_fn_for_testing_ = std::move(fn);
}

void FabricFirmwareInitializer::clear_compile_fn_for_testing() { s_compile_fn_for_testing_ = {}; }

// Thread-local status-override seam — default-constructed (empty std::function) in all threads.
// Only set by tests via set_status_override_fn_for_testing() before MeshDevice::create().
thread_local FabricFirmwareInitializer::StatusOverrideFn FabricFirmwareInitializer::s_status_override_fn_;

void FabricFirmwareInitializer::set_status_override_fn_for_testing(StatusOverrideFn fn) {
    s_status_override_fn_ = std::move(fn);
}

void FabricFirmwareInitializer::clear_status_override_fn_for_testing() { s_status_override_fn_ = {}; }

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
            const auto router_sync_address = control_plane_.get_fabric_context()
                                                 .get_builder_context()
                                                 .get_fabric_router_sync_address_and_status()
                                                 .first;
            constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);

            for (auto* dev : devices_) {
                if (!control_plane_.is_physical_chip_in_fabric_cluster(dev->id())) {
                    continue;
                }
                const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
                const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);

                uint32_t stale_count = 0;
                uint32_t corrupt_count = 0;
                for (const auto& [eth_chan_id, direction] : active_channels) {
                    const auto eth_logical_core =
                        cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
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
        //
        // FIX AM (#42429): if dead-relay devices are already known, skip the health check.
        // dead_relay_devices_ means fabric is already degraded; verify_all_fabric_channels_healthy()
        // would throw on devices whose router sync was skipped (FIX AL), crashing the process
        // before teardown can run.  Tests will still fail at fabric-op time — the error is
        // not hidden, just moved to the correct failure site.
        if (!dead_relay_devices_.empty()) {
            log_warning(
                tt::LogMetal,
                "FabricFirmwareInitializer::configure: {} dead-relay device(s) present; skipping "
                "verify_all_fabric_channels_healthy() to allow clean teardown (FIX AM). "
                "Fabric is degraded — tests will fail at fabric-op time.",
                dead_relay_devices_.size());

            // FIX QU (#42429): Re-assert per-device flags after Device::configure_fabric() resets them.
            //
            // Device::configure_fabric() resets fabric_relay_path_broken_ = false AND
            // fabric_channels_not_ready_for_traffic_ = false at its top (correct for a clean
            // quiesce cycle where fresh firmware is loaded).  But for devices in dead_relay_devices_
            // or mmio_dead_master_chan_devices_, the fabric path IS still degraded — fresh firmware
            // was NOT loaded on the dead channels.  The flag reset leaves both flags as false, so
            // test-fixture guards (FIX QS: is_fabric_relay_path_broken() ||
            // is_fabric_channels_not_ready_for_traffic()) see a healthy-looking cluster and proceed
            // to dispatch tensor operations to devices that have no dispatch kernel.  Those ops then
            // hang for TT_METAL_OPERATION_TIMEOUT_SECONDS before throwing TIMEOUT.
            //
            // Re-assert the correct flags here so any test guard that runs after configure() sees
            // the true degraded state and can SKIP or fail fast (#42429 FIX QU).
            for (auto* dev : devices_) {
                if (!dev) {
                    continue;
                }
                if (dead_relay_devices_.count(dev->id()) > 0) {
                    dev->set_fabric_relay_path_broken();
                    log_warning(
                        tt::LogMetal,
                        "FabricFirmwareInitializer::configure: FIX QU (#42429) — re-asserting "
                        "fabric_relay_path_broken_ for Device {} (dead-relay; "
                        "Device::configure_fabric() reset it to false). "
                        "Test guards will now correctly detect degraded fabric.",
                        dev->id());
                }
                if (mmio_dead_master_chan_devices_.count(dev->id()) > 0) {
                    dev->set_fabric_channels_not_ready_for_traffic();
                    log_warning(
                        tt::LogMetal,
                        "FabricFirmwareInitializer::configure: FIX QU (#42429) — setting "
                        "fabric_channels_not_ready_for_traffic_ for Device {} (MMIO dead-master-chan; "
                        "verify_all_fabric_channels_healthy() was skipped by FIX AM). "
                        "Test guards will now correctly detect degraded fabric.",
                        dev->id());
                }
            }
        } else {
            verify_all_fabric_channels_healthy();
            // FIX RZ2 (#42429): After ring-sync + health verification complete without errors,
            // clear fabric_stale_base_umd_channels_ for any device where the channels are now
            // confirmed healthy (not_ready=false, relay_broken=false).
            //
            // Background: FIX M sets fabric_stale_base_umd_channels_=true when base-UMD relay
            // channels are transitioned via launch_msg instead of soft reset.  The flag is set
            // BEFORE ring-sync runs and was never cleared, even on the healthy path.  This caused
            // two cascading problems:
            //
            //   1. FIX QW (test guards) skipped ALL subsequent tests because stale_base_umd=true,
            //      even when channels had fully transitioned and ring-sync succeeded.
            //
            //   2. FIX RX (TearDown) skipped quiesce_devices() because fabric_broken=true
            //      (stale_base_umd → fabric_broken), leaving ETH channels in a partially
            //      initialized state.  Each improper teardown left MORE channels in base-UMD
            //      state on the next init (2 → 4 → 4 per non-MMIO device), ultimately causing
            //      ring-sync timeouts and ALL t3k_ttnn_tests to skip indefinitely.
            //
            // Fix: if ring-sync passed and health check passed (channels_not_ready=false,
            // relay_broken=false), the channels are running proper fabric firmware. Clear the
            // stale flag so FIX QW allows tests to run and FIX RX allows proper quiesce.
            for (auto* dev : devices_) {
                if (dev && dev->is_fabric_stale_base_umd_channels() &&
                    !dev->is_fabric_channels_not_ready_for_traffic() && !dev->is_fabric_relay_path_broken()) {
                    dev->clear_fabric_stale_base_umd_channels();
                    log_info(
                        tt::LogMetal,
                        "FIX RZ2 (#42429): Device {} base-UMD channels confirmed healthy after "
                        "ring-sync + health check — clearing fabric_stale_base_umd_channels_. "
                        "FIX QW will allow tests to run; FIX RX will allow proper quiesce.",
                        dev->id());
                }
            }
        }
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
    //
    // FIX AU (#42429): Skip WriteToDeviceL1 + l1_barrier for non-MMIO devices whose relay
    // path is already known broken (fabric_relay_path_broken_=true).
    //
    // Root cause of prior bug: when quiesce_and_restart_fabric_workers sets
    // fabric_relay_path_broken_=true for a non-MMIO device (because Phase 2.5 relay reads
    // timed out), the relay ERISC is still running fabric firmware at
    // REMOTE_HANDSHAKE_COMPLETE.  WriteToDeviceL1 for a non-MMIO device goes through the
    // UMD ETH relay (write_core → wait_for_non_mmio_flush), which blocks for 5 s and then
    // throws a UmdException.  That exception propagates out of teardown(), is swallowed by
    // the ScopedDevices destructor, and leaves the entire force-reset second pass (below)
    // unreachable.  Non-MMIO ERISCs are never reset.  When RiscFirmwareInitializer::teardown
    // (FIX AC) then PCIe-resets the MMIO ERISCs, the non-MMIO ERISCs are still running
    // fabric firmware — their ETH link training blocks the rebooting MMIO ERISCs, which
    // write ROM postcode 0x49705180 to edm_status_address (0x18070).  The next session's
    // terminate_stale_erisc_routers sees this on ALL channels and falls back to degraded
    // mode, causing the 61 s CI timeout cascade.
    //
    // Fix: relay-broken non-MMIO devices skip the TERMINATE write and l1_barrier here.
    // Their channels are still collected into `pending` below (get_active_fabric_eth_channels
    // reads from in-memory control-plane state, not device L1), poll reads will throw and
    // keep them in pending, and after the global deadline they are force-reset via
    // assert_risc_reset_at_core + deassert_risc_reset_at_core — completing the cleanup that
    // was previously unreachable.
    for (auto* dev : devices_) {
        if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
            continue;
        }

        // FIX AU-2 (#42429): The original FIX AU skipped TERMINATE write + l1_barrier entirely
        // for non-MMIO devices with broken relay, which left ERISC channels dirty for subsequent
        // CI jobs.  Now we ATTEMPT the write and catch on failure — cleanup must always be tried.
        // If the write throws (relay dead), the channel still proceeds to the force-reset second
        // pass below, which is the safety net.
        const bool is_non_mmio = cluster_.get_associated_mmio_device(dev->id()) != dev->id();
        if (is_non_mmio && dev->is_fabric_relay_path_broken()) {
            log_warning(
                tt::LogMetal,
                "FIX AU-2 (#42429): Device {} (non-MMIO) relay path broken — "
                "attempting TERMINATE write anyway (was previously skipped). "
                "Failure will be caught; channels will still be force-reset.",
                dev->id());
        }

        try {
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
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "FIX AU-2 (#42429): TERMINATE write or l1_barrier failed for Device {}: {} — "
                "channel will proceed to force-reset second pass.",
                dev->id(),
                e.what());
        } catch (...) {
            // UmdException may not inherit from std::exception
            log_warning(
                tt::LogMetal,
                "FIX AU-2 (#42429): TERMINATE write or l1_barrier threw non-std exception for "
                "Device {} (likely UMD relay timeout) — channel will proceed to force-reset.",
                dev->id());
        }
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
    // FIX AJ (#42429): declared here (outside the anonymous block below) so it is visible
    // to the relay drain loop after the block closes.  Populated inside the block when a
    // non-MMIO device's relay path is confirmed dead (diagnostic read or assert_risc_reset
    // threw).  l1_barrier on such a device calls wait_for_non_mmio_flush() which BLOCKS
    // INDEFINITELY rather than throwing, so we must skip it for these devices.
    std::unordered_set<ChipId> relay_dead_devices;
    {
        const auto router_sync_address = builder_ctx.get_fabric_router_sync_address_and_status().first;
        constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
        constexpr uint32_t teardown_timeout_ms = 5000;
        constexpr uint32_t kSpinsBetweenSleeps = 64;

        // Record a single global deadline for the entire ETH poll phase.
        const auto global_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(teardown_timeout_ms);

        // Collect (dev, chan_id, eth_logical_core) for all active ETH channels.
        struct PendingChannel {
            Device* dev;
            uint32_t eth_chan_id;
            CoreCoord eth_logical_core;
        };
        std::vector<PendingChannel> pending;
        // FIX AU (#42429): Channels belonging to relay-broken non-MMIO devices are excluded
        // from `pending` to prevent ReadFromDeviceL1 (which uses the UMD ETH relay) from
        // throwing an uncaught UmdException that escapes the poll loop.  UmdException does
        // not inherit from std::exception, so the existing catch(const std::exception&) would
        // not catch it — the exception would propagate out of teardown(), abort the entire
        // force-reset second pass, and leave non-MMIO ERISCs running fabric firmware.
        // These channels are collected into relay_broken_force_reset and appended to `pending`
        // after the poll loop completes, so they still receive the assert+deassert force-reset.
        std::vector<PendingChannel> relay_broken_force_reset;
        for (auto* dev : devices_) {
            if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
                continue;
            }
            const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
            const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);
            const bool is_non_mmio = cluster_.get_associated_mmio_device(dev->id()) != dev->id();
            for (const auto& [eth_chan_id, direction] : active_channels) {
                PendingChannel ch{
                    dev,
                    eth_chan_id,
                    cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL)};
                if (is_non_mmio && dev->is_fabric_relay_path_broken()) {
                    // Skip relay-broken channels from the poll loop; queue for direct force-reset.
                    relay_broken_force_reset.push_back(ch);
                    relay_dead_devices.insert(dev->id());
                } else {
                    pending.push_back(ch);
                }
            }
        }

        // FIX PE (GAP-56): Track channels that exit the poll loop via clean TERMINATE
        // acknowledgment so we can zero fw_launch_addr for them after the loop.
        // ERISC firmware does not self-clear fw_launch_addr on TERMINATE — it stays
        // non-zero in L1 after a clean exit.  On the next test's reset_cores(),
        // erisc_app_still_running() reads this non-zero value and fires a false-positive
        // 500ms wait, cascading into force-reset for all 4 MMIO devices every test.
        // FIX PC already clears fw_launch_addr on the force-reset path; this covers
        // the clean clean-exit path (e.g. Devices 1, 3 on T3000 after a partial quiesce).
        std::vector<PendingChannel> cleanly_terminated;

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
                            if (status_buf[0] == terminated_val) {
                                cleanly_terminated.push_back(ch);  // FIX PE: capture for fw_launch_addr clear
                                return true;
                            }
                            return false;
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

        // FIX PE (GAP-56): Clear fw_launch_addr for channels that terminated cleanly.
        // These channels passed the TERMINATE handshake but fw_launch_addr is never
        // zeroed by the ERISC on clean exit.  Without this clear, erisc_app_still_running()
        // in the next test's reset_cores() sees a non-zero fw_launch_addr and fires a
        // false-positive 500ms wait → force-reset cascade on every subsequent test.
        if (!cleanly_terminated.empty()) {
            const auto aeth_idx = hal_.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
            const uint32_t fw_launch_addr_val = hal_.get_jit_build_config(aeth_idx, 0, 0).fw_launch_addr;
            for (const auto& ch : cleanly_terminated) {
                try {
                    const auto virtual_eth_coord = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        ch.dev->id(), ch.eth_logical_core, CoreType::ETH);
                    cluster_.write_core_immediate(
                        ch.dev->id(), virtual_eth_coord, std::vector<uint32_t>{0}, fw_launch_addr_val);
                    log_debug(
                        tt::LogMetal,
                        "FabricFirmwareInitializer::teardown: Device {} chan={} FIX PE — "
                        "cleared fw_launch_addr after clean TERMINATE acknowledgment",
                        ch.dev->id(),
                        ch.eth_chan_id);
                } catch (...) {
                    // Best-effort: MMIO writes succeed via PCIe; relay-broken non-MMIO
                    // devices may throw — acceptable, they will be PCIe-reset by
                    // RiscFirmwareInitializer::teardown anyway.
                }
            }
        }

        // FIX AU (#42429): Append relay-broken non-MMIO channels that were excluded from the
        // poll loop.  They are added here — after the poll — so that the force-reset second
        // pass below handles them just like deadline-expired channels (assert + deassert).
        // relay_dead_devices was already populated during pending collection above, so
        // l1_barrier will be skipped for these devices after the force-reset loop.
        if (!relay_broken_force_reset.empty()) {
            log_warning(
                tt::LogMetal,
                "FabricFirmwareInitializer::teardown: {} channel(s) on relay-broken non-MMIO device(s) "
                "bypassed the poll loop — appending to force-reset list. (FIX AU #42429)",
                relay_broken_force_reset.size());
            pending.insert(pending.end(), relay_broken_force_reset.begin(), relay_broken_force_reset.end());
        }

        // Log which channels missed the deadline — critical for diagnosing partial teardown.
        if (!pending.empty()) {
            std::string missed_list;
            for (const auto& ch : pending) {
                if (!missed_list.empty()) {
                    missed_list += ", ";
                }
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
            // FIX AX (#42429): For non-MMIO channels whose device is already confirmed
            // relay-dead (from a prior channel's failed diagnostic read in this same loop),
            // skip the diagnostic read entirely.  Each diagnostic read on a dead-relay
            // non-MMIO device takes the full 5-second UMD timeout before throwing.
            // Skipping saves (N_channels_per_device - 1) × 5 s per affected device.
            // The sentinel 0xDEAD'DEAD remains in status_buf — same as if the read had thrown.
            const bool is_non_mmio_already_dead = cluster_.get_associated_mmio_device(ch.dev->id()) != ch.dev->id() &&
                                                  relay_dead_devices.count(ch.dev->id()) > 0;

            // Diagnostic read: log the last-seen status before asserting reset.
            // Wrapped in try/catch — if the read itself throws (e.g. device unresponsive),
            // we still want to proceed with force_reset_channels_ registration and
            // assert_risc_reset_at_core for ALL remaining channels.  Without this guard,
            // a single bad read would abort the loop, leaving other channels un-reset and
            // skipping devices_.clear() / init_done.erase(key) at the end of teardown.
            std::vector<uint32_t> status_buf(1, 0xDEAD'DEAD);
            if (!is_non_mmio_already_dead) {
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
                    // FIX AJ: relay path is dead — mark device so l1_barrier is skipped below.
                    if (cluster_.get_associated_mmio_device(ch.dev->id()) != ch.dev->id()) {
                        relay_dead_devices.insert(ch.dev->id());
                    }
                } catch (...) {
                    // UmdException does not inherit from std::exception; catch it here so it
                    // cannot abort the force-reset loop.  (FIX AU #42429)
                    log_warning(
                        tt::LogMetal,
                        "FabricFirmwareInitializer::teardown: diagnostic ReadFromDeviceL1 threw "
                        "non-std exception on Device {} chan={} (likely UMD relay timeout) — "
                        "using sentinel status, proceeding with reset",
                        ch.dev->id(),
                        ch.eth_chan_id);
                    if (cluster_.get_associated_mmio_device(ch.dev->id()) != ch.dev->id()) {
                        relay_dead_devices.insert(ch.dev->id());
                    }
                }
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
            // FIX AX (#42429): Skip assert_risc_reset_at_core for non-MMIO channels whose
            // relay path is already confirmed dead (either from FIX AU relay-broken path, or
            // because the diagnostic read above just threw and inserted this device into
            // relay_dead_devices).
            //
            // Root cause of the new hang: assert_risc_reset_at_core on a non-MMIO device goes
            // through the UMD ETH relay (read_non_mmio inside assert_risc_reset).  When the
            // relay ERISC is unresponsive, every call takes the full 5-second UMD timeout before
            // throwing.  With N channels per non-MMIO device and M non-MMIO devices, each channel
            // costs up to 10 s (5 s diagnostic read + 5 s assert), serialized in the loop.
            // On a T3K (4 non-MMIO devices × 2 channels each) that is up to 80 s of serial
            // waiting — enough to blow the CI step timeout.
            //
            // The assert_risc_reset_at_core would fail anyway because the relay is dead — there
            // is no point spending 5 s per channel confirming it.  RiscFirmwareInitializer::
            // teardown (FIX AC) will PCIe-reset the MMIO ETH cores, which restores the ETH PHY
            // link and lets non-MMIO ERISCs boot into base firmware in the next session.  We
            // still register the channel in force_reset_channels_ (done above) so the next
            // session's diagnostics are correct.
            // FIX BU (#42429) supersedes FIX AX-2: confirmed-dead relay channels skip the assert
            // entirely (saves 5s UMD timeout per channel); MMIO/live channels still attempt assert.
            const bool is_non_mmio_relay_dead = cluster_.get_associated_mmio_device(ch.dev->id()) != ch.dev->id() &&
                                                relay_dead_devices.count(ch.dev->id()) > 0;
            if (is_non_mmio_relay_dead) {
                // FIX BU (#42429): The relay is confirmed dead — assert_risc_reset_at_core goes
                // through read_non_mmio which costs 5s per channel (UMD relay timeout). With 4
                // dead channels on each of Devices 4 and 7, this was ~40s of pure timeout per test.
                // RiscFirmwareInitializer::teardown (FIX AC) will PCIe-reset the MMIO ETH channels
                // that serve as relay, restoring the ETH PHY link so non-MMIO ERISCs can reboot
                // into base firmware on the next session. The channel is already registered in
                // force_reset_channels_ above, so next-session diagnostics remain correct.
                log_info(
                    tt::LogMetal,
                    "FIX BU (#42429): Device {} chan={} relay confirmed dead — "
                    "skipping assert_risc_reset_at_core (saves 5s relay timeout). "
                    "RiscFirmwareInitializer::teardown (FIX AC) will PCIe-reset via MMIO relay channels.",
                    ch.dev->id(),
                    ch.eth_chan_id);
            } else {
                // FIX AI (#42429): assert + deassert to restart the ERISC into base UMD firmware.
                //
                // Previously this only called assert_risc_reset_at_core(ALL) without deassert,
                // leaving ALL RISCs (ERISC0/BRISC + subordinate ERISC/NCRISC) in hardware reset.
                // The subsequent teardown_fabric_config() only deasserted ERISC0, leaving NCRISC
                // (which maintains the ETH PHY link) permanently in reset.  On the next test's
                // fabric init, terminate_stale_erisc_routers() probe reads would timeout because:
                //   - Non-MMIO devices: relay path dead (MMIO ETH core NCRISC in reset = PHY down)
                //   - MMIO devices: l1_barrier → wait_for_non_mmio_flush triggered by non-MMIO
                //     device association could hang or timeout
                // Result: corrupt=4 probe_dead=4 on all devices, all ETH channels dead.
                //
                // Fix: immediately deassert after assert (same pattern as FIX AC in
                // risc_firmware_initializer.cpp and the clean path in teardown_fabric_config).
                // This restarts the ERISC into base UMD relay firmware with all RISCs running,
                // preserving the ETH PHY link for the next session's probe reads.
                //
                // FIX DK-2 (#42429): Before force-resetting a channel that is stuck at
                // REMOTE_HANDSHAKE_COMPLETE (0xa1b1c1d1), attempt a graceful termination by
                // writing IMMEDIATELY_TERMINATE to the termination signal address and polling
                // for EDMStatus::TERMINATED for up to ~500ms.  If the firmware acknowledges
                // and exits cleanly we avoid the assert+deassert entirely, which prevents
                // gateway ETH cores from being left mid-handshake with dead firmware after
                // teardown — a state that caused FIX AA to spuriously skip AllGather tests
                // on the next run.  Only fall through to force-reset on timeout.
                if (status_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE)) {
                    bool graceful_exit = false;
                    try {
                        const auto virtual_eth_coord_dk2 = cluster_.get_virtual_coordinate_from_logical_coordinates(
                            ch.dev->id(), ch.eth_logical_core, CoreType::ETH);
                        std::vector<uint32_t> imm_term_signal(
                            1, static_cast<uint32_t>(tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE));
                        cluster_.write_core_immediate(
                            ch.dev->id(), virtual_eth_coord_dk2, imm_term_signal, termination_signal_address);
                        // Poll for TERMINATED for up to ~500ms (50 × 10ms intervals).
                        constexpr uint32_t kDK2PollIntervalMs = 10;
                        constexpr uint32_t kDK2PollTimeoutMs = 500;
                        for (uint32_t elapsed = 0; elapsed < kDK2PollTimeoutMs; elapsed += kDK2PollIntervalMs) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(kDK2PollIntervalMs));
                            std::vector<uint32_t> poll_buf(1, 0);
                            try {
                                detail::ReadFromDeviceL1(
                                    ch.dev, ch.eth_logical_core, router_sync_address, 4, poll_buf, CoreType::ETH);
                            } catch (...) {
                                break;  // unresponsive — fall through to force-reset
                            }
                            if (poll_buf[0] == terminated_val) {
                                graceful_exit = true;
                                cleanly_terminated.push_back(ch);
                                log_info(
                                    tt::LogMetal,
                                    "FIX DK-2 (#42429): Device {} chan={} exited gracefully via "
                                    "IMMEDIATELY_TERMINATE after {}ms — force-reset skipped.",
                                    ch.dev->id(),
                                    ch.eth_chan_id,
                                    elapsed + kDK2PollIntervalMs);
                                break;
                            }
                        }
                    } catch (...) {
                        // Best-effort: if write/read threw (e.g. dead relay on non-MMIO),
                        // fall through to force-reset as before.
                    }
                    if (graceful_exit) {
                        continue;  // skip assert+deassert for this channel
                    }
                    log_warning(
                        tt::LogMetal,
                        "FIX DK-2 (#42429): Device {} chan={} stuck at REMOTE_HANDSHAKE_COMPLETE "
                        "did not respond to IMMEDIATELY_TERMINATE within 500ms — proceeding with "
                        "force-reset.",
                        ch.dev->id(),
                        ch.eth_chan_id);
                }
                try {
                    const auto virtual_eth_coord = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        ch.dev->id(), ch.eth_logical_core, CoreType::ETH);
                    cluster_.assert_risc_reset_at_core(
                        tt_cxy_pair(ch.dev->id(), virtual_eth_coord), tt::umd::RiscType::ALL);
                    cluster_.deassert_risc_reset_at_core(
                        tt_cxy_pair(ch.dev->id(), virtual_eth_coord), tt::umd::RiscType::ALL);
                    // FIX PC: clear ERISC dispatch launch flag after fabric teardown force-reset.
                    // HW reset (assert + deassert) halts the ERISC and restarts base UMD firmware
                    // but does NOT zero L1.  If dispatch firmware was running on this channel,
                    // fw_launch_addr retains its non-zero value from the prior session.
                    // On the next test's reset_cores(), erisc_app_still_running() reads this
                    // non-zero flag → 500ms wait_until_cores_done timeout → another force-reset
                    // → loop on every single-device test open.
                    // (Observed: 140 occurrences in run 25096771728, 146+ in prior runs.)
                    // Clear fw_launch_addr here so reset_cores() sees the core as idle and skips
                    // the stall entirely.
                    try {
                        const auto aeth_idx =
                            hal_.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
                        const uint32_t fw_launch_addr = hal_.get_jit_build_config(aeth_idx, 0, 0).fw_launch_addr;
                        cluster_.write_core_immediate(
                            ch.dev->id(), virtual_eth_coord, std::vector<uint32_t>{0}, fw_launch_addr);
                    } catch (...) {
                        // Best-effort: MMIO devices succeed via PCIe; non-MMIO with a dead relay
                        // may throw — acceptable, FIX PA in reset_cores() handles the fallback.
                    }
                    // FIX XY-2 (#42429): Successful assert+deassert means the relay path is
                    // restored (ERISC rebooted into base firmware).  Clear relay_broken so that
                    // subsequent multicast writes (cleanup, next session init) are not blocked
                    // by the stale relay_broken flag from an earlier transient failure.
                    try {
                        cluster_.clear_relay_broken(ch.dev->id());
                        log_info(
                            tt::LogMetal,
                            "FIX XY-2 (#42429): relay_broken cleared for Device {} chan={} — "
                            "ERISC force-reset succeeded, relay path restored for subsequent writes.",
                            ch.dev->id(),
                            ch.eth_chan_id);
                    } catch (...) {
                        // Best-effort: if clear fails (e.g. chip not remote), harmless.
                    }
                } catch (const std::exception& e) {
                    log_error(
                        tt::LogMetal,
                        "FabricFirmwareInitializer::teardown: assert/deassert_risc_reset_at_core failed on "
                        "Device {} chan={}: {} — ERISC may still be running or halted! "
                        "Next fabric init should expect corrupt state on this channel.",
                        ch.dev->id(),
                        ch.eth_chan_id,
                        e.what());
                    reset_failed_channels.push_back(fmt::format("dev={}/chan={}", ch.dev->id(), ch.eth_chan_id));
                    // FIX AJ: if assert itself threw (relay path completely dead), mark device
                    // so we skip l1_barrier below — l1_barrier on a dead-relay non-MMIO device
                    // blocks indefinitely in wait_for_non_mmio_flush instead of throwing.
                    if (cluster_.get_associated_mmio_device(ch.dev->id()) != ch.dev->id()) {
                        relay_dead_devices.insert(ch.dev->id());
                    }
                }
            }
        }

        if (!reset_failed_channels.empty()) {
            std::string failed_list;
            for (const auto& s : reset_failed_channels) {
                if (!failed_list.empty()) {
                    failed_list += ", ";
                }
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

    // Capture timestamp after all force-resets complete — used by FIX XZ to report
    // elapsed time since deassert, distinguishing "genuinely fast" (non-MMIO processing
    // provided ample boot time) from "suspiciously fast" (just the FIX DS 50ms delay).
    const auto deassert_end = std::chrono::steady_clock::now();

    // FIX XZ (#42429): Wait for MMIO ERISC channels to finish rebooting after force-reset.
    //
    // Root cause: the force-reset loop above (FIX AI) does assert+deassert on channels that
    // did not reach TERMINATED in time.  After deassert, the ERISC begins rebooting into
    // base-UMD firmware (sentinel 0x49706550), but teardown returns immediately without
    // waiting for the reboot to complete.  If the next session starts quickly:
    //   - terminate_stale_erisc_routers() reads via ETH command-queue protocol, which requires
    //     ERISC to be running to service reads → mid-reboot ERISC can't service → probe_dead
    //   - probe_dead on MMIO cascades to relay_timeout on all non-MMIO devices
    //   - All ETH channels dead → SKIP or FAIL
    //
    // FIX TV in run_launch_phase() provides the same wait on the NEXT session's init, but
    // waiting HERE in teardown eliminates the race entirely — the next session always sees
    // fully-booted MMIO ERISC channels.
    //
    // Only poll MMIO channels (PCIe direct reads, no relay needed).  Non-MMIO channels with
    // dead relay are unreachable anyway and handled by FIX NZ/AX guards.
    {
        const uint32_t hb_addr = hal_.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
        if (hb_addr != 0u) {
            // Collect MMIO channels that were force-reset.
            struct MmioResetChannel {
                tt_cxy_pair target;
                uint32_t prev_hb = 0;
                bool nonzero_seen = false;
                bool ready = false;
            };
            std::vector<MmioResetChannel> mmio_reset_chans;
            {
                std::lock_guard<std::mutex> lock(force_reset_channels_mutex_);
                for (const auto& [chip_id, eth_chan_id] : force_reset_channels_) {
                    if (cluster_.get_associated_mmio_device(chip_id) != chip_id) {
                        continue;  // Non-MMIO — skip, unreachable via PCIe
                    }
                    try {
                        const CoreCoord logical_core =
                            cluster_.get_soc_desc(chip_id).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
                        const CoreCoord virt = cluster_.get_virtual_coordinate_from_logical_coordinates(
                            chip_id, logical_core, CoreType::ETH);
                        mmio_reset_chans.push_back({tt_cxy_pair(chip_id, virt), 0, false, false});
                    } catch (...) {
                        log_debug(
                            tt::LogMetal,
                            "FIX BH: Device {} chan={} coord lookup threw non-std exception — skipping MUX poll "
                            "channel",
                            chip_id,
                            eth_chan_id);
                    }
                }
            }

            if (!mmio_reset_chans.empty()) {
                // FIX DV (#42429): extended to 8000ms for mass-boot after tt-smi reset.
                // When tt-smi resets all 24 MMIO channels simultaneously between test binaries,
                // all channels boot concurrently and compete for ETH link training bandwidth with
                // potentially ROM-state non-MMIO peers.  3000ms was insufficient (all 24 failed
                // FIX BH in run 25967409004); 8000ms gives >2× margin for simultaneous boot.
                constexpr int kRebootWaitMs = 8000;
                constexpr auto kPollInterval = std::chrono::milliseconds(10);
                // FIX DS (#42429): Add a 50ms delay before starting the heartbeat poll.
                //
                // Root cause: after assert+deassert (force-reset), the ERISC begins rebooting
                // through ROM.  L1 is zeroed by ROM early in that sequence — but until the zero
                // write reaches the heartbeat address, the pre-reset 0xABCDxxxx value is still
                // visible via PCIe.  The poll loop's first read therefore sees the STALE
                // 0xABCDxxxx immediately, sets mc.ready = true, and reports "confirmed in 0ms"
                // without waiting for an actual reboot.
                //
                // The downstream consequence: teardown returns claiming channels are ready.
                // The next session starts, reads those channels at the ROM postcode 0x49705180
                // (ROM is still executing), FIX BT promotes them to probe_dead, FIX RR
                // re-resets them, and FIX BH polls for 5000ms — often failing because the
                // reboot time budget is shared with the previous teardown's ROM boot.
                //
                // 50ms is sufficient: WH ERISC ROM zeroes L1 within ~10ms of deassert (same
                // reasoning as FIX AR2's 100ms guard in risc_firmware_initializer.cpp).  Using
                // 50ms gives 5× margin so we enter the loop only after the stale pre-reset
                // value is gone, and the poll then correctly waits for the fresh 0xABCDxxxx
                // written by UMD base firmware after ROM boot completes.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                const auto poll_start = std::chrono::steady_clock::now();

                while (true) {
                    bool all_done = true;
                    for (auto& mc : mmio_reset_chans) {
                        if (mc.ready) {
                            continue;
                        }
                        uint32_t hb_val = 0;
                        try {
                            cluster_.read_reg(&hb_val, mc.target, hb_addr);
                        } catch (...) {
                            log_debug(
                                tt::LogMetal,
                                "FIX BH: PCIe read for MUX heartbeat poll at ({},{},{}) threw non-std exception — "
                                "marking ready",
                                mc.target.chip,
                                mc.target.x,
                                mc.target.y);
                            mc.ready = true;  // PCIe read failed — count as done
                            continue;
                        }
                        if (!mc.nonzero_seen) {
                            if (hb_val != 0) {
                                mc.prev_hb = hb_val;
                                mc.nonzero_seen = true;
                                // UMD base firmware writes static 0xABCDxxxx marker — never increments.
                                if ((hb_val >> 16) == 0xABCDu) {
                                    mc.ready = true;
                                }
                            }
                        } else if ((hb_val >> 16) == 0xABCDu || hb_val != mc.prev_hb) {
                            mc.ready = true;
                        }
                        if (!mc.ready) {
                            all_done = false;
                        }
                    }
                    if (all_done) {
                        break;
                    }
                    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::steady_clock::now() - poll_start)
                                                .count();
                    if (elapsed_ms >= kRebootWaitMs) {
                        const auto not_ready = static_cast<int>(std::count_if(
                            mmio_reset_chans.begin(), mmio_reset_chans.end(), [](const MmioResetChannel& mc) {
                                return !mc.ready;
                            }));
                        const auto timeout_since_deassert_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - deassert_end)
                                .count();
                        log_warning(
                            tt::LogAlways,
                            "FIX XZ (#42429): teardown MMIO ETH heartbeat poll timed out after {}ms "
                            "({}ms since deassert); "
                            "{}/{} channel(s) not yet reporting base firmware. "
                            "Next session may see probe_dead on these channels.",
                            elapsed_ms,
                            timeout_since_deassert_ms,
                            not_ready,
                            static_cast<int>(mmio_reset_chans.size()));
                        break;
                    }
                    std::this_thread::sleep_for(kPollInterval);
                }

                const auto total_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - poll_start)
                        .count();
                const auto ready_count = static_cast<int>(
                    std::count_if(mmio_reset_chans.begin(), mmio_reset_chans.end(), [](const MmioResetChannel& mc) {
                        return mc.ready;
                    }));
                const auto since_deassert_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - deassert_end)
                        .count();
                if (ready_count == static_cast<int>(mmio_reset_chans.size())) {
                    log_info(
                        tt::LogAlways,
                        "FIX XZ (#42429): all {} force-reset MMIO ETH channel(s) confirmed "
                        "base firmware heartbeat in {}ms ({}ms since deassert) — clean state for next session.",
                        mmio_reset_chans.size(),
                        total_ms,
                        since_deassert_ms);
                }
            }
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
    // FIX AK (#42429): transitive relay hang guard.
    // The ETH relay network is shared across the mesh.  A dead relay on device X can cause
    // wait_for_non_mmio_flush() to block indefinitely on *adjacent* non-MMIO devices that route
    // through X — even though those devices are not in relay_dead_devices.  try/catch cannot
    // protect against an indefinite block (no exception is thrown).
    // If ANY device has a confirmed dead relay, skip l1_barrier for ALL non-MMIO devices.
    // Cost: UMD relay queues on surviving non-MMIO devices may retain stuck commands; the next
    // session's relay_broken guard handles queue saturation.  Observed: 4+ min hang avoided.
    if (!relay_dead_devices.empty()) {
        std::string dead_list;
        for (const auto dead_id : relay_dead_devices) {
            if (!dead_list.empty()) {
                dead_list += ", ";
            }
            dead_list += std::to_string(dead_id);
        }
        log_warning(
            tt::LogMetal,
            "FabricFirmwareInitializer::teardown: relay-dead device(s) [{}] confirmed; skipping "
            "l1_barrier relay drain for ALL non-MMIO devices to prevent transitive relay hang "
            "(FIX AK). UMD relay queues may retain stuck commands for surviving devices.",
            dead_list);
    } else {
        for (auto* dev : devices_) {
            if (cluster_.get_associated_mmio_device(dev->id()) == dev->id()) {
                // MMIO devices: l1_barrier is always safe, but there is no relay queue to drain.
                continue;
            }
            // FIX AJ (#42429): skip l1_barrier for devices whose relay path was confirmed dead
            // (diagnostic read or assert_risc_reset threw) during the force-reset pass above.
            // l1_barrier() → wait_for_non_mmio_flush() BLOCKS INDEFINITELY on a dead-relay
            // non-MMIO device instead of throwing, so try/catch cannot protect against it.
            if (relay_dead_devices.count(dev->id())) {
                log_warning(
                    tt::LogMetal,
                    "FabricFirmwareInitializer::teardown: skipping relay drain l1_barrier for "
                    "non-MMIO Device {} — relay path confirmed dead during force-reset (FIX AJ); "
                    "relay queue may still have stuck commands but blocking here would hang the job",
                    dev->id());
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
    }

    devices_.clear();
    initialized_.clear();
    // force_reset_channels_ was populated during this teardown and will be consumed by the
    // next session's verify_all_fabric_channels_healthy() for degraded-channel diagnostics.
    // It is cleared at the start of the next teardown() call.
    init_done.erase(key);
}

void FabricFirmwareInitializer::post_teardown() {
    // Reset fabric config.  This call triggers MetalEnvImpl::teardown_fabric_config(), which
    // populates teardown_timed_out_chips_ with any chip IDs where TERMINATED was not reached
    // within the timeout.  Read those IDs after the call and set fabric_teardown_timed_out_ on
    // the corresponding Device objects so FIX AB can hard-reset their MMIO ETH channels at
    // process exit, even if fabric_relay_path_broken_ was not set on any non-MMIO device.
    descriptor_->metal_context().set_fabric_config(tt::tt_fabric::FabricConfig::DISABLED);

    // FIX AB extension: propagate teardown timeout flag to Device objects.
    const auto& timed_out_chips = descriptor_->env_impl().get_teardown_timed_out_chips();
    if (!timed_out_chips.empty() && descriptor_->metal_context().is_device_manager_initialized()) {
        auto& dm = descriptor_->metal_context().device_manager();
        for (const ChipId chip_id : timed_out_chips) {
            IDevice* dev = dm->get_active_device(chip_id);
            if (dev) {
                dev->set_fabric_teardown_timed_out();
                log_warning(
                    tt::LogAlways,
                    "post_teardown: FIX AB extension — chip {} teardown timed out; "
                    "fabric_teardown_timed_out_ set on device for hard-reset at process exit",
                    chip_id);
            }
        }
    }
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
FabricFirmwareInitializer::TerminateStaleResult FabricFirmwareInitializer::terminate_stale_erisc_routers(
    Device* dev, const tt_fabric::FabricBuilderContext& builder_context) const {
    // Channels whose probe L1 read threw (physically dead link — remote ERISC completely
    // unresponsive).  Returned to the caller so configure_fabric_cores() can skip
    // assert_risc_reset_at_core() for them, avoiding the ch7-style indefinite hang.
    std::unordered_set<uint32_t> probe_dead_channels;
    // FIX M (#42429): Channels with base-UMD relay firmware (edm_status == 0x49706550).
    // configure_fabric_cores() must NOT soft-reset these — their BRISC is alive and serving
    // as the ETH relay endpoint for non-MMIO reads.  Soft-resetting halts the relay BRISC and
    // causes all subsequent reads from MMIO→non-MMIO to time out (5s each, then cascade hang).
    std::unordered_set<uint32_t> base_umd_channels;
    // FIX EXT (#42429): External ETH channels — at 0x49706550 but no in-cluster peer.
    // Soft-reset is skipped (preserve relay BRISC) and write_launch_msg_to_core is skipped
    // (FABRIC_1D not loaded — external peer can never complete ETH handshake).
    std::unordered_set<uint32_t> external_umd_channels;

    if (builder_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
        return {probe_dead_channels, false, base_umd_channels, external_umd_channels};
    }

    const auto router_sync_address = builder_context.get_fabric_router_sync_address_and_status().first;
    const auto [term_addr, term_signal] = builder_context.get_fabric_router_termination_address_and_signal();
    constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
    constexpr uint32_t stale_timeout_ms = 100;  // EDM firmware responds in <5ms; base firmware never responds
    constexpr uint32_t kSpinsBetweenSleeps = 64;

    const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
    const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);

    uint32_t corrupt_count = 0;
    uint32_t canary_count = 0;  // channels with 0xA0A0A0A0 or 0xDEADB07E (crashed / launch lost)
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

    // FIX RP PARALLEL (#42429): constants + deferred-channel collection.
    // Sequential FIX RP polled each ROM-postcode channel for up to kRomPostcodePollTotalMs each;
    // on a T3K with 32 channels all at 0x49705180 (dead non-MMIO side, link training stuck) that
    // was 32 × 5s = 160s all timing out.  Instead, collect all such channels first, then poll
    // all of them together with a single shared deadline.
    constexpr uint32_t kRomPostcode = 0x49705180u;
    constexpr uint32_t kBaseUmdFirmwareSentinel =
        static_cast<uint32_t>(tt::tt_metal::EthDiagSentinel::BASE_UMD_FIRMWARE_SENTINEL);
    constexpr uint32_t kRomPostcodePollIntervalMs = 5;
    constexpr uint32_t kRomPostcodePollTotalMs = 5000;  // shared deadline for entire batch
    struct RomPostcodeChan {
        uint32_t eth_chan_id;
        CoreCoord eth_logical_core;
        bool is_non_mmio;
    };
    std::vector<RomPostcodeChan> rom_postcode_deferred;

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
        if (!seam_provided_status) {
            try {
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
                    log_debug(
                        tt::LogMetal,
                        "terminate_stale_erisc_routers: Device {} chan={} TERMINATE write threw non-std exception — "
                        "best effort only",
                        dev->id(),
                        eth_chan_id);
                }
                // Best-effort: zero edm_status_address even on probe-dead channels.
                // If the read threw but the write succeeds (asymmetric failure), this prevents
                // the next session from seeing stale garbage.  If the write also throws, no harm.
                try {
                    std::vector<uint32_t> zero_buf(1, 0);
                    detail::WriteToDeviceL1(dev, eth_logical_core, router_sync_address, zero_buf, CoreType::ETH);
                } catch (...) {
                    log_debug(
                        tt::LogMetal,
                        "terminate_stale_erisc_routers: Device {} chan={} edm_status zero-write threw non-std "
                        "exception — best effort only",
                        dev->id(),
                        eth_chan_id);
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
        }

        if (status_buf[0] == 0 || status_buf[0] == terminated_val) {
            continue;  // clean — nothing to do
        }

        // Fix F4 / FIX K / FIX L (#42429): if the status word is not a valid EDMStatus value,
        // the L1 slot is either the base-UMD-firmware sentinel (0x49706550 = "iPeP" — live relay)
        // or truly corrupt garbage.  Skip the 50ms TERMINATE poll in both cases.
        // See the FIX K / FIX L comment below for why any writes to the relay kill it.
        const bool known_status = is_known_edm_status(status_buf[0]);
        if (!known_status) {
            // FIX K / FIX L (#42429): Handle unknown-status channels WITHOUT disturbing the ETH relay.
            //
            // 0x49706550 ("iPeP") is the base-UMD-firmware sentinel — it means the ERISC is
            // running the stock UMD relay firmware, NOT stale fabric firmware.  This is the NORMAL
            // state on a T3K after a clean boot or proper teardown (ALL active ETH channels on
            // non-fabric-initialized hardware show this value).
            //
            // Base UMD relay firmware monitors BOTH:
            //   1. termination_signal_address — exits gracefully on TERMINATE (FIX K: don't write this)
            //   2. router_sync_address (edm_status_address) — exits when it sees 0 here (FIX L: don't zero)
            //
            // Both writes were killing MMIO Device 0-3 relay ERISCs during PHASE 1.  By the time
            // PHASE 2 tried to soft-reset non-MMIO Device 4 (relay-dependent), the relay was dead
            // → 5 s UMD timeout.
            //
            // For 0x49706550: DO NOTHING.  The firmware is live and serving as a relay; we must
            // not disturb it.  configure_fabric_cores() will issue a BRISC soft-reset which
            // cleanly restarts the ERISC and loads new firmware, regardless of prior state.
            //
            // For other unknown (truly-corrupt) values: send no TERMINATE (we don't know what
            // firmware, if any, is running), but DO zero edm_status_address to break the
            // cascade — corrupt L1 values persist across container restarts and would otherwise
            // re-poison every subsequent session until hardware reset.
            // kBaseUmdFirmwareSentinel and kRomPostcode declared at function scope above (FIX RP PARALLEL).
            // CANARY value written at the very top of fabric_erisc_router.cpp kernel_main()
            // (#42429).  Means the ERISC transitioned away from base-UMD and entered fabric
            // firmware, but crashed before POSTCODE(INITIALIZATION_STARTED).  This is distinct
            // from both a live base-UMD relay (0x49706550) and a valid EDMStatus value — it
            // indicates a firmware crash-before-init that needs a soft-reset + retry.
            static constexpr uint32_t kFabricKernelMainCanary = 0xA0A0A0A0u;
            // HOST PRE-LAUNCH CANARY: written by configure_fabric() BEFORE write_launch_msg_to_core.
            // Disambiguates the 0x49706550 ambiguity: if a prior session sent the launch message
            // and the ERISC crashed before writing 0xA0A0A0A0, L1 would still show 0x49706550
            // (indistinguishable from a live relay).  The host canary breaks this by stamping
            // 0xDEADB07E before the launch message is sent — so this session can detect the gap.
            static constexpr uint32_t kHostPreLaunchCanary =
                static_cast<uint32_t>(EthDiagSentinel::HOST_PRE_LAUNCH_CANARY);
            const bool is_base_umd = (status_buf[0] == kBaseUmdFirmwareSentinel);
            const bool is_canary = (status_buf[0] == kFabricKernelMainCanary);
            const bool is_host_canary = (status_buf[0] == kHostPreLaunchCanary);

            if (is_base_umd) {
                // Live relay firmware — touch nothing.  configure_fabric_cores() handles the
                // firmware transition via write_launch_msg_to_core (no soft reset needed).
                // FIX M: record this channel so configure_fabric_cores() can skip the soft reset.
                // NOTE: do NOT add to probe_dead_channels — base-UMD state is the expected
                // fresh-boot / post-reset condition (all ERISC channels start here).  Adding
                // base-UMD channels to probe_dead_channels falsely marks every device as
                // dead-relay on a clean machine, which prevents dispatch kernel initialization
                // and causes fetch_queue_reserve_back timeouts on the very first run.
                //
                // FIX EXT (#42429): Distinguish in-cluster vs external base-UMD channels.
                // An "external" channel connects to a peer outside the active device set —
                // either an external T3K board or the host.  Loading FABRIC_1D on external
                // channels causes ETH handshake timeouts because the external peer never
                // responds → Phase 5 master chan stuck at STARTED → FIX AL/AM fires →
                // fabric_channels_not_ready_for_traffic_=true → GTEST_SKIP.
                // Route external channels to external_umd_channels: skip soft-reset (preserve
                // relay BRISC) AND skip write_launch_msg_to_core (do not load FABRIC_1D).
                //
                // FIX EXT2 (#42429): Corrected detection — get_ethernet_connections() returns
                // ALL detected ETH connections INCLUDING external T3K boards (with ChipIds
                // outside our local cluster). The original FIX EXT check (any entry in
                // ethernet_connections?) was insufficient: external boards appear there too.
                // Fix: also verify the peer ChipId is in the active devices_ list.
                {
                    bool peer_in_cluster = false;
                    const auto& eth_connections = cluster_.get_ethernet_connections();
                    auto dev_conn_it = eth_connections.find(dev->id());
                    if (dev_conn_it != eth_connections.end()) {
                        auto chan_it = dev_conn_it->second.find(static_cast<int>(eth_chan_id));
                        if (chan_it != dev_conn_it->second.end()) {
                            // Peer chip exists in UMD topology — verify it is in our active cluster.
                            const ChipId peer_chip_id = std::get<0>(chan_it->second);
                            for (const Device* d : devices_) {
                                if (d->id() == peer_chip_id) {
                                    peer_in_cluster = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (peer_in_cluster) {
                        base_umd_channels.insert(eth_chan_id);
                        log_info(
                            tt::LogMetal,
                            "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} "
                            "(base-UMD-firmware sentinel) — relay is live, skipping all writes. "
                            "Added to base_umd_channels to skip soft reset in configure_fabric_cores.",
                            dev->id(),
                            eth_chan_id,
                            status_buf[0]);
                    } else {
                        external_umd_channels.insert(eth_chan_id);
                        log_info(
                            tt::LogMetal,
                            "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} "
                            "(base-UMD-firmware sentinel) — peer ChipId NOT in active device set "
                            "(external T3K board or host). Added to external_umd_channels; "
                            "soft-reset and write_launch_msg_to_core will both be skipped. "
                            "(FIX EXT2 #42429)",
                            dev->id(),
                            eth_chan_id,
                            status_buf[0]);
                    }
                }
            } else if (is_canary) {
                // 0xA0A0A0A0 canary: fabric firmware entered kernel_main() but crashed before
                // POSTCODE(INITIALIZATION_STARTED).  The ERISC is not running any live firmware —
                // send no TERMINATE, but DO add to probe_dead_channels so configure_fabric_cores()
                // will soft-reset it cleanly.  Do NOT zero edm_status_address — the canary will be
                // overwritten by the next firmware load, so there is no cascade risk.
                probe_dead_channels.insert(eth_chan_id);
                canary_count++;
                log_warning(
                    tt::LogMetal,
                    "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} "
                    "(fabric kernel_main canary) — fabric firmware crashed before "
                    "INITIALIZATION_STARTED. Adding to probe_dead_channels for soft-reset "
                    "by configure_fabric_cores (no TERMINATE needed — firmware is dead).",
                    dev->id(),
                    eth_chan_id,
                    status_buf[0]);
            } else if (is_host_canary) {
                // 0xDEADB07E host pre-launch canary: configure_fabric() wrote this before sending
                // the launch message but the ERISC never transitioned to writing 0xA0A0A0A0.
                // The ERISC is not running any live firmware (launch was lost or ERISC crashed
                // before reaching kernel_main).  Treat identically to kFabricKernelMainCanary:
                // flag for soft-reset, do NOT zero L1 (the next firmware load will overwrite it).
                probe_dead_channels.insert(eth_chan_id);
                canary_count++;
                log_warning(
                    tt::LogMetal,
                    "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} "
                    "(host-pre-launch canary 0xDEADB07E) — host wrote launch canary but ERISC "
                    "never wrote firmware canary 0xA0A0A0A0. Soft-reset needed. "
                    "Adding to probe_dead_channels for soft-reset by configure_fabric_cores "
                    "(no TERMINATE needed — firmware is dead).",
                    dev->id(),
                    eth_chan_id,
                    status_buf[0]);
            } else if (status_buf[0] == kRomPostcode) {
                // FIX BT (#42429): ROM boot postcode — promote IMMEDIATELY to probe_dead_channels.
                //
                // Background (FIX RP PARALLEL history):
                //   Session N teardown fires assert_risc_reset_at_core on non-MMIO channels →
                //   ERISCs reset → BRISC ROM writes 0x49705180 to edm_status_address as a
                //   power-on init postcode → before the ERISC finishes booting to UMD relay
                //   firmware (which would overwrite with 0x49706550), close_device() is called
                //   with relay already marked broken → UMD relay never completes the write.
                //   Session N+1: terminate_stale_erisc_routers reads 0x49705180.
                //
                // FIX RP PARALLEL (earlier) collected all ROM-postcode channels into a batch
                // and polled them against a 5s shared deadline before promoting to
                // probe_dead_channels.  The intent was to allow ERISCs mid-boot to finish.
                //
                // FIX BT: the 5s batch poll is ALWAYS wasted — here's why:
                //   0x49705180 is the BRISC ROM power-on initialization postcode.  An ERISC
                //   stuck at this postcode is frozen in ROM boot waiting for a PCIe-triggered
                //   reset (deassert_risc_reset).  It will NOT write a new value to
                //   edm_status_address on its own — polling forever produces no transition.
                //   The PCIe reset is provided by FIX RR in configure_fabric_cores(), which
                //   runs AFTER terminate_stale_erisc_routers() returns.  Waiting 5s here
                //   achieves nothing — the channel is stuck and FIX RR is the only recovery.
                //
                // Promoting immediately to probe_dead_channels lets FIX RR attempt the soft
                // reset without burning 5s of wall time first.  If FIX RR also fails, the
                // channel is confirmed dead and configure_fabric throws (caught as SKIP by
                // FIX BR in SetUp()).
                //
                // Do NOT zero edm_status_address — 0x49705180 is a valid ROM postcode, not
                // garbage; zeroing it mid-boot could interfere with the ROM init sequence.
                // Do NOT send TERMINATE — there is no firmware to receive it during ROM boot.
                const bool is_non_mmio = cluster_.get_associated_mmio_device(dev->id()) != dev->id();
                probe_dead_channels.insert(eth_chan_id);
                corrupt_count++;
                log_info(
                    tt::LogMetal,
                    "terminate_stale_erisc_routers: FIX BT Device {} chan={} ROM postcode "
                    "0x{:08x} — immediately promoted to probe_dead_channels (no 5s poll). "
                    "FIX RR in configure_fabric_cores() will attempt PCIe soft-reset. "
                    "(is_non_mmio={})",
                    dev->id(),
                    eth_chan_id,
                    kRomPostcode,
                    is_non_mmio);
            } else {
                // Truly corrupt / unknown value — no TERMINATE (unknown firmware), but zero
                // edm_status_address to prevent cascade into the next session.
                log_error(
                    tt::LogMetal,
                    "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} is NOT a "
                    "valid EDMStatus value and NOT the base-UMD sentinel — ERISC L1 appears CORRUPT "
                    "(see #42429). NOT sending TERMINATE. Zeroing edm_status_address to prevent "
                    "cascade. Adding to probe_dead_channels so configure_fabric treats soft-reset "
                    "failure as pre-known (degraded mode) instead of throwing.",
                    dev->id(),
                    eth_chan_id,
                    status_buf[0]);
                // Cascade prevention: zero edm_status_address so the NEXT session sees a clean 0
                // instead of the same garbage value.  Bare-metal L1 is NOT cleared on process exit,
                // so without this the corrupt status persists across container restarts forever.
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
                    log_debug(
                        tt::LogMetal,
                        "terminate_stale_erisc_routers: Device {} chan={} cascade-prevention zero-write threw non-std "
                        "exception — channel may be unreachable",
                        dev->id(),
                        eth_chan_id);
                }
                // FIX O (#42429): Add truly-corrupt channels to probe_dead_channels.
                //
                // While the probe L1 read SUCCEEDED (relay functional), deassert_risc_reset requires
                // an operational NOC — which may be dead if a previous session left this channel in
                // assert_risc_reset state.  Marking it pre-known dead lets configure_fabric run in
                // degraded mode (skip those channels with a warning) rather than throwing
                // "newly-dead ETH channel(s)".  The zero-write above ensures the NEXT session sees
                // a clean 0.
                // NOTE: base-UMD (0x49706550) and ROM postcode (0x49705180) values are handled by
                // is_base_umd and FIX RP branches above — they do NOT reach this else-clause.
                // NOTE: base-UMD channels (is_base_umd == true) are intentionally excluded here —
                // they are normal fresh-boot state and must NOT be treated as dead-relay.
                probe_dead_channels.insert(eth_chan_id);
                corrupt_count++;
            }
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
                detail::ReadFromDeviceL1(dev, eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
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

    // FIX BT (#42429): ROM-postcode channels are now immediately promoted to probe_dead_channels
    // in the per-channel loop above (was: FIX RP PARALLEL — batch poll with 5s shared deadline).
    // rom_postcode_deferred will always be empty after FIX BT; this block is kept as a safety
    // net in case future code paths add to it again, but in practice it never executes.
    if (!rom_postcode_deferred.empty()) {
        log_info(
            tt::LogMetal,
            "terminate_stale_erisc_routers: FIX RP Device {} — parallel polling {} ROM-postcode "
            "channels with {}ms shared deadline (was up to {}ms sequential)",
            dev->id(),
            rom_postcode_deferred.size(),
            kRomPostcodePollTotalMs,
            rom_postcode_deferred.size() * kRomPostcodePollTotalMs);

        const auto rp_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kRomPostcodePollTotalMs);
        std::vector<RomPostcodeChan> rp_remaining = rom_postcode_deferred;

        while (!rp_remaining.empty() && std::chrono::steady_clock::now() < rp_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kRomPostcodePollIntervalMs));
            std::vector<RomPostcodeChan> still_waiting;
            for (const auto& ci : rp_remaining) {
                std::vector<uint32_t> poll_buf(1, 0);
                try {
                    detail::ReadFromDeviceL1(dev, ci.eth_logical_core, router_sync_address, 4, poll_buf, CoreType::ETH);
                } catch (...) {
                    // channel became unresponsive during poll — treat as dead
                    probe_dead_channels.insert(ci.eth_chan_id);
                    corrupt_count++;
                    continue;
                }
                if (poll_buf[0] == kBaseUmdFirmwareSentinel) {
                    base_umd_channels.insert(ci.eth_chan_id);
                    log_info(
                        tt::LogMetal,
                        "terminate_stale_erisc_routers: FIX RP Device {} chan={} ROM postcode "
                        "transitioned to base-UMD sentinel 0x{:08x} — ERISC booted. "
                        "Added to base_umd_channels. (is_non_mmio={})",
                        dev->id(),
                        ci.eth_chan_id,
                        kBaseUmdFirmwareSentinel,
                        ci.is_non_mmio);
                } else if (poll_buf[0] != kRomPostcode) {
                    // unexpected transition — treat as dead
                    log_warning(
                        tt::LogMetal,
                        "terminate_stale_erisc_routers: FIX RP Device {} chan={} ROM postcode "
                        "transitioned to unexpected value 0x{:08x} — treating as dead. "
                        "(is_non_mmio={})",
                        dev->id(),
                        ci.eth_chan_id,
                        poll_buf[0],
                        ci.is_non_mmio);
                    probe_dead_channels.insert(ci.eth_chan_id);
                    corrupt_count++;
                } else {
                    still_waiting.push_back(ci);  // still 0x49705180 — keep polling
                }
            }
            rp_remaining = std::move(still_waiting);
        }

        // Anything still waiting after the shared deadline → probe_dead
        for (const auto& ci : rp_remaining) {
            log_warning(
                tt::LogMetal,
                "terminate_stale_erisc_routers: FIX RP Device {} chan={} ROM postcode "
                "0x{:08x} did NOT transition within {}ms shared deadline. "
                "Adding to probe_dead_channels. (is_non_mmio={})",
                dev->id(),
                ci.eth_chan_id,
                kRomPostcode,
                kRomPostcodePollTotalMs,
                ci.is_non_mmio);
            probe_dead_channels.insert(ci.eth_chan_id);
            corrupt_count++;
        }
    }

    if (corrupt_count > 0 || canary_count > 0 || stale_running_count > 0 || !base_umd_channels.empty()) {
        log_info(
            tt::LogMetal,
            "terminate_stale_erisc_routers: Device {} summary: corrupt={} "
            "canary={} (0xA0A0A0A0 firmware or 0xDEADB07E host-pre-launch) "
            "stale_running={} stale_term_timeout={} (of stale_running) probe_dead={} base_umd={}",
            dev->id(),
            corrupt_count,
            canary_count,
            stale_running_count,
            stale_timeout_count,
            probe_dead_channels.size(),
            base_umd_channels.size());
    }

    return {
        std::move(probe_dead_channels), relay_broken, std::move(base_umd_channels), std::move(external_umd_channels)};
}

// Quiesce/Teardown Phase Protocol
// ================================
// T3K fabric shutdown follows a strict phase ordering to safely drain
// in-flight commands and terminate ERISC firmware without hanging.
// The same phases run for normal init, but quiesce mode adds Phases 3-5.
// Phases 1-2 live in this file; Phases 3-6 live in device.cpp.
//
// PHASE 1 - Dead Channel Detection  [compile_and_configure_fabric → terminate_stale_erisc_routers]
//   Probe each active ETH channel via relay read to find already-dead channels before
//   attempting firmware writes that would hang waiting on a dead peer.
//   Outcome sets:
//     probe_dead_channels  — relay read timed out (channel unresponsive)
//     relay_broken         — 3+ timeouts on a single device (entire relay path gone)
//     base_umd_channels    — reads back 0x49706550 (base UMD firmware sentinel)
//   FIX E2: any non-MMIO device with probe_dead → added to dead_relay_devices_ (skip dispatch init)
//   FIX H:  once relay_broken for an MMIO host, skip probing all non-MMIO devices behind it
//
// PHASE 2 - Fabric Configure  [configure_fabric / configure_fabric_cores]
//   Write EDM firmware to active ETH channels; dead channels identified in Phase 1 are skipped.
//   FIX M:   skip soft-reset for base_umd channels (BRISC halt kills the relay needed for UMD comms)
//   FIX C:   skip WriteRuntimeArgs / ConfigureDeviceWithProgram for dead ETH cores
//
// PHASE 2.5 - Post-configure Dead Channel Catch
//   If configure_fabric throws for a non-MMIO device, set fabric_relay_path_broken_ (FIX AN).
//   FIX AO/AP: skip all further relay-dependent ops for broken-relay non-MMIO devices so that
//              subsequent phases (teardown writes, sync polls) do not hang on a dead relay.
//
// PHASE 3 - ETH Launch  [device.cpp]
//   Launch the quiesce (or init) kernel on each ETH channel and poll for a STARTED heartbeat.
//   Quiesce path: 3-pass launch per FIX AE (non-MMIO first, then MMIO, then drain confirm).
//   FIX AF: poll STARTED between non-MMIO and MMIO launches to catch partial-mesh failures early.
//   FIX AD: skip MMIO ETH soft-reset in quiesce mode (already performed in Phase 2).
//
// PHASE 4 - Router Sync  [wait_for_fabric_router_sync]
//   Poll all devices for EDM_STATUS_STARTED to confirm firmware is live before traffic.
//   Skip dead_relay_devices_ (FIX G). TT_THROW on timeout for non-MMIO devices only.
//
// PHASE 5 - AllGather  [quiesce mode only, device.cpp]
//   Run AllGather op to drain the dispatch queue before teardown.
//   Failure here is fatal — quiesce guarantees a clean state or it is an error.
//
// PHASE 5b - ETH Health Verification  [verify_all_fabric_channels_healthy]
//   Read edm_status from each active ETH channel to confirm healthy firmware.
//   Skip dead_relay_devices_ (FIX G).
//   FIX AK:  non-fatal for partial-mesh configs (not all channels required).
//   FIX W/AK: channels reading 0xDEADECE7 / 0xDEAD5B5B → all-dead clean return.
//   FIX AM:  set fabric_channels_not_ready_for_traffic_ on health check failure so callers
//            can skip dispatch rather than hanging on a non-functional fabric.
//
// PHASE 6 - Teardown  [device.cpp]
//   Poll ETH channels for TERMINATED status; force-reset any that do not respond in time.
//   FIX AI:   force-reset targets all RISCs (not just ERISC) on the unresponsive channel.
//   FIX F5a:  unconditional force-reset on timeout — do not leave firmware in an unknown state.
//   FIX AP/AO: skip relay-dependent teardown writes for fabric_relay_path_broken_ devices.
//   FIX F2.5: force-reset unresponsive ERISC before overwriting its L1 region.
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
    uint32_t suppressed_count = 0;
    std::vector<Device*> compiled_devices;
    compiled_devices.reserve(events.size());
    for (const auto& event : events) {
        try {
            compiled_devices.push_back(event.get());
        } catch (...) {
            if (!first_ex) {
                first_ex = std::current_exception();
            } else {
                suppressed_count++;
            }
        }
    }
    // Rethrow now that all tasks have completed — no orphaned threads remain.
    if (first_ex) {
        if (suppressed_count > 0) {
            log_error(
                tt::LogMetal,
                "compile_and_configure_fabric: {} additional device(s) also threw (suppressed by first exception)",
                suppressed_count);
        }
        std::rethrow_exception(first_ex);
    }

    // A fresh fabric configure session must recompute degraded topology from the current
    // probe results. Leaving these sets or per-device flags populated from an earlier open
    // would make quiesce skip router readiness on an MMIO device that may be healthy now.
    dead_relay_devices_.clear();
    mmio_dead_peer_devices_.clear();
    mmio_dead_master_chan_devices_.clear();
    // FIX BE (#42429): Clear per-cycle state that was never reset between teardown+reinit
    // cycles, causing progressive accumulation of base-UMD channel bookkeeping.
    //
    // Root cause: after each teardown+reinit cycle, channels that were running fabric firmware
    // get force-reset back to base-UMD (edm_status=0x49706550).  On the next init cycle,
    // terminate_stale_erisc_routers() correctly identifies them as base-UMD.  But these
    // four members were never cleared at cycle start:
    //   - external_umd_channels_map_: stale entries from prior cycle cause get_external()
    //     to return wrong channel sets for devices that went through FIX H fast-path
    //   - has_base_umd_channels_: once true, never reverted — causes extended ring-sync
    //     timeouts (30s instead of 10s) on cycles where no base-UMD channels exist
    //   - timeout_on_base_umd_devices_: accumulated across cycles, causing health checks
    //     to skip channels on devices that were healthy in the current cycle
    //   - ring_sync_already_timed_out_: once set, fast-skips ring sync for all subsequent
    //     devices in every future cycle — even when the ring is healthy
    //
    // Fix: clear all four at the start of each compile_and_configure_fabric() call, alongside
    // the existing clears for dead_relay_devices_ et al.  Each cycle recomputes them from
    // fresh probe results.
    // FIX BE (#42429): log when stale state is present so CI logs capture
    // whether progressive accumulation occurred before clearing.
    if (!external_umd_channels_map_.empty() || has_base_umd_channels_ || !timeout_on_base_umd_devices_.empty() ||
        ring_sync_already_timed_out_) {
        log_debug(
            tt::LogMetal,
            "compile_and_configure_fabric: FIX BE (#42429) clearing stale per-cycle state: "
            "external_umd_map.size()={} has_base_umd={} timeout_devs.size()={} ring_sync_timed_out={}",
            external_umd_channels_map_.size(),
            has_base_umd_channels_,
            timeout_on_base_umd_devices_.size(),
            ring_sync_already_timed_out_);
    }
    external_umd_channels_map_.clear();
    has_base_umd_channels_ = false;
    timeout_on_base_umd_devices_.clear();
    ring_sync_already_timed_out_ = false;
    for (auto* dev : compiled_devices) {
        if (dev) {
            dev->set_fabric_is_mmio_dead_peer_device(false);
        }
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
    // FIX H (#42429): Once a non-MMIO device confirms its relay is broken, all subsequent
    // non-MMIO devices behind the SAME MMIO host share the same fate — the relay path to
    // those remote chips routes through that MMIO device's ETH relay ERISCs.  Rather than
    // paying 3×15s probe timeouts per additional non-MMIO device, track which MMIO hosts
    // have a confirmed broken relay and fast-path non-MMIO devices behind them.
    //
    // Galaxy note: Galaxy has multiple MMIO chips, each with its own relay path.  A broken
    // relay on one MMIO chip does NOT imply the relay behind other MMIO chips is broken.
    // The per-MMIO-host set below correctly scopes the fast-path to the affected MMIO host.

    // PHASE 1: Probe ALL devices FIRST before any configure_fabric() call.
    // At this point all ETH relay ERISCs are still running base UMD firmware and can service
    // the relay read protocol used by terminate_stale_erisc_routers().
    std::unordered_map<ChipId, std::unordered_set<uint32_t>> probe_dead_channels_map;
    // FIX M (#42429): channels with base-UMD relay firmware (0x49706550) — configure_fabric_cores()
    // must skip soft reset for these to avoid killing the ETH relay endpoint.
    std::unordered_map<ChipId, std::unordered_set<uint32_t>> base_umd_channels_map;
    // relay_broken_mmio_hosts: set of MMIO chip IDs whose relay path is confirmed broken.
    // Non-MMIO devices behind a broken MMIO host are fast-pathed (FIX H) without probing.
    std::unordered_set<ChipId> relay_broken_mmio_hosts;
    for (auto* dev : compiled_devices) {
        if (dev) {
            const bool is_non_mmio = (cluster_.get_associated_mmio_device(dev->id()) != dev->id());
            const ChipId mmio_host = cluster_.get_associated_mmio_device(dev->id());

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

            if (is_non_mmio && relay_broken_mmio_hosts.count(mmio_host)) {
                // FIX H: relay already confirmed broken for MMIO host {} — skip the
                // 3×15s probe timeout sequence entirely for non-MMIO devices behind that host.
                // Mark ALL active ETH channels dead so configure_fabric() skips every
                // relay-routed write (ETH reset, runtime args, ConfigureDeviceWithProgram,
                // l1_barrier).  This cuts ~135s off the hang path (3 skipped devices × 3
                // timeouts × 15s) when the relay ERISCs were left in a corrupt mid-handshake
                // state by a prior abrupt process termination (#42429).
                relay_broken = true;
                const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
                const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);
                for (const auto& [chan_id, dir] : active_channels) {
                    probe_dead_channels.insert(chan_id);
                }
                log_warning(
                    tt::LogMetal,
                    "compile_and_configure_fabric: Device {} non-MMIO ETH relay confirmed broken "
                    "via MMIO host {} — skipping terminate_stale_erisc_routers() probe "
                    "(FIX H #42429). Marking all {} ETH channel(s) as dead.",
                    dev->id(),
                    mmio_host,
                    probe_dead_channels.size());
            } else {
                // FIX AP dependency (#42429): terminate_stale_erisc_routers() must run before
                // configure_fabric() — it quiesces existing ERISC router state that could cause
                // phantom traffic or handshake confusion during fabric bringup.
                auto result = terminate_stale_erisc_routers(dev, builder_context);
                probe_dead_channels = std::move(result.probe_dead_channels);
                relay_broken = result.relay_broken;
                base_umd_channels_map[dev->id()] = std::move(result.base_umd_channels);
                // FIX TH2 (#42429): Record that this session has base-UMD channels so
                // get_fabric_router_sync_timeout_ms() can extend the per-device timeout.
                // NOTE: external_umd_channels do NOT set has_base_umd_channels_ — they never
                // participate in ring-sync, so they cannot cause a ring-barrier timeout that
                // would require the extended 30s timeout window (FIX TH2/TI).
                if (!base_umd_channels_map[dev->id()].empty()) {
                    has_base_umd_channels_ = true;
                }
                // FIX EXT (#42429): collect external channels (no firmware loaded on these).
                external_umd_channels_map_[dev->id()] = std::move(result.external_umd_channels);
            }

            // FIX E2 (#42429): Only mark non-MMIO devices as dead-relay when ETH relay is
            // compromised.  MMIO devices dispatch via PCIe — their ETH relay channel health
            // does not affect dispatch kernel writes.  Applying this to MMIO devices causes
            // dispatch to be skipped entirely even though it would succeed over PCIe.
            if (is_non_mmio && (relay_broken || !probe_dead_channels.empty())) {
                dead_relay_devices_.insert(dev->id());
                // FIX E2 + FIX AY gap fix (#42429): Set fabric_relay_path_broken_ on the device so
                // RiscFirmwareInitializer::teardown() includes it in relay_broken_non_mmio.  Without
                // this, devices in the probe_dead path (firmware never loaded, relay never established)
                // escape the FIX AY + FIX AC teardown cleanup because those fixes are gated on
                // relay_broken_non_mmio, which is only populated from is_fabric_relay_path_broken() and
                // is_fabric_channels_not_ready_for_traffic().  Skipping FIX AY leaves stale corrupt
                // ERISC firmware on non-MMIO chips, which blocks UMD gateway heartbeat probes on the
                // next open and causes "ASIC not found in chip_topology_mapping_" TT_FATAL.
                dev->set_fabric_relay_path_broken();
                log_warning(
                    tt::LogMetal,
                    "compile_and_configure_fabric: Device {} ETH relay compromised (relay_broken={}, "
                    "probe_dead_channels={}) — marking as dead-relay device and setting "
                    "fabric_relay_path_broken_ so FIX AY/AC teardown fires. "
                    "Dispatch kernel initialization will be skipped (#42429 FIX E2).",
                    dev->id(),
                    relay_broken,
                    probe_dead_channels.size());
                if (relay_broken) {
                    relay_broken_mmio_hosts.insert(mmio_host);
                }
            }

            probe_dead_channels_map[dev->id()] = std::move(probe_dead_channels);
        }
    }

    // FIX M2 (#42429): Secondary check — for MMIO device channels showing 0x49706550 whose
    // peer non-MMIO device is already confirmed dead-relay, allow a hard soft-reset instead
    // of skipping.
    //
    // FIX M skips soft reset for channels in base_umd_channels_map to avoid halting the ETH
    // relay BRISC while non-MMIO devices depend on it for relay reads.  This protection is
    // correct when the peer non-MMIO device is alive and issuing relay reads.  But when the
    // peer device is in dead_relay_devices_ (confirmed unreachable, no firmware loaded), the
    // relay on that specific channel has NOTHING to serve.  Removing it from base_umd_channels
    // lets configure_fabric_cores() perform a normal ERISC0 soft-reset, giving a clean slate.
    //
    // NOTE: We do NOT add these channels to probe_dead_channels_map.  Soft-reset (assert +
    // deassert ERISC0) is safe here — the peer is dead, no relay reads are in flight by the
    // time we reach PHASE 2 configure.  After reset, write_launch_msg_to_core loads fabric
    // firmware; FIX I handles the fact that the peer handshake will never complete.
    if (!dead_relay_devices_.empty()) {
        const auto& eth_connections_m2 = cluster_.get_ethernet_connections();
        for (auto* dev : compiled_devices) {
            if (!dev) {
                continue;
            }
            // Only MMIO devices can have base-UMD relay channels.
            if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
                continue;
            }
            auto& base_umd_chans = base_umd_channels_map[dev->id()];
            if (base_umd_chans.empty()) {
                continue;
            }
            auto dev_conn_it = eth_connections_m2.find(dev->id());
            if (dev_conn_it == eth_connections_m2.end()) {
                continue;
            }
            std::vector<uint32_t> to_force_reset;
            for (const uint32_t chan : base_umd_chans) {
                auto chan_conn_it = dev_conn_it->second.find(static_cast<int>(chan));
                if (chan_conn_it == dev_conn_it->second.end()) {
                    continue;
                }
                const ChipId peer_chip_id = std::get<0>(chan_conn_it->second);
                if (dead_relay_devices_.count(peer_chip_id) > 0) {
                    to_force_reset.push_back(chan);
                    log_warning(
                        tt::LogMetal,
                        "compile_and_configure_fabric: FIX M2 (#42429) — Device {} chan={} "
                        "shows 0x49706550 (base-UMD relay) but peer Device {} is confirmed "
                        "dead-relay. Removing from base_umd_channels so configure_fabric_cores() "
                        "performs a hard soft-reset instead of skipping (relay has nothing to serve).",
                        dev->id(),
                        chan,
                        peer_chip_id);
                }
            }
            for (const uint32_t chan : to_force_reset) {
                base_umd_chans.erase(chan);
            }
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
    // FIX EXT (#42429): empty set used as default when a device has no external channels.
    static const std::unordered_set<uint32_t> kEmptyChannelSet;
    auto get_external = [&](ChipId id) -> const std::unordered_set<uint32_t>& {
        auto it = external_umd_channels_map_.find(id);
        return it != external_umd_channels_map_.end() ? it->second : kEmptyChannelSet;
    };
    // Pass 1: non-MMIO devices (relay-dependent — must run before MMIO ETH switches fw)
    for (auto* dev : compiled_devices) {
        if (dev && cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            dev->configure_fabric(
                probe_dead_channels_map[dev->id()], base_umd_channels_map[dev->id()], get_external(dev->id()));
            configured_count++;
        }
    }
    // Pass 2: MMIO devices (PCIe-direct — safe to configure after non-MMIO relay ops complete)
    // FIX DJ (#42429): Parallelize MMIO configure_fabric() across all MMIO devices.
    // Sequential cost: N_mmio × (kFIX_BH_BootWaitMs + kFIX_DI_RetryWaitMs) = 4 × 6s = 24s.
    // Parallel cost: max(device_times) ≈ 6s regardless of MMIO device count.
    {
        std::vector<std::shared_future<void>> mmio_futs;
        for (auto* dev : compiled_devices) {
            if (dev && cluster_.get_associated_mmio_device(dev->id()) == dev->id()) {
                auto probe_dead = probe_dead_channels_map[dev->id()];
                auto base_umd  = base_umd_channels_map[dev->id()];
                auto ext       = get_external(dev->id());
                mmio_futs.emplace_back(detail::async([dev, probe_dead, base_umd, ext]() mutable {
                    dev->configure_fabric(probe_dead, base_umd, ext);
                }));
                configured_count++;
            }
        }
        std::exception_ptr first_ex;
        for (auto& fut : mmio_futs) {
            try {
                fut.get();
            } catch (...) {
                if (!first_ex) first_ex = std::current_exception();
            }
        }
        if (first_ex) std::rethrow_exception(first_ex);
    }
    log_info(tt::LogMetal, "Fabric initialized on {} devices", configured_count);

    // FIX SB2 (#42429): When FIX M fires on any MMIO relay channel, that channel transitions
    // from UMD relay firmware to fabric EDM firmware via launch_msg without soft-reset.
    // After PHASE 2, the MMIO ERISC now runs fabric EDM firmware — it no longer serves the
    // UMD relay protocol.  Any subsequent relay read/write from non-MMIO devices through that
    // MMIO host will HANG indefinitely (the ERISC accepts the relay handshake but routes the
    // request as EDM traffic, so the read never completes).
    //
    // The ENTRY snapshot 6-second deadline in quiesce_and_restart_fabric_workers() only fires
    // if relay reads THROW exceptions.  A blocking (hanging) read is NOT caught by try/catch
    // and is NOT bounded by a deadline check — the thread is simply blocked.
    //
    // Proactively mark all non-MMIO devices behind affected MMIO hosts as relay-broken so
    // the ENTRY snapshot, Phase 2.5, and Phase 3 all skip their relay reads (FIX R guard:
    // fabric_relay_path_broken_ && !is_mmio_capable()).
    // FIX BE2 (#42429): iterate compiled_devices for deterministic log ordering
    // (base_umd_channels_map is unordered — direct iteration gives non-deterministic logs)
    for (auto* mmio_dev : compiled_devices) {
        if (!mmio_dev) {
            continue;
        }
        // Only process MMIO devices
        if (cluster_.get_associated_mmio_device(mmio_dev->id()) != mmio_dev->id()) {
            continue;
        }
        auto it = base_umd_channels_map.find(mmio_dev->id());
        if (it == base_umd_channels_map.end() || it->second.empty()) {
            continue;
        }
        const auto mmio_id = mmio_dev->id();
        const auto& base_umd_chans = it->second;
        // This MMIO host had FIX M channels — its ERISC now runs EDM firmware, not UMD relay.
        for (auto* dev : compiled_devices) {
            if (!dev) {
                continue;
            }
            // Only non-MMIO devices are affected; MMIO reads its own L1 via PCIe directly.
            if (cluster_.get_associated_mmio_device(dev->id()) != mmio_id) {
                continue;
            }
            if (dev->id() == mmio_id) {
                continue;  // skip the MMIO host itself
            }
            if (dead_relay_devices_.count(dev->id()) > 0) {
                continue;  // already marked broken via other mechanism
            }
            // FIX SB2-R (#42429): Only mark relay broken if this non-MMIO device's probe
            // detected actual problems (dead channels / timeouts).  On a clean boot, all ETH
            // channels start at 0x49706550 (base-UMD sentinel) and probe cleanly — the FIX M
            // transition is the *intended* path and the relay is NOT broken.  Marking it broken
            // on every clean boot caused the entire T3K fabric to show as degraded, skipping
            // all tests.  Only fail-fast when the probe itself flagged this device.
            const auto& probe_dead = probe_dead_channels_map.count(dev->id()) ? probe_dead_channels_map.at(dev->id())
                                                                              : std::unordered_set<uint32_t>{};
            if (probe_dead.empty()) {
                log_debug(
                    tt::LogMetal,
                    "compile_and_configure_fabric: Device {} (non-MMIO) behind MMIO host {} had "
                    "{} FIX M channel(s) but probe was clean — relay_broken NOT set (FIX SB2-R "
                    "#42429). EDM firmware transition is expected on clean boot.",
                    dev->id(),
                    mmio_id,
                    base_umd_chans.size());
                continue;
            }
            dev->set_fabric_relay_path_broken();
            log_warning(
                tt::LogMetal,
                "compile_and_configure_fabric: Device {} (non-MMIO) relay path marked broken — "
                "MMIO host {} had {} FIX M channel(s) AND device probe detected {} dead "
                "channel(s). UMD relay reads through this MMIO host will HANG. ENTRY snapshot / "
                "Phase 2.5 / Phase 3 relay reads skipped. (#42429 FIX SB2-R)",
                dev->id(),
                mmio_id,
                base_umd_chans.size(),
                probe_dead.size());
        }
    }

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
                    dev->set_fabric_is_mmio_dead_peer_device(true);
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

        // FIX I2 (#42429): Transitive closure — if an MMIO device's master channel connects to
        // another MMIO device that's already in mmio_dead_peer_devices_, it should also be added.
        // A dead-peer MMIO device can never complete its fabric handshake, so any MMIO device
        // whose master channel routes through it is also stuck.  Repeat to fixed point.
        // This mirrors propagate_dead_mmio_peers() in RiscFirmwareInitializer.
        bool transitive_changed = true;
        while (transitive_changed) {
            transitive_changed = false;
            for (auto* tdev : compiled_devices) {
                if (!tdev) {
                    continue;
                }
                if (cluster_.get_associated_mmio_device(tdev->id()) != tdev->id()) {
                    continue;  // not MMIO
                }
                if (dead_relay_devices_.count(tdev->id()) > 0) {
                    continue;  // already dead-relay
                }
                if (mmio_dead_peer_devices_.count(tdev->id()) > 0) {
                    continue;  // already in set
                }
                const auto t_master_chan = builder_context.get_fabric_master_router_chan(tdev->id());
                auto t_conn_it = eth_connections.find(tdev->id());
                if (t_conn_it == eth_connections.end()) {
                    continue;
                }
                for (const auto& [t_eth_chan, t_peer_info] : t_conn_it->second) {
                    if (static_cast<int>(t_eth_chan) != static_cast<int>(t_master_chan)) {
                        continue;
                    }
                    const ChipId t_peer_chip_id = std::get<0>(t_peer_info);
                    if (mmio_dead_peer_devices_.count(t_peer_chip_id) > 0) {
                        mmio_dead_peer_devices_.insert(tdev->id());
                        tdev->set_fabric_is_mmio_dead_peer_device(true);
                        log_warning(
                            tt::LogMetal,
                            "compile_and_configure_fabric: Device {} transitively in "
                            "mmio_dead_peer_devices_ — master chan={} connects to dead-peer "
                            "Device {}. (#42429 FIX I2 transitive)",
                            tdev->id(),
                            t_master_chan,
                            t_peer_chip_id);
                        transitive_changed = true;
                    }
                    break;  // master_chan found
                }
            }
        }
    }

    // FIX AN (#42429): Identify MMIO devices whose own master router ETH channel was excluded
    // from configure_fabric_cores() (was in probe_dead_channels — L1 corrupt or channel
    // unresponsive). This is independent of dead_relay_devices_: the fault is local to the MMIO
    // device itself, not a peer relay. No firmware was loaded on that master channel, so
    // wait_for_fabric_router_sync() would spin for the full timeout (10s per device) with no
    // chance of success. Track them here so sync can be skipped.
    //
    // FIX ST (#42429): Use the device's effective fabric_pre_dead_channels_ (post-FIX-RR) rather
    // than the original probe_dead_channels_map (pre-FIX-RR).  When FIX RR recovers the master
    // channel via PCIe-direct soft reset, it is removed from fabric_pre_dead_channels_ and
    // firmware IS loaded on it — the channel is no longer dead.  Using probe_dead_channels_map
    // here caused mmio_dead_master_chan_devices_ to be populated even after recovery, making
    // verify_all_fabric_channels_healthy() set channels_not_ready=true and skip AllGather every
    // session (the same symptom FIX RS fixed for configure_fabric, now fixed here for FIX AN).
    for (auto* dev : compiled_devices) {
        if (!dev) {
            continue;
        }
        // Only MMIO devices — non-MMIO are already handled via dead_relay_devices_.
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }
        // Already in dead_relay_devices_ — sync is already skipped via FIX G.
        if (dead_relay_devices_.count(dev->id()) > 0) {
            continue;
        }
        const auto master_chan = builder_context.get_fabric_master_router_chan(dev->id());
        // FIX ST: use effective (post-FIX-RR) pre-dead set from the device, not probe_dead_channels_map.
        const auto& effective_pre_dead = dev->get_fabric_pre_dead_channels();
        if (effective_pre_dead.count(master_chan) > 0) {
            mmio_dead_master_chan_devices_.insert(dev->id());
            log_warning(
                tt::LogMetal,
                "compile_and_configure_fabric: Device {} MMIO master router chan={} is pre-dead "
                "(excluded from configure_fabric — L1 corrupt or unresponsive, and NOT recovered by FIX RR). "
                "Sync will be skipped. (#42429 FIX AN / FIX ST)",
                dev->id(),
                master_chan);
        } else if (
            probe_dead_channels_map.count(dev->id()) > 0 &&
            probe_dead_channels_map.at(dev->id()).count(master_chan) > 0) {
            // Master chan was in probe_dead but FIX RR recovered it — firmware IS loaded.
            log_info(
                tt::LogMetal,
                "compile_and_configure_fabric: Device {} MMIO master router chan={} was probe-dead "
                "but recovered by FIX RR — NOT adding to mmio_dead_master_chan_devices_. "
                "Firmware loaded; sync will proceed normally. (#42429 FIX ST)",
                dev->id(),
                master_chan);
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
        // FIX TJ (#42429): if a prior device already timed out on the base-UMD ring barrier,
        // all remaining devices will also time out (ring barrier requires every member).
        // Skip the full 30s wait and immediately mark this device as timed-out to avoid
        // N×30s sequential overhead (e.g. 8 devices × 30s = 4 min on a broken cluster).
        if (ring_sync_already_timed_out_ && has_base_umd_channels_) {
            timeout_on_base_umd_devices_.insert(dev->id());
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_router_sync: Device {} skipped — ring sync already timed out "
                "on an earlier device; ring barrier cannot complete (FIX TJ #42429).",
                dev->id());
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
        // FIX AN (#42429): skip MMIO devices whose own master router channel was excluded
        // from configure_fabric (pre-dead — L1 corrupt). No firmware on the local master
        // channel → sync value will never appear → avoid the 10s per-device timeout.
        if (mmio_dead_master_chan_devices_.count(dev->id()) > 0) {
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_router_sync: Device {} own master router chan is pre-dead "
                "(no fabric firmware loaded on it — L1 was corrupt at init). "
                "Skipping router sync. (#42429 FIX AN)",
                dev->id());
            return;
        }

        const auto master_router_chan = builder_context.get_fabric_master_router_chan(dev->id());

        // FIX EXT (#42429): skip if master router channel is external (no in-cluster peer).
        // Firmware was NOT loaded on external channels (write_launch_msg_to_core skipped),
        // so the ring-sync value will never appear.  Skip cleanly — no timeout, no FIX TI.
        {
            auto ext_it = external_umd_channels_map_.find(dev->id());
            if (ext_it != external_umd_channels_map_.end() && ext_it->second.count(master_router_chan) > 0) {
                log_info(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Device {} master chan={} is an external ETH "
                    "channel (no in-cluster peer, firmware not loaded) — skipping ring sync "
                    "cleanly. (FIX EXT #42429)",
                    dev->id(),
                    master_router_chan);
                return;
            }
        }
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
                // FIX AL (#42429): relay path broken for this device's master router channel.
                // Convert to log_error + return instead of TT_THROW to avoid crashing the process
                // when a mesh neighbor is dead-relay (seen: Device 0 sync fails because Device 3
                // dead-relay blocks ring completion → TT_THROW → signal 6 → torn-down state).
                // Fabric on this device is unusable; tests will fail at fabric-op time.
                log_error(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Device {} master chan={} read FAILED ({}). "
                    "Skipping router sync — fabric on this device is unusable (FIX AL).",
                    dev->id(),
                    master_router_chan,
                    read_ex.what());
                return;
            } catch (...) {
                log_error(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Device {} master chan={} read failed (unknown "
                    "exception). Skipping router sync — fabric unusable (FIX AL).",
                    dev->id(),
                    master_router_chan);
                return;
            }
            if (master_router_status[0] == expected_status) {
                break;
            }
            // Also accept READY_FOR_TRAFFIC — a peer may have already written it.
            if (master_router_status[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC)) {
                break;
            }
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            // FIX AO (#42429): STARTED early-exit for initial ring sync.  STARTED (0xa0b0c0d0)
            // means firmware booted but ETH handshake hasn't completed.  In normal operation
            // this takes <1ms.  If the master channel is still at STARTED after 1s, its peer
            // is not responding — most commonly an out-of-mesh device whose corresponding
            // channel is in base-UMD mode.  Skip cleanly instead of waiting the full timeout_ms
            // (10-120s), which wastes 10s+ per device on every init cycle.
            constexpr uint32_t kStartedEarlyExitMs = 1000;
            if (master_router_status[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::STARTED) &&
                elapsed_ms > kStartedEarlyExitMs) {
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Device {} master chan={} stuck at STARTED "
                    "(0x{:08x}) after {}ms — peer is not responding (likely out-of-mesh). "
                    "Skipping ring sync cleanly.  (FIX AO #42429)",
                    dev->id(),
                    master_router_chan,
                    master_router_status[0],
                    elapsed_ms);
                if (has_base_umd_channels_) {
                    timeout_on_base_umd_devices_.insert(dev->id());
                    ring_sync_already_timed_out_ = true;
                }
                return;
            }
            // FIX BG (#42429): host-pre-launch (0xdeadb07e) early-exit.  This sentinel means
            // write_launch_msg was called but ERISC has not yet started executing firmware.  In
            // normal operation the transition is instantaneous (<1ms).  If the master channel
            // is still at 0xdeadb07e after 2s, the PCIe-direct soft reset (FIX RR) succeeded
            // at the PCIe level but the ERISC is not actually running — treat as dead master
            // channel and skip the ring sync instead of burning the full timeout_ms (10s).
            constexpr uint32_t kPreLaunchEarlyExitMs = 2000;
            if (master_router_status[0] == 0xdeadb07eu && elapsed_ms > kPreLaunchEarlyExitMs) {
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Device {} master chan={} stuck at host-pre-launch "
                    "(0xdeadb07e) after {}ms — ERISC not executing after soft-reset (FIX RR). "
                    "Marking device as dead-master-chan. (FIX BG #42429)",
                    dev->id(),
                    master_router_chan,
                    elapsed_ms);
                mmio_dead_master_chan_devices_.insert(dev->id());
                if (has_base_umd_channels_) {
                    timeout_on_base_umd_devices_.insert(dev->id());
                    ring_sync_already_timed_out_ = true;
                }
                return;
            }
            if (elapsed_ms > timeout_ms) {
                // FIX AL (#42429): router sync timed out — mesh fabric is partially broken
                // (likely a dead-relay neighbor holding up the ring handshake; see Job 932).
                // Convert to log_error + return instead of TT_THROW to avoid crashing.
                // Fabric on this device is degraded; tests fail at fabric-op time rather than here.
                log_error(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Timeout after {} ms on Device {} "
                    "(master chan={}, sync addr=0x{:08x}). Expected 0x{:08x}, got 0x{:08x}. "
                    "Skipping — fabric on this device is degraded (FIX AL).",
                    timeout_ms,
                    dev->id(),
                    master_router_chan,
                    router_sync_address,
                    expected_status,
                    master_router_status[0]);
                // FIX TI (#42429): when base-UMD channels are present the ring barrier signal
                // may never propagate (inter-rank quiesce can exceed even the 30s FIX TH2
                // window).  Track this device so verify_all_fabric_channels_healthy() skips
                // its channels instead of failing the 150ms health-check retry loop.
                if (has_base_umd_channels_) {
                    timeout_on_base_umd_devices_.insert(dev->id());
                    // FIX TJ (#42429): signal that the ring barrier has failed — all subsequent
                    // devices in the polling loop will be fast-skipped instead of waiting 30s each.
                    ring_sync_already_timed_out_ = true;
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_router_sync: Device {} recorded in "
                        "timeout_on_base_umd_devices_ — health check will skip its channels; "
                        "ring_sync_already_timed_out_ set to fast-skip remaining devices "
                        "(FIX TI + FIX TJ #42429).",
                        dev->id());
                }
                return;
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
        // FIX AN (#42429): skip MMIO devices whose own master router channel was excluded from
        // configure_fabric (pre-dead — L1 corrupt). No firmware was loaded on that channel, so
        // the channel will never be at READY_FOR_TRAFFIC.
        if (mmio_dead_master_chan_devices_.count(dev->id()) > 0) {
            log_warning(
                tt::LogMetal,
                "verify_all_fabric_channels_healthy: Device {} own master router chan is pre-dead "
                "(no fabric firmware loaded — L1 was corrupt at init). "
                "Marking fabric_channels_not_ready_for_traffic_ so callers can skip AllGather. "
                "Skipping health check. (#42429 FIX AN / FIX QD)",
                dev->id());
            // FIX QD (#42429): Set the not-ready flag so test fixtures (e.g. MeshDevice1x4Fixture)
            // can detect this state and issue GTEST_SKIP() instead of running an AllGather that
            // will hang because no fabric firmware was loaded on the master router channel.
            dev->set_fabric_channels_not_ready_for_traffic();
            continue;
        }
        // FIX TI (#42429): skip devices where wait_for_fabric_router_sync timed out while
        // base-UMD channels were present.  The ring barrier signal never propagated to
        // LOCAL_HANDSHAKE_COMPLETE, so the master ERISC never broadcast READY_FOR_TRAFFIC to
        // subordinate channels.  All channels on this device will be stuck at 0xa1b1c1d1
        // (REMOTE_HANDSHAKE_COMPLETE) indefinitely; the 150ms kMaxRetries window is too short
        // to wait for them.  Mark fabric not-ready so AllGather test fixtures skip cleanly.
        if (timeout_on_base_umd_devices_.count(dev->id()) > 0) {
            log_warning(
                tt::LogMetal,
                "verify_all_fabric_channels_healthy: Device {} ring barrier timed out during "
                "base-UMD channel quiesce (channels stuck at REMOTE_HANDSHAKE_COMPLETE). "
                "Marking fabric_channels_not_ready_for_traffic_ and fabric_ring_sync_timed_out_ "
                "so callers can skip AllGather and FIX BA skips relay_broken_non_mmio. "
                "(#42429 FIX TI + FIX TK)",
                dev->id());
            dev->set_fabric_channels_not_ready_for_traffic();
            // FIX TK (#42429): also set ring_sync_timed_out so RiscFirmwareInitializer::teardown()
            // FIX BA does NOT add this device to relay_broken_non_mmio.  The ring sync timeout does
            // NOT mean the relay is broken — the ETH channels are mid-transition from base-UMD
            // firmware via launch_msg (FIX M).  Triggering FIX AC (PCIe reset of MMIO ETH) in this
            // state causes ALL MMIO ETH heartbeats to time out (5s × 24 cores), leaving the
            // machine with only 4/8 chips visible after the job exits.
            dev->set_fabric_ring_sync_timed_out();
            continue;
        }
        const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
        const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);
        // FIX EXT (#42429): look up this device's external channels once per device.
        const std::unordered_set<uint32_t>* ext_chans_ptr = nullptr;
        {
            auto ext_it = external_umd_channels_map_.find(dev->id());
            if (ext_it != external_umd_channels_map_.end() && !ext_it->second.empty()) {
                ext_chans_ptr = &ext_it->second;
            }
        }
        for (const auto& [eth_chan_id, direction] : active_channels) {
            // FIX EXT (#42429): skip external channels — firmware was not loaded on them,
            // so they will never be at READY_FOR_TRAFFIC.  They are expected to remain at
            // 0x49706550 (base-UMD) indefinitely; treat as non-participant, not a failure.
            if (ext_chans_ptr != nullptr && ext_chans_ptr->count(eth_chan_id) > 0) {
                log_info(
                    tt::LogMetal,
                    "verify_all_fabric_channels_healthy: Device {} chan={} is external ETH "
                    "(no in-cluster peer, firmware not loaded) — skipping health check. "
                    "(FIX EXT #42429)",
                    dev->id(),
                    eth_chan_id);
                continue;
            }
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
            classification = fmt::format("STILL_INITIALIZING (status={})", edm_status_name(status_enum));
            initializing_count++;
        }

        // Strategy 11 (#42429): ETH link status post-failure diagnostic.
        // For MMIO devices (PCIe-direct readable), read the ETH link error status register
        // (Wormhole 0x1440) to diagnose if the PHY link is not trained, which would explain
        // channels stuck at STARTED (0xa0b0c0d0).
        // Wormhole ETH_LINK_ERR_STATUS_ADDR = 0x1440; codes >= 11 = not connected.
        std::string eth_link_diag;
        {
            const ChannelInfo* ch_info = nullptr;
            for (const auto& ch : channels_to_check) {
                if (ch.dev->id() == fc.device_id && ch.eth_chan_id == fc.eth_chan_id) {
                    ch_info = &ch;
                    break;
                }
            }
            if (ch_info != nullptr && ch_info->dev->is_mmio_capable()) {
                constexpr uint32_t kEthLinkErrStatusAddr = 0x1440;
                std::vector<uint32_t> eth_link_buf(1, 0xBEEF);
                try {
                    detail::ReadFromDeviceL1(
                        ch_info->dev, ch_info->eth_logical_core, kEthLinkErrStatusAddr, 4,
                        eth_link_buf, CoreType::ETH);
                    if (eth_link_buf[0] == 0) {
                        eth_link_diag = " ETH_link_status=0(OK)";
                    } else if (eth_link_buf[0] >= 11) {
                        eth_link_diag = fmt::format(
                            " ETH_link_status={}(NOT_CONNECTED — link untrained!)", eth_link_buf[0]);
                    } else {
                        eth_link_diag = fmt::format(" ETH_link_status={}(config_error)", eth_link_buf[0]);
                    }
                } catch (...) {
                    eth_link_diag = " ETH_link_status=<read_failed>";
                }
            }
        }

        log_error(
            tt::LogMetal,
            "verify_all_fabric_channels_healthy: Device {} chan={} actual=0x{:08x} expected=0x{:08x} — {}{}",
            fc.device_id,
            fc.eth_chan_id,
            fc.actual_status,
            expected_status,
            classification,
            eth_link_diag);

        failure_details += fmt::format(
            "  dev={} chan={} status=0x{:08x} ({}){}\n",
            fc.device_id, fc.eth_chan_id, fc.actual_status, classification, eth_link_diag);
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
    // FIX TH3 (#42429): Base-UMD channels transition via launch_msg instead of soft reset.
    // After base-UMD quiesce + new firmware launch + ring handshake, they need more time.
    // T3K has up to 16 base-UMD channels after a tt-smi -r warm-up cycle; each is polled
    // sequentially. 30s (3x) was insufficient — observed stuck at 0xa1b1c1d1 throughout.
    // 120s (12x) gives each channel a full 7.5s window even in worst-case 16-channel scenario.
    const uint32_t base_timeout = timeout.value_or(10000);
    if (has_base_umd_channels_ && !timeout.has_value()) {
        const uint32_t extended = base_timeout * 12;
        log_info(
            tt::LogMetal,
            "FIX TH3 (#42429): base-UMD channels detected — extending fabric_router_sync_timeout "
            "from {} ms to {} ms (12x) to allow relay quiesce + ring handshake (up to 16 channels).",
            base_timeout,
            extended);
        return extended;
    }
    return base_timeout;
}

}  // namespace tt::tt_metal
