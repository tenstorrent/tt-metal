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


namespace tt::tt_metal {

namespace {

// Returns true iff `status` is one of the well-known EDMStatus sentinel values written by
// a live fabric ERISC router at some point in its lifecycle. Any other nonzero value at
// the router_sync_address indicates the L1 slot is corrupt or has been overwritten by
// unrelated NOC traffic — the ERISC is NOT running recognizable firmware and the
// TERMINATE handshake will not complete.
//
// See tt_metal/fabric/fabric_edm_packet_header.hpp for the authoritative enum list.
bool is_known_edm_status(uint32_t status) {
    switch (status) {
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::STARTED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TXQ_INITIALIZED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::WORKER_INTERFACES_INITIALIZED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::VCS_OPENED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED):
        case static_cast<uint32_t>(tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE): return true;
        default: return false;
    }
}

// NOTE: If a new EDMStatus enumerator is added, update is_known_edm_status() above.

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

}  // namespace

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
    TT_FATAL(
        !init_done.contains(InitializerKey::Dispatch),
        "FabricFirmwareInitializer must be torn down after DispatchKernelInitializer");
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

            cluster_.l1_barrier(dev->id());

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
                    detail::ReadFromDeviceL1(dev, mux_core, status_addr, 4, status_buf, CoreType::WORKER);
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
                        std::vector<uint32_t> status_buf(1, 0);
                        detail::ReadFromDeviceL1(
                            ch.dev, ch.eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
                        return status_buf[0] == terminated_val;
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

        // Force-reset any channels that did not terminate within the global deadline.
        // GAP 5: Record force-reset channels so the next verify_all_fabric_channels_healthy()
        // can distinguish "was force-reset" from "corrupt from prior crash".
        for (const auto& ch : pending) {
            std::vector<uint32_t> status_buf(1, 0);
            detail::ReadFromDeviceL1(
                ch.dev, ch.eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
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
            try {
                const auto virtual_eth_coord = cluster_.get_virtual_coordinate_from_logical_coordinates(
                    ch.dev->id(), ch.eth_logical_core, CoreType::ETH);
                cluster_.assert_risc_reset_at_core(
                    tt_cxy_pair(ch.dev->id(), virtual_eth_coord), tt::umd::RiscType::ALL);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "FabricFirmwareInitializer::teardown: assert_risc_reset_at_core failed on "
                    "Device {} chan={}: {}",
                    ch.dev->id(),
                    ch.eth_chan_id,
                    e.what());
            }
        }
    }

    devices_.clear();
    initialized_.clear();
    // Don't clear force_reset_channels_ here — preserve it for the next session's
    // verify_all_fabric_channels_healthy() to use for degraded-channel diagnostics.
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
void FabricFirmwareInitializer::terminate_stale_erisc_routers(
    Device* dev, const tt_fabric::FabricBuilderContext& builder_context) const {
    if (builder_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
        return;
    }

    const auto router_sync_address = builder_context.get_fabric_router_sync_address_and_status().first;
    const auto [term_addr, term_signal] = builder_context.get_fabric_router_termination_address_and_signal();
    constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
    constexpr uint32_t stale_timeout_ms = 50;  // EDM firmware responds in <5ms; base firmware never responds
    constexpr uint32_t kSpinsBetweenSleeps = 64;

    const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
    const auto& active_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);

    uint32_t corrupt_count = 0;
    uint32_t stale_running_count = 0;
    uint32_t stale_timeout_count = 0;

    for (const auto& [eth_chan_id, direction] : active_channels) {
        const auto eth_logical_core =
            cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);

        std::vector<uint32_t> status_buf(1, 0);
        detail::ReadFromDeviceL1(dev, eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);

        if (status_buf[0] == 0 || status_buf[0] == terminated_val) {
            continue;  // clean — nothing to do
        }

        // Fix F4 (#42429): if the status word is not a valid EDMStatus value, the L1 slot is
        // corrupt — either from a prior process's mid-handshake crash (e.g. close_finish
        // spinning forever on a lost EDM ACK, then BRISC-halted by Device::close()), or from
        // unrelated stray NOC traffic. In either case, the TERMINATE probe will time out
        // because no recognizable firmware is running at the peer.  Skip the 50ms wait per
        // channel (saves ~42s on a T3K with ~840 corrupt channels) but still send the
        // TERMINATE write as a best-effort — if there IS firmware behind the garbage word,
        // it gets a chance to notice.  configure_fabric_cores() will clear the L1 and load
        // the new router firmware; whether that recovers cleanly is observed downstream at
        // wait_for_fabric_endpoint_ready.
        const bool known_status = is_known_edm_status(status_buf[0]);
        if (!known_status) {
            log_error(
                tt::LogMetal,
                "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} is NOT a "
                "valid EDMStatus value — ERISC L1 appears CORRUPT (likely stuck-mid-handshake "
                "from a prior process; see #42429). Sending TERMINATE best-effort, NOT polling "
                "(would time out). configure_fabric_cores() will reset L1 next.",
                dev->id(),
                eth_chan_id,
                status_buf[0]);

            std::vector<uint32_t> term_buf(1, static_cast<uint32_t>(term_signal));
            detail::WriteToDeviceL1(dev, eth_logical_core, term_addr, term_buf, CoreType::ETH);
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
        detail::WriteToDeviceL1(dev, eth_logical_core, term_addr, term_buf, CoreType::ETH);

        // Poll for TERMINATED
        const auto stale_start = std::chrono::steady_clock::now();
        uint32_t spin_counter = 0;
        bool terminated_ok = false;
        while (true) {
            detail::ReadFromDeviceL1(dev, eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
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
            "stale_term_timeout={} (of stale_running)",
            dev->id(),
            corrupt_count,
            stale_running_count,
            stale_timeout_count);
    }
}

void FabricFirmwareInitializer::compile_and_configure_fabric() {
    std::vector<std::shared_future<Device*>> events;
    events.reserve(devices_.size());
    for (auto* dev : devices_) {
        events.emplace_back(detail::async([dev]() {
            if (dev->compile_fabric()) {
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

    size_t configured_count = 0;
    for (const auto& event : events) {
        auto* dev = event.get();
        if (dev) {
            // Fix A: probe for stale ERISC firmware on all active channels BEFORE
            // configure_fabric_cores() clears L1 and loads the new firmware image.
            // This gives old firmware a chance to terminate cleanly rather than being
            // interrupted mid-execution by an L1 overwrite.
            terminate_stale_erisc_routers(dev, builder_context);

            dev->configure_fabric();
            configured_count++;
        }
    }
    log_info(tt::LogMetal, "Fabric initialized on {} devices", configured_count);
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

        const auto master_router_chan = builder_context.get_fabric_master_router_chan(dev->id());
        const auto master_router_logical_core =
            cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(master_router_chan, CoordSystem::LOGICAL);

        const auto [router_sync_address, expected_status] = builder_context.get_fabric_router_sync_address_and_status();
        std::vector<std::uint32_t> master_router_status{0};

        auto start_time = std::chrono::steady_clock::now();
        while (master_router_status[0] != expected_status) {
            detail::ReadFromDeviceL1(
                dev, master_router_logical_core, router_sync_address, 4, master_router_status, CoreType::ETH);
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
    constexpr uint32_t kRetryDelayMs = 10;

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
            detail::ReadFromDeviceL1(
                ch.dev, ch.eth_logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
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
