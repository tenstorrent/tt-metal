// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "fabric_firmware_initializer.hpp"

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

    devices_.clear();
    initialized_.clear();
    init_done.erase(key);
}

void FabricFirmwareInitializer::post_teardown() {
    // Reset fabric config
    descriptor_->metal_context().set_fabric_config(tt::tt_fabric::FabricConfig::DISABLED);
}

bool FabricFirmwareInitializer::is_initialized() const { return initialized_.test(); }

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

    size_t configured_count = 0;
    for (const auto& event : events) {
        auto* dev = event.get();
        if (dev) {
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

        // Q1: Stale-firmware probe.
        // configure_fabric_cores() clears edm_status_address to 0 before loading the new
        // firmware image.  If we read a non-zero, non-TERMINATED value here the ETH core
        // was NOT cleanly reset between sessions (previous firmware is still running).
        // Send TERMINATE and wait up to 2 s; fall back to ERISC hard-reset if it won't stop.
        {
            constexpr uint32_t terminated_val =
                static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
            detail::ReadFromDeviceL1(
                dev, master_router_logical_core, router_sync_address, 4, master_router_status, CoreType::ETH);
            if (master_router_status[0] != 0 && master_router_status[0] != terminated_val) {
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_router_sync: Device {} ETH master chan={} edm_status=0x{:08x} "
                    "before new firmware handshake (expected 0 or TERMINATED=0x{:08x}) — "
                    "stale firmware detected; sending TERMINATE",
                    dev->id(),
                    master_router_chan,
                    master_router_status[0],
                    terminated_val);
                auto [term_addr, term_signal] =
                    builder_context.get_fabric_router_termination_address_and_signal();
                std::vector<uint32_t> term_buf(1, static_cast<uint32_t>(term_signal));
                detail::WriteToDeviceL1(
                    dev, master_router_logical_core, term_addr, term_buf, CoreType::ETH);
                constexpr uint32_t stale_timeout_ms = 2000;
                auto stale_start = std::chrono::steady_clock::now();
                while (true) {
                    detail::ReadFromDeviceL1(
                        dev, master_router_logical_core, router_sync_address, 4, master_router_status, CoreType::ETH);
                    if (master_router_status[0] == terminated_val) {
                        break;
                    }
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now() - stale_start)
                                       .count();
                    if (elapsed > stale_timeout_ms) {
                        log_warning(
                            tt::LogMetal,
                            "wait_for_fabric_router_sync: Stale ETH firmware on Device {} did not "
                            "terminate within {}ms — asserting ERISC reset",
                            dev->id(),
                            stale_timeout_ms);
                        const auto virtual_eth_coord =
                            cluster_.get_virtual_coordinate_from_logical_coordinates(
                                dev->id(), master_router_logical_core, CoreType::ETH);
                        cluster_.assert_risc_reset_at_core(
                            tt_cxy_pair(dev->id(), virtual_eth_coord), tt::umd::RiscType::ALL);
                        break;
                    }
                }
                master_router_status[0] = 0;  // reset for the main handshake poll below
            }
        }

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

uint32_t FabricFirmwareInitializer::get_fabric_router_sync_timeout_ms() const {
    if (rtoptions_.get_simulator_enabled()) {
        return 15000;
    }
    auto timeout = rtoptions_.get_fabric_router_sync_timeout_ms();
    return timeout.value_or(10000);
}

}  // namespace tt::tt_metal
