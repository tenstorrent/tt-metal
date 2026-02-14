// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_firmware_initializer.hpp"

#include <chrono>

#include <tt_stl/assert.hpp>
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
    initialized_ = true;
}

void FabricFirmwareInitializer::teardown() {
    if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::TERMINATE_FABRIC)) {
        devices_.clear();
        initialized_ = false;
        return;
    }

    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        devices_.clear();
        initialized_ = false;
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
    initialized_ = false;
}

void FabricFirmwareInitializer::post_teardown() {
    // Reset fabric config
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
}

bool FabricFirmwareInitializer::is_initialized() const { return initialized_; }

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

    for (const auto& event : events) {
        auto* dev = event.get();
        if (dev) {
            dev->configure_fabric();
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

uint32_t FabricFirmwareInitializer::get_fabric_router_sync_timeout_ms() const {
    if (rtoptions_.get_simulator_enabled()) {
        return 15000;
    }
    auto timeout = rtoptions_.get_fabric_router_sync_timeout_ms();
    return timeout.value_or(10000);
}

}  // namespace tt::tt_metal
