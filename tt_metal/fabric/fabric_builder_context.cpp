// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

FabricBuilderContext::FabricBuilderContext(const FabricContext& fabric_context) : fabric_context_(fabric_context) {
    // Create default router config
    router_config_ = create_edm_config();

    // Create router config with mux extension for all directions
    for (size_t direction = 0; direction < eth_chan_directions::COUNT; direction++) {
        router_with_mux_config_[direction] =
            create_edm_config(FabricTensixConfig::MUX, static_cast<eth_chan_directions>(direction));
    }

    // Initialize tensix config later after routing tables are configured
    tensix_config_ = nullptr;

    // Initialize per-device build state
    num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    auto num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
    if (num_devices_ != 4 && num_pcie_devices == 4) {
        // Add dispatch devices for multi-host setups
        num_devices_ += num_pcie_devices;
    }
    master_router_chans_.resize(num_devices_, UNINITIALIZED_MASTER_ROUTER_CHAN);
    num_initialized_routers_.resize(num_devices_, UNINITIALIZED_ROUTERS);
}

std::unique_ptr<FabricEriscDatamoverConfig> FabricBuilderContext::create_edm_config(
    FabricTensixConfig fabric_tensix_config, eth_chan_directions direction) const {
    auto edm_options = FabricEriscDatamoverOptions{
        .fabric_tensix_config = fabric_tensix_config,
        .direction = direction,
    };

    const auto channel_buffer_size_bytes = fabric_context_.get_fabric_channel_buffer_size_bytes();
    const auto topology = fabric_context_.get_fabric_topology();

    return std::make_unique<FabricEriscDatamoverConfig>(channel_buffer_size_bytes, topology, edm_options);
}

FabricEriscDatamoverConfig& FabricBuilderContext::get_fabric_router_config(
    FabricTensixConfig fabric_tensix_config, eth_chan_directions direction) const {
    switch (fabric_tensix_config) {
        case FabricTensixConfig::DISABLED:
        case FabricTensixConfig::UDM:
            TT_FATAL(router_config_ != nullptr, "Error, fabric router config is uninitialized");
            return *router_config_;
        case FabricTensixConfig::MUX:
            TT_FATAL(
                router_with_mux_config_[direction] != nullptr,
                "Error, fabric router config with mux extension is uninitialized for direction {}",
                direction);
            return *router_with_mux_config_[direction].get();
        default: TT_FATAL(false, "Error, invalid fabric_tensix_config: {}", fabric_tensix_config);
    }
}

void FabricBuilderContext::set_num_fabric_initialized_routers(ChipId chip_id, size_t num_routers) {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        num_initialized_routers_[chip_id] == UNINITIALIZED_ROUTERS,
        "Error, tried to set num initialized routers again for device {}",
        chip_id);
    num_initialized_routers_[chip_id] = num_routers;
}

uint32_t FabricBuilderContext::get_num_fabric_initialized_routers(ChipId chip_id) const {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        num_initialized_routers_[chip_id] != UNINITIALIZED_ROUTERS,
        "Error, querying num initialized routers for an unknown device {}",
        chip_id);
    return num_initialized_routers_[chip_id];
}

void FabricBuilderContext::set_fabric_master_router_chan(ChipId chip_id, chan_id_t chan_id) {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        master_router_chans_[chip_id] == UNINITIALIZED_MASTER_ROUTER_CHAN,
        "Error, tried to set master router channel again for the same device {}",
        chip_id);
    master_router_chans_[chip_id] = chan_id;
}

chan_id_t FabricBuilderContext::get_fabric_master_router_chan(ChipId chip_id) const {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        master_router_chans_[chip_id] != UNINITIALIZED_MASTER_ROUTER_CHAN,
        "Error, querying master router channel for an unknown device {}",
        chip_id);
    return master_router_chans_[chip_id];
}

std::vector<size_t> FabricBuilderContext::get_fabric_router_addresses_to_clear() const {
    std::vector<size_t> addresses_to_clear = {
        router_config_->edm_local_sync_address, router_config_->edm_local_tensix_sync_address};

    if (router_config_->sender_txq_id != router_config_->receiver_txq_id) {
        addresses_to_clear.push_back(router_config_->to_sender_channel_remote_ack_counters_base_addr);
        addresses_to_clear.push_back(router_config_->to_sender_channel_remote_completion_counters_base_addr);
        addresses_to_clear.push_back(router_config_->receiver_channel_remote_ack_counters_base_addr);
        addresses_to_clear.push_back(router_config_->receiver_channel_remote_completion_counters_base_addr);
    }

    return addresses_to_clear;
}

std::pair<uint32_t, uint32_t> FabricBuilderContext::get_fabric_router_sync_address_and_status() const {
    return std::make_pair(router_config_->edm_status_address, EDMStatus::LOCAL_HANDSHAKE_COMPLETE);
}

std::optional<std::pair<uint32_t, EDMStatus>> FabricBuilderContext::get_fabric_router_ready_address_and_signal() const {
    return std::make_pair(router_config_->edm_status_address, EDMStatus::READY_FOR_TRAFFIC);
}

std::pair<uint32_t, uint32_t> FabricBuilderContext::get_fabric_router_termination_address_and_signal() const {
    return std::make_pair(router_config_->termination_signal_address, TerminationSignal::IMMEDIATELY_TERMINATE);
}

FabricTensixDatamoverConfig& FabricBuilderContext::get_tensix_config() const {
    TT_FATAL(tensix_config_ != nullptr, "Error, fabric tensix config is uninitialized");
    return *tensix_config_;
}

void FabricBuilderContext::initialize_tensix_config() {
    TT_FATAL(tensix_config_ == nullptr, "Trying to re-initialize fabric tensix config");

    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    if (fabric_tensix_config != FabricTensixConfig::DISABLED) {
        // Now it's safe to call get_active_fabric_eth_channels() because
        // configure_routing_tables_for_fabric_ethernet_channels() has already run
        tensix_config_ = std::make_unique<FabricTensixDatamoverConfig>();
    }
}

}  // namespace tt::tt_fabric
