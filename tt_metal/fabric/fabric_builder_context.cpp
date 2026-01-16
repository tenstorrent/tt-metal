// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

void FabricBuilderContext::compute_max_channel_counts() {
    // Create channel mappings for all router types that exist in this fabric
    const auto topology = fabric_context_.get_fabric_topology();

    std::vector<FabricRouterChannelMapping> possible_mappings;

    // Always have MESH routers
    possible_mappings.emplace_back(
        topology,
        false,  // no tensix
        RouterVariant::MESH,
        intermesh_vc_config_.requires_vc1 ? &intermesh_vc_config_ : nullptr,
        false);

    // If Z routers exist in this fabric, add Z_ROUTER mapping
    if (intermesh_vc_config_.router_type == IntermeshRouterType::Z_INTERMESH) {
        possible_mappings.emplace_back(
            topology,
            false,  // no tensix
            RouterVariant::Z_ROUTER,
            &intermesh_vc_config_,
            true);
    }

    // Compute max channel counts across all router types in this fabric
    max_sender_channels_per_vc_.fill(0);
    max_receiver_channels_per_vc_.fill(0);

    for (const auto& mapping : possible_mappings) {
        uint32_t num_vcs = mapping.get_num_virtual_channels();
        for (uint32_t vc = 0; vc < num_vcs; ++vc) {
            max_sender_channels_per_vc_[vc] = std::max(
                max_sender_channels_per_vc_[vc],
                static_cast<std::size_t>(mapping.get_num_sender_channels_for_vc(vc)));
            max_receiver_channels_per_vc_[vc] = std::max(
                max_receiver_channels_per_vc_[vc],
                static_cast<std::size_t>(1u));  // Always 1 receiver per VC
        }
    }
}

FabricBuilderContext::FabricBuilderContext(const FabricContext& fabric_context) : fabric_context_(fabric_context) {
    this->intermesh_vc_config_ = this->compute_intermesh_vc_config();

    // Compute max channel counts for this fabric instance
    compute_max_channel_counts();

    // Create configs using computed max
    router_config_ = create_edm_config();
    for (size_t direction = 0; direction < eth_chan_directions::COUNT; direction++) {
        router_with_mux_config_[direction] =
            create_edm_config(FabricTensixConfig::MUX, static_cast<eth_chan_directions>(direction));
    }

    tensix_config_ = nullptr;

    // Initialize per-device build state
    num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    auto num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
    if (num_devices_ != 4 && num_pcie_devices == 4) {
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

    return std::make_unique<FabricEriscDatamoverConfig>(
        fabric_context_.get_fabric_channel_buffer_size_bytes(),
        fabric_context_.get_fabric_topology(),
        edm_options,
        max_sender_channels_per_vc_,      // Max for this fabric instance
        max_receiver_channels_per_vc_);   // Max for this fabric instance
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

IntermeshVCConfig FabricBuilderContext::compute_intermesh_vc_config() const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Check if multiple meshes exist
    const auto& mesh_ids = mesh_graph.get_mesh_ids();
    constexpr size_t single_mesh_count = 1;
    if (mesh_ids.size() <= single_mesh_count) {
        return IntermeshVCConfig::disabled();
    }

    // Check if intermesh connections exist (use inter_mesh_connectivity which has actual parsed connections)
    const auto& inter_mesh_connectivity = mesh_graph.get_inter_mesh_connectivity();

    // Count total intermesh connections across all meshes
    size_t total_intermesh_connections = 0;
    for (const auto& mesh_connections : inter_mesh_connectivity) {
        for (const auto& chip_connections : mesh_connections) {
            total_intermesh_connections += chip_connections.size();
        }
    }

    if (total_intermesh_connections == 0) {
        return IntermeshVCConfig::disabled();
    }

    // Detect Z vs XY intermesh by checking for Z-direction connections in inter-mesh connectivity
    bool has_z_routers = false;
    for (const auto& mesh_connections : inter_mesh_connectivity) {
        for (const auto& chip_connections : mesh_connections) {
            for (const auto& [dst_mesh_id, router_edge] : chip_connections) {
                if (router_edge.port_direction == RoutingDirection::Z) {
                    has_z_routers = true;
                    break;
                }
            }
            if (has_z_routers) {
                break;
            }
        }
        if (has_z_routers) {
            break;
        }
    }

    // Default to FULL_MESH when intermesh exists
    // TODO: Implement detection logic for:
    //   - EDGE_ONLY: Check if workload only needs edge nodes (optimization)
    //   - FULL_MESH_WITH_PASS_THROUGH: Check if any mesh forwards traffic between other meshes
    constexpr bool needs_mesh_pass_through = false;

    auto config =
        needs_mesh_pass_through ? IntermeshVCConfig::full_mesh_with_pass_through() : IntermeshVCConfig::full_mesh();

    // Set router type based on detection
    config.router_type = has_z_routers ? IntermeshRouterType::Z_INTERMESH : IntermeshRouterType::XY_INTERMESH;

    return config;
}


}  // namespace tt::tt_fabric
