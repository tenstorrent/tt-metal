// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>
#include <vector>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/host_api.hpp>
#include <enchantum/enchantum.hpp>
#include "erisc_datamover_builder.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

std::unordered_map<MeshId, bool> FabricContext::check_for_wrap_around_mesh() const {
    std::unordered_map<MeshId, bool> wrap_around_mesh;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    for (const auto& mesh_id : mesh_ids) {
        // We can wrap around mesh if the corner chip (logical chip 0) has exactly 2 connections
        const uint32_t corner_chip_id = 0;
        uint32_t corner_chip_connections = 0;
        for (const auto& direction : FabricContext::routing_directions) {
            if (!control_plane.get_intra_chip_neighbors(FabricNodeId(mesh_id, corner_chip_id), direction).empty()) {
                corner_chip_connections++;
            }
        }

        wrap_around_mesh[mesh_id] = (corner_chip_connections == 2);
    }
    return wrap_around_mesh;
}

tt::tt_fabric::Topology FabricContext::get_topology_from_config(tt::tt_fabric::FabricConfig fabric_config) {
    switch (fabric_config) {
        case tt::tt_fabric::FabricConfig::FABRIC_1D: return tt::tt_fabric::Topology::Linear;
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: return tt::tt_fabric::Topology::Ring;
        case tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE: return tt::tt_fabric::Topology::NeighborExchange;
        case tt::tt_fabric::FabricConfig::FABRIC_2D: return tt::tt_fabric::Topology::Mesh;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X:
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y:
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY: return tt::tt_fabric::Topology::Torus;
        case tt::tt_fabric::FabricConfig::DISABLED:
        case tt::tt_fabric::FabricConfig::CUSTOM:
            TT_THROW("Unsupported fabric config: {}", enchantum::to_string(fabric_config));
    }
    return tt::tt_fabric::Topology::Linear;
}

size_t FabricContext::compute_packet_header_size_bytes() const {
    bool udm_enabled =
        tt::tt_metal::MetalContext::instance().get_fabric_udm_mode() == tt::tt_fabric::FabricUDMMode::ENABLED;
    if (udm_enabled) {
        // UDM mode only supports 2D routing
        TT_FATAL(this->is_2D_routing_enabled(), "UDM mode only supports 2D routing");
        return sizeof(tt::tt_fabric::UDMHybridMeshPacketHeader);
    } else {
        if (this->is_2D_routing_enabled()) {
            return sizeof(tt::tt_fabric::HybridMeshPacketHeader);
        } else {
            return sizeof(tt::tt_fabric::PacketHeader);
        }
    }
}

size_t FabricContext::compute_max_payload_size_bytes() const {
    if (this->is_2D_routing_enabled()) {
        return tt::tt_fabric::FabricEriscDatamoverBuilder::default_mesh_packet_payload_size_bytes;
    } else {
        return tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;
    }
}

FabricContext::FabricContext(tt::tt_fabric::FabricConfig fabric_config) {
    TT_FATAL(
        fabric_config != tt::tt_fabric::FabricConfig::DISABLED,
        "Trying to initialize fabric context for disabled fabric config");
    this->fabric_config_ = fabric_config;

    this->wrap_around_mesh_ = this->check_for_wrap_around_mesh();
    this->topology_ = FabricContext::get_topology_from_config(fabric_config);

    this->is_2D_routing_enabled_ = is_2D_topology(this->topology_);
    this->bubble_flow_control_enabled_ = is_ring_or_torus(this->topology_);

    this->packet_header_size_bytes_ = this->compute_packet_header_size_bytes();
    this->max_payload_size_bytes_ = this->compute_max_payload_size_bytes();
    this->channel_buffer_size_bytes_ = this->packet_header_size_bytes_ + this->max_payload_size_bytes_;

    // Query tensix config from MetalContext at init time
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    this->tensix_enabled_ = (fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED);

    // Compute intermesh VC configuration (requires ControlPlane to be initialized)
    // this->intermesh_vc_config_ = this->compute_intermesh_vc_config();

    // Builder context will be lazy-initialized on first access
    builder_context_ = nullptr;

    set_routing_mode(this->topology_);
}

// Destructor needed because of unique_ptr with forward-declared FabricBuilderContext
FabricContext::~FabricContext() = default;

// Used to check whether a physical mesh has external torus connections, to enable Ring topology.
// Note: is_wrap_around_mesh is true if the mesh does NOT have external torus connections.
// Returning true tells the fabric code that it must fold the internal connections on the corner chips in order to form
// a "torus mesh"
bool FabricContext::is_wrap_around_mesh(MeshId mesh_id) const {
    auto it = this->wrap_around_mesh_.find(mesh_id);
    TT_FATAL(it != this->wrap_around_mesh_.end(), "Querying wrap around mesh for an unknown mesh id");
    return it->second;
}

bool FabricContext::is_switch_mesh(MeshId mesh_id) const {
    // Stub: returns false for now (all meshes are compute meshes)
    // TODO: Implement when switch mesh support lands - delegate to ControlPlane
    (void)mesh_id;  // Unused for now
    return false;
}

bool FabricContext::has_z_router_on_device(ChipId device_id) const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& inter_mesh_connectivity = mesh_graph.get_inter_mesh_connectivity();

    // Iterate through all meshes to find which one contains this device
    const auto& mesh_ids = mesh_graph.get_mesh_ids();
    for (const auto& mesh_id : mesh_ids) {
        const auto& mesh_connections = inter_mesh_connectivity[*mesh_id];

        // Check if this device ID is within this mesh's connectivity map
        if (device_id < mesh_connections.size()) {
            const auto& chip_connections = mesh_connections[device_id];

            // Check if any connection from this chip uses Z direction
            for (const auto& [dst_mesh_id, router_edge] : chip_connections) {
                if (router_edge.port_direction == RoutingDirection::Z) {
                    return true;
                }
            }
        }
    }

    return false;
}

// ============ Builder Context Access ============

FabricBuilderContext& FabricContext::get_builder_context() {
    if (!builder_context_) {
        builder_context_ = std::make_unique<FabricBuilderContext>(*this);
    }
    return *builder_context_;
}

const FabricBuilderContext& FabricContext::get_builder_context() const {
    if (!builder_context_) {
        builder_context_ = std::make_unique<FabricBuilderContext>(*this);
    }
    return *builder_context_;
}

bool FabricContext::need_deadlock_avoidance_support(eth_chan_directions direction) const {
    if (topology_ == Topology::Ring) {
        return true;
    } else if (topology_ == Topology::Torus) {
        const auto fabric_type = get_fabric_type(fabric_config_);
        // if we are not torused along a dimension, we dont need deadlock avoidance for that direction
        const bool is_north_south =
            (direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH);
        const bool is_east_west = (direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST);

        const bool torus_mismatch = (fabric_type == FabricType::TORUS_X && is_north_south) ||
                                    (fabric_type == FabricType::TORUS_Y && is_east_west);

        return !torus_mismatch;
    }

    return false;
}


}  // namespace tt::tt_fabric
