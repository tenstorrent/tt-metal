// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>
#include <vector>
#include <map>
#include <algorithm>
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

uint32_t FabricContext::get_max_1d_hops_from_topology() const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // For 1D topologies: max hops determined by larger dimension of the mesh shape
    // A 1D chain can be laid out as rows×1 or 1×cols, so we take max(rows, cols)
    // Max hops = dimension_size - 1 (edges between nodes)
    auto mesh_ids = mesh_graph.get_mesh_ids();
    uint32_t max_hops = 0;

    for (const auto& mesh_id : mesh_ids) {
        auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
        uint32_t rows = mesh_shape[0];
        uint32_t cols = mesh_shape[1];

        // Max hops is the larger dimension minus 1
        uint32_t mesh_max_hops = std::max(rows, cols);
        if (mesh_max_hops > 0) {
            mesh_max_hops -= 1;  // Convert size to hops (0-indexed)
        }
        max_hops = std::max(max_hops, mesh_max_hops);
    }

    TT_FATAL(max_hops > 0, "No chips found in mesh topology - cannot determine 1D hop count");

    return max_hops;
}

uint32_t FabricContext::get_max_2d_hops_from_topology() const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // For 2D mesh: max hops determined by XY-routing formula
    // Max hops = (rows - 1) + (cols - 1) for worst-case corner-to-corner routing
    auto mesh_ids = mesh_graph.get_mesh_ids();
    uint32_t max_hops = 0;

    for (const auto& mesh_id : mesh_ids) {
        auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
        uint32_t rows = mesh_shape[0];
        uint32_t cols = mesh_shape[1];

        // XY-routing: worst case is diagonal corner-to-corner
        uint32_t mesh_max_hops = (rows > 0 ? rows - 1 : 0) + (cols > 0 ? cols - 1 : 0);
        max_hops = std::max(max_hops, mesh_max_hops);
    }

    TT_FATAL(max_hops > 0, "No chips found in mesh topology - cannot determine 2D hop count");

    return max_hops;
}

uint32_t FabricContext::compute_2d_pkt_hdr_route_buffer_size(uint32_t max_hops) const {
    // Map hop count to discrete route buffer sizes (8, 16, 24, 32 bytes)
    // These sizes are chosen to:
    // 1. Align with 16-byte packet header boundaries (compiler adds tail padding)
    // 2. Provide good coverage for common mesh sizes without excessive granularity
    // 3. Balance memory efficiency vs simplicity (4 discrete sizes instead of per-hop sizing)
    if (max_hops <= 8) {
        return 8;
    } else if (max_hops <= 16) {
        return 16;
    } else if (max_hops <= 24) {
        return 24;
    } else {
        return Limits::MAX_2D_ROUTE_BUFFER_SIZE;  // Maximum route buffer size
    }
}

void FabricContext::compute_packet_specifications() {
    // Query topology to determine optimal header sizes
    if (this->is_2D_routing_enabled()) {
        // 2D mode: only set 2D-related values
        max_2d_hops_ = get_max_2d_hops_from_topology();
        routing_2d_buffer_size_ = compute_2d_pkt_hdr_route_buffer_size(max_2d_hops_);
    } else {
        // 1D mode: only set 1D-related values
        max_1d_hops_ = get_max_1d_hops_from_topology();

        // Current memory map limits: ROUTING_PATH_SIZE_1D = 256 bytes supports max 32 hops
        // (32 chips × 8 bytes per routing entry = 256 bytes)
        // Support for >32 hops requires L1 memory map updates and larger routing tables
        TT_FATAL(
            max_1d_hops_ <= Limits::MAX_1D_HOPS,
            "1D routing with >32 hops (max_hops={}) requires L1 memory map updates. "
            "Current allocation (ROUTING_PATH_SIZE_1D = 256 bytes) supports max {} hops.",
            max_1d_hops_,
            Limits::MAX_1D_HOPS);
    }

    // Compute actual packet sizes based on topology
    packet_header_size_bytes_ = compute_packet_header_size_bytes();
    max_payload_size_bytes_ = compute_max_payload_size_bytes();
    channel_buffer_size_bytes_ = packet_header_size_bytes_ + max_payload_size_bytes_;
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

size_t FabricContext::get_1d_header_size(uint32_t extension_words) const {
    // Use explicit template instantiation for compile-time type safety
    switch (extension_words) {
        case 0: return sizeof(tt::tt_fabric::LowLatencyPacketHeaderT<0>);
        case 1: return sizeof(tt::tt_fabric::LowLatencyPacketHeaderT<1>);
        default: TT_THROW("Unsupported extension words: {}", extension_words);
    }
}

size_t FabricContext::get_2d_header_size(uint32_t route_buffer_size) const {
    // Use explicit template instantiation for compile-time type safety
    switch (route_buffer_size) {
        case 8: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<8>);
        case 16: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<16>);
        case 24: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<24>);
        case 32: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<32>);
        default: TT_THROW("Unsupported 2D route buffer size: {}", route_buffer_size);
    }
}

size_t FabricContext::get_udm_header_size(uint32_t route_buffer_size) const {
    // UDM header = base 2D header + UDM control fields
    return get_2d_header_size(route_buffer_size) + sizeof(tt::tt_fabric::UDMControlFields);
}

size_t FabricContext::compute_packet_header_size_bytes() const {
    bool udm_enabled =
        tt::tt_metal::MetalContext::instance().get_fabric_udm_mode() == tt::tt_fabric::FabricUDMMode::ENABLED;

    if (udm_enabled) {
        TT_FATAL(this->is_2D_routing_enabled(), "UDM mode only supports 2D routing");
        return get_udm_header_size(routing_2d_buffer_size_);
    } else if (this->is_2D_routing_enabled()) {
        return get_2d_header_size(routing_2d_buffer_size_);
    } else {
        return get_1d_header_size(get_1d_pkt_hdr_extension_words());
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
    this->topology_ = this->get_topology_from_config(fabric_config);

    this->is_2D_routing_enabled_ = is_2D_topology(this->topology_);
    this->bubble_flow_control_enabled_ = is_ring_or_torus(this->topology_);

    // Compute packet specifications based on topology
    this->compute_packet_specifications();

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

std::map<std::string, std::string> FabricContext::get_fabric_kernel_defines() const {
    std::map<std::string, std::string> defines;

    // Add routing mode define
    defines["ROUTING_MODE"] = std::to_string(static_cast<uint32_t>(this->topology_));

    // Add UDM mode define
    bool udm_enabled =
        tt::tt_metal::MetalContext::instance().get_fabric_udm_mode() == tt::tt_fabric::FabricUDMMode::ENABLED;
    defines["UDM_MODE"] = udm_enabled ? "1" : "0";

    // Add dynamic packet header sizing defines based on topology
    if (this->is_2D_routing_enabled()) {
        // 2D routing: inject route buffer size
        defines["FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE"] = std::to_string(routing_2d_buffer_size_);
    } else {
        // 1D routing: inject extension words
        uint32_t extension_words = get_1d_pkt_hdr_extension_words();
        defines["FABRIC_1D_PKT_HDR_EXTENSION_WORDS"] = std::to_string(extension_words);
    }

    return defines;
}

}  // namespace tt::tt_fabric
