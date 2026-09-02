// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include <enchantum/enchantum.hpp>
#include <tt_stl/reflection.hpp>
#include "erisc_datamover_builder.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "fabric/hw/inc/fabric_routing_mode.h"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

std::ostream& operator<<(std::ostream& os, const tt::tt_fabric::Topology& topology) {
    ttsl::reflection::operator<<(os, topology);
    return os;
}

std::unordered_map<MeshId, bool> FabricContext::check_for_wrap_around_mesh(const ControlPlane& control_plane) const {
    std::unordered_map<MeshId, bool> wrap_around_mesh;

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

uint32_t FabricContext::get_max_1d_hops_from_topology(const ControlPlane& control_plane) const {
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Extract mesh shapes from topology
    std::vector<MeshShape> mesh_shapes;
    auto mesh_ids = mesh_graph.get_mesh_ids();
    mesh_shapes.reserve(mesh_ids.size());
    for (const auto& mesh_id : mesh_ids) {
        mesh_shapes.push_back(mesh_graph.get_mesh_shape(mesh_id));
    }

    // Use helper function for hop calculation
    return compute_max_1d_hops(mesh_shapes);
}

uint32_t FabricContext::get_max_2d_hops_from_topology(const ControlPlane& control_plane) const {
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Extract mesh shapes from topology
    std::vector<MeshShape> mesh_shapes;
    auto mesh_ids = mesh_graph.get_mesh_ids();
    mesh_shapes.reserve(mesh_ids.size());
    for (const auto& mesh_id : mesh_ids) {
        mesh_shapes.push_back(mesh_graph.get_mesh_shape(mesh_id));
    }

    // Use helper function for hop calculation
    return compute_max_2d_hops(mesh_shapes);
}

// Every 2D packet carries action maps rather than a hop program, so its route buffer holds rows + cols
// bytes -- two more than the (rows-1) + (cols-1) Manhattan hop count.
uint32_t FabricContext::get_max_2d_route_buffer_bytes_from_topology(const ControlPlane& control_plane) const {
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Every 2D mesh: the action maps occupy Y + X bytes, two more than the (Y-1) + (X-1) hop count
    // the tiers were sized from, and after the flip every 2D mesh carries them.
    //
    // compute_packet_specifications() takes the max of this and the hop count, so this can only grow
    // the buffer. On [32,4] the extra two bytes cross a tier boundary (35 -> 51, i.e. a 96 B header
    // becomes 112 B); Phase 4.4 gives those bytes back by retiring is_mcast_active, which dropped the
    // header base 61 -> 60 and moved the tiers to 36/52/67, so [32,4] lands on 36 and stays at 96 B.
    // Every other in-tree shape stays on its current tier. (routing_fields was NOT retired -- the
    // profiler still decodes it.)
    uint32_t max_bytes = 0;
    for (const auto& mesh_id : mesh_graph.get_mesh_ids()) {
        // Must match the shape get_2d_kernel_defines emits as FABRIC_2D_MESH_*_SIZE, GLOBAL
        // scope included, since that is what the kernel widens against.
        const auto shape = control_plane.get_physical_mesh_shape(mesh_id, MeshScope::GLOBAL);
        max_bytes = std::max(max_bytes, static_cast<uint32_t>(shape[0]) + static_cast<uint32_t>(shape[1]));
    }
    return max_bytes;
}

uint32_t FabricContext::compute_1d_pkt_hdr_extension_words(uint32_t max_hops) const {
    // Precondition: max_hops validated by compute_packet_specifications()

    // Base routing word supports 16 hops; extension words add 16 hops each
    // ExtensionWords=0: 1-16 hops (48B header)
    // ExtensionWords=1: 17-32 hops (64B header)

    return (max_hops - 1) / ROUTING_1D_HOPS_PER_WORD;
}

// required_route_bytes is the hop count for the legacy hop program, or rows + cols for 2D action maps.
// Precondition: bounds validated by compute_packet_specifications().
uint32_t FabricContext::compute_2d_pkt_hdr_route_buffer_size(uint32_t required_route_bytes) const {
    // Route buffer tiers aligned to packet header size boundaries
    for (const auto& tier : ROUTING_2D_BUFFER_TIERS) {
        if (required_route_bytes <= tier.max_hops) {
            return tier.buffer_size;
        }
    }

    return Limits::MAX_2D_ROUTE_BUFFER_SIZE;
}

void FabricContext::compute_packet_specifications(
    const ControlPlane& control_plane, const tt_metal::Hal& hal, tt::ARCH arch) {
    // Query topology to determine optimal header sizes
    if (is_2D_routing_enabled_) {
        // 2D mode: query topology and validate against limits
        max_2d_hops_ = get_max_2d_hops_from_topology(control_plane);

        if (max_2d_hops_ == 0) {
            log_warning(
                tt::LogFabric,
                "Max 2D routing hops were determined as 0, check mesh topology before running fabric workloads, "
                "defaulting to: {}",
                Limits::MAX_2D_HOPS);
            // NOTE: Default to a non-zero value as we might be running tests in a simulated/mock environment
            max_2d_hops_ = Limits::MAX_2D_HOPS;
        }

        // Validate 2D topology against route buffer limits
        // Each byte in route buffer encodes 1 hop, so max_hops cannot exceed buffer size
        TT_FATAL(
            max_2d_hops_ <= Limits::MAX_2D_HOPS,
            "2D routing with {} hops exceeds maximum {} hops supported by {}B route buffer.",
            max_2d_hops_,
            Limits::MAX_2D_HOPS,
            Limits::MAX_2D_ROUTE_BUFFER_SIZE);

        // Size for whichever encoding demands more, so neither can overrun the buffer.
        const uint32_t required_2d_route_bytes =
            std::max(max_2d_hops_, get_max_2d_route_buffer_bytes_from_topology(control_plane));

        TT_FATAL(
            required_2d_route_bytes <= Limits::MAX_2D_ROUTE_BUFFER_SIZE,
            "2D routing requires a {}B route buffer, exceeding the maximum of {}B.",
            required_2d_route_bytes,
            Limits::MAX_2D_ROUTE_BUFFER_SIZE);

        routing_2d_buffer_size_ = compute_2d_pkt_hdr_route_buffer_size(required_2d_route_bytes);
    } else {
        // 1D mode: query topology and validate against limits
        max_1d_hops_ = get_max_1d_hops_from_topology(control_plane);

        if (max_1d_hops_ == 0) {
            log_warning(
                tt::LogFabric,
                "Max 1D routing hops were determined as 0, check mesh topology before running fabric workloads, "
                "defaulting to: {}",
                Limits::MAX_1D_HOPS);
            // NOTE: Default to a non-zero value as we might be running tests in a simulated/mock environment
            max_1d_hops_ = Limits::MAX_1D_HOPS;
        }

        // Validate 1D topology against memory map limits
        // ROUTING_PATH_SIZE_1D = 1024 bytes / 16 bytes per entry = 64 chips max (63 hops)
        TT_FATAL(
            max_1d_hops_ <= Limits::MAX_1D_HOPS,
            "1D routing with {} hops exceeds maximum {} hops (ROUTING_PATH_SIZE_1D = 1024 bytes limit).",
            max_1d_hops_,
            Limits::MAX_1D_HOPS);

        routing_1d_extension_words_ = compute_1d_pkt_hdr_extension_words(max_1d_hops_);
    }

    // Compute actual packet sizes based on topology
    packet_header_size_bytes_ = compute_packet_header_size_bytes(control_plane);
    max_payload_size_bytes_ = compute_max_payload_size_bytes(hal, arch);
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
        case 2: return sizeof(tt::tt_fabric::LowLatencyPacketHeaderT<2>);
        case 3: return sizeof(tt::tt_fabric::LowLatencyPacketHeaderT<3>);
        default: TT_THROW("Unsupported extension words: {}", extension_words);
    }
}

size_t FabricContext::get_2d_header_size(uint32_t route_buffer_size) const {
    // Use explicit template instantiation for compile-time type safety
    // Only max-capacity tiers per header size (20, 36, 52, 67) to avoid switch bloat
    switch (route_buffer_size) {
        case 20: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<20>);  // 80B header, max capacity
        case 36: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<36>);  // 96B header, max capacity
        case 52: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<52>);  // 112B header, max capacity
        case 67: return sizeof(tt::tt_fabric::HybridMeshPacketHeaderT<67>);  // 128B header, max capacity
        default: TT_THROW("Unsupported 2D route buffer size: {}", route_buffer_size);
    }
}

size_t FabricContext::get_udm_header_size(uint32_t route_buffer_size) const {
    // UDM header = base 2D header + UDM control fields
    return get_2d_header_size(route_buffer_size) + sizeof(tt::tt_fabric::UDMControlFields);
}

size_t FabricContext::compute_packet_header_size_bytes(const ControlPlane& control_plane) const {
    bool udm_enabled = control_plane.get_fabric_udm_mode() == tt::tt_fabric::FabricUDMMode::ENABLED;

    if (udm_enabled) {
        TT_FATAL(is_2D_routing_enabled_, "UDM mode only supports 2D routing");
        return get_udm_header_size(routing_2d_buffer_size_);
    }

    if (is_2D_routing_enabled_) {
        return get_2d_header_size(routing_2d_buffer_size_);
    }

    return get_1d_header_size(routing_1d_extension_words_);
}

size_t FabricContext::compute_max_payload_size_bytes(const tt_metal::Hal& hal, tt::ARCH arch) const {
    // If user provided override, validate and use it
    if (router_config_.max_packet_payload_size_bytes.has_value()) {
        return validate_and_apply_packet_size(hal, arch, router_config_.max_packet_payload_size_bytes.value());
    }
    // Default behavior
    if (is_2D_routing_enabled_) {
        return tt::tt_fabric::FabricEriscDatamoverBuilder::default_mesh_packet_payload_size_bytes;
    }
    return tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;
}

FabricContext::FabricContext(
    const ControlPlane& control_plane,
    const tt_metal::Hal& hal,
    tt::ARCH arch,
    bool is_ubb_galaxy,
    tt::tt_fabric::FabricConfig fabric_config,
    const FabricRouterConfig& router_config) :
    router_config_(router_config), is_ubb_galaxy_(is_ubb_galaxy) {
    // === Initialization order critical - dependencies flow downward ===
    // fabric_config_ → topology_ → routing flags → packet specs

    // Step 1: Validate and store base configuration
    TT_FATAL(
        fabric_config != tt::tt_fabric::FabricConfig::DISABLED,
        "Trying to initialize fabric context for disabled fabric config");
    this->fabric_config_ = fabric_config;

    // Step 2: Derive topology (depends on: fabric_config_)
    this->topology_ = FabricContext::get_topology_from_config(fabric_config);
    this->wrap_around_mesh_ = this->check_for_wrap_around_mesh(control_plane);

    // Step 3: Compute routing flags (depends on: topology_)
    this->is_2D_routing_enabled_ = is_2D_topology(this->topology_);
    this->bubble_flow_control_enabled_ = is_ring_or_torus(this->topology_);

    // Step 4: Compute and validate routing mode (depends on: topology_)
    this->compute_routing_mode();

    // Step 5: Compute packet specifications (depends on: routing flags)
    this->compute_packet_specifications(control_plane, hal, arch);

    // Step 6: Additional independent configs
    auto fabric_tensix_config = control_plane.get_fabric_tensix_config();
    this->tensix_enabled_ = (fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED);

    // Compute intermesh VC configuration (requires ControlPlane to be initialized)
    // this->intermesh_vc_config_ = this->compute_intermesh_vc_config();

    builder_context_ = nullptr;
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

bool FabricContext::need_deadlock_avoidance_support(
    const ControlPlane& control_plane, const FabricNodeId& fabric_node_id, eth_chan_directions direction) const {
    // The topology token cannot answer this under express routing: express chords can close a protected
    // ring on an axis the descriptor advertises as a line, so Topology::Torus would drop the bubble on a
    // ring that needs it. Asked per axis rather than per chip, since a per-node answer would elide the
    // guard on leaf chips and this flag also selects first-level ACK and the upstream credit path.
    if (control_plane.express_routing_enabled(fabric_node_id.mesh_id)) {
        return control_plane.mesh_has_protected_ring_in_axis_of(
            fabric_node_id.mesh_id, control_plane.eth_direction_to_routing_direction(direction));
    }

    if (topology_ == Topology::Ring) {
        return true;
    }
    if (topology_ == Topology::Torus) {
        const auto fabric_type = get_fabric_type(fabric_config_, is_ubb_galaxy_);
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

std::map<std::string, std::string> FabricContext::get_2d_kernel_defines(
    const ControlPlane& control_plane, MeshId mesh_id) const {
    if (!is_2D_routing_enabled_) {
        return {};
    }
    // GLOBAL scope is required: the L1 2D route table is packed against the global shape and indexed
    // by global chip ids, so a local-scope shape would desync the encode from the table.
    const auto mesh_shape = control_plane.get_physical_mesh_shape(mesh_id, MeshScope::GLOBAL);

    // Every 2D worker needs the global shape to encode destination-major action maps.
    std::map<std::string, std::string> defines{
        {"FABRIC_2D_MESH_Y_SIZE", std::to_string(mesh_shape[0])},
        {"FABRIC_2D_MESH_X_SIZE", std::to_string(mesh_shape[1])},
    };

    // Express enablement no longer selects the codec. It widens worker-side Z-port capacity and
    // rejects express with UDM; keeping it conditional avoids growing non-express connection managers.
    if (control_plane.express_routing_enabled(mesh_id)) {
        defines.emplace("FABRIC_EXPRESS_ENABLED", "1");
    }
    return defines;
}

std::map<std::string, std::string> FabricContext::get_fabric_kernel_defines(const ControlPlane& control_plane) const {
    std::map<std::string, std::string> defines;

    // Only emit defines if routing mode has been computed
    if (routing_mode_ == ROUTING_MODE_UNDEFINED) {
        return defines;  // Return empty map
    }

    // Add routing mode define
    defines["ROUTING_MODE"] = std::to_string(routing_mode_);

    // Add UDM mode define - only define it when enabled (not "0"), since header checks with #ifdef
    bool udm_enabled = control_plane.get_fabric_udm_mode() == tt::tt_fabric::FabricUDMMode::ENABLED;
    if (udm_enabled) {
        defines["UDM_MODE"] = "1";
    }

    // Add dynamic packet header sizing defines based on topology
    if (is_2D_routing_enabled_) {
        // 2D routing: inject route buffer size
        defines["FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE"] = std::to_string(routing_2d_buffer_size_);
    } else {
        // 1D routing: inject extension words
        defines["FABRIC_1D_PKT_HDR_EXTENSION_WORDS"] = std::to_string(routing_1d_extension_words_);
    }

    return defines;
}

void FabricContext::compute_routing_mode() {
    // Compute routing mode from topology configuration
    // This consolidates the logic from fabric_host_utils.cpp's set_routing_mode functions

    // Determine dimension based on topology
    uint32_t dimension = is_2D_routing_enabled_ ? 2 : 1;
    TT_FATAL(dimension == 1 || dimension == 2, "Invalid dimension {}. Supported dimensions are 1 or 2", dimension);

    // Build routing mode from topology flags
    uint16_t mode = 0;
    if (topology_ == Topology::Ring) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_RING);
    } else if (topology_ == Topology::Linear) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_LINE);
    } else if (topology_ == Topology::NeighborExchange) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_NEIGHBOR_EXCHANGE);
    } else if (topology_ == Topology::Mesh) {
        mode |= (ROUTING_MODE_2D | ROUTING_MODE_MESH);
    } else if (topology_ == Topology::Torus) {
        mode |= (ROUTING_MODE_2D | ROUTING_MODE_TORUS);
    }

    mode |= ROUTING_MODE_LOW_LATENCY;

    // Validate dimension flags are orthogonal (only one can be set)
    TT_FATAL(
        __builtin_popcount(mode & (ROUTING_MODE_1D | ROUTING_MODE_2D | ROUTING_MODE_3D)) == 1,
        "Only one dimension mode (1D, 2D, 3D) can be active at once");

    // Validate topology flags are orthogonal
    TT_FATAL(
        __builtin_popcount(
            mode & (ROUTING_MODE_RING | ROUTING_MODE_LINE | ROUTING_MODE_NEIGHBOR_EXCHANGE | ROUTING_MODE_MESH |
                    ROUTING_MODE_TORUS)) == 1,
        "Only one topology mode (RING, LINE, NEIGHBOR_EXCHANGE, MESH, TORUS) can be active at once");

    // Validate 1D can't be used with MESH or TORUS
    TT_FATAL(
        !(mode & ROUTING_MODE_1D) || !(mode & (ROUTING_MODE_MESH | ROUTING_MODE_TORUS)),
        "1D routing mode cannot be combined with MESH or TORUS topology");

    // Validate 2D can't be used with LINE or RING
    TT_FATAL(
        !(mode & ROUTING_MODE_2D) || !(mode & (ROUTING_MODE_LINE | ROUTING_MODE_RING | ROUTING_MODE_NEIGHBOR_EXCHANGE)),
        "2D routing mode cannot be combined with LINE or RING or NEIGHBOR_EXCHANGE topology");

    routing_mode_ = mode;
}

size_t FabricContext::validate_and_apply_packet_size(
    const tt_metal::Hal& hal, tt::ARCH arch, size_t requested_size) const {
    // Get architecture-specific limit from single source of truth
    size_t max_allowed = FabricEriscDatamoverBuilder::get_max_packet_payload_size_for_arch(arch);

    TT_FATAL(
        requested_size <= max_allowed,
        "Requested packet size {} exceeds maximum {} for {}",
        requested_size,
        max_allowed,
        tt::arch_to_str(arch));

    TT_FATAL(requested_size > 0, "Packet size must be greater than 0");

    // Validate alignment (must be L1-aligned for NOC transfers)
    const auto alignment = hal.get_alignment(tt::tt_metal::HalMemType::L1);
    TT_FATAL(requested_size % alignment == 0, "Packet size {} must be {}-byte aligned", requested_size, alignment);

    return requested_size;
}

}  // namespace tt::tt_fabric
