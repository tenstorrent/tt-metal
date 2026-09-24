// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>

#include <algorithm>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>

#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal::experimental::tt_fabric {

// Generate fixed ASIC position pinnings for Galaxy topology to ensure QSFP links align with fabric mesh
// corner nodes (and the mesh is not folded). Shared by generate_rank_bindings (Phase 1) and ControlPlane
// (Phase 2) so the galaxy pin placement is identical in both stages.
//
// * o o * < Corners pinned with *
// o o o o
// o o o o
// * o o * < Corners pinned with *
std::vector<PinningConstraint> get_galaxy_fixed_asic_position_pinnings_for_mesh(
    MeshId mesh_id, const tt::tt_metal::distributed::MeshShape& mesh_shape, bool hard_pin_node_0, bool nw_corner_only) {
    std::vector<PinningConstraint> pinning_groups;

    // Sub-galaxy slices: pin only the NW corner (node 0) to any tray-corner ASIC (asic_location==1 on
    // trays 1..4). The host-rank partition may land on any tray, so tray 1 alone is unsatisfiable.
    if (nw_corner_only) {
        pinning_groups.push_back(
            {{FabricNodeId{mesh_id, 0}},
             {AsicPosition{1, 1}, AsicPosition{2, 1}, AsicPosition{3, 1}, AsicPosition{4, 1}}});
        return pinning_groups;
    }

    // Get all 4 possible corner ASIC positions
    std::vector<AsicPosition> corner_asic_positions;
    corner_asic_positions.reserve(4);
    corner_asic_positions.emplace_back(AsicPosition{1, 1});  // Top left corner
    corner_asic_positions.emplace_back(AsicPosition{2, 1});  // Top right corner
    corner_asic_positions.emplace_back(AsicPosition{3, 1});  // Bottom left corner
    corner_asic_positions.emplace_back(AsicPosition{4, 1});  // Bottom right corner

    // Generate corner fabric node IDs for this mesh
    std::vector<FabricNodeId> corner_fabric_node_ids;
    corner_fabric_node_ids.reserve(4);
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, 0});
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, mesh_shape[1] - 1});
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, mesh_shape[1] * (mesh_shape[0] - 1)});
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, (mesh_shape[1] * mesh_shape[0]) - 1});

    pinning_groups.reserve(corner_fabric_node_ids.size());
    for (const auto& corner_fabric_node_id : corner_fabric_node_ids) {
        // Special case: Hard pin NW corner (fabric node id 0) to asic 1 tray 1 if requested.
        if (corner_fabric_node_id == FabricNodeId{mesh_id, 0} && hard_pin_node_0) {
            pinning_groups.push_back({{corner_fabric_node_id}, {AsicPosition{1, 1}}});
            continue;
        }

        pinning_groups.push_back({{corner_fabric_node_id}, corner_asic_positions});
    }

    return pinning_groups;
}

namespace {

// Apply many-to-many pinning groups as a single required constraint per group. Each group is applied
// independently, filtered down to what exists here: fabric nodes belonging to another logical mesh are
// dropped, as are ASIC positions absent from this physical mesh. A group left with nothing to say is
// skipped, so a group naming positions that only exist on some meshes constrains just those meshes. A
// mesh that declares pins must end up with at least one of them applied, otherwise it would map
// unpinned without anyone noticing.
std::optional<std::string> apply_pinning_groups(
    ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>& intra_mesh_constraints,
    const std::vector<PinningConstraint>& pinning_groups,
    MeshId logical_mesh_id,
    const std::map<AsicPosition, std::set<tt::tt_metal::AsicID>>& asic_positions_to_asic_ids) {
    bool any_group_for_mesh = false;
    bool any_group_applied = false;
    for (const auto& group : pinning_groups) {
        std::set<FabricNodeId> fabric_nodes;
        for (const auto& fabric_node : group.fabric_nodes) {
            if (fabric_node.mesh_id != logical_mesh_id) {
                continue;
            }
            fabric_nodes.insert(fabric_node);
        }
        if (fabric_nodes.empty()) {
            continue;
        }
        any_group_for_mesh = true;

        std::set<tt::tt_metal::AsicID> asic_ids;
        for (const auto& position : group.asic_positions) {
            auto it = asic_positions_to_asic_ids.find(position);
            if (it == asic_positions_to_asic_ids.end()) {
                log_trace(
                    tt::LogFabric,
                    "Pinned ASIC position (tray_id: {}, asic_location: {}) not found in physical topology; skipping",
                    position.first.get(),
                    position.second.get());
                continue;
            }
            asic_ids.insert(it->second.begin(), it->second.end());
        }

        if (asic_ids.empty()) {
            continue;
        }

        if (!intra_mesh_constraints.add_required_constraint(fabric_nodes, asic_ids)) {
            return fmt::format(
                "fabric nodes in a pinning group have pinned ASIC positions present in the physical mesh but none "
                "lie in each node's host-rank partition (conflicts with rank bindings)");
        }
        any_group_applied = true;
    }

    if (any_group_for_mesh && !any_group_applied) {
        return fmt::format(
            "Pinned ASIC positions of every pinning group for mesh {} were not found among the physical ASICs "
            "participating in this mesh",
            logical_mesh_id.get());
    }

    return std::nullopt;
}

// Extract requested inter-mesh connections and ports from MeshGraphDescriptor (same logic as
// MeshGraph::initialize_from_mgd).
std::pair<::tt::tt_fabric::RequestedIntermeshConnections, ::tt::tt_fabric::RequestedIntermeshPorts>
get_requested_intermesh_from_mgd(const ::tt::tt_fabric::MeshGraphDescriptor& mgd) {
    ::tt::tt_fabric::RequestedIntermeshConnections requested_intermesh_connections;
    ::tt::tt_fabric::RequestedIntermeshPorts requested_intermesh_ports;

    if (!mgd.has_connections_of_type("FABRIC")) {
        return {requested_intermesh_connections, requested_intermesh_ports};
    }

    for (::tt::tt_fabric::ConnectionId conn_id : mgd.connections_by_type("FABRIC")) {
        const auto& connection_data = mgd.get_connection(conn_id);
        const auto& src_instance = mgd.get_instance(connection_data.nodes[0]);
        const auto& dst_instance = mgd.get_instance(connection_data.nodes[1]);

        bool is_device_level = (src_instance.kind == ::tt::tt_fabric::NodeKind::Device) &&
                               (dst_instance.kind == ::tt::tt_fabric::NodeKind::Device);

        if (is_device_level) {
            const auto& src_mesh_instance = mgd.get_instance(src_instance.hierarchy.back());
            const auto& dst_mesh_instance = mgd.get_instance(dst_instance.hierarchy.back());
            const uint32_t src_mesh_id_val = src_mesh_instance.local_id;
            const uint32_t dst_mesh_id_val = dst_mesh_instance.local_id;
            requested_intermesh_ports[src_mesh_id_val][dst_mesh_id_val].push_back(
                {src_instance.local_id, dst_instance.local_id, connection_data.count});
        } else {
            const uint32_t src_mesh_id_val = src_instance.local_id;
            const uint32_t dst_mesh_id_val = dst_instance.local_id;
            requested_intermesh_connections[src_mesh_id_val][dst_mesh_id_val] = connection_data.count;
        }
    }
    return {requested_intermesh_connections, requested_intermesh_ports};
}

LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph_impl(
    const std::map<MeshId, ::tt::tt_fabric::AdjacencyGraph<FabricNodeId>>& mesh_adjacency_graphs,
    const ::tt::tt_fabric::RequestedIntermeshConnections& requested_intermesh_connections,
    const ::tt::tt_fabric::RequestedIntermeshPorts& requested_intermesh_ports) {
    // This function handles both strict mode (requested_intermesh_ports) and relaxed mode
    // (requested_intermesh_connections) intermesh connections:
    // - Strict mode: Creates fabric node-level exit nodes (LogicalExitNode with mesh_id and fabric_node_id)
    // - Relaxed mode: Creates mesh-level exit nodes (LogicalExitNode with mesh_id only, no fabric_node_id)
    // TODO: Add support for mixing STRICT and RELAXED policies in the same graph.
    // Currently, MGD validation prevents mixing policies, but when this feature is added,
    // this function will need to handle both simultaneously, creating appropriate exit node types
    // based on the connection type.
    using namespace ::tt::tt_fabric;

    LogicalMultiMeshGraph logical_multi_mesh_graph;

    for (const auto& [mesh_id, adjacency_graph] : mesh_adjacency_graphs) {
        logical_multi_mesh_graph.mesh_adjacency_graphs_[mesh_id] = adjacency_graph;
    }

    AdjacencyGraph<MeshId>::AdjacencyMap mesh_level_adjacency_map;
    std::map<MeshId, AdjacencyGraph<LogicalExitNode>::AdjacencyMap> exit_node_adjacency_maps;

    if (!requested_intermesh_ports.empty()) {
        for (const auto& [src_mesh_id_val, dst_mesh_map] : requested_intermesh_ports) {
            MeshId src_mesh_id(src_mesh_id_val);

            // Initialize exit node adjacency map for this mesh if needed
            if (!exit_node_adjacency_maps.contains(src_mesh_id)) {
                exit_node_adjacency_maps[src_mesh_id] = AdjacencyGraph<LogicalExitNode>::AdjacencyMap();
            }

            for (const auto& [dst_mesh_id_val, port_list] : dst_mesh_map) {
                MeshId dst_mesh_id(dst_mesh_id_val);
                // Skip self-connections
                if (dst_mesh_id != src_mesh_id) {
                    // Initialize exit node adjacency map for destination mesh if needed
                    if (!exit_node_adjacency_maps.contains(dst_mesh_id)) {
                        exit_node_adjacency_maps[dst_mesh_id] = AdjacencyGraph<LogicalExitNode>::AdjacencyMap();
                    }

                    // Add connections based on num_channels from each port entry
                    // Each tuple is (src_device, dst_device, num_channels)
                    for (const auto& port_entry : port_list) {
                        uint32_t src_device = std::get<0>(port_entry);
                        uint32_t dst_device = std::get<1>(port_entry);
                        uint32_t num_channels = std::get<2>(port_entry);

                        // Create LogicalExitNodes for exit nodes (fabric node-level exit nodes)
                        LogicalExitNode src_exit_node{src_mesh_id, FabricNodeId(src_mesh_id, src_device)};
                        LogicalExitNode dst_exit_node{dst_mesh_id, FabricNodeId(dst_mesh_id, dst_device)};

                        // Add to mesh-level adjacency map (multiple entries for multiple channels)
                        for (uint32_t i = 0; i < num_channels; ++i) {
                            mesh_level_adjacency_map[src_mesh_id].push_back(dst_mesh_id);
                        }

                        // Add to exit node graphs (multiple entries for multiple channels)
                        // Only add in the direction specified - the descriptor already handles bidirectional entries
                        for (uint32_t i = 0; i < num_channels; ++i) {
                            exit_node_adjacency_maps[src_mesh_id][src_exit_node].push_back(dst_exit_node);
                        }
                    }
                }
            }
        }
    }

    // Process requested_intermesh_connections (mesh-level connections, no device specified) if it exists
    // Mapping: src_mesh -> dst_mesh -> num_channels
    // Create mesh-level exit nodes for mesh-level connections (no device specified)
    // Note: Using LogicalExitNode as map key ensures no duplicates - same mesh-level exit node
    // (LogicalExitNode{mesh_id, nullopt}) will only appear once per mesh, with all neighbors added to it
    if (!requested_intermesh_connections.empty()) {
        for (const auto& [src_mesh_id_val, dst_mesh_map] : requested_intermesh_connections) {
            MeshId src_mesh_id(src_mesh_id_val);

            // Initialize exit node adjacency map for this mesh if needed
            if (!exit_node_adjacency_maps.contains(src_mesh_id)) {
                exit_node_adjacency_maps[src_mesh_id] = AdjacencyGraph<LogicalExitNode>::AdjacencyMap();
            }

            for (const auto& [dst_mesh_id_val, num_channels] : dst_mesh_map) {
                MeshId dst_mesh_id(dst_mesh_id_val);
                // Skip self-connections
                if (dst_mesh_id != src_mesh_id) {
                    // Initialize exit node adjacency map for destination mesh if needed
                    if (!exit_node_adjacency_maps.contains(dst_mesh_id)) {
                        exit_node_adjacency_maps[dst_mesh_id] = AdjacencyGraph<LogicalExitNode>::AdjacencyMap();
                    }

                    // Create a single mesh-level exit node for this source mesh (will be reused for all connections)
                    // The map key ensures this exit node only appears once, even if we reference it multiple times
                    LogicalExitNode src_exit_node{src_mesh_id, std::nullopt};
                    LogicalExitNode dst_exit_node{dst_mesh_id, std::nullopt};

                    // Add connections based on num_channels (multiple connections between same meshes)
                    for (uint32_t i = 0; i < num_channels; ++i) {
                        mesh_level_adjacency_map[src_mesh_id].push_back(dst_mesh_id);
                        // Add to exit node graphs (multiple entries for multiple channels)
                        // The map key ensures src_exit_node only appears once, with all neighbors accumulated
                        exit_node_adjacency_maps[src_mesh_id][src_exit_node].push_back(dst_exit_node);
                    }
                }
            }
        }
    }

    // Ensure all meshes are represented as nodes in the mesh-level graph, even if they have no connections
    // This is important for single-mesh scenarios where there are no inter-mesh connections
    for (const auto& [mesh_id, _] : mesh_adjacency_graphs) {
        if (!mesh_level_adjacency_map.contains(mesh_id)) {
            mesh_level_adjacency_map[mesh_id] = std::vector<MeshId>();
        }
    }

    // Build mesh-level graph from adjacency map
    logical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(mesh_level_adjacency_map);

    for (const auto& [mesh_id, _] : mesh_adjacency_graphs) {
        auto exit_node_it = exit_node_adjacency_maps.find(mesh_id);
        if (exit_node_it != exit_node_adjacency_maps.end() && !exit_node_it->second.empty()) {
            logical_multi_mesh_graph.mesh_exit_node_graphs_[mesh_id] =
                AdjacencyGraph<LogicalExitNode>(exit_node_it->second);
        } else {
            // Initialize empty graph for meshes with no exit nodes
            logical_multi_mesh_graph.mesh_exit_node_graphs_[mesh_id] = AdjacencyGraph<LogicalExitNode>();
        }
    }
    return logical_multi_mesh_graph;
}

}  // namespace

namespace {

MeshId remap_local_mesh_id(const std::map<MeshId, MeshId>& local_to_global, MeshId local) {
    const auto it = local_to_global.find(local);
    return (it != local_to_global.end()) ? it->second : local;
}

std::vector<TopologyMappingResult> split_mapping_to_local_parts(
    const TopologyMappingResult& global, const std::vector<std::map<MeshId, MeshId>>& local_to_global) {
    if (local_to_global.empty()) {
        return {global};
    }
    std::vector<TopologyMappingResult> parts(local_to_global.size());
    for (auto& part : parts) {
        part.success = global.success;
        part.error_message = global.error_message;
        part.stats = global.stats;
    }
    for (const auto& [fabric_node, asic] : global.fabric_node_to_asic) {
        for (std::size_t part_index = 0; part_index < local_to_global.size(); ++part_index) {
            for (const auto& [local, global_mesh] : local_to_global[part_index]) {
                if (global_mesh != fabric_node.mesh_id) {
                    continue;
                }
                const FabricNodeId local_node(local, fabric_node.chip_id);
                parts[part_index].fabric_node_to_asic.insert({local_node, asic});
                parts[part_index].asic_to_fabric_node.insert({asic, local_node});
            }
        }
    }
    return parts;
}

void remap_fabric_node_ranks_to_global(
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& dest,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& local_ranks,
    const std::map<MeshId, MeshId>& local_to_global) {
    for (const auto& [local_mesh, node_ranks] : local_ranks) {
        const MeshId global_mesh = remap_local_mesh_id(local_to_global, local_mesh);
        for (const auto& [fabric_node, rank] : node_ranks) {
            dest[global_mesh][FabricNodeId(global_mesh, fabric_node.chip_id)] = rank;
        }
    }
}

void remap_asic_ranks_to_global(
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& dest,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& local_ranks,
    const std::map<MeshId, MeshId>& local_to_global) {
    for (const auto& [local_mesh, asic_ranks] : local_ranks) {
        dest[remap_local_mesh_id(local_to_global, local_mesh)].insert(asic_ranks.begin(), asic_ranks.end());
    }
}

void validate_shared_inter_mesh_policy(
    const std::vector<const ::tt::tt_fabric::MeshGraphDescriptor*>& mesh_graph_descriptors) {
    std::optional<bool> shared_relaxed;
    std::size_t shared_index = 0;
    for (std::size_t index = 0; index < mesh_graph_descriptors.size(); ++index) {
        TT_FATAL(mesh_graph_descriptors[index] != nullptr, "Mesh graph descriptor {} is null", index);
        if (!mesh_graph_descriptors[index]->is_inter_mesh_policy_specified()) {
            continue;
        }
        const bool relaxed = mesh_graph_descriptors[index]->is_inter_mesh_policy_relaxed();
        if (!shared_relaxed.has_value()) {
            shared_relaxed = relaxed;
            shared_index = index;
            continue;
        }
        TT_FATAL(
            relaxed == *shared_relaxed,
            "Mesh graph descriptors merged into one topology must agree on the inter-mesh channel policy, but "
            "descriptor {} is {} while descriptor {} is {}. Mixed policies are not supported yet: the merged solve "
            "applies one policy to every seam, so one descriptor's policy would be applied to the other's. "
            "See https://github.com/tenstorrent/tt-metal/issues/49960",
            shared_index,
            *shared_relaxed ? "RELAXED" : "STRICT",
            index,
            relaxed ? "RELAXED" : "STRICT");
    }
}

}  // namespace

LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor) {
    auto mesh_adjacency_graphs = ::tt::tt_fabric::build_adjacency_graph_logical(mesh_graph_descriptor);
    auto [requested_intermesh_connections, requested_intermesh_ports] =
        get_requested_intermesh_from_mgd(mesh_graph_descriptor);
    return build_logical_multi_mesh_adjacency_graph_impl(
        mesh_adjacency_graphs, requested_intermesh_connections, requested_intermesh_ports);
}

/**
 * @brief Build a flat PhysicalAdjacencyMap from PhysicalSystemDescriptor
 *
 * Builds a complete flat adjacency map including all connections (both intra-mesh and intermesh),
 * with multiple entries per channel (one edge per Ethernet link).
 */
PhysicalAdjacencyMap build_flat_adjacency_map_from_psd(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    PhysicalAdjacencyMap flat_adj;

    // Diagnostics (debug only): count how many ethernet links are local (intra-host) vs global (cross-host),
    // and how many distinct cross-host ASIC pairs survive into the flat graph. Cross-host links are the
    // inter-galaxy fabric seams; if they are under-represented here the physical mesh-level adjacency will be
    // too sparse for the topology mapper to embed the MGD.
    std::size_t local_links = 0;
    std::size_t global_links = 0;
    std::set<std::pair<tt::tt_metal::AsicID, tt::tt_metal::AsicID>> cross_host_pairs;

    // Isolated ASICs (no eth links) must still appear so 1x1 / single-chip systems can be seated.
    for (const auto& [asic_id, unused_desc] : physical_system_descriptor.get_asic_descriptors()) {
        (void)unused_desc;
        flat_adj[asic_id];
    }

    // Go through all connections in the physical system descriptor
    for (const auto& host_name : physical_system_descriptor.get_all_hostnames()) {
        for (const auto& [src_asic_id, asic_connections] : physical_system_descriptor.get_asic_topology(host_name)) {
            for (const auto& asic_connection : asic_connections) {
                auto dst_asic_id = asic_connection.first;

                // Skip self-connections
                if (src_asic_id == dst_asic_id) {
                    continue;
                }

                const auto& eth_connections = asic_connection.second;
                // Add each neighbor multiple times based on number of ethernet connections (channels)
                for (const auto& eth_conn : eth_connections) {
                    flat_adj[src_asic_id].push_back(dst_asic_id);
                    if (eth_conn.is_local) {
                        ++local_links;
                    } else {
                        ++global_links;
                        cross_host_pairs.emplace(
                            std::min(src_asic_id, dst_asic_id), std::max(src_asic_id, dst_asic_id));
                    }
                }
            }
        }
    }

    log_debug(
        tt::LogFabric,
        "build_flat_adjacency_map_from_psd: {} ASIC node(s), {} local eth link(s), {} cross-host eth link(s), {} "
        "distinct cross-host ASIC pair(s)",
        flat_adj.size(),
        local_links,
        global_links,
        cross_host_pairs.size());

    return flat_adj;
}

namespace {

struct MeshPhysicalLayout {
    std::unordered_set<tt::tt_metal::AsicID> asics;
    std::map<LogicalChipId, AsicPosition> mesh_node_to_asic_position;
};

std::map<MeshId, MeshPhysicalLayout> mesh_physical_layouts_from_assigned_meshes(
    const std::vector<::tt::tt_fabric::PlacedMesh>& assigned_meshes) {
    std::map<MeshId, MeshPhysicalLayout> layouts;
    for (const auto& placed : assigned_meshes) {
        if (placed.placement.asics.empty()) {
            continue;
        }
        MeshPhysicalLayout& layout = layouts[placed.mesh_id];
        layout.asics = placed.placement.asics;
        layout.mesh_node_to_asic_position = placed.placement.mesh_node_to_asic_position;
    }
    return layouts;
}

PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::map<MeshId, MeshPhysicalLayout>& mesh_layouts) {
    // Build asic_id_to_mesh_rank map from mesh layouts using the caller's MeshIds as-is.
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [mesh_id, layout] : mesh_layouts) {
        for (const auto& asic_id : layout.asics) {
            // Default to rank 0 - proper rank assignment would come from hostname_to_asics or other config
            asic_id_to_mesh_rank[mesh_id][asic_id] = MeshHostRankId{0};
        }
    }

    // Build a map from AsicID to MeshId for quick lookup
    std::unordered_map<tt::tt_metal::AsicID, MeshId> asic_id_to_mesh_id;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            asic_id_to_mesh_id[asic_id] = mesh_id;
        }
    }

    // Build per-mesh adjacency maps (only intra-mesh connections)
    std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap> mesh_adjacency_maps;
    std::map<MeshId, AdjacencyGraph<PhysicalExitNode>::AdjacencyMap> exit_node_adjacency_maps;
    AdjacencyGraph<MeshId>::AdjacencyMap mesh_level_adjacency_map;

    // Initialize adjacency maps for all meshes and ensure all ASICs are included
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        mesh_adjacency_maps[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap();
        exit_node_adjacency_maps[mesh_id] = AdjacencyGraph<PhysicalExitNode>::AdjacencyMap();
        // Initialize all ASICs in this mesh with empty neighbor lists
        for (const auto& [asic_id, _] : asic_map) {
            mesh_adjacency_maps[mesh_id][asic_id] = std::vector<tt::tt_metal::AsicID>();
        }
    }

    // Process each ASIC in the flat adjacency graph
    for (const auto& src_asic_id : flat_adjacency_graph.get_nodes()) {
        auto src_mesh_id_it = asic_id_to_mesh_id.find(src_asic_id);
        if (src_mesh_id_it == asic_id_to_mesh_id.end()) {
            // ASIC not in any mesh assignment, skip it
            continue;
        }
        MeshId src_mesh_id = src_mesh_id_it->second;

        // Process each neighbor
        const auto& neighbors = flat_adjacency_graph.get_neighbors(src_asic_id);
        for (const auto& dst_asic_id : neighbors) {
            auto dst_mesh_id_it = asic_id_to_mesh_id.find(dst_asic_id);
            if (dst_mesh_id_it == asic_id_to_mesh_id.end()) {
                // Neighbor not in any mesh assignment, skip it
                continue;
            }
            MeshId dst_mesh_id = dst_mesh_id_it->second;

            if (src_mesh_id == dst_mesh_id) {
                // Intra-mesh connection: add to mesh adjacency map
                mesh_adjacency_maps[src_mesh_id][src_asic_id].push_back(dst_asic_id);
            } else {
                // Intermesh connection: add to exit node graph and mesh-level graph
                // Create PhysicalExitNode objects with mesh_id populated
                PhysicalExitNode src_exit_node{src_mesh_id, src_asic_id};
                PhysicalExitNode dst_exit_node{dst_mesh_id, dst_asic_id};
                exit_node_adjacency_maps[src_mesh_id][src_exit_node].push_back(dst_exit_node);
                mesh_level_adjacency_map[src_mesh_id].push_back(dst_mesh_id);
            }
        }
    }

    // Build PhysicalMultiMeshGraph
    PhysicalMultiMeshGraph physical_multi_mesh_graph;

    // Convert adjacency maps to graphs
    for (const auto& [mesh_id, adj_map] : mesh_adjacency_maps) {
        physical_multi_mesh_graph.mesh_adjacency_graphs_[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>(adj_map);
    }

    // Convert exit node adjacency maps to graphs
    // Initialize exit node graphs for all meshes (even if empty)
    for (const auto& [mesh_id, _] : asic_id_to_mesh_rank) {
        auto exit_node_it = exit_node_adjacency_maps.find(mesh_id);
        if (exit_node_it != exit_node_adjacency_maps.end() && !exit_node_it->second.empty()) {
            physical_multi_mesh_graph.mesh_exit_node_graphs_[mesh_id] =
                AdjacencyGraph<PhysicalExitNode>(exit_node_it->second);
        } else {
            // Initialize empty graph for meshes with no exit nodes
            physical_multi_mesh_graph.mesh_exit_node_graphs_[mesh_id] = AdjacencyGraph<PhysicalExitNode>();
        }
    }

    // Ensure all meshes are represented in mesh-level graph, even if they have no connections
    for (const auto& [mesh_id, _] : asic_id_to_mesh_rank) {
        if (!mesh_level_adjacency_map.contains(mesh_id)) {
            mesh_level_adjacency_map[mesh_id] = std::vector<MeshId>();
        }
    }

    // Build mesh-level graph from adjacency map
    physical_multi_mesh_graph.mesh_level_graph_ = ::tt::tt_fabric::AdjacencyGraph<MeshId>(mesh_level_adjacency_map);

    for (const auto& [mesh_id, layout] : mesh_layouts) {
        if (!layout.mesh_node_to_asic_position.empty()) {
            physical_multi_mesh_graph.mesh_pgd_pinnings_[mesh_id] = layout.mesh_node_to_asic_position;
        }
    }

    return physical_multi_mesh_graph;
}

PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::vector<::tt::tt_fabric::PlacedMesh>& assigned_meshes) {
    return build_hierarchical_from_flat_graph(
        flat_adjacency_graph, mesh_physical_layouts_from_assigned_meshes(assigned_meshes));
}

// Helper function to build ASIC positions to ASIC IDs map
std::map<AsicPosition, std::set<tt::tt_metal::AsicID>> build_asic_positions_map(
    const ::tt::tt_fabric::AdjacencyGraph<tt::tt_metal::AsicID>& physical_graph, const TopologyMappingConfig& config) {
    std::map<AsicPosition, std::set<tt::tt_metal::AsicID>> asic_positions_to_asic_ids;
    if (!config.asic_positions.empty()) {
        for (const auto& asic_id : physical_graph.get_nodes()) {
            auto pos_it = config.asic_positions.find(asic_id);
            if (pos_it != config.asic_positions.end()) {
                asic_positions_to_asic_ids[pos_it->second].insert(asic_id);
            }
        }
    }
    return asic_positions_to_asic_ids;
}

// Helper function to determine inter-mesh validation mode
::tt::tt_fabric::ConnectionValidationMode determine_inter_mesh_validation_mode(const TopologyMappingConfig& config) {
    if (config.inter_mesh_validation_mode.has_value()) {
        return config.inter_mesh_validation_mode.value();
    }
    return ::tt::tt_fabric::ConnectionValidationMode::RELAXED;
}

// Helper function to determine intra-mesh validation mode
::tt::tt_fabric::ConnectionValidationMode determine_intra_mesh_validation_mode(
    const TopologyMappingConfig& config, MeshId logical_mesh_id) {
    auto config_mode_it = config.mesh_validation_modes.find(logical_mesh_id);
    if (config_mode_it != config.mesh_validation_modes.end()) {
        return config_mode_it->second;
    }
    return ::tt::tt_fabric::ConnectionValidationMode::RELAXED;
}

// Helper function to add rank binding constraints. Only called when config.disable_rank_bindings is false.
//
// Purpose: Build rank_to_asics so that fabric nodes of rank R can only map to ASICs in rank_to_asics[R].
// The topology solver then chooses a valid 1:1 mapping respecting connectivity.
void add_rank_binding_constraints(
    ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>& intra_mesh_constraints,
    const TopologyMappingConfig& config,
    MeshId logical_mesh_id,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    if (!fabric_node_id_to_mesh_rank.contains(logical_mesh_id)) {
        return;
    }
    const auto& fabric_node_ranks = fabric_node_id_to_mesh_rank.at(logical_mesh_id);

    // When asic_id_to_mesh_rank has no entry for this mesh, treat all physical ASICs as UNSET
    std::map<tt::tt_metal::AsicID, MeshHostRankId> asic_ranks_unset;
    if (!asic_id_to_mesh_rank.contains(logical_mesh_id)) {
        for (const auto& [_, asic_set] : config.hostname_to_asics) {
            for (const auto& asic_id : asic_set) {
                asic_ranks_unset[asic_id] = ::tt::tt_fabric::MESH_HOST_RANK_UNSET;
            }
        }
    }
    const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_ranks =
        asic_id_to_mesh_rank.contains(logical_mesh_id) ? asic_id_to_mesh_rank.at(logical_mesh_id) : asic_ranks_unset;

    // Group fabric nodes by rank: rank_to_fabric_nodes[R] = { fabric nodes that must map to rank R's ASICs }
    std::map<MeshHostRankId, std::set<FabricNodeId>> rank_to_fabric_nodes;
    for (const auto& [fabric_node, rank] : fabric_node_ranks) {
        rank_to_fabric_nodes[rank].insert(fabric_node);
    }

    // rank_to_asics[R] = { ASICs that fabric nodes of rank R may map to }
    std::map<MeshHostRankId, std::set<tt::tt_metal::AsicID>> rank_to_asics;

    if (config.hostname_to_asics.empty()) {
        // Legacy path: no host grouping. Each ASIC with explicit rank goes to that rank's pool.
        for (const auto& [asic_id, rank] : asic_ranks) {
            if (rank != ::tt::tt_fabric::MESH_HOST_RANK_UNSET) {
                rank_to_asics[rank].insert(asic_id);
            }
        }
    } else {
        // Host-grouped path: config.hostname_to_asics defines which ASICs belong to which host.
        // Constraint: all ASICs on the same host must map to fabric nodes of the same rank
        // (ControlPlane/TopologyMapper "same-host same-rank" invariant).

        std::unordered_set<tt::tt_metal::AsicID> asics_in_host_config;
        for (const auto& [_, asic_set] : config.hostname_to_asics) {
            asics_in_host_config.insert(asic_set.begin(), asic_set.end());
        }

        // Legacy ASICs (not in any host in config): add by explicit rank.
        for (const auto& [asic_id, rank] : asic_ranks) {
            if (rank != ::tt::tt_fabric::MESH_HOST_RANK_UNSET && !asics_in_host_config.contains(asic_id)) {
                rank_to_asics[rank].insert(asic_id);
            }
        }

        // Per-host: classify as explicitly bound (has rank) or UNSET (all ASICs have MESH_HOST_RANK_UNSET).
        std::set<MeshHostRankId> claimed_ranks;
        std::vector<std::set<tt::tt_metal::AsicID>> unset_hosts;
        unset_hosts.reserve(config.hostname_to_asics.size());

        for (const auto& [hostname, asic_set] : config.hostname_to_asics) {
            std::set<tt::tt_metal::AsicID> host_asics_in_mesh;
            std::optional<MeshHostRankId> host_rank;
            for (const auto& asic_id : asic_set) {
                auto it = asic_ranks.find(asic_id);
                if (it == asic_ranks.end()) {
                    continue;
                }
                host_asics_in_mesh.insert(asic_id);
                if (it->second != ::tt::tt_fabric::MESH_HOST_RANK_UNSET) {
                    if (host_rank.has_value() && host_rank.value() != it->second) {
                        TT_THROW(
                            "Host consistency violated: host {} has ASICs with inconsistent ranks ({} and {}). "
                            "Each host in the PSD must have exactly one rank binding.",
                            hostname,
                            host_rank->get(),
                            it->second.get());
                    }
                    host_rank = it->second;
                }
            }
            if (host_asics_in_mesh.empty()) {
                continue;
            }

            if (host_rank.has_value()) {
                claimed_ranks.insert(host_rank.value());
                for (const auto& asic_id : host_asics_in_mesh) {
                    rank_to_asics[host_rank.value()].insert(asic_id);
                }
            } else {
                unset_hosts.push_back(std::move(host_asics_in_mesh));
            }
        }

        // -----------------------------------------------------------------------
        // UNSET hosts: no pre-assignment of ranks. Solver picks assignment.
        // -----------------------------------------------------------------------
        // Constraint: each host's ASICs must all map to fabric nodes of the same rank
        // (same-host same-rank). We add UNSET ASICs to all unclaimed ranks' pools and
        // set a same-rank-groups constraint so the solver rejects splits during DFS.
        // If large meshes hit DFS limits, consider pruning (e.g., host↔rank matching).
        // -----------------------------------------------------------------------
        if (!unset_hosts.empty()) {
            std::vector<MeshHostRankId> unclaimed_ranks;
            unclaimed_ranks.reserve(rank_to_fabric_nodes.size());
            for (const auto& [r, fn_set] : rank_to_fabric_nodes) {
                if (!fn_set.empty() && !claimed_ranks.contains(r)) {
                    unclaimed_ranks.push_back(r);
                }
            }
            if (unclaimed_ranks.empty()) {
                TT_THROW(
                    "Rank bindings: {} host(s) have no rank binding but all mesh ranks are already claimed. "
                    "Either assign ranks to these hosts or ensure enough ranks exist.",
                    unset_hosts.size());
            }

            for (const auto& r : unclaimed_ranks) {
                for (const auto& host_asics : unset_hosts) {
                    rank_to_asics[r].insert(host_asics.begin(), host_asics.end());
                }
            }

            // Same-group: fabric ranks that use UNSET host pools (unclaimed_ranks); one target group per such rank.
            // Global partitions: UNSET hosts only (unset_hosts). Claimed ranks are pinned by rank_to_asics below and
            // are not part of this host↔rank matching. Solver assigns target groups to distinct UNSET partitions
            // (not index-aligned).
            std::vector<std::set<FabricNodeId>> target_groups;
            target_groups.reserve(unclaimed_ranks.size());
            for (const auto& r : unclaimed_ranks) {
                auto it = rank_to_fabric_nodes.find(r);
                if (it != rank_to_fabric_nodes.end() && !it->second.empty()) {
                    target_groups.push_back(it->second);
                }
            }
            std::vector<std::set<tt::tt_metal::AsicID>> global_groups(unset_hosts.begin(), unset_hosts.end());
            // set_same_rank_groups_constraint matches same-rank target groups (ranks) to global host
            // partitions injectively -- one DISTINCT partition slot per rank, so it needs nt <= ng. When the
            // MGD declares more mesh_host_ranks (nt) than there are UNSET physical hosts (ng = G), replicate
            // the G real host ASIC pools round-robin up to nt slots so each host backs ceil/floor(nt/G) slots.
            //
            // This only sets the per-host SLOT CAPACITY (how many ranks a host may hold) -- it does NOT place
            // any chips itself. Example: G = 2 hosts, nt = 4 ranks
            //
            //   base_partitions (G real host pools) : [ H0 ][ H1 ]
            //   replicate  slot i <- host (i % G)   :   H0    H1    H0    H1
            //   global_groups (nt = 4 slots)        : [ H0 ][ H1 ][ H0 ][ H1 ]   (H0,H1 each back 2 slots)
            //
            // set_same_rank_groups_constraint then matches the nt ranks to these slots (injective on slots,
            // NOT index-aligned), so 2 ranks land on H0 and 2 on H1. The SOLVER -- not this loop -- then
            // carves the actual disjoint, connectivity-preserving ASIC slice per rank within its host's pool;
            // the same-rank constraint keeps every rank inside ONE physical host (no galaxy straddling):
            //
            //        H0 pool                 H1 pool
            //     [ rank0 | rank1 ]       [ rank2 | rank3 ]      (each '|' separates a disjoint chip slice)
            //
            //   * G = 1, N ranks  -> single galaxy split into N host-ranks (mock single-host discovery)
            //   * G hosts, N ranks -> N/G ranks per galaxy (e.g. a dual/quad galaxy mock assembled from G
            //     adjacent SP4 galaxy descriptors, split into the MGD's finer host grid)
            //
            // Round-robin keeps the per-host slot counts balanced (they differ by at most 1 when G does not
            // divide nt), matching the balanced host grids these MGDs declare.
            if (!global_groups.empty() && target_groups.size() > global_groups.size()) {
                const std::vector<std::set<tt::tt_metal::AsicID>> base_partitions = global_groups;
                global_groups.clear();
                global_groups.reserve(target_groups.size());
                for (size_t i = 0; i < target_groups.size(); ++i) {
                    global_groups.push_back(base_partitions[i % base_partitions.size()]);
                }
            }
            if (!intra_mesh_constraints.set_same_rank_groups_constraint(target_groups, global_groups)) {
                TT_THROW(
                    "Failed to set same-rank groups constraint for mesh {} (rank/host partition matching "
                    "infeasible with current rank bindings).",
                    logical_mesh_id.get());
            }
        }
    }

    // Add required constraint: fabric nodes of rank R can only map to ASICs in rank_to_asics[R].
    for (const auto& [rank, fabric_nodes] : rank_to_fabric_nodes) {
        auto asic_it = rank_to_asics.find(rank);
        if (asic_it != rank_to_asics.end() && !asic_it->second.empty()) {
            if (!intra_mesh_constraints.add_required_constraint(fabric_nodes, asic_it->second)) {
                TT_THROW(
                    "Failed to add required constraint for rank bindings in mesh {} for rank {}",
                    logical_mesh_id.get(),
                    rank);
            }
        }
    }
}

// Helper function to build pinning constraints.
// Only applies pinnings whose ASIC positions exist on the current physical grouping; absent positions are
// skipped. Returns an error if a present pinning conflicts with rank bindings (spill).
std::optional<std::string> add_pinning_constraints(
    ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>& intra_mesh_constraints,
    const std::map<AsicPosition, std::set<tt::tt_metal::AsicID>>& asic_positions_to_asic_ids,
    const TopologyMappingConfig& config,
    MeshId logical_mesh_id) {
    return apply_pinning_groups(intra_mesh_constraints, config.pinnings, logical_mesh_id, asic_positions_to_asic_ids);
}

// Add the PGD-derived layout as PREFERRED (soft) intra-mesh constraints. Must be called AFTER the hard
// rank/exit/MGD-pin constraints so it only biases ASIC choice where they leave freedom; soft constraints never
// make the solve infeasible. `mesh_pgd_pinnings` is keyed by physical MeshId; the entry for the mapped
// `physical_mesh_id` is a logical-chip-id -> ASIC position (TrayID + ASICLocation) layout, which is re-keyed onto
// the logical FabricNodeId being solved for. Each pinned position is resolved to concrete ASIC(s) via
// `asic_positions_to_asic_ids` and then restricted to `physical_mesh_node_set` (this physical mesh's sub-graph),
// so a preference can never reference an out-of-mesh node. No-op when there is no pinning for this physical mesh.
void add_pgd_pinning_preferred_constraints(
    ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>& intra_mesh_constraints,
    const std::map<MeshId, std::map<LogicalChipId, AsicPosition>>& mesh_pgd_pinnings,
    const std::map<AsicPosition, std::set<tt::tt_metal::AsicID>>& asic_positions_to_asic_ids,
    const std::unordered_set<tt::tt_metal::AsicID>& physical_mesh_node_set,
    MeshId logical_mesh_id,
    MeshId physical_mesh_id) {
    auto mesh_it = mesh_pgd_pinnings.find(physical_mesh_id);
    if (mesh_it == mesh_pgd_pinnings.end()) {
        return;
    }
    for (const auto& [chip_id, asic_position] : mesh_it->second) {
        // Resolve the pinned physical position back to the ASIC(s) sitting at that tray/asic-location, restricted
        // to this physical mesh's sub-graph (within one mesh footprint a position resolves to a single ASIC).
        auto pos_it = asic_positions_to_asic_ids.find(asic_position);
        if (pos_it == asic_positions_to_asic_ids.end()) {
            continue;
        }
        std::set<tt::tt_metal::AsicID> preferred_asics;
        for (const auto& asic_id : pos_it->second) {
            if (physical_mesh_node_set.contains(asic_id)) {
                preferred_asics.insert(asic_id);
            }
        }
        if (preferred_asics.empty()) {
            continue;
        }
        intra_mesh_constraints.add_preferred_constraint(FabricNodeId(logical_mesh_id, chip_id), preferred_asics);
    }
}

// Parallel physical inter-mesh edges from one exit ASIC to a destination mesh (each edge is one link / channel).
uint32_t max_physical_exit_edges_per_asic_toward_mesh(
    const ::tt::tt_fabric::AdjacencyGraph<PhysicalExitNode>& physical_exit_node_graph, MeshId dst_physical_mesh_id) {
    uint32_t max_toward_dst = 0;
    for (const auto& src_exit : physical_exit_node_graph.get_nodes()) {
        uint32_t count = 0;
        for (const auto& dst_exit : physical_exit_node_graph.get_neighbors(src_exit)) {
            if (dst_exit.mesh_id == dst_physical_mesh_id) {
                count++;
            }
        }
        max_toward_dst = std::max(max_toward_dst, count);
    }
    return max_toward_dst;
}

// Total physical inter-mesh links from this mesh toward dst_physical_mesh_id (sum over exit ASICs).
uint32_t total_physical_exit_edges_toward_mesh(
    const ::tt::tt_fabric::AdjacencyGraph<PhysicalExitNode>& physical_exit_node_graph, MeshId dst_physical_mesh_id) {
    uint32_t total = 0;
    for (const auto& src_exit : physical_exit_node_graph.get_nodes()) {
        for (const auto& dst_exit : physical_exit_node_graph.get_neighbors(src_exit)) {
            if (dst_exit.mesh_id == dst_physical_mesh_id) {
                total++;
            }
        }
    }
    return total;
}

// Helper function to add exit node constraints
// Constrains certain exit node ASICs on the physical graph to be mappable to exit node fabric nodes in the logical
// graph
// Returns nullopt on success, or a reason string if constraints cannot be satisfied
// (e.g., no valid physical exit nodes or over-constrained)
std::optional<std::string> add_exit_node_constraints(
    ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>& intra_mesh_constraints,
    const std::unordered_map<MeshId, MeshId>& mesh_mappings,
    const ::tt::tt_fabric::AdjacencyGraph<FabricNodeId>& logical_graph,
    const ::tt::tt_fabric::AdjacencyGraph<LogicalExitNode>& logical_exit_node_graph,
    const ::tt::tt_fabric::AdjacencyGraph<PhysicalExitNode>& physical_exit_node_graph,
    ::tt::tt_fabric::ConnectionValidationMode inter_mesh_validation_mode) {
    std::unordered_map<MeshId, std::set<tt::tt_metal::AsicID>> valid_physical_exit_nodes_by_mesh;
    std::set<FabricNodeId> valid_logical_exit_nodes(logical_graph.get_nodes().begin(), logical_graph.get_nodes().end());

    // Build reverse map: physical mesh ID -> logical mesh ID
    std::unordered_map<MeshId, MeshId> physical_to_logical_mesh;
    for (const auto& [logical_mesh_id, physical_mesh_id] : mesh_mappings) {
        physical_to_logical_mesh[physical_mesh_id] = logical_mesh_id;
    }

    // Get the valid physical exit nodes for each mesh direction
    // Map them by physical mesh ID (not logical mesh ID) since we'll look them up by physical mesh ID later
    // Only process exit nodes for physical meshes that are mapped to logical meshes
    for (const auto& src_exit_node : physical_exit_node_graph.get_nodes()) {
        // Skip if source physical mesh is not mapped to any logical mesh
        if (!physical_to_logical_mesh.contains(src_exit_node.mesh_id)) {
            continue;
        }

        // Get the valid logical exit nodes for this source exit node
        const auto& dst_exit_nodes = physical_exit_node_graph.get_neighbors(src_exit_node);

        // Loop through all destination exit nodes (can be multiple)
        // Only process exit nodes where both source and destination physical meshes are mapped
        for (const auto& dst_exit_node : dst_exit_nodes) {
            // Skip if destination physical mesh is not mapped to any logical mesh
            // This can happen when there are more physical meshes than logical meshes
            if (!physical_to_logical_mesh.contains(dst_exit_node.mesh_id)) {
                continue;
            }

            // Use the mapped physical mesh ID as the key (which is the same as dst_exit_node.mesh_id)
            valid_physical_exit_nodes_by_mesh[dst_exit_node.mesh_id].insert(src_exit_node.asic_id);
        }
    }

    for (const auto& src_exit_node : logical_exit_node_graph.get_nodes()) {
        const auto& dst_exit_nodes = logical_exit_node_graph.get_neighbors(src_exit_node);

        if (src_exit_node.fabric_node_id.has_value()) {
            // Fabric node-level: parallel edges to the same destination mesh share one required constraint.
            // num_logical_exit_nodes_assigned counts duplicate exit edges per (fabric node, logical dst mesh); it
            // cannot exceed the number of physical exit ASICs toward that destination mesh.
            std::map<std::pair<FabricNodeId, MeshId>, uint32_t> num_logical_exit_nodes_assigned_per_fabric_dst;
            for (const auto& dst_exit_node : dst_exit_nodes) {
                num_logical_exit_nodes_assigned_per_fabric_dst[{
                    src_exit_node.fabric_node_id.value(), dst_exit_node.mesh_id}]++;
            }
            for (const auto& [fabric_dst_key, num_logical_exit_nodes_assigned] :
                 num_logical_exit_nodes_assigned_per_fabric_dst) {
                const auto& [fabric_node_id, dst_logical_mesh] = fabric_dst_key;
                auto mesh_mapping_it = mesh_mappings.find(dst_logical_mesh);
                TT_ASSERT(
                    mesh_mapping_it != mesh_mappings.end(),
                    "Mesh mapping missing for logical mesh ID {} (destination exit node mesh ID)",
                    dst_logical_mesh.get());

                const auto& mapped_physical_dst_mesh_id = mesh_mappings.at(dst_logical_mesh);
                auto valid_physical_exit_nodes_it = valid_physical_exit_nodes_by_mesh.find(mapped_physical_dst_mesh_id);
                if (valid_physical_exit_nodes_it == valid_physical_exit_nodes_by_mesh.end()) {
                    return fmt::format(
                        "no physical exit ASICs toward physical mesh {} (logical destination mesh {}) for "
                        "fabric node (mesh {}, chip {})",
                        mapped_physical_dst_mesh_id.get(),
                        dst_logical_mesh.get(),
                        fabric_node_id.mesh_id.get(),
                        fabric_node_id.chip_id);
                }
                const auto& valid_physical_exit_nodes = valid_physical_exit_nodes_it->second;
                if (num_logical_exit_nodes_assigned > valid_physical_exit_nodes.size()) {
                    return fmt::format(
                        "fabric node (mesh {}, chip {}) has {} logical exit(s) toward logical mesh {} but only "
                        "{} physical exit ASIC(s) toward mapped physical mesh {}",
                        fabric_node_id.mesh_id.get(),
                        fabric_node_id.chip_id,
                        num_logical_exit_nodes_assigned,
                        dst_logical_mesh.get(),
                        valid_physical_exit_nodes.size(),
                        mapped_physical_dst_mesh_id.get());
                }
                if (!intra_mesh_constraints.add_required_constraint(fabric_node_id, valid_physical_exit_nodes)) {
                    return fmt::format(
                        "required exit-node constraint infeasible for fabric node (mesh {}, chip {}) toward "
                        "logical mesh {} ({} candidate physical exit ASIC(s))",
                        fabric_node_id.mesh_id.get(),
                        fabric_node_id.chip_id,
                        dst_logical_mesh.get(),
                        valid_physical_exit_nodes.size());
                }
            }
            continue;
        }

        // Mesh-level: one cardinality constraint per destination logical mesh. Each duplicate neighbor is one logical
        // inter-mesh channel. Per-ASIC parallel link counts and total link count come only from the physical exit graph
        // toward the mapped physical destination mesh. In RELAXED mode, channel demand for pair math is capped by that
        // physical link total (not logical multiplicity alone).
        std::map<MeshId, uint32_t> num_logical_exit_nodes_assigned_per_dst_mesh;
        for (const auto& dst_exit_node : dst_exit_nodes) {
            num_logical_exit_nodes_assigned_per_dst_mesh[dst_exit_node.mesh_id]++;
        }

        for (const auto& [dst_logical_mesh, num_logical_exit_nodes_assigned] :
             num_logical_exit_nodes_assigned_per_dst_mesh) {
            auto mesh_mapping_it = mesh_mappings.find(dst_logical_mesh);
            TT_ASSERT(
                mesh_mapping_it != mesh_mappings.end(),
                "Mesh mapping missing for logical mesh ID {} (destination exit node mesh ID)",
                dst_logical_mesh.get());

            const auto& mapped_physical_dst_mesh_id = mesh_mappings.at(dst_logical_mesh);
            auto valid_physical_exit_nodes_it = valid_physical_exit_nodes_by_mesh.find(mapped_physical_dst_mesh_id);
            if (valid_physical_exit_nodes_it == valid_physical_exit_nodes_by_mesh.end()) {
                return fmt::format(
                    "no physical exit ASICs toward physical mesh {} (logical destination mesh {})",
                    mapped_physical_dst_mesh_id.get(),
                    dst_logical_mesh.get());
            }
            const auto& valid_physical_exit_nodes = valid_physical_exit_nodes_it->second;

            const size_t max_mappable_exit_pairs =
                std::min(valid_logical_exit_nodes.size(), valid_physical_exit_nodes.size());

            const uint32_t total_physical_links_toward_dst =
                total_physical_exit_edges_toward_mesh(physical_exit_node_graph, mapped_physical_dst_mesh_id);
            const uint32_t max_edges_per_exit_asic =
                max_physical_exit_edges_per_asic_toward_mesh(physical_exit_node_graph, mapped_physical_dst_mesh_id);
            const uint32_t physical_links_per_exit_asic = std::max(1u, max_edges_per_exit_asic);

            uint32_t channels_for_pair_count = num_logical_exit_nodes_assigned;
            if (inter_mesh_validation_mode == ::tt::tt_fabric::ConnectionValidationMode::RELAXED) {
                channels_for_pair_count = std::min(num_logical_exit_nodes_assigned, total_physical_links_toward_dst);
            }

            const uint32_t required_exit_pair_count =
                (channels_for_pair_count + physical_links_per_exit_asic - 1) / physical_links_per_exit_asic;

            uint32_t effective_exit_pair_min_count = required_exit_pair_count;
            if (inter_mesh_validation_mode == ::tt::tt_fabric::ConnectionValidationMode::RELAXED) {
                effective_exit_pair_min_count = static_cast<uint32_t>(
                    std::min(static_cast<size_t>(required_exit_pair_count), max_mappable_exit_pairs));
                if (effective_exit_pair_min_count < num_logical_exit_nodes_assigned) {
                    log_debug(
                        tt::LogFabric,
                        "Relaxed mode: mesh-level exit toward logical mesh {}: {} logical channel(s), {} physical "
                        "link(s) toward mapped mesh → {} channel(s) for pair math (up to {} parallel link(s)/exit "
                        "ASIC); need at least {} (fabric_node, exit-ASIC) pair(s); exit cardinality min_count {} "
                        "(mappable pair cap {}).",
                        dst_logical_mesh.get(),
                        num_logical_exit_nodes_assigned,
                        total_physical_links_toward_dst,
                        channels_for_pair_count,
                        physical_links_per_exit_asic,
                        required_exit_pair_count,
                        effective_exit_pair_min_count,
                        max_mappable_exit_pairs);
                }
            } else if (required_exit_pair_count > max_mappable_exit_pairs) {
                return fmt::format(
                    "mesh-level exit toward logical mesh {} needs {} (fabric_node, exit-ASIC) pair(s) but only "
                    "{} pair(s) are mappable ({} logical exit node(s), {} physical exit ASIC(s), {} physical "
                    "link(s), {} link(s)/exit ASIC)",
                    dst_logical_mesh.get(),
                    required_exit_pair_count,
                    max_mappable_exit_pairs,
                    valid_logical_exit_nodes.size(),
                    valid_physical_exit_nodes.size(),
                    total_physical_links_toward_dst,
                    physical_links_per_exit_asic);
            }

            if (effective_exit_pair_min_count == 0) {
                return fmt::format(
                    "mesh-level exit toward logical mesh {} has zero effective exit pairs ({} logical "
                    "channel(s), {} physical link(s))",
                    dst_logical_mesh.get(),
                    num_logical_exit_nodes_assigned,
                    total_physical_links_toward_dst);
            }

            if (!intra_mesh_constraints.add_cardinality_constraint(
                    valid_logical_exit_nodes, valid_physical_exit_nodes, effective_exit_pair_min_count)) {
                return fmt::format(
                    "exit-node cardinality constraint infeasible toward logical mesh {} (min_count={}, {} "
                    "logical exit node(s), {} physical exit ASIC(s))",
                    dst_logical_mesh.get(),
                    effective_exit_pair_min_count,
                    valid_logical_exit_nodes.size(),
                    valid_physical_exit_nodes.size());
            }
        }
    }

    return std::nullopt;
}

}  // anonymous namespace

namespace {

void print_logical_adjacency_map(const LogicalMultiMeshGraph& multi_mesh_graph) {
    multi_mesh_graph.mesh_level_graph_.print_adjacency_map("Logical Mesh-Level Graph", true);
    for (const auto& [mesh_id, graph] : multi_mesh_graph.mesh_adjacency_graphs_) {
        graph.print_adjacency_map(fmt::format("Logical Mesh {} Internal Graph", mesh_id.get()), true);
    }
}

void print_physical_adjacency_map(const PhysicalMultiMeshGraph& multi_mesh_graph) {
    multi_mesh_graph.mesh_level_graph_.print_adjacency_map("Physical Mesh-Level Graph", true);
    for (const auto& [mesh_id, graph] : multi_mesh_graph.mesh_adjacency_graphs_) {
        graph.print_adjacency_map(fmt::format("Physical Mesh {} Internal Graph", mesh_id.get()), true);
    }
}

void record_failing_meshes(TopologyMappingResult& result, MeshId logical_mesh_id, MeshId physical_mesh_id) {
    result.stats.failing_logical_mesh = logical_mesh_id.get();
    result.stats.failing_physical_mesh = physical_mesh_id.get();
}

void apply_placement_stats(TopologyMappingStats& stats, const ::tt::tt_fabric::PlacementSolveStats& placement) {
    stats.placement_attempted = placement.master_solve_attempted;
    stats.placement_success = placement.master_solve_success;
    stats.candidate_lists_complete = placement.candidate_lists_complete;
    stats.meshes_total = placement.meshes_total;
    stats.meshes_placed = placement.meshes_placed;
    stats.placement_candidates = placement.master_candidates_enumerated;
    stats.placement_growth_rounds = placement.master_growth_rounds;
    stats.placement_sat_attempts = placement.master_sat_attempts;
    stats.placement_sat_vars = placement.master_sat_vars;
    stats.placement_sat_clauses = placement.master_sat_clauses;
    stats.inner_solver_calls = placement.inner_solver_calls;
    stats.candidates_generated = placement.candidates_generated;
}

std::string format_sat_placement_failure(const ::tt::tt_fabric::PlacementSolveStats& stats) {
    if (!stats.master_solve_attempted) {
        return "SAT joint placement did not start: no MESH groupings or empty logical graph";
    }
    if (stats.master_sat_attempts == 0 && !stats.master_solve_success) {
        return fmt::format(
            "SAT joint placement session was not ready (a mesh has no grouping variants); meshes_total={}",
            stats.meshes_total);
    }
    return fmt::format(
        "SAT joint placement: no placement found after {} attempt(s) and {} growth cycle(s) over {} "
        "candidate(s); candidate lists {} -- the UNSAT verdict is {}",
        stats.master_sat_attempts,
        stats.master_growth_rounds,
        stats.master_candidates_enumerated,
        stats.candidate_lists_complete ? "COMPLETE" : "TRUNCATED",
        stats.candidate_lists_complete ? "trustworthy" : "NOT trustworthy");
}

void set_mapping_failure(TopologyMappingResult& result, std::string failure_stage, std::string error_message) {
    result.success = false;
    result.stats.failure_stage = std::move(failure_stage);
    result.error_message = std::move(error_message);
    result.stats.warnings.push_back(result.error_message);
    log_warning(tt::LogFabric, "{}", result.error_message);
}

}  // namespace

std::string TopologyMappingStats::to_string() const {
    return fmt::format(
        "TopologyMappingStats:\n"
        "  failure_stage: {}  failing meshes: logical={} physical={}\n"
        "  seated grouping: '{}' type '{}'\n"
        "  intra: {} n_target={} n_global={} required={} preferred={}/{} dfs_calls={} sat_solves={}\n"
        "  placement: attempted={} success={} lists_complete={} meshes={}/{} candidates={} "
        "growth={} sat_attempts={} vars={} clauses={}\n"
        "  inner_solver_calls={} candidates_generated={} warnings={}",
        failure_stage.empty() ? "-" : failure_stage,
        failing_logical_mesh.has_value() ? std::to_string(*failing_logical_mesh) : "-",
        failing_physical_mesh.has_value() ? std::to_string(*failing_physical_mesh) : "-",
        seated_grouping_name,
        seated_grouping_type,
        intra_used_sat ? "SAT" : "DFS",
        intra_n_target,
        intra_n_global,
        intra_required_satisfied,
        intra_preferred_satisfied,
        intra_preferred_total,
        intra_dfs_calls,
        intra_sat_solve_calls,
        placement_attempted,
        placement_success,
        candidate_lists_complete,
        meshes_placed,
        meshes_total,
        placement_candidates,
        placement_growth_rounds,
        placement_sat_attempts,
        placement_sat_vars,
        placement_sat_clauses,
        inner_solver_calls,
        candidates_generated,
        warnings.size());
}

std::optional<std::vector<std::pair<FabricNodeId, FabricNodeId>>> assign_non_colliding_hops(
    const std::vector<std::vector<std::pair<FabricNodeId, FabricNodeId>>>& candidates) {
    using HopPair = std::pair<FabricNodeId, FabricNodeId>;
    const std::size_t num_hops = candidates.size();

    // Visit the most-constrained sets first (fewest candidates) so the search prunes quickly.
    std::vector<std::size_t> visit_order(num_hops);
    for (std::size_t i = 0; i < num_hops; i++) {
        visit_order[i] = i;
    }
    std::stable_sort(visit_order.begin(), visit_order.end(), [&](std::size_t a, std::size_t b) {
        return candidates[a].size() < candidates[b].size();
    });

    std::vector<std::optional<HopPair>> selected(num_hops);
    std::set<FabricNodeId> used_nodes;
    std::function<bool(std::size_t)> assign = [&](std::size_t k) -> bool {
        if (k == num_hops) {
            return true;
        }
        const std::size_t hop = visit_order[k];
        for (const auto& pair : candidates[hop]) {
            if (used_nodes.contains(pair.first) || used_nodes.contains(pair.second)) {
                continue;
            }
            selected[hop] = pair;
            used_nodes.insert(pair.first);
            used_nodes.insert(pair.second);
            if (assign(k + 1)) {
                return true;
            }
            used_nodes.erase(pair.first);
            used_nodes.erase(pair.second);
            selected[hop].reset();
        }
        return false;
    };
    if (!assign(0)) {
        return std::nullopt;
    }

    std::vector<HopPair> hops;
    hops.reserve(num_hops);
    for (auto& hop : selected) {
        hops.push_back(*hop);
    }
    return hops;
}

// Complete the intra-mesh (fabric-node -> ASIC) mapping for one fixed inter-mesh placement.
//
// Complete the intra-mesh (fabric-node -> ASIC) mapping for one fixed identity placement.
// If any mesh pair's intra-mesh mapping is infeasible, the whole placement is rejected
// (returned result has success == false). The enumerator forbids/retries rejected placements.
TopologyMappingResult complete_intra_mesh_for_placement(
    const std::unordered_map<MeshId, MeshId>& mesh_mappings,
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const TopologyMappingConfig& config,
    ::tt::tt_fabric::ConnectionValidationMode inter_mesh_validation_mode,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank,
    std::optional<std::pair<MeshId, MeshId>>* failing_pair_out) {
    using namespace ::tt::tt_fabric;

    TopologyMappingResult result;

    for (const auto& [logical_mesh_id, physical_mesh_id] : mesh_mappings) {
        const auto& logical_graph = adjacency_map_logical.mesh_adjacency_graphs_.at(logical_mesh_id);
        const auto& physical_graph = adjacency_map_physical.mesh_adjacency_graphs_.at(physical_mesh_id);

        // Exit-node graphs (use a shared empty graph when a mesh has none, as elsewhere in this file).
        const AdjacencyGraph<LogicalExitNode>* logical_exit_node_graph_ptr = nullptr;
        auto logical_exit_node_it = adjacency_map_logical.mesh_exit_node_graphs_.find(logical_mesh_id);
        if (logical_exit_node_it != adjacency_map_logical.mesh_exit_node_graphs_.end()) {
            logical_exit_node_graph_ptr = &logical_exit_node_it->second;
        } else {
            static const AdjacencyGraph<LogicalExitNode> empty_logical_exit_node_graph;
            logical_exit_node_graph_ptr = &empty_logical_exit_node_graph;
        }
        const auto& logical_exit_node_graph = *logical_exit_node_graph_ptr;

        const AdjacencyGraph<PhysicalExitNode>* physical_exit_node_graph_ptr = nullptr;
        auto physical_exit_node_it = adjacency_map_physical.mesh_exit_node_graphs_.find(physical_mesh_id);
        if (physical_exit_node_it != adjacency_map_physical.mesh_exit_node_graphs_.end()) {
            physical_exit_node_graph_ptr = &physical_exit_node_it->second;
        } else {
            static const AdjacencyGraph<PhysicalExitNode> empty_physical_exit_node_graph;
            physical_exit_node_graph_ptr = &empty_physical_exit_node_graph;
        }
        const auto& physical_exit_node_graph = *physical_exit_node_graph_ptr;

        ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID> intra_mesh_constraints;

        if (!config.disable_rank_bindings) {
            add_rank_binding_constraints(
                intra_mesh_constraints, config, logical_mesh_id, fabric_node_id_to_mesh_rank, asic_id_to_mesh_rank);
        }

        if (!logical_exit_node_graph.get_nodes().empty() && !physical_exit_node_graph.get_nodes().empty()) {
            auto exit_node_constraints_failure = add_exit_node_constraints(
                intra_mesh_constraints,
                mesh_mappings,
                logical_graph,
                logical_exit_node_graph,
                physical_exit_node_graph,
                inter_mesh_validation_mode);
            if (exit_node_constraints_failure.has_value()) {
                record_failing_meshes(result, logical_mesh_id, physical_mesh_id);
                set_mapping_failure(
                    result,
                    "exit_node_constraints",
                    fmt::format(
                        "intra-mesh mapping failed: exit-node constraints for logical mesh {} -> physical mesh {}: {}",
                        logical_mesh_id.get(),
                        physical_mesh_id.get(),
                        *exit_node_constraints_failure));
                if (failing_pair_out) {
                    *failing_pair_out = {logical_mesh_id, physical_mesh_id};
                }
                return result;
            }
        }

        auto asic_positions_to_asic_ids = build_asic_positions_map(physical_graph, config);
        auto pinning_constraint_failure =
            add_pinning_constraints(intra_mesh_constraints, asic_positions_to_asic_ids, config, logical_mesh_id);
        if (pinning_constraint_failure.has_value()) {
            record_failing_meshes(result, logical_mesh_id, physical_mesh_id);
            set_mapping_failure(
                result,
                "mgd_pinning",
                fmt::format(
                    "intra-mesh mapping failed: MGD pinning constraints for logical mesh {} -> physical mesh {}: {}",
                    logical_mesh_id.get(),
                    physical_mesh_id.get(),
                    *pinning_constraint_failure));
            if (failing_pair_out) {
                *failing_pair_out = {logical_mesh_id, physical_mesh_id};
            }
            return result;
        }

        // Mirror the single-solution path: when the physical graph carries PGD-derived pinnings, bias the
        // intra-mesh solve toward the PGD-chosen layout via PREFERRED (soft) constraints, added after the hard
        // rank/exit/MGD-pin constraints so they only influence ASIC choice where hard constraints leave freedom.
        if (!adjacency_map_physical.mesh_pgd_pinnings_.empty()) {
            const auto& physical_mesh_nodes = physical_graph.get_nodes();
            const std::unordered_set<tt::tt_metal::AsicID> physical_mesh_node_set(
                physical_mesh_nodes.begin(), physical_mesh_nodes.end());
            add_pgd_pinning_preferred_constraints(
                intra_mesh_constraints,
                adjacency_map_physical.mesh_pgd_pinnings_,
                asic_positions_to_asic_ids,
                physical_mesh_node_set,
                logical_mesh_id,
                physical_mesh_id);
        }

        auto validation_mode = determine_intra_mesh_validation_mode(config, logical_mesh_id);

        auto sub_mapping = ::tt::tt_fabric::solve_topology_mapping(
            logical_graph, physical_graph, intra_mesh_constraints, validation_mode, /*quiet_mode=*/true);
        if (!sub_mapping.success) {
            record_failing_meshes(result, logical_mesh_id, physical_mesh_id);
            result.stats.intra_used_sat = sub_mapping.stats.used_sat;
            result.stats.intra_n_target = sub_mapping.stats.n_target;
            result.stats.intra_n_global = sub_mapping.stats.n_global;
            result.stats.intra_required_satisfied = sub_mapping.constraint_stats.required_satisfied;
            result.stats.intra_preferred_satisfied = sub_mapping.constraint_stats.preferred_satisfied;
            result.stats.intra_preferred_total = sub_mapping.constraint_stats.preferred_total;
            result.stats.intra_dfs_calls = sub_mapping.stats.dfs_calls;
            result.stats.intra_sat_solve_calls = sub_mapping.stats.sat_solve_calls;
            result.stats.warnings.insert(
                result.stats.warnings.end(), sub_mapping.warnings.begin(), sub_mapping.warnings.end());
            const std::string solver_error =
                sub_mapping.error_message.empty() ? "intra-mesh solver returned no mapping" : sub_mapping.error_message;
            set_mapping_failure(
                result,
                "intra_mesh_solve",
                fmt::format(
                    "intra-mesh mapping failed: logical mesh {} ({} node(s)) -> physical mesh {} ({} asic(s)): {} "
                    "[{} required={} preferred={}/{} n_target={} n_global={}]",
                    logical_mesh_id.get(),
                    logical_graph.get_nodes().size(),
                    physical_mesh_id.get(),
                    physical_graph.get_nodes().size(),
                    solver_error,
                    sub_mapping.stats.used_sat ? "SAT" : "DFS",
                    sub_mapping.constraint_stats.required_satisfied,
                    sub_mapping.constraint_stats.preferred_satisfied,
                    sub_mapping.constraint_stats.preferred_total,
                    sub_mapping.stats.n_target,
                    sub_mapping.stats.n_global));
            for (const auto& [fabric_node, asic] : sub_mapping.target_to_global) {
                result.fabric_node_to_asic.insert({fabric_node, asic});
                result.asic_to_fabric_node.insert({asic, fabric_node});
            }
            if (failing_pair_out) {
                *failing_pair_out = {logical_mesh_id, physical_mesh_id};
            }
            return result;
        }
        for (const auto& [fabric_node, asic] : sub_mapping.target_to_global) {
            result.fabric_node_to_asic.insert({fabric_node, asic});
            result.asic_to_fabric_node.insert({asic, fabric_node});
        }
    }

    result.success = true;
    return result;
}

// ─────────────────────── MultiMeshSolutionEnumerator (SAT seating + identity intra) ───────────────────────
void MultiMeshSolutionEnumerator::fill_host_and_asic_positions_from_psd() {
    if (physical_system_descriptor_ == nullptr) {
        return;
    }
    const bool fill_hosts = config_.hostname_to_asics.empty();
    const bool fill_positions = config_.asic_positions.empty();
    if (!fill_hosts && !fill_positions) {
        return;
    }
    for (const auto& [asic_id, desc] : physical_system_descriptor_->get_asic_descriptors()) {
        if (fill_hosts) {
            config_.hostname_to_asics[desc.host_name].insert(asic_id);
        }
        if (fill_positions) {
            config_.asic_positions[asic_id] = std::make_pair(desc.tray_id, desc.asic_location);
        }
    }
}

MultiMeshSolutionEnumerator::MultiMeshSolutionEnumerator(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const TopologyMappingConfig& config,
    bool unique_shapes,
    const std::optional<PinningsByMesh>& pinnings,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) :
    MultiMeshSolutionEnumerator(
        physical_system_descriptor,
        physical_grouping_descriptor,
        std::vector<MultiMeshMappingPart>{
            MultiMeshMappingPart{&mesh_graph_descriptor, pinnings, fabric_node_id_to_mesh_rank, asic_id_to_mesh_rank}},
        config,
        unique_shapes) {}

::tt::tt_fabric::MeshGraphDescriptor MultiMeshSolutionEnumerator::init_from_parts(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::vector<MultiMeshMappingPart>& parts,
    std::optional<PinningsByMesh>& session_pinnings) {
    using namespace ::tt::tt_fabric;

    std::vector<const MeshGraphDescriptor*> mesh_graph_descriptors;
    mesh_graph_descriptors.reserve(parts.size());
    for (const auto& part : parts) {
        TT_FATAL(part.mesh_graph_descriptor != nullptr, "MultiMeshMappingPart is missing a mesh graph descriptor");
        mesh_graph_descriptors.push_back(part.mesh_graph_descriptor);
    }
    TT_FATAL(
        !mesh_graph_descriptors.empty(), "MultiMeshSolutionEnumerator requires at least one mesh graph descriptor");

    validate_shared_inter_mesh_policy(mesh_graph_descriptors);

    MeshGraphDescriptor merged =
        MeshGraphDescriptor::merge(mesh_graph_descriptors, &per_part_local_to_global_mesh_ids_);
    logical_ = build_logical_multi_mesh_adjacency_graph(merged);

    if (config_.mesh_validation_modes.empty()) {
        for (std::size_t mgd_index = 0; mgd_index < per_part_local_to_global_mesh_ids_.size(); ++mgd_index) {
            const MeshGraphDescriptor& mgd = *mesh_graph_descriptors[mgd_index];
            for (const auto& [local_mesh_id, global_mesh_id] : per_part_local_to_global_mesh_ids_[mgd_index]) {
                config_.mesh_validation_modes[global_mesh_id] = mgd.is_intra_mesh_policy_relaxed(local_mesh_id)
                                                                    ? ConnectionValidationMode::RELAXED
                                                                    : ConnectionValidationMode::STRICT;
            }
        }
    }
    // Local pinnings and host ranks from each part are remapped with the merge maps.
    if (config_.pinnings.empty()) {
        for (std::size_t mgd_index = 0;
             mgd_index < parts.size() && mgd_index < per_part_local_to_global_mesh_ids_.size();
             ++mgd_index) {
            if (!parts[mgd_index].pinnings.has_value()) {
                continue;
            }
            const auto& local_to_global = per_part_local_to_global_mesh_ids_[mgd_index];
            for (const auto& [_, groups] : *parts[mgd_index].pinnings) {
                for (const auto& group : groups) {
                    ::tt::tt_fabric::AsicPinningGroup remapped;
                    remapped.asic_positions = group.asic_positions;
                    remapped.board_revision = group.board_revision;
                    remapped.fabric_nodes.reserve(group.fabric_nodes.size());
                    for (const auto& fabric_node : group.fabric_nodes) {
                        remapped.fabric_nodes.emplace_back(
                            remap_local_mesh_id(local_to_global, fabric_node.mesh_id), fabric_node.chip_id);
                    }
                    config_.pinnings.push_back(std::move(remapped));
                }
            }
        }
    }
    for (std::size_t mgd_index = 0; mgd_index < parts.size() && mgd_index < per_part_local_to_global_mesh_ids_.size();
         ++mgd_index) {
        remap_fabric_node_ranks_to_global(
            fabric_node_id_to_mesh_rank_,
            parts[mgd_index].fabric_node_id_to_mesh_rank,
            per_part_local_to_global_mesh_ids_[mgd_index]);
        remap_asic_ranks_to_global(
            asic_id_to_mesh_rank_,
            parts[mgd_index].asic_id_to_mesh_rank,
            per_part_local_to_global_mesh_ids_[mgd_index]);
    }
    if (!config_.inter_mesh_validation_mode.has_value()) {
        std::optional<ConnectionValidationMode> specified_mode;
        for (const auto* mgd : mesh_graph_descriptors) {
            if (!mgd->is_inter_mesh_policy_specified()) {
                continue;
            }
            specified_mode = mgd->is_inter_mesh_policy_relaxed() ? ConnectionValidationMode::RELAXED
                                                                 : ConnectionValidationMode::STRICT;
            break;
        }
        config_.inter_mesh_validation_mode = specified_mode.value_or(ConnectionValidationMode::STRICT);
    }
    inter_mesh_validation_mode_ = determine_inter_mesh_validation_mode(config_);
    fill_host_and_asic_positions_from_psd();
    flat_graph_ = AdjacencyGraph<tt::tt_metal::AsicID>(build_flat_adjacency_map_from_psd(physical_system_descriptor));

    // Session keys pinnings by merged/global MeshId (remapped above).
    PinningsByMesh by_mesh = merged.get_pinnings();
    drop_inactive_revision_pinnings(by_mesh, physical_system_descriptor);
    if (!config_.pinnings.empty()) {
        merge_pinnings_by_mesh(by_mesh, config_.pinnings);
        drop_inactive_revision_pinnings(by_mesh, physical_system_descriptor);
    }
    if (!by_mesh.empty()) {
        session_pinnings = std::move(by_mesh);
    }
    return merged;
}

MultiMeshSolutionEnumerator::MultiMeshSolutionEnumerator(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const std::vector<MultiMeshMappingPart>& parts,
    const TopologyMappingConfig& config,
    bool unique_shapes) :
    physical_system_descriptor_(&physical_system_descriptor), config_(config) {
    using namespace ::tt::tt_fabric;
    std::optional<PinningsByMesh> session_pinnings;
    MeshGraphDescriptor merged = init_from_parts(physical_system_descriptor, parts, session_pinnings);
    placement_stats_ = std::make_unique<PlacementSolveStats>();
    placement_session_ = std::make_unique<SatPlacementEnumerationSession>(
        physical_grouping_descriptor,
        merged,
        physical_system_descriptor,
        placement_stats_.get(),
        session_pinnings,
        asic_id_to_mesh_rank_,
        unique_shapes,
        fabric_node_id_to_mesh_rank_);
}

MultiMeshSolutionEnumerator::MultiMeshSolutionEnumerator(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const TopologyMappingConfig& config,
    bool unique_shapes,
    const std::optional<PinningsByMesh>& pinnings,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) :
    physical_system_descriptor_(&physical_system_descriptor), config_(config) {
    using namespace ::tt::tt_fabric;
    const std::vector<MultiMeshMappingPart> parts{
        MultiMeshMappingPart{&mesh_graph_descriptor, pinnings, fabric_node_id_to_mesh_rank, asic_id_to_mesh_rank}};
    std::optional<PinningsByMesh> session_pinnings;
    MeshGraphDescriptor merged = init_from_parts(physical_system_descriptor, parts, session_pinnings);
    placement_stats_ = std::make_unique<PlacementSolveStats>();
    placement_session_ = std::make_unique<SatPlacementEnumerationSession>(
        merged,
        physical_system_descriptor,
        placement_stats_.get(),
        session_pinnings,
        asic_id_to_mesh_rank_,
        unique_shapes);
}

MultiMeshSolutionEnumerator::MultiMeshSolutionEnumerator(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::vector<MultiMeshMappingPart>& parts,
    const TopologyMappingConfig& config,
    bool unique_shapes) :
    physical_system_descriptor_(&physical_system_descriptor), config_(config) {
    using namespace ::tt::tt_fabric;
    std::optional<PinningsByMesh> session_pinnings;
    MeshGraphDescriptor merged = init_from_parts(physical_system_descriptor, parts, session_pinnings);
    placement_stats_ = std::make_unique<PlacementSolveStats>();
    placement_session_ = std::make_unique<SatPlacementEnumerationSession>(
        merged,
        physical_system_descriptor,
        placement_stats_.get(),
        session_pinnings,
        asic_id_to_mesh_rank_,
        unique_shapes);
}

MultiMeshSolutionEnumerator::MultiMeshSolutionEnumerator(MultiMeshSolutionEnumerator&&) noexcept = default;
MultiMeshSolutionEnumerator& MultiMeshSolutionEnumerator::operator=(MultiMeshSolutionEnumerator&&) noexcept = default;
MultiMeshSolutionEnumerator::~MultiMeshSolutionEnumerator() = default;

std::vector<TopologyMappingResult> MultiMeshSolutionEnumerator::unsuccessful_parts() const {
    TopologyMappingResult result;
    if (last_failed_.has_value()) {
        result = *last_failed_;
    }
    result.success = false;
    if (placement_stats_ != nullptr) {
        apply_placement_stats(result.stats, *placement_stats_);
        const std::string sat_status = format_sat_placement_failure(*placement_stats_);
        result.stats.warnings.push_back(sat_status);
        result.stats.warnings.push_back(placement_stats_->to_string());
        result.stats.warnings.push_back(result.stats.to_string());
        if (result.error_message.empty()) {
            result.stats.failure_stage = "sat_placement";
            result.error_message =
                fmt::format("map_multi_mesh_to_physical: no valid placement+intra-mesh mapping found: {}", sat_status);
        }
    }
    if (result.error_message.empty()) {
        result.stats.failure_stage = "sat_placement";
        result.error_message = "map_multi_mesh_to_physical: no valid placement+intra-mesh mapping found";
    }
    log_warning(tt::LogFabric, "{}", result.error_message);
    return split_mapping_to_local_parts(result, per_part_local_to_global_mesh_ids_);
}

std::vector<TopologyMappingResult> MultiMeshSolutionEnumerator::next() {
    using namespace ::tt::tt_fabric;
    if (placement_session_ == nullptr || physical_system_descriptor_ == nullptr) {
        if (emitted_ == 0 && !yielded_failure_) {
            yielded_failure_ = true;
            return unsuccessful_parts();
        }
        return {};
    }

    auto give_up = [&]() -> std::vector<TopologyMappingResult> {
        if (emitted_ == 0 && !yielded_failure_) {
            yielded_failure_ = true;
            return unsuccessful_parts();
        }
        return {};
    };

    while (true) {
        AssignedMeshes seating = placement_session_->next();
        if (seating.empty()) {
            return give_up();
        }

        PhysicalMultiMeshGraph physical = build_hierarchical_from_flat_graph(flat_graph_, seating);
        print_logical_adjacency_map(logical_);
        print_physical_adjacency_map(physical);
        std::unordered_map<MeshId, MeshId> identity;
        identity.reserve(seating.size());
        for (const PlacedMesh& placed : seating) {
            identity.emplace(placed.mesh_id, placed.mesh_id);
        }

        std::optional<std::pair<MeshId, MeshId>> intra_failing_pair;
        TopologyMappingResult full = complete_intra_mesh_for_placement(
            identity,
            logical_,
            physical,
            config_,
            inter_mesh_validation_mode_,
            asic_id_to_mesh_rank_,
            fabric_node_id_to_mesh_rank_,
            &intra_failing_pair);
        if (placement_stats_ != nullptr) {
            apply_placement_stats(full.stats, *placement_stats_);
        }
        if (full.success) {
            ++emitted_;
            return split_mapping_to_local_parts(full, per_part_local_to_global_mesh_ids_);
        }

        const PlacedMesh* failed_placed = nullptr;
        if (intra_failing_pair.has_value()) {
            for (const PlacedMesh& placed : seating) {
                if (placed.mesh_id == intra_failing_pair->first) {
                    failed_placed = &placed;
                    break;
                }
            }
        }
        if (failed_placed != nullptr) {
            full.stats.seated_grouping_name = failed_placed->grouping_name;
            full.stats.seated_grouping_type = failed_placed->grouping_type;
            full.error_message = fmt::format(
                "{} (seated as grouping '{}' type '{}' with {} asic(s))",
                full.error_message,
                failed_placed->grouping_name,
                failed_placed->grouping_type,
                failed_placed->placement.asics.size());
            full.stats.warnings.push_back(full.error_message);
            log_warning(tt::LogFabric, "{}", full.error_message);
        }
        last_failed_ = full;

        if (failed_placed == nullptr || failed_placed->placement.asics.empty()) {
            continue;
        }

        if (!placement_session_->add_forbidden_constraint(*failed_placed)) {
            return give_up();
        }
    }
}

TopologyMappingResult map_multi_mesh_to_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const TopologyMappingConfig& config,
    const std::optional<PinningsByMesh>& pinnings,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    const auto local_parts = map_multi_mesh_to_physical(
        physical_system_descriptor,
        physical_grouping_descriptor,
        std::vector<MultiMeshMappingPart>{
            MultiMeshMappingPart{&mesh_graph_descriptor, pinnings, fabric_node_id_to_mesh_rank, asic_id_to_mesh_rank}},
        config);
    if (local_parts.empty()) {
        TopologyMappingResult result;
        result.success = false;
        result.stats.failure_stage = "sat_placement";
        result.error_message = "map_multi_mesh_to_physical: no valid placement+intra-mesh mapping found";
        result.stats.warnings.push_back(result.error_message);
        log_warning(tt::LogFabric, "{}", result.error_message);
        return result;
    }
    return local_parts.front();
}

TopologyMappingResult map_multi_mesh_to_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const TopologyMappingConfig& config,
    const std::optional<PinningsByMesh>& pinnings,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    MultiMeshSolutionEnumerator enumerator(
        physical_system_descriptor,
        mesh_graph_descriptor,
        config,
        /*unique_shapes=*/false,
        pinnings,
        asic_id_to_mesh_rank,
        fabric_node_id_to_mesh_rank);
    const auto local_parts = enumerator.next();
    if (local_parts.empty()) {
        TopologyMappingResult result;
        result.success = false;
        result.stats.failure_stage = "sat_placement";
        result.error_message = "map_multi_mesh_to_physical: no valid placement+intra-mesh mapping found";
        result.stats.warnings.push_back(result.error_message);
        log_warning(tt::LogFabric, "{}", result.error_message);
        return result;
    }
    return local_parts.front();
}

std::vector<TopologyMappingResult> map_multi_mesh_to_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const std::vector<MultiMeshMappingPart>& parts,
    const TopologyMappingConfig& config) {
    MultiMeshSolutionEnumerator enumerator(
        physical_system_descriptor,
        physical_grouping_descriptor,
        parts,
        config,
        /*unique_shapes=*/false);
    return enumerator.next();
}

}  // namespace tt::tt_metal::experimental::tt_fabric
