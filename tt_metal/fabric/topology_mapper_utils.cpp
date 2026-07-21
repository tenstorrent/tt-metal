// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <limits>
#include <map>
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
#include <cstdlib>

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
std::vector<std::pair<FabricNodeId, std::vector<AsicPosition>>> get_galaxy_fixed_asic_position_pinnings_for_mesh(
    MeshId mesh_id,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    bool hard_pin_node_0,
    bool nw_corner_only) {
    std::vector<std::pair<FabricNodeId, std::vector<AsicPosition>>> fixed_asic_position_pinnings;

    // Sub-galaxy slices: pin only the NW corner (node 0) to any tray-corner ASIC (asic_location==1 on
    // trays 1..4). The host-rank partition may land on any tray, so tray 1 alone is unsatisfiable.
    if (nw_corner_only) {
        fixed_asic_position_pinnings.emplace_back(
            FabricNodeId{mesh_id, 0},
            std::vector<AsicPosition>{AsicPosition{1, 1}, AsicPosition{2, 1}, AsicPosition{3, 1}, AsicPosition{4, 1}});
        return fixed_asic_position_pinnings;
    }

    // Get all 4 possible corner ASIC positions
    std::vector<AsicPosition> corner_asic_positions;
    corner_asic_positions.emplace_back(AsicPosition{1, 1});  // Top left corner
    corner_asic_positions.emplace_back(AsicPosition{2, 1});  // Top right corner
    corner_asic_positions.emplace_back(AsicPosition{3, 1});  // Bottom left corner
    corner_asic_positions.emplace_back(AsicPosition{4, 1});  // Bottom right corner

    // Generate corner fabric node IDs for this mesh
    std::vector<FabricNodeId> corner_fabric_node_ids;
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, 0});
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, mesh_shape[1] - 1});
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, mesh_shape[1] * (mesh_shape[0] - 1)});
    corner_fabric_node_ids.emplace_back(FabricNodeId{mesh_id, (mesh_shape[1] * mesh_shape[0]) - 1});

    fixed_asic_position_pinnings.reserve(corner_fabric_node_ids.size());
    for (const auto& corner_fabric_node_id : corner_fabric_node_ids) {
        // Special case: Hard pin NW corner (fabric node id 0) to asic 1 tray 1 if requested.
        if (corner_fabric_node_id == FabricNodeId{mesh_id, 0} && hard_pin_node_0) {
            fixed_asic_position_pinnings.emplace_back(
                corner_fabric_node_id, std::vector<AsicPosition>{AsicPosition{1, 1}});
            continue;
        }

        fixed_asic_position_pinnings.emplace_back(corner_fabric_node_id, corner_asic_positions);
    }

    return fixed_asic_position_pinnings;
}

TopologyMappingResult map_mesh_to_physical(
    MeshId mesh_id,
    const LogicalAdjacencyMap& logical_adjacency,
    const PhysicalAdjacencyMap& physical_adjacency,
    const std::map<FabricNodeId, MeshHostRankId>& node_to_host_rank,
    const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_to_host_rank,
    const TopologyMappingConfig& config) {
    TopologyMappingResult result;

    using namespace ::tt::tt_fabric;

    // Convert maps to AdjacencyGraph format
    const AdjacencyGraph<FabricNodeId> target_graph(logical_adjacency);
    const AdjacencyGraph<tt::tt_metal::AsicID> global_graph(physical_adjacency);

    // Build constraints
    MappingConstraints<FabricNodeId, tt::tt_metal::AsicID> constraints;

    // Add mesh host rank constraints (trait-based constraint)
    if (!constraints.add_required_trait_constraint(node_to_host_rank, asic_to_host_rank)) {
        result.success = false;
        result.error_message = "Failed to add required trait constraint for mesh host rank";
        return result;
    }

    // Add pinning constraints if any
    // Build position trait maps for pinning constraints
    if (!config.pinnings.empty()) {
        std::map<FabricNodeId, AsicPosition> fabric_node_to_position;
        std::map<tt::tt_metal::AsicID, AsicPosition> asic_to_position;

        // Build fabric node to position map from pinnings
        for (const auto& [pos, fabric_node] : config.pinnings) {
            if (fabric_node.mesh_id != mesh_id) {
                continue;  // pin for another mesh
            }

            // Validate that the fabric node exists in logical adjacency
            bool found = logical_adjacency.contains(fabric_node);
            if (!found) {
                result.success = false;
                result.error_message =
                    fmt::format("Pinned fabric node {} not found in logical mesh {}", fabric_node, mesh_id.get());
                return result;
            }

            // Check for duplicate pinnings
            auto [it, inserted] = fabric_node_to_position.try_emplace(fabric_node, pos);
            if (!inserted) {
                const auto& prev_pos = it->second;
                result.success = false;
                result.error_message = fmt::format(
                    "Fabric node {} in mesh {} is pinned to multiple ASIC positions: (tray {}, loc {}) and (tray {}, "
                    "loc {})",
                    fabric_node,
                    mesh_id.get(),
                    *prev_pos.first,
                    *prev_pos.second,
                    *pos.first,
                    *pos.second);
                return result;
            }
        }

        // Build ASIC to position map from config.asic_positions
        std::unordered_set<tt::tt_metal::AsicID> physical_node_set;
        for (const auto& [asic_id, _] : physical_adjacency) {
            physical_node_set.insert(asic_id);
        }
        for (const auto& [asic_id, pos] : config.asic_positions) {
            // Only include ASICs that are in the physical adjacency graph
            if (physical_node_set.contains(asic_id)) {
                asic_to_position[asic_id] = pos;
            }
        }

        // Validate that all pinned positions exist
        for (const auto& [fabric_node, pos] : fabric_node_to_position) {
            bool found = false;
            for (const auto& [asic_id, asic_pos] : asic_to_position) {
                if (asic_pos.first == pos.first && asic_pos.second == pos.second) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                result.success = false;
                result.error_message = fmt::format(
                    "Pinned ASIC position (tray {}, loc {}) not found among physical ASICs participating in mesh {}",
                    *pos.first,
                    *pos.second,
                    mesh_id.get());
                return result;
            }
        }

        // Add position-based trait constraint
        if (!fabric_node_to_position.empty() && !asic_to_position.empty()) {
            // Convert AsicPosition to a comparable type for trait constraint
            // We'll use a string representation as the trait value
            std::map<FabricNodeId, std::string> fabric_node_traits;
            std::map<tt::tt_metal::AsicID, std::string> asic_traits;

            for (const auto& [fabric_node, pos] : fabric_node_to_position) {
                std::string trait = fmt::format("tray_{}_loc_{}", *pos.first, *pos.second);
                fabric_node_traits[fabric_node] = trait;
            }

            for (const auto& [asic_id, pos] : asic_to_position) {
                std::string trait = fmt::format("tray_{}_loc_{}", *pos.first, *pos.second);
                asic_traits[asic_id] = trait;
            }

            // Add required trait constraint for pinning
            if (!constraints.add_required_trait_constraint(fabric_node_traits, asic_traits)) {
                result.success = false;
                result.error_message = fmt::format(
                    "Failed to add required trait constraint for pinned ASIC positions in mesh {}", mesh_id.get());
                return result;
            }

            // Log pinnings
            std::string pinnings_str;
            bool first = true;
            for (const auto& [fabric_node, pos] : fabric_node_to_position) {
                if (!first) {
                    pinnings_str += ", ";
                }
                first = false;
                pinnings_str += fmt::format(
                    "fabric_node={} (mesh_id={}, chip_id={}) -> ASIC position (tray={}, loc={})",
                    fabric_node,
                    fabric_node.mesh_id.get(),
                    fabric_node.chip_id,
                    *pos.first,
                    *pos.second);
            }
            log_info(
                tt::LogFabric,
                "TopologyMapper: Using {} pinning(s) for mesh {}: [{}]",
                fabric_node_to_position.size(),
                mesh_id.get(),
                pinnings_str);
        }
    }

    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED;
    auto mode_it = config.mesh_validation_modes.find(mesh_id);
    if (mode_it != config.mesh_validation_modes.end()) {
        validation_mode = mode_it->second;
    }

    // Solve using topology solver
    // Catch exceptions from constraint validation and convert to failure result
    MappingResult<FabricNodeId, tt::tt_metal::AsicID> solver_result;
    try {
        solver_result = solve_topology_mapping(target_graph, global_graph, constraints, validation_mode);
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        return result;
    }

    // Convert result
    result.success = solver_result.success;
    result.error_message = solver_result.error_message;

    if (solver_result.success) {
        // Convert bidirectional mappings
        for (const auto& [target, global] : solver_result.target_to_global) {
            result.fabric_node_to_asic[target] = global;
            result.asic_to_fabric_node.emplace(global, target);
        }
    }

    return result;
}

std::map<MeshId, LogicalAdjacencyMap> build_adjacency_map_logical(const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    // Build adjacency graphs using topology solver
    auto adjacency_graphs = ::tt::tt_fabric::build_adjacency_graph_logical(mesh_graph);

    // Convert from AdjacencyGraph format to map format
    std::map<MeshId, LogicalAdjacencyMap> result;
    for (const auto& [mesh_id, graph] : adjacency_graphs) {
        LogicalAdjacencyMap logical_map;
        for (const auto& node : graph.get_nodes()) {
            logical_map[node] = graph.get_neighbors(node);
        }
        result[mesh_id] = logical_map;
    }
    return result;
}

std::map<MeshId, PhysicalAdjacencyMap> build_adjacency_map_physical(
    tt::tt_metal::ClusterType cluster_type,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // Build adjacency graphs using topology solver
    auto adjacency_graphs =
        ::tt::tt_fabric::build_adjacency_graph_physical(cluster_type, physical_system_descriptor, asic_id_to_mesh_rank);

    // Convert from AdjacencyGraph format to map format
    std::map<MeshId, PhysicalAdjacencyMap> result;
    for (const auto& [mesh_id, graph] : adjacency_graphs) {
        PhysicalAdjacencyMap physical_map;
        for (const auto& node : graph.get_nodes()) {
            physical_map[node] = graph.get_neighbors(node);
        }
        result[mesh_id] = physical_map;
    }
    return result;
}

namespace {

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

// Mesh-level FABRIC edges (same extraction as get_requested_intermesh_from_mgd) restricted to instances sharing
// one mesh descriptor name. Nodes are MeshId(instance.local_id) for those instances only.
AdjacencyGraph<MeshId> build_mgd_mesh_level_subgraph_for_mesh_descriptor_name(
    const ::tt::tt_fabric::MeshGraphDescriptor& mgd,
    const std::string& mesh_descriptor_name,
    const ::tt::tt_fabric::RequestedIntermeshConnections& intermesh_mesh_level_edges) {
    using namespace ::tt::tt_fabric;

    std::unordered_set<uint32_t> mesh_id_in_descriptor;
    for (GlobalNodeId global_id : mgd.instances_by_name(mesh_descriptor_name)) {
        const auto& inst = mgd.get_instance(global_id);
        if (inst.kind != NodeKind::Mesh) {
            continue;
        }
        mesh_id_in_descriptor.insert(inst.local_id);
    }

    AdjacencyGraph<MeshId>::AdjacencyMap adj;
    for (uint32_t mid : mesh_id_in_descriptor) {
        adj[MeshId(mid)] = {};
    }

    std::set<std::pair<uint32_t, uint32_t>> undirected_seen;
    for (const auto& [src_u32, dst_map] : intermesh_mesh_level_edges) {
        if (!mesh_id_in_descriptor.contains(src_u32)) {
            continue;
        }
        for (const auto& [dst_u32, edge_count] : dst_map) {
            (void)edge_count;
            if (!mesh_id_in_descriptor.contains(dst_u32)) {
                continue;
            }
            if (src_u32 == dst_u32) {
                continue;
            }
            const auto ends = std::minmax(src_u32, dst_u32);
            if (!undirected_seen.insert(ends).second) {
                continue;
            }
            MeshId src(src_u32);
            MeshId dst(dst_u32);
            adj[src].push_back(dst);
            adj[dst].push_back(src);
        }
    }
    return AdjacencyGraph<MeshId>(adj);
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

LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor) {
    auto mesh_adjacency_graphs = ::tt::tt_fabric::build_adjacency_graph_logical(mesh_graph_descriptor);
    auto [requested_intermesh_connections, requested_intermesh_ports] =
        get_requested_intermesh_from_mgd(mesh_graph_descriptor);
    return build_logical_multi_mesh_adjacency_graph_impl(
        mesh_adjacency_graphs, requested_intermesh_connections, requested_intermesh_ports);
}

LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    // This function handles both strict mode (requested_intermesh_ports) and relaxed mode
    // (requested_intermesh_connections) intermesh connections - see build_logical_multi_mesh_adjacency_graph_impl.
    auto mesh_adjacency_graphs = ::tt::tt_fabric::build_adjacency_graph_logical(mesh_graph);
    const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
    const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();

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

// ============================================================================
// build_physical_multi_mesh_adjacency_graph
//
// Given a description of the logical mesh topology (MGD) and the real hardware
// layout (PSD), figure out which physical chips should be assigned to each
// logical mesh and return the result as a PhysicalMultiMeshGraph.
//
// OVERVIEW (when there are multiple different mesh shapes):
//
//   For each distinct mesh shape (e.g. "prefill", "decode"):
//     1. Ask the grouping descriptor (PGD) where that shape actually fits on
//        the real hardware — this gives a list of candidate chip placements.
//     2. Build a graph showing how those candidate placements connect to each
//        other at the mesh level.
//     3. Set up a solver that can lazily enumerate valid ways to assign logical
//        meshes to physical placements, one solution at a time.
//
//   Round-by-round search for a conflict-free assignment:
//     Round k (k = 1, 2, ...):
//       a. Ask each mesh's solver for one more candidate placement (so each
//          mesh now has k options cached).
//       b. Walk every combination of one option per mesh and check that no two
//          options claim the same physical chip. Chips are tracked with
//          compact 64-bit bitmasks so the check is very fast.
//          Only combinations that use at least one option that is new this
//          round are tested — combinations made entirely of older options were
//          already checked in a previous round.
//       c. If a fully conflict-free assignment is found, build and return the
//          final graph.
//     Stops when a solution is found or all solvers run out of options.
//
//   FAST PATH: if all meshes have the same shape (e.g. every galaxy is a
//   4×8 tile), the PGD result is returned directly with no solver needed.
// ============================================================================
namespace {

std::map<MeshId, MeshPhysicalLayout> mesh_physical_layouts_from_psd_placements(
    const std::vector<::tt::tt_fabric::PsdPlacement>& placements) {
    std::map<MeshId, MeshPhysicalLayout> layouts;
    for (std::size_t i = 0; i < placements.size(); ++i) {
        const MeshId mesh_id{static_cast<std::uint32_t>(i)};
        MeshPhysicalLayout& layout = layouts[mesh_id];
        layout.asics = placements[i].asics;
        layout.mesh_node_to_asic_position = placements[i].mesh_node_to_asic_position;
    }
    return layouts;
}

}  // namespace

PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const std::optional<std::vector<PinningConstraint>>& pinnings) {
    using namespace ::tt::tt_fabric;

    // -------------------------------------------------------------------------
    // Phase 1: Build a complete chip-level connection graph and assign each
    // chip a compact integer ID.
    //
    // The connection graph lists every Ethernet link between chips across the
    // whole cluster. It is built once here and reused in three later steps:
    //   - Finding which chip groups can satisfy a requested mesh shape
    //   - Converting chip sets to bitmasks for fast overlap detection
    //   - Assembling the final result graph
    //
    // The compact integer IDs let us represent any set of chips as a small
    // array of 64-bit words (one bit per chip). Checking whether two chip sets
    // overlap then becomes a fast word-by-word AND loop instead of a slower
    // hash-set intersection.
    // -------------------------------------------------------------------------
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(build_flat_adjacency_map_from_psd(physical_system_descriptor));

    std::unordered_map<tt::tt_metal::AsicID, std::uint32_t> asic_to_dense_index;
    {
        const auto& flat_nodes = flat_graph.get_nodes();
        asic_to_dense_index.reserve(flat_nodes.size());
        for (std::uint32_t i = 0; i < flat_nodes.size(); ++i) {
            asic_to_dense_index.emplace(flat_nodes[i], i);
        }
    }
    const std::uint32_t cluster_asic_count = static_cast<std::uint32_t>(flat_graph.get_nodes().size());
    // How many 64-bit words are needed so we have one bit per chip.
    const std::size_t asic_word_count = (static_cast<std::size_t>(cluster_asic_count) + 63u) / 64u;

    // -------------------------------------------------------------------------
    // Phase 2: Ask the grouping descriptor (PGD) which chip groups on the real
    // hardware can satisfy each mesh shape requested by the logical descriptor.
    //
    // The result is a nested map. We only look at the "MESH" granularity here:
    // for each mesh shape name (e.g. "prefill"), we get a list of candidate
    // chip groups that have the right topology to host that shape.
    // -------------------------------------------------------------------------
    auto valid_groupings_map = physical_grouping_descriptor.get_valid_groupings_for_mgd(
        mesh_graph_descriptor, physical_system_descriptor, pinnings);

    TT_FATAL(valid_groupings_map.contains("MESH"), "Internal error: MESH grouping not found in valid groupings map");
    TT_FATAL(
        !valid_groupings_map.at("MESH").empty(),
        "Internal error: Physical grouping descriptor was not able to find mesh groupings");

    // -------------------------------------------------------------------------
    // Phase 3: For each mesh shape, locate every valid placement on the real
    // hardware and build a connection graph for those placements.
    //
    // placements_by_shape[name][i]  = i-th PSD placement: ASIC footprint plus grouping (with
    //                                 mesh_node_to_asic_position from PGD<->MGD match).
    // mesh_physical_graphs        = shape name → connection graph built from all candidate placements.
    // -------------------------------------------------------------------------
    std::unordered_map<std::string, PhysicalMultiMeshGraph> mesh_physical_graphs;
    std::unordered_map<std::string, std::vector<::tt::tt_fabric::PsdPlacement>> placements_by_shape;
    // Pre-computed chip bitmask for every placement group, keyed by mesh shape
    // name. group_bits_by_name[name][i] = the asic_word_count-word bitset for
    // the i-th candidate placement for that shape. Built once here so that
    // compute_solution_bitset (called once per SAT solution) is a simple
    // word-OR loop instead of a per-chip hash lookup.
    std::unordered_map<std::string, std::vector<std::vector<std::uint64_t>>> group_bits_by_name;

    for (const auto& [mesh_name, groupings] : valid_groupings_map.at("MESH")) {
        // find_all_in_psd returns PSD placements (ASIC footprint + grouping with mesh_node_to_asic_position).
        std::vector<std::string> find_all_errors;
        const auto placements = physical_grouping_descriptor.find_all_in_psd(
            groupings, physical_system_descriptor, flat_graph, &find_all_errors);
        if (placements.empty()) {
            for (const auto& error : find_all_errors) {
                log_error(
                    tt::LogFabric,
                    "Physical groupings adjacency: '{}' found no PSD placements from {} committed grouping(s): {}",
                    mesh_name,
                    groupings.size(),
                    error);
            }
        }
        // Build the shape graph from placements: ASIC footprints and PGD pinning both come from each PsdPlacement.
        const auto mesh_layouts = mesh_physical_layouts_from_psd_placements(placements);
        mesh_physical_graphs[mesh_name] = build_hierarchical_from_flat_graph(flat_graph, mesh_layouts);

        // Pre-compute one bitmask per candidate placement for this shape, straight from each placement's footprint.
        auto& gbits = group_bits_by_name[mesh_name];
        gbits.reserve(placements.size());
        for (const auto& placement : placements) {
            std::vector<std::uint64_t> word_vec(asic_word_count, 0);
            for (const auto& asic : placement.asics) {
                auto di = asic_to_dense_index.find(asic);
                TT_FATAL(
                    di != asic_to_dense_index.end(),
                    "ASIC from placement not found in PSD flat graph (dense index)");
                const std::uint32_t idx = di->second;
                TT_FATAL(
                    (idx >> 6) < asic_word_count,
                    "Dense ASIC index {} out of range for bitset of {} words ({} ASICs total)",
                    idx,
                    asic_word_count,
                    cluster_asic_count);
                word_vec[idx >> 6] |= (std::uint64_t{1} << (idx & 63));
            }
            gbits.push_back(std::move(word_vec));
        }

        // Record the placements for this shape so later phases can look them up by shape name. The mapping to
        // logical MeshIds is decided by the solver later, so no logical MeshId is needed here.
        placements_by_shape[mesh_name] = placements;
    }

    // -------------------------------------------------------------------------
    // Phase 4: Fast path — only one mesh shape used.
    //
    // If the whole logical descriptor uses a single mesh shape (e.g. every
    // galaxy is the same 4×8 tile), the grouping descriptor's placements
    // already cover the entire system. We can skip the solver entirely and
    // return the pre-built graph right away.
    // -------------------------------------------------------------------------
    const auto& mesh_shape_entries = valid_groupings_map.at("MESH");
    if (mesh_shape_entries.size() == 1) {
        const std::string& sole_mesh_name = mesh_shape_entries.begin()->first;
        const auto sole_it = mesh_physical_graphs.find(sole_mesh_name);
        TT_FATAL(
            sole_it != mesh_physical_graphs.end(),
            "Single mesh shape '{}' missing PSD-derived PhysicalMultiMeshGraph",
            sole_mesh_name);
        return sole_it->second;
    }

    // -------------------------------------------------------------------------
    // Phase 5: Set up the incremental solver state for each mesh shape.
    //
    // MeshEnumState bundles everything needed to drive one shape's solver and
    // remember what it has found so far:
    //   logical_graph   — the pattern we want to place: either the mesh
    //                     instances from the logical descriptor, or the full
    //                     physical coarse topology when only one shape exists
    //                     and all placements need to be assigned.
    //   physical_graph  — the hardware mesh-level connectivity for this shape.
    //   session         — the incremental solver; it keeps prior constraints in
    //                     memory so each call only adds new work.
    //   excluded        — solutions already returned; fed back to the solver so
    //                     it doesn't repeat them.
    //   solutions       — all solutions found so far, in the order returned.
    //   solution_bits   — a chip bitmask for each solution; used for fast
    //                     overlap checks in the packing search below.
    //   embedding_sizes — how many meshes each solution places; used to try
    //                     larger placements first (better pruning).
    //   exhausted       — set to true once the solver finds no more options.
    // -------------------------------------------------------------------------
    struct MeshEnumState {
        AdjacencyGraph<MeshId> logical_graph;
        AdjacencyGraph<MeshId> physical_graph;
        MappingConstraints<MeshId, MeshId> constraints;
        TopologyMappingEnumerationSession<MeshId, MeshId> session;
        std::vector<std::map<MeshId, MeshId>> excluded;
        std::vector<MappingResult<MeshId, MeshId>> solutions;
        std::vector<std::vector<std::uint64_t>> solution_bits;
        std::vector<std::size_t> embedding_sizes;
        bool exhausted = false;
    };

    // Read the inter-mesh connections from the logical descriptor once here;
    // they are reused when building the pattern graph for each mesh shape.
    const auto [mgd_intermesh_mesh_level, mgd_intermesh_ports] =
        get_requested_intermesh_from_mgd(mesh_graph_descriptor);
    (void)mgd_intermesh_ports;

    // Count how many mesh shapes actually have a hardware-derived placement
    // graph. Used below to decide whether to expand the pattern to cover the
    // full physical topology.
    std::size_t descriptor_names_with_psd = 0;
    for (const auto& mn : mesh_graph_descriptor.get_all_mesh_names()) {
        if (mesh_physical_graphs.contains(mn)) {
            ++descriptor_names_with_psd;
        }
    }

    std::unordered_map<std::string, MeshEnumState> mesh_enum_states;

    for (const auto& mesh_name : mesh_graph_descriptor.get_all_mesh_names()) {
        const auto physical_it = mesh_physical_graphs.find(mesh_name);
        if (physical_it == mesh_physical_graphs.end()) {
            log_warning(
                tt::LogFabric,
                "No PSD-derived PhysicalMultiMeshGraph for mesh descriptor '{}'; skipping mesh-level topology solve",
                mesh_name);
            continue;
        }
        const AdjacencyGraph<MeshId>& physical_mesh_level = physical_it->second.mesh_level_graph_;

        // Build the subset of the logical descriptor's mesh-level graph that
        // belongs to this shape: nodes are the logical mesh IDs for this
        // shape name, edges are FABRIC inter-mesh connections among them.
        AdjacencyGraph<MeshId> mgd_mesh_level_graph = build_mgd_mesh_level_subgraph_for_mesh_descriptor_name(
            mesh_graph_descriptor, mesh_name, mgd_intermesh_mesh_level);

        // Pattern expansion: when the logical descriptor declares fewer
        // instances than the hardware has placements (e.g. one "galaxy" entry
        // but 16 physical galaxies), widen the pattern to match the hardware
        // topology so the solver can assign every physical placement.
        // This widening is only safe when there is a single shape; with
        // multiple shapes each shape's chip set would cover the whole system,
        // making conflict-free assignment impossible.
        const bool expand_to_psd_coarse =
            descriptor_names_with_psd == 1 &&
            mgd_mesh_level_graph.get_nodes().size() < physical_mesh_level.get_nodes().size();
        AdjacencyGraph<MeshId> logical_mesh_level_graph =
            expand_to_psd_coarse ? physical_mesh_level : mgd_mesh_level_graph;

        // Register the solver state for this shape. No solver work runs yet;
        // the first solution is pulled lazily when round 1 starts in Phase 6.
        MeshEnumState state;
        state.logical_graph = std::move(logical_mesh_level_graph);
        state.physical_graph = physical_mesh_level;
        mesh_enum_states.emplace(mesh_name, std::move(state));
    }

    // Order shapes so the most constrained (fewest available placements) is
    // tried first. This fail-first heuristic prunes dead-end branches earlier
    // in the DFS and typically reduces combinations tested significantly.
    // Ties are broken lexicographically for determinism.
    std::vector<std::string> mesh_order;
    mesh_order.reserve(mesh_enum_states.size());
    for (const auto& [name, _state] : mesh_enum_states) {
        mesh_order.push_back(name);
    }
    std::sort(mesh_order.begin(), mesh_order.end(), [&](const std::string& a, const std::string& b) {
        const auto a_it = mesh_physical_graphs.find(a);
        const auto b_it = mesh_physical_graphs.find(b);
        const std::size_t a_count =
            (a_it != mesh_physical_graphs.end()) ? a_it->second.mesh_level_graph_.get_nodes().size() : 0;
        const std::size_t b_count =
            (b_it != mesh_physical_graphs.end()) ? b_it->second.mesh_level_graph_.get_nodes().size() : 0;
        if (a_count != b_count) {
            return a_count < b_count;  // fewer placements = more constrained = first
        }
        return a < b;
    });

    const std::size_t n_meshes = mesh_order.size();

    // -------------------------------------------------------------------------
    // Phase 6: Round-by-round search for a conflict-free chip assignment.
    //
    // BITMASK HELPERS
    // Each cached solution carries a bitmask with one bit per chip in the
    // cluster. Three helpers operate on these bitmasks word by word:
    //
    //   bitset_disjoint(a, b) — true if a and b share no chip (no bit set in
    //                           both), i.e. the two placements don't overlap.
    //   mark_used(dst, src)   — dst |= src  (mark src's chips as taken).
    //   unmark(dst, src)      — dst &= ~src (release src's chips on backtrack).
    //
    // These are tight 64-bit word loops that the compiler can auto-vectorise.
    // The cost per combination is proportional to cluster size in words, not
    // to the number of chips per mesh.
    // -------------------------------------------------------------------------
    std::vector<std::uint64_t> occupied_asics(asic_word_count, 0);

    auto bitset_disjoint = [asic_word_count](
                               const std::vector<std::uint64_t>& cand,
                               const std::vector<std::uint64_t>& occupied) -> bool {
        for (std::size_t i = 0; i < asic_word_count; ++i) {
            if (cand[i] & occupied[i]) {
                return false;
            }
        }
        return true;
    };
    auto mark_used = [asic_word_count](
                         std::vector<std::uint64_t>& dst, const std::vector<std::uint64_t>& src) {
        for (std::size_t i = 0; i < asic_word_count; ++i) {
            dst[i] |= src[i];
        }
    };
    auto unmark = [asic_word_count](
                      std::vector<std::uint64_t>& dst, const std::vector<std::uint64_t>& src) {
        for (std::size_t i = 0; i < asic_word_count; ++i) {
            dst[i] &= ~src[i];
        }
    };

    // Build the chip bitmask for one solver solution by OR-ing together the
    // pre-computed per-group bitsets. No hash lookups in this hot path.
    auto compute_solution_bitset = [&](const std::string& mesh_name,
                                       const MappingResult<MeshId, MeshId>& solution) {
        std::vector<std::uint64_t> bits(asic_word_count, 0);
        const auto& group_bits = group_bits_by_name.at(mesh_name);
        for (const auto& [logical_mesh_id, physical_mesh_id] : solution.target_to_global) {
            TT_FATAL(
                physical_mesh_id.get() < group_bits.size(),
                "Physical mesh index {} out of range for group_bits (logical MeshId {})",
                physical_mesh_id.get(),
                logical_mesh_id.get());
            const auto& gbits = group_bits[physical_mesh_id.get()];
            for (std::size_t w = 0; w < asic_word_count; ++w) {
                bits[w] |= gbits[w];
            }
        }
        return bits;
    };

    // Ask a shape's solver for the next distinct placement, cache the result,
    // and return true. Returns false (and marks the shape exhausted) when the
    // solver has no more options. Logs a note if a single solver call takes
    // more than 2 s so slow calls are visible in the logs.
    constexpr std::chrono::milliseconds kSlowSatThreshold{2000};
    auto pull_next_solution = [&](const std::string& mesh_name) -> bool {
        MeshEnumState& s = mesh_enum_states.at(mesh_name);
        if (s.exhausted) {
            return false;
        }
        const auto t_begin = std::chrono::steady_clock::now();
        MappingResult<MeshId, MeshId> result = s.session.next(
            s.logical_graph,
            s.physical_graph,
            s.constraints,
            s.excluded,
            ConnectionValidationMode::STRICT,
            /*quiet_mode=*/true,
            TopologyMappingSolverEngine::Sat,
            /*unique_shapes=*/true);
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_begin);
        if (elapsed >= kSlowSatThreshold) {
            log_info(
                tt::LogFabric,
                "Topology mapper: SAT next() for mesh '{}' took {}ms (success={}, cache size now {})",
                mesh_name,
                elapsed.count(),
                result.success,
                s.solutions.size() + (result.success ? 1 : 0));
        }
        if (!result.success) {
            log_info(
                tt::LogFabric,
                "Topology mapper: mesh '{}' enumeration exhausted at {} solutions",
                mesh_name,
                s.solutions.size());
            s.exhausted = true;
            return false;
        }
        std::vector<std::uint64_t> bits = compute_solution_bitset(mesh_name, result);
        s.excluded.push_back(result.target_to_global);
        s.solution_bits.push_back(std::move(bits));
        s.embedding_sizes.push_back(result.target_to_global.size());
        s.solutions.push_back(std::move(result));
        return true;
    };

    // -----------------------------------------------------------------------
    // DisjointPackingSearch — depth-first search over all combinations of one
    // placement per mesh shape.
    //
    // At each level of the recursion we try every cached placement for one
    // shape (largest first), skip any that overlap chips already claimed by
    // earlier levels, and recurse deeper if it fits.
    //
    // "Test each combination only once" rule: in round k we only check
    // combinations that include at least one placement that is new this round
    // (cache index k-1). Any combination made entirely of older placements was
    // already checked in a previous round. This is enforced by carrying a
    // flag (frontier_satisfied) through the recursion; at the leaf level we
    // only accept a candidate if the flag is already set or the candidate
    // itself is the new one.
    // -----------------------------------------------------------------------
    std::vector<MeshEnumState*> mesh_state_ptrs(n_meshes, nullptr);
    for (std::size_t d = 0; d < n_meshes; ++d) {
        mesh_state_ptrs[d] = &mesh_enum_states.at(mesh_order[d]);
    }
    std::vector<std::vector<std::size_t>> try_order_per_depth(n_meshes);
    std::vector<std::size_t> chosen_index(n_meshes, 0);

    struct DisjointPackingSearch {
        const std::vector<MeshEnumState*>& states;
        const std::vector<std::vector<std::size_t>>& try_order_per_depth;
        std::vector<std::size_t>& chosen_index;
        std::vector<std::uint64_t>& occupied_asics;
        std::size_t round;       // current round number (1-based)
        std::size_t target_idx;  // = round - 1; the "new" cache index this round
        std::size_t n_meshes;
        std::size_t& combinations_tested;
        std::size_t& timer_check_mask;
        std::chrono::steady_clock::time_point& last_progress_log_time;
        const std::chrono::steady_clock::time_point& search_start_time;
        decltype(bitset_disjoint)& is_disjoint;
        decltype(mark_used)& mark_used_fn;
        decltype(unmark)& unmark_fn;
        // frontier_reachable_from[d] = true if any depth >= d has a solution
        // with index target_idx (i.e., can still satisfy the new-this-round
        // rule). Computed once before each search call and used to prune
        // interior subtrees that can never satisfy the frontier.
        std::vector<bool> frontier_reachable_from;
        std::chrono::seconds progress_interval{10};

        // Returns true (and fills chosen_index[depth..n_meshes)) when it finds
        // a conflict-free assignment that satisfies the new-this-round rule.
        bool run(std::size_t depth, bool frontier_satisfied) {
            // Interior pruning: if no depth from here to the leaf can provide
            // the required new-this-round solution, skip this whole subtree.
            if (!frontier_satisfied && !frontier_reachable_from[depth]) {
                return false;
            }

            const MeshEnumState& s = *states[depth];
            const std::vector<std::size_t>& order = try_order_per_depth[depth];
            const bool is_leaf = (depth + 1 == n_meshes);

            if (is_leaf) {
                for (std::size_t si : order) {
                    if (si >= round) {
                        continue;
                    }
                    // Only accept this leaf if the "new this round" rule is
                    // already satisfied, or if this candidate is itself new.
                    if (!frontier_satisfied && si != target_idx) {
                        continue;
                    }
                    ++combinations_tested;
                    // Log progress periodically so a long search isn't silent.
                    if ((combinations_tested & timer_check_mask) == 0) {
                        const auto now = std::chrono::steady_clock::now();
                        if (now - last_progress_log_time >= progress_interval) {
                            const auto elapsed_sec =
                                std::chrono::duration_cast<std::chrono::seconds>(now - search_start_time).count();
                            log_info(
                                tt::LogFabric,
                                "Topology mapper round-robin: tested {} mesh-level combinations in {}s (round {})",
                                combinations_tested,
                                elapsed_sec,
                                round);
                            last_progress_log_time = now;
                        }
                    }
                    if (!is_disjoint(s.solution_bits[si], occupied_asics)) {
                        continue;
                    }
                    chosen_index[depth] = si;
                    return true;
                }
                return false;
            }

            // Interior level: try each candidate, claim its chips, recurse
            // into the next level, and release the chips if we backtrack.
            for (std::size_t si : order) {
                if (si >= round) {
                    continue;
                }
                if (!is_disjoint(s.solution_bits[si], occupied_asics)) {
                    continue;
                }
                mark_used_fn(occupied_asics, s.solution_bits[si]);
                chosen_index[depth] = si;
                const bool child_frontier = frontier_satisfied || (si == target_idx);
                if (run(depth + 1, child_frontier)) {
                    return true;
                }
                unmark_fn(occupied_asics, s.solution_bits[si]);
            }
            return false;
        }
    };

    // -----------------------------------------------------------------------
    // Main loop: keep growing each shape's solution cache by one per round,
    // then search for a conflict-free assignment among all cached options.
    // -----------------------------------------------------------------------
    bool found_disjoint_combination = false;
    constexpr std::size_t kProgressCheckMask = 4095;  // sample timer every 4096 leaf tests
    std::size_t combinations_tested = 0;
    std::size_t timer_check_mask_ref = kProgressCheckMask;
    const auto search_start_time = std::chrono::steady_clock::now();
    auto last_progress_log_time = search_start_time;
    auto last_round_log_time = search_start_time;
    constexpr std::chrono::seconds kRoundLogInterval{10};

    for (std::size_t round = 1; n_meshes != 0 && !found_disjoint_combination; ++round) {
        // -- Pull phase: ask each shape's solver for one more placement -------
        // Each shape that still has options advances by one step. Timing is
        // logged separately so solver time is visible apart from search time.
        const auto t_pull = std::chrono::steady_clock::now();
        bool any_progress = false;
        for (std::size_t d = 0; d < n_meshes; ++d) {
            if (pull_next_solution(mesh_order[d])) {
                any_progress = true;
            }
        }
        const auto pull_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_pull).count();

        const auto now = std::chrono::steady_clock::now();
        if (round == 1 || (now - last_round_log_time) >= kRoundLogInterval || pull_ms >= 1000) {
            std::string cache_summary;
            cache_summary.reserve(n_meshes * 24);
            for (std::size_t d = 0; d < n_meshes; ++d) {
                if (d) {
                    cache_summary += ", ";
                }
                const MeshEnumState& s = *mesh_state_ptrs[d];
                cache_summary +=
                    fmt::format("{}={}{}", mesh_order[d], s.solutions.size(), s.exhausted ? "(exhausted)" : "");
            }
            log_info(
                tt::LogFabric,
                "Topology mapper round-robin round {}: pulled in {}ms, caches=[{}], combinations tested so far={}",
                round,
                pull_ms,
                cache_summary,
                combinations_tested);
            last_round_log_time = now;
        }

        // -- Termination check ------------------------------------------------
        // If no shape's cache grew to size `round` this iteration, every
        // possible combination was already covered by an earlier round.
        // There is nothing new to try, so we can stop.
        bool any_can_hit_frontier = false;
        for (std::size_t d = 0; d < n_meshes; ++d) {
            if (mesh_state_ptrs[d]->solutions.size() >= round) {
                any_can_hit_frontier = true;
                break;
            }
        }
        if (!any_can_hit_frontier) {
            (void)any_progress;
            break;
        }

        // -- Build the per-shape candidate order for this round ---------------
        // Candidates are ordered by descending embedding size (more meshes
        // placed first), ties broken by ascending arrival index. Instead of
        // a full re-sort each round, we insert the single new element (if any)
        // into its correct position in the already-sorted vector — O(k) scan
        // rather than O(k log k) sort over the same data.
        for (std::size_t d = 0; d < n_meshes; ++d) {
            const MeshEnumState& s = *mesh_state_ptrs[d];
            const std::size_t hi = std::min<std::size_t>(s.solutions.size(), round);
            std::vector<std::size_t>& order = try_order_per_depth[d];

            if (hi <= order.size()) {
                continue;  // no new solution this round for this shape
            }
            // Exactly one new solution was added: index `hi - 1`.
            const std::size_t new_idx = hi - 1;
            const std::size_t new_size = s.embedding_sizes[new_idx];

            // Find insertion point: after all candidates with a strictly
            // larger embedding size. Among equals, new_idx is largest so it
            // goes last (preserving ascending-index tie-breaking).
            auto pos = order.begin();
            while (pos != order.end() && s.embedding_sizes[*pos] > new_size) {
                ++pos;
            }
            order.insert(pos, new_idx);
        }

        // -- Packing search ---------------------------------------------------
        // Pre-compute frontier_reachable_from[d]: true if any depth >= d has
        // at least one solution with index target_idx (= round - 1). This
        // suffix array lets DisjointPackingSearch prune entire subtrees that
        // can never satisfy the new-this-round frontier rule.
        std::vector<bool> frontier_reachable(n_meshes + 1, false);
        for (std::size_t d = n_meshes; d-- > 0;) {
            const bool this_depth_can_hit = (mesh_state_ptrs[d]->solutions.size() > round - 1);
            frontier_reachable[d] = this_depth_can_hit || frontier_reachable[d + 1];
        }

        std::fill(occupied_asics.begin(), occupied_asics.end(), 0);
        const std::size_t combos_before = combinations_tested;
        DisjointPackingSearch search{
            mesh_state_ptrs,
            try_order_per_depth,
            chosen_index,
            occupied_asics,
            round,
            round - 1,
            n_meshes,
            combinations_tested,
            timer_check_mask_ref,
            last_progress_log_time,
            search_start_time,
            bitset_disjoint,
            mark_used,
            unmark,
            std::move(frontier_reachable)};
        found_disjoint_combination = search.run(0, /*frontier_satisfied=*/false);
        log_debug(
            tt::LogFabric,
            "Topology mapper round-robin round {} complete: +{} combinations (total {}), found={}",
            round,
            combinations_tested - combos_before,
            combinations_tested,
            found_disjoint_combination);
    }

    log_info(
        tt::LogFabric,
        "Topology mapper round-robin finished: {} mesh-level combinations tested in {}s, success={}",
        combinations_tested,
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - search_start_time).count(),
        found_disjoint_combination);

    // -------------------------------------------------------------------------
    // Phase 7: Build the final result graph from the winning placements.
    //
    // For each shape, chosen_index[d] identifies which cached solution won.
    // That solution maps logical mesh IDs to physical mesh IDs. We look up the
    // chip set for each physical placement and store it at the logical mesh
    // index in combined_mesh_groupings.
    //
    // It is safe if two shapes use the same logical MeshId because the bitmask
    // check above already guarantees their chip sets are disjoint. The union
    // written at that index is therefore correct.
    // -------------------------------------------------------------------------
    if (!found_disjoint_combination) {
        std::string solution_counts;
        std::string mesh_names_str;
        for (std::size_t d = 0; d < n_meshes; ++d) {
            if (d) {
                solution_counts += ", ";
                mesh_names_str += ", ";
            }
            const MeshEnumState& s = *mesh_state_ptrs[d];
            solution_counts +=
                fmt::format("{}={}{}", mesh_order[d], s.solutions.size(), s.exhausted ? "(exhausted)" : "");
            mesh_names_str += mesh_order[d];
        }
        TT_THROW(
            "Topology mapper failed to find disjoint placements for mesh descriptors [{}] on a system with {} ASICs. "
            "Solution counts per mesh: [{}]",
            mesh_names_str,
            cluster_asic_count,
            solution_counts);
    }

    // Find the highest logical mesh ID across all winning solutions so we know
    // how large to make the combined groupings vector.
    MeshId max_logical{0};
    for (std::size_t j = 0; j < n_meshes; ++j) {
        for (const auto& [logical_mesh_id, _phys] :
             mesh_state_ptrs[j]->solutions[chosen_index[j]].target_to_global) {
            if (logical_mesh_id.get() > max_logical.get()) {
                max_logical = logical_mesh_id;
            }
        }
    }

    // Re-key each shape's chosen placements under the logical MeshId the solver assigned them, so
    // combined_placements[logical] carries that mesh's ASIC footprint and PGD pinning.
    std::vector<::tt::tt_fabric::PsdPlacement> combined_placements(max_logical.get() + 1);
    for (std::size_t j = 0; j < n_meshes; ++j) {
        const std::string& mesh_name = mesh_order[j];
        const MappingResult<MeshId, MeshId>& picked = mesh_state_ptrs[j]->solutions[chosen_index[j]];
        TT_FATAL(!picked.target_to_global.empty(), "Empty mesh-level mapping for mesh descriptor '{}'", mesh_name);
        const auto& placements = placements_by_shape.at(mesh_name);
        for (const auto& [logical_mesh_id, physical_mesh_id] : picked.target_to_global) {
            TT_FATAL(
                physical_mesh_id.get() < placements.size(),
                "Physical mesh index {} out of range for placements (logical MeshId {})",
                physical_mesh_id.get(),
                logical_mesh_id.get());
            const auto& placement = placements[physical_mesh_id.get()];
            auto& combined = combined_placements[logical_mesh_id.get()];
            combined.asics.insert(placement.asics.begin(), placement.asics.end());
            for (const auto& [chip_id, asic_position] : placement.mesh_node_to_asic_position) {
                combined.mesh_node_to_asic_position[chip_id] = asic_position;
            }
        }
    }

    return build_hierarchical_from_flat_graph(flat_graph, combined_placements);
}

namespace {

// Map logical MeshId (MGD local_id) -> mesh/switch definition name used as the MESH key in
// get_valid_groupings_for_mgd (e.g. MeshId{0} -> "M0").
std::unordered_map<MeshId, std::string> logical_mesh_id_to_mgd_instance_name(
    const tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor) {
    std::unordered_map<MeshId, std::string> mesh_id_to_name;
    for (const auto global_id : mesh_graph_descriptor.all_meshes()) {
        const auto& instance = mesh_graph_descriptor.get_instance(global_id);
        mesh_id_to_name.emplace(MeshId{instance.local_id}, instance.name);
    }
    for (const auto global_id : mesh_graph_descriptor.all_switches()) {
        const auto& instance = mesh_graph_descriptor.get_instance(global_id);
        mesh_id_to_name.emplace(MeshId{instance.local_id}, instance.name);
    }
    return mesh_id_to_name;
}

// Attach PGD preferred pinnings onto an already-built rank-bound physical graph. For each mesh already
// present on the graph, look up its MGD type name and copy the committed MESH grouping's
// mesh_node_to_asic_position onto mesh_pgd_pinnings_ (no footprint rediscovery).
void assign_pgd_pinnings_to_rank_bound_physical_graph(
    PhysicalMultiMeshGraph& physical_multi_mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const std::optional<std::vector<PinningConstraint>>& pinnings) {
    using namespace ::tt::tt_fabric;

    if (physical_multi_mesh_graph.mesh_adjacency_graphs_.empty()) {
        return;
    }

    const auto valid_groupings_map = physical_grouping_descriptor.get_valid_groupings_for_mgd(
        mesh_graph_descriptor, physical_system_descriptor, pinnings);
    if (!valid_groupings_map.contains("MESH") || valid_groupings_map.at("MESH").empty()) {
        log_debug(
            tt::LogFabric,
            "Rank-bound PGD pinning enrichment: no MESH groupings from get_valid_groupings_for_mgd; leaving "
            "mesh_pgd_pinnings_ empty");
        return;
    }

    const auto& mesh_groupings_by_name = valid_groupings_map.at("MESH");
    const auto mesh_id_to_instance_name = logical_mesh_id_to_mgd_instance_name(mesh_graph_descriptor);

    std::size_t assigned = 0;
    for (const auto& [logical_mesh_id, _] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto name_it = mesh_id_to_instance_name.find(logical_mesh_id);
        if (name_it == mesh_id_to_instance_name.end()) {
            log_debug(
                tt::LogFabric,
                "Rank-bound PGD pinning enrichment: logical mesh {} has no MGD mesh/switch instance; skipping",
                logical_mesh_id.get());
            continue;
        }

        const auto groupings_it = mesh_groupings_by_name.find(name_it->second);
        if (groupings_it == mesh_groupings_by_name.end() || groupings_it->second.empty()) {
            log_debug(
                tt::LogFabric,
                "Rank-bound PGD pinning enrichment: no committed MESH groupings for MGD type '{}' "
                "(logical mesh {}); skipping",
                name_it->second,
                logical_mesh_id.get());
            continue;
        }

        bool matched = false;
        for (const auto& grouping : groupings_it->second) {
            if (grouping.mesh_node_to_asic_position.empty()) {
                continue;
            }
            physical_multi_mesh_graph.mesh_pgd_pinnings_[logical_mesh_id] = grouping.mesh_node_to_asic_position;
            ++assigned;
            matched = true;
            log_debug(
                tt::LogFabric,
                "Rank-bound PGD pinning enrichment: assigned pinning ({} chips) for logical mesh {} "
                "(MGD type '{}')",
                grouping.mesh_node_to_asic_position.size(),
                logical_mesh_id.get(),
                name_it->second);
            break;
        }
        if (!matched) {
            log_debug(
                tt::LogFabric,
                "Rank-bound PGD pinning enrichment: committed MESH groupings for MGD type '{}' "
                "(logical mesh {}) have empty mesh_node_to_asic_position; skipping",
                name_it->second,
                logical_mesh_id.get());
        }
    }

    log_info(
        tt::LogFabric,
        "Rank-bound PGD pinning enrichment: assigned pinnings to {}/{} logical mesh(es)",
        assigned,
        physical_multi_mesh_graph.mesh_adjacency_graphs_.size());
}

}  // namespace

PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const std::optional<std::vector<PinningConstraint>>& pinnings) {
    auto physical_multi_mesh_graph =
        build_physical_multi_mesh_adjacency_graph(physical_system_descriptor, asic_id_to_mesh_rank);
    assign_pgd_pinnings_to_rank_bound_physical_graph(
        physical_multi_mesh_graph,
        physical_system_descriptor,
        physical_grouping_descriptor,
        mesh_graph_descriptor,
        pinnings);
    return physical_multi_mesh_graph;
}

PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // Build flat adjacency map from PhysicalSystemDescriptor
    PhysicalAdjacencyMap flat_adj = build_flat_adjacency_map_from_psd(physical_system_descriptor);

    // Convert asic_id_to_mesh_rank to an explicit MeshId -> ASIC-set map (same MeshIds preserved).
    if (asic_id_to_mesh_rank.empty()) {
        return PhysicalMultiMeshGraph{};
    }
    std::map<MeshId, std::unordered_set<tt::tt_metal::AsicID>> mesh_groupings;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            mesh_groupings[mesh_id].insert(asic_id);
        }
    }

    // Convert to AdjacencyGraph and use the common algorithm
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    PhysicalMultiMeshGraph result = build_hierarchical_from_flat_graph(flat_graph, mesh_groupings);

    return result;
}

PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::vector<::tt::tt_fabric::PsdPlacement>& placements) {
    return build_hierarchical_from_flat_graph(
        flat_adjacency_graph, mesh_physical_layouts_from_psd_placements(placements));
}

PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::map<MeshId, std::unordered_set<tt::tt_metal::AsicID>>& mesh_groupings,
    const std::map<MeshId, std::map<LogicalChipId, tt::tt_metal::ASICPosition>>& mesh_pgd_pinnings) {
    std::map<MeshId, MeshPhysicalLayout> mesh_layouts;
    for (const auto& [mesh_id, asics] : mesh_groupings) {
        mesh_layouts[mesh_id].asics = asics;
    }
    for (const auto& [mesh_id, pinning] : mesh_pgd_pinnings) {
        mesh_layouts[mesh_id].mesh_node_to_asic_position = pinning;
    }
    return build_hierarchical_from_flat_graph(flat_adjacency_graph, mesh_layouts);
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

namespace {

std::optional<std::string> hostname_for_asic_from_hostname_map(
    tt::tt_metal::AsicID asic_id, const std::map<std::string, std::set<tt::tt_metal::AsicID>>& hostname_to_asics) {
    for (const auto& [hostname, asics] : hostname_to_asics) {
        if (asics.contains(asic_id)) {
            return hostname;
        }
    }
    return std::nullopt;
}

// Minimal host cover for inter-mesh mapping: partition physical meshes by host, then apply preferred
// constraints so unbound logical meshes tend to pack onto fewer hosts.
// Only called when the physical graph is not already identity-bound via asic_id_to_mesh_rank (Phase 1).
// TODO: This can be removed and replaced with cost heuristics when using a SAT solver because preferred
// constraints aren't very effective here
// https://github.com/tenstorrent/tt-metal/issues/40640
void add_inter_mesh_minimal_host_cover_from_hostname_map(
    const TopologyMappingConfig& config,
    const PhysicalMultiMeshGraph& physical_graph,
    const AdjacencyGraph<MeshId>& mesh_logical_level_graph,
    ::tt::tt_fabric::MappingConstraints<MeshId, MeshId>& inter_mesh_constraints) {
    if (config.hostname_to_asics.empty()) {
        return;
    }

    std::set<MeshId> logical_target_set(
        mesh_logical_level_graph.get_nodes().begin(), mesh_logical_level_graph.get_nodes().end());
    if (logical_target_set.size() <= 1) {
        return;
    }

    // Build global_mesh_groups in one pass: one group per host for single-host meshes, singleton for multi-host.
    std::vector<std::set<MeshId>> global_mesh_groups;
    std::map<std::string, std::size_t> host_group_index;
    for (const auto& [phys_mesh_id, adj] : physical_graph.mesh_adjacency_graphs_) {
        if (adj.get_nodes().empty()) {
            continue;
        }
        std::set<std::string> hosts_for_mesh;
        for (const auto& asic_id : adj.get_nodes()) {
            auto hostname = hostname_for_asic_from_hostname_map(asic_id, config.hostname_to_asics);
            if (hostname.has_value()) {
                hosts_for_mesh.insert(*hostname);
            }
        }
        if (hosts_for_mesh.size() == 1) {
            auto [it, inserted] = host_group_index.try_emplace(*hosts_for_mesh.begin(), global_mesh_groups.size());
            if (inserted) {
                global_mesh_groups.emplace_back();
            }
            global_mesh_groups[it->second].insert(phys_mesh_id);
        } else {
            global_mesh_groups.push_back({phys_mesh_id});
        }
    }
    if (global_mesh_groups.empty()) {
        return;
    }

    const auto [single_group_fits, preferred_globals] =
        ::tt::tt_fabric::PhysicalGroupingDescriptor::find_minimum_coverage_group(
            logical_target_set, global_mesh_groups);
    if (single_group_fits) {
        std::vector<std::set<MeshId>> target_groups;
        target_groups.push_back(logical_target_set);
        if (inter_mesh_constraints.set_same_rank_groups_constraint(target_groups, global_mesh_groups)) {
            return;
        }
        log_warning(
            tt::LogFabric,
            "Inter-mesh host alignment: failed to set same-rank groups constraint; falling back to preferred globals");
    }

    // No single host covers all targets. Constrain the HOST COUNT (not a specific set of hosts): register the host
    // partitions as same-rank global groups, then cap how many of them the mapping may occupy at the capacity lower
    // bound k_min = ceil(num_targets / max host capacity). The solver enforces "at most k_min host groups occupied"
    // as a hard cardinality constraint but is free to choose WHICH k_min hosts (any combination), so it picks a
    // connectivity-feasible set instead of being pinned to one greedy cover. If that hard cap is unsatisfiable the
    // SAT backend backs down to the soft minimize objective (best-effort fewest groups), so we still return a valid,
    // near-minimal mapping.
    (void)single_group_fits;
    (void)preferred_globals;

    // Register the host partitions as same-rank GLOBAL groups (no target groups -> no hard co-location; this only
    // exposes per-mesh host membership so the occupancy cardinality can be built).
    if (!inter_mesh_constraints.set_same_rank_groups_constraint(/*target_groups=*/{}, global_mesh_groups)) {
        log_warning(
            tt::LogFabric, "Inter-mesh host alignment: failed to register host partitions as same-rank global groups");
        return;
    }

    std::size_t max_group_capacity = 0;
    for (const auto& grp : global_mesh_groups) {
        max_group_capacity = std::max(max_group_capacity, grp.size());
    }
    if (max_group_capacity == 0) {
        return;
    }
    const std::size_t k_min = (logical_target_set.size() + max_group_capacity - 1) / max_group_capacity;

    inter_mesh_constraints.set_max_same_rank_groups_used(k_min);      // HARD: fit within k_min hosts (any k_min)
    inter_mesh_constraints.set_minimize_same_rank_groups_used(true);  // SOFT fallback: minimize if the cap is infeasible

    log_debug(
        tt::LogFabric,
        "Inter-mesh host alignment: capping host-group usage at k_min={} (targets={}, max host capacity={})",
        k_min,
        logical_target_set.size(),
        max_group_capacity);
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

// Helper function to build inter-mesh constraints.
// Phase 2 (non-empty asic_id_to_mesh_rank): physical MeshIds match logical MeshIds, so each rank-bound
// logical mesh is hard-pinned to itself and host-cover bias is skipped.
// Phase 1 (empty asic_id_to_mesh_rank): pinnings + host-cover drive the inter-mesh mapping.
::tt::tt_fabric::MappingConstraints<MeshId, MeshId> build_inter_mesh_constraints(
    const TopologyMappingConfig& config,
    const PhysicalMultiMeshGraph& physical_graph,
    const AdjacencyGraph<MeshId>& mesh_logical_level_graph,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    ::tt::tt_fabric::MappingConstraints<MeshId, MeshId> inter_mesh_constraints;

    std::map<MeshId, std::set<MeshId>> mesh_level_pinnings;
    for (const auto& [pos, fabric_node] : config.pinnings) {
        for (const auto& [physical_mesh_id, physical_mesh_graph] : physical_graph.mesh_adjacency_graphs_) {
            auto asic_position_map = build_asic_positions_map(physical_mesh_graph, config);
            if (asic_position_map.contains(pos)) {
                mesh_level_pinnings[fabric_node.mesh_id].insert(physical_mesh_id);
            }
        }
    }
    // Make sure to restrict so that MGD and other pinnings can be respected always, and shrinks search space
    for (const auto& [mesh_id, physical_meshes] : mesh_level_pinnings) {
        if (!physical_meshes.empty()) {
            inter_mesh_constraints.add_required_constraint(mesh_id, physical_meshes);
        }
    }

    if (!config.disable_rank_bindings && !asic_id_to_mesh_rank.empty()) {
        for (const auto& mesh_id : mesh_logical_level_graph.get_nodes()) {
            if (asic_id_to_mesh_rank.contains(mesh_id)) {
                inter_mesh_constraints.add_required_constraint(mesh_id, mesh_id);
            }
        }
    }

    add_inter_mesh_minimal_host_cover_from_hostname_map(
        config, physical_graph, mesh_logical_level_graph, inter_mesh_constraints);
    return inter_mesh_constraints;
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
    // Build the pinning constraints from config.pinnings
    // Group pinnings by fabric_node (since config.pinnings is position -> fabric_node)
    std::map<FabricNodeId, std::vector<AsicPosition>> fabric_node_to_positions;
    for (const auto& [position, fabric_node] : config.pinnings) {
        // Only check the pinnings for the current mesh
        if (fabric_node.mesh_id != logical_mesh_id) {
            continue;
        }
        fabric_node_to_positions[fabric_node].push_back(position);
    }

    // Apply pinning constraints that exist on this physical grouping
    for (const auto& [fabric_node, positions] : fabric_node_to_positions) {
        std::set<tt::tt_metal::AsicID> asic_ids;

        // Convert the ASIC positions to ASIC IDs; skip positions absent from the physical mesh
        for (const auto& position : positions) {
            auto it = asic_positions_to_asic_ids.find(position);
            if (it == asic_positions_to_asic_ids.end()) {
                log_trace(
                    tt::LogFabric,
                    "Pinned ASIC position (tray_id: {}, asic_location: {}) to fabric node id (mesh_id: {}, chip_id: "
                    "{}) from MGD not found in physical topology; skipping",
                    position.first.get(),
                    position.second.get(),
                    fabric_node.mesh_id.get(),
                    fabric_node.chip_id);
                continue;
            }
            asic_ids.insert(it->second.begin(), it->second.end());
        }

        if (!asic_ids.empty()) {
            if (!intra_mesh_constraints.add_required_constraint(fabric_node, asic_ids)) {
                return fmt::format(
                    "fabric node (mesh={}, chip={}) has pinned ASIC positions present in the physical mesh but none "
                    "lie in the node's host-rank partition (conflicts with rank bindings)",
                    fabric_node.mesh_id.get(),
                    fabric_node.chip_id);
            }
        }
    }

    return std::nullopt;
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
// Returns true if constraints were successfully added, false if constraints cannot be satisfied
// (e.g., no valid physical exit nodes or over-constrained)
bool add_exit_node_constraints(
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
                    return false;
                }
                const auto& valid_physical_exit_nodes = valid_physical_exit_nodes_it->second;
                if (num_logical_exit_nodes_assigned > valid_physical_exit_nodes.size()) {
                    return false;
                }
                if (!intra_mesh_constraints.add_required_constraint(fabric_node_id, valid_physical_exit_nodes)) {
                    return false;
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
                return false;
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
                return false;
            }

            if (effective_exit_pair_min_count == 0) {
                return false;
            }

            if (!intra_mesh_constraints.add_cardinality_constraint(
                    valid_logical_exit_nodes, valid_physical_exit_nodes, effective_exit_pair_min_count)) {
                return false;
            }
        }
    }

    return true;
}

// Helper function to build detailed inter-mesh mapping error message
std::string build_inter_mesh_mapping_error_message(
    unsigned int retry_attempt,
    const std::vector<MeshId>& logical_meshes,
    const std::vector<MeshId>& physical_meshes,
    ::tt::tt_fabric::ConnectionValidationMode inter_mesh_validation_mode,
    const std::string& solver_error_message,
    const std::vector<std::pair<MeshId, MeshId>>& failed_mesh_pairs) {
    // Build logical meshes string
    std::string logical_meshes_str;
    bool first = true;
    for (const auto& mesh_id : logical_meshes) {
        if (!first) {
            logical_meshes_str += ", ";
        }
        first = false;
        logical_meshes_str += std::to_string(mesh_id.get());
    }

    // Build physical meshes string
    std::string physical_meshes_str;
    first = true;
    for (const auto& mesh_id : physical_meshes) {
        if (!first) {
            physical_meshes_str += ", ";
        }
        first = false;
        physical_meshes_str += std::to_string(mesh_id.get());
    }

    // Build failed pairs string
    std::string failed_pairs_str;
    if (!failed_mesh_pairs.empty()) {
        failed_pairs_str = " Failed mesh pairs from previous attempts: [";
        first = true;
        for (const auto& [logical_id, physical_id] : failed_mesh_pairs) {
            if (!first) {
                failed_pairs_str += ", ";
            }
            first = false;
            failed_pairs_str += fmt::format("(logical={}, physical={})", logical_id.get(), physical_id.get());
        }
        failed_pairs_str += "].";
    }

    // Convert validation mode to string
    std::string validation_mode_str;
    switch (inter_mesh_validation_mode) {
        case ::tt::tt_fabric::ConnectionValidationMode::STRICT: validation_mode_str = "STRICT"; break;
        case ::tt::tt_fabric::ConnectionValidationMode::RELAXED: validation_mode_str = "RELAXED"; break;
    }

    return fmt::format(
        "Inter-mesh mapping failed after {} attempt(s). "
        "Logical meshes being mapped: [{}] ({} total). "
        "Physical meshes available: [{}] ({} total). "
        "Failed mesh pair configurations tried: {} out of {} possible combinations. "
        "Inter-mesh validation mode: {}. "
        "Solver error: {}.{}",
        retry_attempt,
        logical_meshes_str,
        logical_meshes.size(),
        physical_meshes_str,
        physical_meshes.size(),
        failed_mesh_pairs.size(),
        logical_meshes.size() * physical_meshes.size(),
        validation_mode_str,
        solver_error_message,
        failed_pairs_str);
}

// Helper function to handle adding forbidden constraint and check if mapping should continue
// Returns false if mapping should return early (overconstrained), true if should continue
bool handle_forbidden_constraint(
    ::tt::tt_fabric::MappingConstraints<MeshId, MeshId>& inter_mesh_constraints,
    MeshId logical_mesh_id,
    MeshId physical_mesh_id,
    std::vector<std::pair<MeshId, MeshId>>& failed_mesh_pairs,
    std::vector<std::pair<MeshId, MeshId>>& current_attempt_failed_pairs,
    unsigned int retry_attempt,
    const std::vector<MeshId>& logical_meshes,
    const std::vector<MeshId>& physical_meshes,
    ::tt::tt_fabric::ConnectionValidationMode inter_mesh_validation_mode,
    TopologyMappingResult& result,
    const std::string& error_context) {
    if (!inter_mesh_constraints.add_forbidden_constraint(logical_mesh_id, physical_mesh_id)) {
        // If adding forbidden constraint causes overconstrained nodes (no valid mappings left),
        // this means we've exhausted all possibilities for this logical mesh.
        // Treat this as a failure and return with an appropriate error message.
        // Update failed pairs to include the current one that caused the failure
        failed_mesh_pairs.insert(
            failed_mesh_pairs.end(), current_attempt_failed_pairs.begin(), current_attempt_failed_pairs.end());
        failed_mesh_pairs.emplace_back(logical_mesh_id, physical_mesh_id);

        // Count how many times this logical mesh failed to map
        size_t failed_count_for_this_mesh = 0;
        for (const auto& [log_id, phys_id] : failed_mesh_pairs) {
            if (log_id == logical_mesh_id) {
                failed_count_for_this_mesh++;
            }
        }

        log_info(
            tt::LogFabric,
            "Multi-mesh mapping failed after {} attempt(s): Tried {} different mesh configurations. "
            "Logical mesh {} failed to map to {} out of {} physical meshes. "
            "Total failed mesh pair combinations: {}",
            retry_attempt,
            failed_mesh_pairs.size(),
            logical_mesh_id.get(),
            failed_count_for_this_mesh,
            physical_meshes.size(),
            failed_mesh_pairs.size());

        std::string solver_error_message = fmt::format(
            "All mapping possibilities exhausted for logical mesh {} after trying {} different mesh "
            "configurations. "
            "{}: failed to add forbidden constraint",
            logical_mesh_id.get(),
            failed_mesh_pairs.size(),
            error_context);

        result.success = false;
        result.error_message = build_inter_mesh_mapping_error_message(
            retry_attempt,
            logical_meshes,
            physical_meshes,
            inter_mesh_validation_mode,
            solver_error_message,
            failed_mesh_pairs);
        return false;  // Indicate that mapping should return early
    }
    current_attempt_failed_pairs.emplace_back(logical_mesh_id, physical_mesh_id);
    return true;  // Indicate that mapping should continue
}

}  // anonymous namespace

namespace {

template <typename NodeId>
std::string format_adjacency_degree_histogram(const AdjacencyGraph<NodeId>& graph) {
    std::map<std::size_t, std::size_t> degree_hist;
    for (const auto& node : graph.get_nodes()) {
        const auto& neighbors = graph.get_neighbors(node);
        std::set<NodeId> unique_neighbors(neighbors.begin(), neighbors.end());
        degree_hist[unique_neighbors.size()]++;
    }

    std::string hist_str = "{";
    bool first = true;
    for (const auto& [degree, count] : degree_hist) {
        if (!first) {
            hist_str += ", ";
        }
        first = false;
        hist_str += fmt::format("{}:{}", degree, count);
    }
    hist_str += "}";
    return hist_str;
}

template <typename NodeId>
std::string format_intra_mesh_degree_histograms(const std::map<MeshId, AdjacencyGraph<NodeId>>& mesh_graphs) {
    if (mesh_graphs.empty()) {
        return "(none)";
    }

    std::string hist_str;
    bool first = true;
    for (const auto& [mesh_id, graph] : mesh_graphs) {
        if (!first) {
            hist_str += ", ";
        }
        first = false;
        hist_str += fmt::format("mesh{} {}", mesh_id.get(), format_adjacency_degree_histogram(graph));
    }
    return hist_str;
}

}  // namespace

void log_logical_multi_mesh_adjacency_histograms(const LogicalMultiMeshGraph& multi_mesh_graph) {
    log_info(
        tt::LogFabric,
        "Logical multi-mesh adjacency: intermesh degree histogram {}; intra-mesh degree histograms {}",
        format_adjacency_degree_histogram(multi_mesh_graph.mesh_level_graph_),
        format_intra_mesh_degree_histograms(multi_mesh_graph.mesh_adjacency_graphs_));
}

void log_physical_multi_mesh_adjacency_histograms(const PhysicalMultiMeshGraph& multi_mesh_graph) {
    log_info(
        tt::LogFabric,
        "Physical multi-mesh adjacency: intermesh degree histogram {}; intra-mesh degree histograms {}",
        format_adjacency_degree_histogram(multi_mesh_graph.mesh_level_graph_),
        format_intra_mesh_degree_histograms(multi_mesh_graph.mesh_adjacency_graphs_));
}

TopologyMappingResult map_multi_mesh_to_physical(
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const TopologyMappingConfig& config,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    using namespace ::tt::tt_fabric;

    if (config.strict_mode) {
        log_warning(
            tt::LogFabric,
            "TopologyMappingConfig::strict_mode is deprecated and has no effect. "
            "Set mesh_validation_modes and/or inter_mesh_validation_mode explicitly.");
    }

    // Step 1: Run Mesh to Mesh mapping algorithm
    const auto& mesh_logical_graph = adjacency_map_logical.mesh_level_graph_;
    const auto& mesh_physical_graph = adjacency_map_physical.mesh_level_graph_;

    // Build inter-mesh constraints and determine validation mode
    auto inter_mesh_constraints =
        build_inter_mesh_constraints(config, adjacency_map_physical, mesh_logical_graph, asic_id_to_mesh_rank);
    auto inter_mesh_validation_mode = determine_inter_mesh_validation_mode(config);

    // Track statistics for error reporting
    unsigned int retry_attempt = 0;
    std::vector<std::pair<MeshId, MeshId>> failed_mesh_pairs;
    std::vector<MeshId> logical_meshes;
    std::vector<MeshId> physical_meshes;

    // Collect logical and physical mesh IDs for error reporting
    for (const auto& mesh_id : mesh_logical_graph.get_nodes()) {
        logical_meshes.push_back(mesh_id);
    }
    for (const auto& mesh_id : mesh_physical_graph.get_nodes()) {
        physical_meshes.push_back(mesh_id);
    }

    // Log initial mapping setup
    log_trace(
        tt::LogFabric,
        "Starting multi-mesh mapping: {} logical mesh(es) to {} physical mesh(es)",
        logical_meshes.size(),
        physical_meshes.size());

    bool success = false;

    TopologyMappingResult result;

    // Incremental inter-mesh enumeration: encode hard CNF once, then append blocking clauses for each
    // rejected full mesh-level assignment (intra-mesh failure) instead of re-solving from scratch.
    TopologyMappingEnumerationSession<MeshId, MeshId> inter_mesh_session;

    // Maximum retry attempts to prevent infinite loops
    // This should be sufficient for most cases: if we have N logical meshes and M physical meshes,
    // worst case is N*M attempts (trying each logical mesh with each physical mesh)
    const unsigned int max_retry_attempts = (logical_meshes.size() * physical_meshes.size()) + 1;
    log_debug(tt::LogFabric, "Maximum retry attempts: {}", max_retry_attempts);

    while (!success) {
        retry_attempt++;

        // Safety check to prevent infinite loops
        if (retry_attempt > max_retry_attempts) {
            log_info(
                tt::LogFabric, "Multi-mesh mapping failed: Maximum retry attempts ({}) exceeded", max_retry_attempts);
            result.success = false;
            result.error_message = build_inter_mesh_mapping_error_message(
                retry_attempt - 1,
                logical_meshes,
                physical_meshes,
                inter_mesh_validation_mode,
                fmt::format(
                    "Maximum retry attempts ({}) exceeded. This indicates a problem with the mapping constraints or "
                    "topology.",
                    max_retry_attempts),
                failed_mesh_pairs);
            return result;
        }

        // Use quiet mode for retry attempts (failures are expected during retries)
        // Only log errors if this is the final attempt
        bool quiet_mode = (retry_attempt < max_retry_attempts);

        log_trace(
            tt::LogFabric,
            "Multi-mesh mapping attempt {}/{}: Trying inter-mesh mapping",
            retry_attempt,
            max_retry_attempts);
        if (!failed_mesh_pairs.empty()) {
            log_debug(tt::LogFabric, "Failed mesh pairs from previous attempts: {}", failed_mesh_pairs.size());
        }

        // Perform inter-mesh mapping (incremental: reuses prior SAT encoding across retries)
        auto solver_result = ::tt::tt_fabric::solve_topology_mapping(
            mesh_logical_graph, mesh_physical_graph, inter_mesh_constraints, inter_mesh_validation_mode, quiet_mode);

        // If the solver fails, return error results for all meshes with detailed information
        if (!solver_result.success) {
            log_info(tt::LogFabric, "Multi-mesh mapping attempt {} failed: Inter-mesh mapping failed", retry_attempt);
            log_debug(tt::LogFabric, "Inter-mesh mapping error: {}", solver_result.error_message);
            result.success = false;
            result.error_message = build_inter_mesh_mapping_error_message(
                retry_attempt,
                logical_meshes,
                physical_meshes,
                inter_mesh_validation_mode,
                solver_result.error_message,
                failed_mesh_pairs);
            return result;
        }

        // Log successful inter-mesh mapping
        log_trace(
            tt::LogFabric,
            "Attempt {}: Inter-mesh mapping succeeded, found {} mesh pair(s)",
            retry_attempt,
            solver_result.target_to_global.size());

        unsigned int mapped_mesh_pairs = 0;
        std::vector<std::pair<MeshId, MeshId>> current_attempt_failed_pairs;

        auto reject_mesh_pair_mapping =
            [&](MeshId logical_mesh_id, MeshId physical_mesh_id, const std::string& failure_reason) -> bool {
            log_trace(
                tt::LogFabric,
                "Attempt {}: Rejecting logical mesh {} -> physical mesh {} mapping: {}",
                retry_attempt,
                logical_mesh_id.get(),
                physical_mesh_id.get(),
                failure_reason);
            return handle_forbidden_constraint(
                inter_mesh_constraints,
                logical_mesh_id,
                physical_mesh_id,
                failed_mesh_pairs,
                current_attempt_failed_pairs,
                retry_attempt,
                logical_meshes,
                physical_meshes,
                inter_mesh_validation_mode,
                result,
                fmt::format(
                    "Intra-mesh mapping failure for logical mesh {} -> physical mesh {}: {}",
                    logical_mesh_id.get(),
                    physical_mesh_id.get(),
                    failure_reason));
        };

        // Step 2: For each mesh mapping, do the sub mapping for fabric node id to asic id
        std::unordered_map<MeshId, MeshId> mesh_mappings(
            solver_result.target_to_global.begin(), solver_result.target_to_global.end());
        log_trace(tt::LogFabric, "Attempt {}: Mapping {} mesh pair(s)", retry_attempt, mesh_mappings.size());
        for (const auto& [logical_mesh_id, physical_mesh_id] : mesh_mappings) {
            log_trace(
                tt::LogFabric,
                "Attempt {}: Mapping mesh {} -> {}",
                retry_attempt,
                logical_mesh_id.get(),
                physical_mesh_id.get());
            // Get the logical graph and the physical graph
            const auto& logical_graph = adjacency_map_logical.mesh_adjacency_graphs_.at(logical_mesh_id);
            const auto& physical_graph = adjacency_map_physical.mesh_adjacency_graphs_.at(physical_mesh_id);

            // Get logical exit node graph (safe access - use empty graph if not initialized)
            // Some tests create LogicalMultiMeshGraph manually without initializing exit node graphs
            const AdjacencyGraph<LogicalExitNode>* logical_exit_node_graph_ptr = nullptr;
            auto logical_exit_node_it = adjacency_map_logical.mesh_exit_node_graphs_.find(logical_mesh_id);
            if (logical_exit_node_it != adjacency_map_logical.mesh_exit_node_graphs_.end()) {
                logical_exit_node_graph_ptr = &logical_exit_node_it->second;
            } else {
                // Use a temporary empty graph if not found
                static const AdjacencyGraph<LogicalExitNode> empty_logical_exit_node_graph;
                logical_exit_node_graph_ptr = &empty_logical_exit_node_graph;
            }
            const auto& logical_exit_node_graph = *logical_exit_node_graph_ptr;

            // Get physical exit node graph (safe access - use empty graph if not initialized)
            // Some tests create PhysicalMultiMeshGraph manually without initializing exit node graphs
            const AdjacencyGraph<PhysicalExitNode>* physical_exit_node_graph_ptr = nullptr;
            auto physical_exit_node_it = adjacency_map_physical.mesh_exit_node_graphs_.find(physical_mesh_id);
            if (physical_exit_node_it != adjacency_map_physical.mesh_exit_node_graphs_.end()) {
                physical_exit_node_graph_ptr = &physical_exit_node_it->second;
            } else {
                // Use a temporary empty graph if not found
                static const AdjacencyGraph<PhysicalExitNode> empty_physical_exit_node_graph;
                physical_exit_node_graph_ptr = &empty_physical_exit_node_graph;
            }
            const auto& physical_exit_node_graph = *physical_exit_node_graph_ptr;

            // Build intra-mesh constraints
            ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID> intra_mesh_constraints;

            // Add rank binding constraints only when rank bindings are enabled
            if (!config.disable_rank_bindings) {
                add_rank_binding_constraints(
                    intra_mesh_constraints, config, logical_mesh_id, fabric_node_id_to_mesh_rank, asic_id_to_mesh_rank);
            }

            // Add exit node constraints (only if exit node graphs are not empty)
            // Since we initialize empty graphs for all meshes, we check if they have nodes before adding constraints
            if (!logical_exit_node_graph.get_nodes().empty() && !physical_exit_node_graph.get_nodes().empty()) {
                bool exit_node_constraints_success = add_exit_node_constraints(
                    intra_mesh_constraints,
                    mesh_mappings,
                    logical_graph,
                    logical_exit_node_graph,
                    physical_exit_node_graph,
                    inter_mesh_validation_mode);

                if (!exit_node_constraints_success) {
                    if (!reject_mesh_pair_mapping(
                            logical_mesh_id,
                            physical_mesh_id,
                            "exit node constraints cannot be satisfied (no valid physical exit nodes or "
                            "over-constrained)")) {
                        return result;
                    }
                    continue;
                }
            }

            // Build ASIC positions map and add pinning constraints
            auto asic_positions_to_asic_ids = build_asic_positions_map(physical_graph, config);

            auto pinning_constraint_failure =
                add_pinning_constraints(intra_mesh_constraints, asic_positions_to_asic_ids, config, logical_mesh_id);

            if (pinning_constraint_failure.has_value()) {
                if (!reject_mesh_pair_mapping(logical_mesh_id, physical_mesh_id, pinning_constraint_failure.value())) {
                    return result;
                }
                continue;
            }

            // When the physical graph carries PGD-derived pinnings, bias the intra-mesh solve toward the
            // PGD-chosen layout via PREFERRED (soft) constraints. Added AFTER the hard rank/exit/MGD-pin
            // constraints above so they only influence ASIC choice where the hard constraints leave freedom.
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

            // Determine validation mode
            auto validation_mode = determine_intra_mesh_validation_mode(config, logical_mesh_id);

            // Perform the sub mapping for the fabric node id to the asic id
            auto sub_mapping = ::tt::tt_fabric::solve_topology_mapping(
                logical_graph, physical_graph, intra_mesh_constraints, validation_mode, quiet_mode);

            if (!sub_mapping.success) {
                const std::string failure_reason =
                    sub_mapping.error_message.empty() ? "intra-mesh topology solver failed" : sub_mapping.error_message;
                if (!reject_mesh_pair_mapping(logical_mesh_id, physical_mesh_id, failure_reason)) {
                    return result;
                }
            } else {
                mapped_mesh_pairs++;
                // Add the mapping to the result using MappingResult directly
                for (const auto& [fabric_node, asic] : sub_mapping.target_to_global) {
                    result.fabric_node_to_asic.insert({fabric_node, asic});
                    result.asic_to_fabric_node.insert({asic, fabric_node});
                }
            }
        }

        // Update failed pairs list
        failed_mesh_pairs.insert(
            failed_mesh_pairs.end(), current_attempt_failed_pairs.begin(), current_attempt_failed_pairs.end());

        // If all mesh pairs were mapped we can stop the loop
        if (mapped_mesh_pairs == mesh_mappings.size()) {
            success = true;
            log_trace(
                tt::LogFabric,
                "Multi-mesh mapping succeeded after {} attempt(s): {} mesh pair(s) mapped",
                retry_attempt,
                mapped_mesh_pairs);
        } else {
            // Remove all the results that were added so far and start over
            log_trace(
                tt::LogFabric,
                "Attempt {}: Only {}/{} mesh pair(s) mapped, retrying",
                retry_attempt,
                mapped_mesh_pairs,
                mesh_mappings.size());
            result.fabric_node_to_asic.clear();
            result.asic_to_fabric_node.clear();
        }
    }

    result.success = success;

    return result;
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

namespace {

// Complete the intra-mesh (fabric-node -> ASIC) mapping for one fixed inter-mesh placement.
//
// Mirrors the per-mesh-pair intra-mesh solve inside map_multi_mesh_to_physical, minus the retry/forbid
// machinery: if any mesh pair's intra-mesh mapping is infeasible, the whole placement is rejected
// (returned result has success == false). Callers that enumerate placements simply skip rejected ones.
TopologyMappingResult complete_intra_mesh_for_placement(
    const std::unordered_map<MeshId, MeshId>& mesh_mappings,
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const TopologyMappingConfig& config,
    ::tt::tt_fabric::ConnectionValidationMode inter_mesh_validation_mode,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
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
            const bool exit_node_constraints_success = add_exit_node_constraints(
                intra_mesh_constraints,
                mesh_mappings,
                logical_graph,
                logical_exit_node_graph,
                physical_exit_node_graph,
                inter_mesh_validation_mode);
            if (!exit_node_constraints_success) {
                result.success = false;
                return result;
            }
        }

        auto asic_positions_to_asic_ids = build_asic_positions_map(physical_graph, config);
        auto pinning_constraint_failure =
            add_pinning_constraints(intra_mesh_constraints, asic_positions_to_asic_ids, config, logical_mesh_id);
        if (pinning_constraint_failure.has_value()) {
            result.success = false;
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
            result.success = false;
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

// Canonical, order-independent signature of a full mapping (std::map iteration is sorted), used to
// deduplicate solutions that come out of distinct inter-mesh placements but resolve to the same assignment.
std::string full_mapping_signature(const TopologyMappingResult& result) {
    std::string signature;
    signature.reserve(result.fabric_node_to_asic.size() * 24);
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        signature += fmt::format("{}:{}->{};", *fabric_node.mesh_id, fabric_node.chip_id, *asic);
    }
    return signature;
}

}  // namespace

std::vector<TopologyMappingResult> map_multi_mesh_to_physical_n(
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const TopologyMappingConfig& config,
    std::size_t max_solutions,
    bool unique_shapes,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    using namespace ::tt::tt_fabric;

    std::vector<TopologyMappingResult> solutions;

    const auto& mesh_logical_graph = adjacency_map_logical.mesh_level_graph_;
    const auto& mesh_physical_graph = adjacency_map_physical.mesh_level_graph_;

    auto inter_mesh_constraints =
        build_inter_mesh_constraints(config, adjacency_map_physical, mesh_logical_graph, asic_id_to_mesh_rank);
    auto inter_mesh_validation_mode = determine_inter_mesh_validation_mode(config);

    // Enumerate distinct inter-mesh placements (which physical meshes / hosts host each logical mesh).
    // max_solutions is passed straight through: 0 means "all up to the solver's internal safety cap".
    // unique_shapes collapses placements that occupy the same physical-mesh set.
    std::vector<MappingResult<MeshId, MeshId>> placements = ::tt::tt_fabric::solve_topology_mapping_n(
        mesh_logical_graph,
        mesh_physical_graph,
        inter_mesh_constraints,
        /*max_solutions=*/max_solutions,
        inter_mesh_validation_mode,
        /*quiet_mode=*/true,
        TopologyMappingSolverEngine::Auto,
        unique_shapes);

    log_info(
        tt::LogFabric,
        "map_multi_mesh_to_physical_n: enumerated {} inter-mesh placement(s) (max_solutions={}, unique_shapes={})",
        placements.size(),
        max_solutions,
        unique_shapes);

    std::set<std::string> seen_signatures;
    for (const auto& placement : placements) {
        if (!placement.success) {
            continue;
        }

        std::unordered_map<MeshId, MeshId> mesh_mappings(
            placement.target_to_global.begin(), placement.target_to_global.end());

        TopologyMappingResult full = complete_intra_mesh_for_placement(
            mesh_mappings,
            adjacency_map_logical,
            adjacency_map_physical,
            config,
            inter_mesh_validation_mode,
            asic_id_to_mesh_rank,
            fabric_node_id_to_mesh_rank);
        if (!full.success) {
            continue;
        }

        // Skip solutions whose full assignment we have already emitted.
        if (!seen_signatures.insert(full_mapping_signature(full)).second) {
            continue;
        }

        solutions.push_back(std::move(full));
        if (max_solutions != 0 && solutions.size() >= max_solutions) {
            break;
        }
    }

    log_info(tt::LogFabric, "map_multi_mesh_to_physical_n: returning {} distinct solution(s)", solutions.size());
    return solutions;
}

}  // namespace tt::tt_metal::experimental::tt_fabric
