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

#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal::experimental::tt_fabric {

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
                for ([[maybe_unused]] const auto& eth_conn : eth_connections) {
                    flat_adj[src_asic_id].push_back(dst_asic_id);
                }
            }
        }
    }

    return flat_adj;
}

PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor) {
    using namespace ::tt::tt_fabric;

    log_info(tt::LogFabric, "Building flat adjacency map from PSD");

    // Build flat adjacency once; reused for find_all_in_psd and build_hierarchical_from_flat_graph (avoids a second
    // O(|PSD|) pass inside find_all_in_psd).
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(build_flat_adjacency_map_from_psd(physical_system_descriptor));

    // Dense 0..N-1 indices for ASICs in this PSD (fast bitsets for disjoint packing).
    std::unordered_map<tt::tt_metal::AsicID, std::uint32_t> asic_to_dense_index;
    {
        const auto& flat_nodes = flat_graph.get_nodes();
        asic_to_dense_index.reserve(flat_nodes.size() * 2);
        for (std::uint32_t i = 0; i < flat_nodes.size(); ++i) {
            asic_to_dense_index.emplace(flat_nodes[i], i);
        }
    }
    const std::uint32_t cluster_asic_count = static_cast<std::uint32_t>(flat_graph.get_nodes().size());
    const std::size_t used_asic_word_count = (static_cast<std::size_t>(cluster_asic_count) + 63u) / 64u;

    log_info(tt::LogFabric, "Getting valid groupings map from MGD and PGD");

    // Get valid groupings map from MGD and PGD
    auto valid_groupings_map =
        physical_grouping_descriptor.get_valid_groupings_for_mgd(mesh_graph_descriptor, physical_system_descriptor);

    log_info(tt::LogFabric, "Got {} valid groupings map from MGD and PGD", valid_groupings_map.size());

    // Get groupings for mesh level mappings
    TT_FATAL(valid_groupings_map.contains("MESH"), "Internal error: MESH grouping not found in valid groupings map");
    TT_FATAL(
        !valid_groupings_map.at("MESH").empty(),
        "Internal error: Physical grouping descriptor was not able to find mesh groupings");

    using GroupKey = size_t;

    // Find all possible mappings of mesh groupings to the PSD
    std::vector<std::unordered_set<tt::tt_metal::AsicID>> groupings_by_index;
    std::unordered_map<std::string, std::unordered_set<GroupKey>> mesh_type_to_index;
    std::unordered_map<std::string, std::size_t> mesh_type_num_instances;
    std::unordered_map<std::string, PhysicalMultiMeshGraph> mesh_type_to_physical_graph;
    std::unordered_map<MeshId, std::vector<std::unordered_set<tt::tt_metal::AsicID>>> mesh_id_to_placed_groupings;

    // Break each mesh down to a physical multi-mesh graph
    for (const auto& [mesh_name, groupings] : valid_groupings_map.at("MESH")) {
        // Find all possible mappings of mesh groupings to the PSD
        const auto placed_groupings =
            physical_grouping_descriptor.find_all_in_psd(groupings, physical_system_descriptor);

        // Count the number of instances of this mesh
        mesh_type_num_instances[mesh_name] = mesh_graph_descriptor.instances_by_name(mesh_name).size();

        // Add the groupings to the index
        for (const auto& placed_grouping : placed_groupings) {
            const GroupKey index = groupings_by_index.size();
            groupings_by_index.push_back(placed_grouping);
            mesh_type_to_index[mesh_name].insert(index);
        }

        // Build a physical multi-mesh graph for this mesh; keep the same per-mesh-instance ASIC sets keyed by logical
        // MeshId (instance local_id) for later lookups without mesh descriptor name.
        mesh_type_to_physical_graph[mesh_name] = build_hierarchical_from_flat_graph(flat_graph, placed_groupings);
        for (::tt::tt_fabric::GlobalNodeId gid : mesh_graph_descriptor.instances_by_name(mesh_name)) {
            const auto& inst = mesh_graph_descriptor.get_instance(gid);
            if (inst.kind != ::tt::tt_fabric::NodeKind::Mesh) {
                continue;
            }
            mesh_id_to_placed_groupings[MeshId(inst.local_id)] = placed_groupings;
        }
    }

    // Fast path for single mesh shape
    const auto& mesh_shape_entries = valid_groupings_map.at("MESH");
    if (mesh_shape_entries.size() == 1) {
        const std::string& sole_mesh_name = mesh_shape_entries.begin()->first;
        const auto sole_it = mesh_type_to_physical_graph.find(sole_mesh_name);
        TT_FATAL(
            sole_it != mesh_type_to_physical_graph.end(),
            "Single mesh shape '{}' missing PSD-derived PhysicalMultiMeshGraph",
            sole_mesh_name);
        log_info(
            tt::LogFabric,
            "Single mesh descriptor shape '{}': skipping mesh-level solve and disjoint packing",
            sole_mesh_name);
        return sole_it->second;
    }

    // For each mesh shape that's in the MGD
    const auto [mgd_intermesh_mesh_level, _] = get_requested_intermesh_from_mgd(mesh_graph_descriptor);
    std::unordered_map<std::string, AdjacencyGraph<MeshId>> mesh_to_logical_graph;
    // Per-mesh-descriptor lazy enumeration state. Solutions are pulled one at a time via session.next() and cached
    // here so the round-robin diagonal search below can revisit them without redoing solver work.
    //
    // Each cached solution carries a precomputed *full-width* ASIC bitset (vector<uint64_t> of length
    // used_asic_word_count). Disjointness, set, and clear during the recursive packing search become tight
    // word-by-word loops over fixed-size vectors instead of sparse index lookups — that mirrors the dense bitset
    // intent of the original eager implementation while keeping the lazy enumeration semantics.
    struct MeshEnumState {
        AdjacencyGraph<MeshId> logical_graph;
        AdjacencyGraph<MeshId> physical_graph;
        MappingConstraints<MeshId, MeshId> constraints;
        TopologyMappingEnumerationSession<MeshId, MeshId> session;
        std::vector<std::map<MeshId, MeshId>> excluded;        // fed back into session.next()
        std::vector<MappingResult<MeshId, MeshId>> solutions;  // cached results, one per pulled solution
        std::vector<std::vector<std::uint64_t>> bitset_sets;   // parallel: solution -> full-width ASIC bitset
        std::vector<std::size_t> embedding_sizes;              // parallel: target_to_global.size() for each solution
        bool exhausted = false;
    };
    std::unordered_map<std::string, MeshEnumState> mesh_enum_states;
    // PGD/PSD mesh placement rows are keyed in mesh_id_to_placed_groupings by MGD instance local_id only.
    // Mesh-level solutions may use logical MeshIds beyond those ids when we expand the pattern graph — always resolve
    // placed_groupings through the first mesh instance of this descriptor name.
    std::unordered_map<std::string, MeshId> mesh_name_to_placed_groupings_anchor;

    // Expand MGD mesh-level pattern to match PSD coarse placement count only when this MGD/PGD path solves exactly one
    // mesh descriptor name against PSD. Using the full physical coarse graph as the logical pattern makes each
    // solution's dense ASIC set span every placement row for that descriptor; that is correct for packing one shape
    // (e.g. single BH galaxy → N meshes) but makes dense sets overlap the entire PSD for each descriptor, so pairwise-
    // disjoint packing across multiple mesh types becomes impossible.
    std::size_t mesh_descriptor_names_with_psd_mesh = 0;
    for (const auto& mn : mesh_graph_descriptor.get_all_mesh_names()) {
        if (mesh_type_to_physical_graph.contains(mn)) {
            ++mesh_descriptor_names_with_psd_mesh;
        }
    }

    for (const auto& mesh_name : mesh_graph_descriptor.get_all_mesh_names()) {
        const auto physical_it = mesh_type_to_physical_graph.find(mesh_name);
        if (physical_it == mesh_type_to_physical_graph.end()) {
            log_warning(
                tt::LogFabric,
                "No PSD-derived PhysicalMultiMeshGraph for mesh descriptor '{}'; skipping mesh-level topology solve",
                mesh_name);
            continue;
        }
        const AdjacencyGraph<MeshId>& physical_mesh_level = physical_it->second.mesh_level_graph_;

        // Mesh-level subgraph from MGD intermesh edges (nodes = mesh instance local_ids in the descriptor).
        AdjacencyGraph<MeshId> mgd_mesh_level_graph = build_mgd_mesh_level_subgraph_for_mesh_descriptor_name(
            mesh_graph_descriptor, mesh_name, mgd_intermesh_mesh_level);

        // When the descriptor lists fewer mesh instances than PGD/PSD exposes as distinct coarse placements (e.g. one
        // top_level_instance while find_all_in_psd yields 16 disjoint meshes), solving with only the MGD subgraph
        // maps a single logical mesh node onto one physical placement. Duplicate the PSD-derived coarse topology as the
        // mapping pattern so every placement can appear in target_to_global without editing the MGD instance count.
        // Only when |mesh descriptors with PSD| == 1 — see mesh_descriptor_names_with_psd_mesh above.
        const bool expand_mgd_logical_to_match_psd_coarse_meshes =
            mesh_descriptor_names_with_psd_mesh == 1 &&
            mgd_mesh_level_graph.get_nodes().size() < physical_mesh_level.get_nodes().size();
        AdjacencyGraph<MeshId> logical_mesh_level_graph = mgd_mesh_level_graph;
        if (expand_mgd_logical_to_match_psd_coarse_meshes) {
            logical_mesh_level_graph = physical_mesh_level;
        }
        mesh_to_logical_graph[mesh_name] = logical_mesh_level_graph;

        MeshId placed_groupings_lookup_mesh_id{0};
        bool found_mesh_instance = false;
        for (::tt::tt_fabric::GlobalNodeId gid : mesh_graph_descriptor.instances_by_name(mesh_name)) {
            const auto& inst = mesh_graph_descriptor.get_instance(gid);
            if (inst.kind != ::tt::tt_fabric::NodeKind::Mesh) {
                continue;
            }
            placed_groupings_lookup_mesh_id = MeshId(inst.local_id);
            found_mesh_instance = true;
            break;
        }
        TT_FATAL(found_mesh_instance, "No mesh instances for descriptor '{}' in MGD", mesh_name);
        mesh_name_to_placed_groupings_anchor[mesh_name] = placed_groupings_lookup_mesh_id;

        // Stage the lazy enumeration session for this descriptor. No solver work is performed here — solutions are
        // pulled on demand below by the round-robin diagonal enumeration. unique_shapes=true matches the prior
        // solve_topology_mapping_n contract (image-set equivalence classes).
        MeshEnumState state;
        state.logical_graph = std::move(logical_mesh_level_graph);
        state.physical_graph = physical_mesh_level;
        mesh_enum_states.emplace(mesh_name, std::move(state));
    }

    // Stable mesh ordering for the diagonal enumeration. mesh_enum_states keys = descriptors that have a PSD-derived
    // physical graph; that's the same set the previous implementation packed across.
    std::vector<std::string> mesh_order;
    mesh_order.reserve(mesh_enum_states.size());
    for (const auto& [name, _state] : mesh_enum_states) {
        mesh_order.push_back(name);
    }
    std::sort(mesh_order.begin(), mesh_order.end());

    // Word-aligned ASIC bitset operations. Solutions carry full-width vectors so disjoint/set/clear are tight
    // word loops (vectorizable; one branch per word). This is the core of the speed parity with the eager
    // implementation: every recursive step touches O(used_asic_word_count) words instead of iterating sparse
    // dense-index lists.
    std::vector<std::uint64_t> used_asic_bits(used_asic_word_count, 0);
    auto bitset_disjoint = [used_asic_word_count](
                               const std::vector<std::uint64_t>& cand,
                               const std::vector<std::uint64_t>& occupied) -> bool {
        const std::uint64_t* a = cand.data();
        const std::uint64_t* b = occupied.data();
        for (std::size_t i = 0; i < used_asic_word_count; ++i) {
            if (a[i] & b[i]) {
                return false;
            }
        }
        return true;
    };
    auto bitset_or_into = [used_asic_word_count](
                              std::vector<std::uint64_t>& dst, const std::vector<std::uint64_t>& src) {
        std::uint64_t* d = dst.data();
        const std::uint64_t* s = src.data();
        for (std::size_t i = 0; i < used_asic_word_count; ++i) {
            d[i] |= s[i];
        }
    };
    auto bitset_andnot_from = [used_asic_word_count](
                                  std::vector<std::uint64_t>& dst, const std::vector<std::uint64_t>& src) {
        std::uint64_t* d = dst.data();
        const std::uint64_t* s = src.data();
        for (std::size_t i = 0; i < used_asic_word_count; ++i) {
            d[i] &= ~s[i];
        }
    };

    // Build the full-width ASIC bitset for one mesh-level solution against this descriptor's placed_groupings.
    // Length is fixed at used_asic_word_count for every cached solution so the recursive search can OR/AND-NOT
    // word-aligned vectors with no per-bit indirection.
    auto compute_bitset_for_solution = [&](const std::string& mesh_name,
                                           const MappingResult<MeshId, MeshId>& solution) {
        std::vector<std::uint64_t> bits(used_asic_word_count, 0);
        const auto& placed_groupings_for_mesh =
            mesh_id_to_placed_groupings.at(mesh_name_to_placed_groupings_anchor.at(mesh_name));
        for (const auto& [logical_mesh_id, physical_mesh_id] : solution.target_to_global) {
            TT_FATAL(
                physical_mesh_id.get() < placed_groupings_for_mesh.size(),
                "Physical mesh index {} out of range for placed_groupings (logical MeshId {})",
                physical_mesh_id.get(),
                logical_mesh_id.get());
            const auto& asics = placed_groupings_for_mesh[physical_mesh_id.get()];
            for (const auto& asic : asics) {
                auto di = asic_to_dense_index.find(asic);
                TT_FATAL(
                    di != asic_to_dense_index.end(), "ASIC from placement not found in PSD flat graph (dense index)");
                const std::uint32_t i = di->second;
                bits[i >> 6] |= (std::uint64_t{1} << (i & 63));
            }
        }
        return bits;
    };

    // Pull and cache the next mesh-level solution from a descriptor's enumeration session. Returns true iff a new
    // solution was appended; sets exhausted=true once next() reports failure (no more distinct embeddings). Logs
    // when a single SAT next() takes longer than kSlowPullThresholdMs so it's visible whether a long pause is
    // inside the SAT solver vs the disjoint-packing search.
    constexpr std::chrono::milliseconds kSlowPullThresholdMs{2000};
    auto pull_next_solution = [&](const std::string& mesh_name) -> bool {
        MeshEnumState& s = mesh_enum_states.at(mesh_name);
        if (s.exhausted) {
            return false;
        }
        const auto t_pull_begin = std::chrono::steady_clock::now();
        MappingResult<MeshId, MeshId> result = s.session.next(
            s.logical_graph,
            s.physical_graph,
            s.constraints,
            s.excluded,
            ConnectionValidationMode::STRICT,
            /*quiet_mode=*/true,
            TopologyMappingSolverEngine::Sat,
            /*unique_shapes=*/true);
        const auto pull_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_pull_begin);
        if (pull_ms >= kSlowPullThresholdMs) {
            log_info(
                tt::LogFabric,
                "Topology mapper: SAT next() for mesh '{}' took {}ms (success={}, cache size now {})",
                mesh_name,
                pull_ms.count(),
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
        std::vector<std::uint64_t> bits = compute_bitset_for_solution(mesh_name, result);
        s.excluded.push_back(result.target_to_global);
        s.bitset_sets.push_back(std::move(bits));
        s.embedding_sizes.push_back(result.target_to_global.size());
        s.solutions.push_back(std::move(result));
        return true;
    };

    // Diagonal round-robin enumeration over the per-mesh caches.
    //
    // Round k (k = 1, 2, ...): pull one new solution from each non-exhausted mesh, growing each cache to size k
    // (or to its terminal exhausted size). Then enumerate every "new" combination — those whose maximum chosen
    // *raw cache index* across descriptors equals k-1 — and check pairwise-disjoint ASIC bitsets. Combinations
    // from earlier rounds (max raw index < k-1) were already tested, so each combination is visited exactly once.
    // Within a round each descriptor iterates its cached solutions in `try_order` (largest target_to_global first,
    // breaking ties by smaller bitset cardinality is implicit via SAT enumeration order) so feasible packings tend
    // to be hit early; the frontier check stays on raw cache indices to preserve the visit-once invariant.
    //
    // Hot-path constants captured outside the recursion: pointers to MeshEnumState per depth, and per-round
    // try_order indices. This eliminates string hashing, std::function indirection, and repeated map lookups
    // inside the recursive packing — the original eager code's bitset DFS only touched plain arrays.
    const std::size_t n_meshes = mesh_order.size();
    std::vector<MeshEnumState*> mesh_state_ptrs(n_meshes, nullptr);
    for (std::size_t d = 0; d < n_meshes; ++d) {
        mesh_state_ptrs[d] = &mesh_enum_states.at(mesh_order[d]);
    }
    std::vector<std::vector<std::size_t>> try_order_per_depth(n_meshes);

    std::vector<std::size_t> chosen_index(n_meshes, 0);
    bool found_disjoint_combination = false;

    // Liveness instrumentation: count every full leaf combination considered (one disjoint check across all
    // descriptors). Emit a heartbeat every PackingSearch::kProgressInterval wall-clock seconds so long-running
    // searches don't appear hung. Counter is monotonic across rounds; the timer is checked every 4096 leaves to
    // keep the per-leaf overhead negligible (steady_clock::now() is not free).
    constexpr std::size_t kProgressTimerCheckMask = 4095;  // check timer every 4096 leaves (power-of-two mask)
    std::size_t combinations_tested = 0;
    const auto search_start_time = std::chrono::steady_clock::now();
    auto last_progress_log_time = search_start_time;

    // Recursive packing search. depth ∈ [0, n_meshes), iterates its descriptor's `try_order` filtered to raw
    // indices < round. The frontier (visit-once) check is applied at the leaf — a combination is "new" iff at
    // least one chosen raw index equals round-1.
    struct PackingSearch {
        const std::vector<MeshEnumState*>& mesh_state_ptrs;
        const std::vector<std::vector<std::size_t>>& try_order_per_depth;
        std::vector<std::size_t>& chosen_index;
        std::vector<std::uint64_t>& used_asic_bits;
        std::size_t round;
        std::size_t target_idx;
        std::size_t n_meshes;
        std::size_t& combinations_tested;
        std::size_t& timer_check_mask;
        std::chrono::steady_clock::time_point& last_progress_log_time;
        const std::chrono::steady_clock::time_point& search_start_time;
        // Lambdas captured by reference (no std::function indirection).
        decltype(bitset_disjoint)& bitset_disjoint_fn;
        decltype(bitset_or_into)& bitset_or_into_fn;
        decltype(bitset_andnot_from)& bitset_andnot_from_fn;
        std::chrono::seconds progress_interval{10};

        bool run(std::size_t depth, bool frontier_used) {
            const MeshEnumState& s = *mesh_state_ptrs[depth];
            const std::vector<std::size_t>& order = try_order_per_depth[depth];
            const std::size_t order_size = order.size();
            const bool is_leaf = (depth + 1 == n_meshes);
            if (is_leaf) {
                // Leaf: each iteration is one full combination. Skip whole leaf scan if frontier can't be hit.
                const bool can_hit_frontier_here = (target_idx < s.solutions.size());
                if (!frontier_used && !can_hit_frontier_here) {
                    return false;
                }
                for (std::size_t i = 0; i < order_size; ++i) {
                    const std::size_t si = order[i];
                    if (si >= round) {
                        // try_order is sorted but capped at raw index < round; entries beyond `round` (if any) are
                        // skipped. The expected layout is order_size == round, so this branch is rarely taken.
                        continue;
                    }
                    if (!frontier_used && si != target_idx) {
                        continue;  // frontier-must-hit-here: only si == target_idx satisfies the visit-once rule
                    }
                    ++combinations_tested;
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
                    if (!bitset_disjoint_fn(s.bitset_sets[si], used_asic_bits)) {
                        continue;
                    }
                    chosen_index[depth] = si;
                    return true;
                }
                return false;
            }
            for (std::size_t i = 0; i < order_size; ++i) {
                const std::size_t si = order[i];
                if (si >= round) {
                    continue;
                }
                const auto& cand = s.bitset_sets[si];
                if (!bitset_disjoint_fn(cand, used_asic_bits)) {
                    continue;
                }
                bitset_or_into_fn(used_asic_bits, cand);
                chosen_index[depth] = si;
                const bool next_frontier_used = frontier_used || (si == target_idx);
                if (run(depth + 1, next_frontier_used)) {
                    return true;
                }
                bitset_andnot_from_fn(used_asic_bits, cand);
            }
            return false;
        }
    };

    std::size_t timer_check_mask_ref = kProgressTimerCheckMask;
    auto last_round_log_time = search_start_time;
    constexpr std::chrono::seconds kRoundLogInterval{10};
    for (std::size_t round = 1; n_meshes != 0 && !found_disjoint_combination; ++round) {
        // Per-round pull phase: pull one more solution per non-exhausted mesh. A mesh becomes "exhausted" when
        // the solver reports no further embeddings; its cache size is then the upper bound contributed to all
        // subsequent rounds. Time the pull phase separately so a log line attributes any long pause to SAT
        // enumeration rather than the disjoint-packing search.
        const auto t_pull_phase_begin = std::chrono::steady_clock::now();
        bool any_progress = false;
        for (std::size_t d = 0; d < n_meshes; ++d) {
            if (pull_next_solution(mesh_order[d])) {
                any_progress = true;
            }
        }
        const auto pull_phase_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_pull_phase_begin)
                .count();
        // Periodic round summary every ~10s of wall clock so the user sees per-mesh cache growth even when no
        // combinations have been tested yet (e.g. during a slow first round of SAT enumeration).
        const auto now = std::chrono::steady_clock::now();
        if (round == 1 || (now - last_round_log_time) >= kRoundLogInterval || pull_phase_ms >= 1000) {
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
                pull_phase_ms,
                cache_summary,
                combinations_tested);
            last_round_log_time = now;
        }
        // Termination: if no mesh can contribute index `round-1` (i.e. every cache has size < round), the full
        // Cartesian product across all caches has already been enumerated by the previous round.
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
        // Per-depth try_order: raw cache indices [0, min(cache_size, round)) sorted by descending embedding size,
        // tie-break by ascending bitset population (smaller footprint preferred — leaves more room for other
        // descriptors). This restores the "prefer larger embedding" heuristic of the original eager implementation
        // while preserving diagonal correctness via the leaf-level frontier check on raw cache indices.
        for (std::size_t d = 0; d < n_meshes; ++d) {
            const MeshEnumState& s = *mesh_state_ptrs[d];
            const std::size_t hi = std::min<std::size_t>(s.solutions.size(), round);
            std::vector<std::size_t>& order = try_order_per_depth[d];
            order.resize(hi);
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                if (s.embedding_sizes[a] != s.embedding_sizes[b]) {
                    return s.embedding_sizes[a] > s.embedding_sizes[b];
                }
                return a < b;  // stable tiebreak by raw arrival order
            });
        }

        std::fill(used_asic_bits.begin(), used_asic_bits.end(), 0);
        const std::size_t combos_before_round = combinations_tested;
        PackingSearch search{
            mesh_state_ptrs,
            try_order_per_depth,
            chosen_index,
            used_asic_bits,
            round,
            round - 1,
            n_meshes,
            combinations_tested,
            timer_check_mask_ref,
            last_progress_log_time,
            search_start_time,
            bitset_disjoint,
            bitset_or_into,
            bitset_andnot_from};
        found_disjoint_combination = search.run(0, /*frontier_used=*/false);
        const std::size_t combos_this_round = combinations_tested - combos_before_round;
        log_debug(
            tt::LogFabric,
            "Topology mapper round-robin round {} complete: +{} combinations (total {}), found={}",
            round,
            combos_this_round,
            combinations_tested,
            found_disjoint_combination);
    }
    {
        const auto total_elapsed_sec =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - search_start_time)
                .count();
        log_info(
            tt::LogFabric,
            "Topology mapper round-robin finished: {} mesh-level combinations tested in {}s, success={}",
            combinations_tested,
            total_elapsed_sec,
            found_disjoint_combination);
    }

    PhysicalMultiMeshGraph result;
    if (found_disjoint_combination) {
        MeshId max_logical{0};
        for (size_t j = 0; j < mesh_order.size(); ++j) {
            const auto& mapping = mesh_enum_states.at(mesh_order[j]).solutions.at(chosen_index[j]).target_to_global;
            for (const auto& [logical_mesh_id, _] : mapping) {
                if (logical_mesh_id.get() > max_logical.get()) {
                    max_logical = logical_mesh_id;
                }
            }
        }

        std::vector<std::unordered_set<tt::tt_metal::AsicID>> combined_mesh_groupings(max_logical.get() + 1);
        for (size_t j = 0; j < mesh_order.size(); ++j) {
            const std::string& mesh_name = mesh_order[j];
            const MappingResult<MeshId, MeshId>& picked = mesh_enum_states.at(mesh_name).solutions.at(chosen_index[j]);
            TT_FATAL(!picked.target_to_global.empty(), "Empty mesh-level mapping for mesh descriptor '{}'", mesh_name);
            // placed_groupings rows are indexed by physical_mesh_id (within this descriptor's PSD placement).
            // Resolve via the anchor MGD instance: logical MeshIds after pattern expansion are not all registered in
            // mesh_id_to_placed_groupings.
            const auto& placed_groupings_for_mesh =
                mesh_id_to_placed_groupings.at(mesh_name_to_placed_groupings_anchor.at(mesh_name));
            for (const auto& [logical_mesh_id, physical_mesh_id] : picked.target_to_global) {
                TT_FATAL(
                    physical_mesh_id.get() < placed_groupings_for_mesh.size(),
                    "Physical mesh index {} out of range for placed_groupings (logical MeshId {})",
                    physical_mesh_id.get(),
                    logical_mesh_id.get());
                const auto& asics = placed_groupings_for_mesh[physical_mesh_id.get()];
                auto& slot = combined_mesh_groupings[logical_mesh_id.get()];
                slot.insert(asics.begin(), asics.end());
            }
        }
        result = build_hierarchical_from_flat_graph(flat_graph, combined_mesh_groupings);
    }

    if (!found_disjoint_combination) {
        TT_THROW("Topology mapper failed to find solution for mesh graph descriptors on this physical system");
    }

    return result;
}

PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // Build flat adjacency map from PhysicalSystemDescriptor
    PhysicalAdjacencyMap flat_adj = build_flat_adjacency_map_from_psd(physical_system_descriptor);

    // Convert asic_id_to_mesh_rank to mesh_groupings format
    // Find the maximum mesh ID to determine vector size
    if (asic_id_to_mesh_rank.empty()) {
        return PhysicalMultiMeshGraph{};
    }
    MeshId max_mesh_id{0};
    for (const auto& [mesh_id, _] : asic_id_to_mesh_rank) {
        if (mesh_id.get() > max_mesh_id.get()) {
            max_mesh_id = mesh_id;
        }
    }
    std::vector<std::unordered_set<tt::tt_metal::AsicID>> mesh_groupings(max_mesh_id.get() + 1);
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            mesh_groupings[mesh_id.get()].insert(asic_id);
        }
    }

    // Convert to AdjacencyGraph and use the common algorithm
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    PhysicalMultiMeshGraph result = build_hierarchical_from_flat_graph(flat_graph, mesh_groupings);

    return result;
}

PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::vector<std::unordered_set<tt::tt_metal::AsicID>>& mesh_groupings) {
    // Build asic_id_to_mesh_rank map from mesh groupings
    // Each element in mesh_groupings represents one mesh, with index becoming the MeshId
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (size_t i = 0; i < mesh_groupings.size(); ++i) {
        MeshId mesh_id{static_cast<uint32_t>(i)};
        for (const auto& asic_id : mesh_groupings[i]) {
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

// Minimal host cover for inter-mesh mapping: partition physical meshes by host, then apply.
// `rank_bound_logical_to_physical`: logical meshes already fixed by rank bindings → skip those and their physical
// meshes for host-alignment bias (empty when rank bindings are disabled).
// TODO: THis can be removed and replaced with cost hieristics when using a SAT solver because preferred constraints
// aren't very effective here
// https://github.com/tenstorrent/tt-metal/issues/40640
void add_inter_mesh_minimal_host_cover_from_hostname_map(
    const TopologyMappingConfig& config,
    const PhysicalMultiMeshGraph& physical_graph,
    const AdjacencyGraph<MeshId>& mesh_logical_level_graph,
    ::tt::tt_fabric::MappingConstraints<MeshId, MeshId>& inter_mesh_constraints,
    const std::map<MeshId, MeshId>& rank_bound_logical_to_physical) {
    if (config.hostname_to_asics.empty()) {
        return;
    }

    std::set<MeshId> bound_physical_mesh_ids;
    for (const auto& [_, physical_mesh_id] : rank_bound_logical_to_physical) {
        bound_physical_mesh_ids.insert(physical_mesh_id);
    }

    std::set<MeshId> logical_target_set;
    for (const MeshId& m : mesh_logical_level_graph.get_nodes()) {
        if (!rank_bound_logical_to_physical.contains(m)) {
            logical_target_set.insert(m);
        }
    }
    if (logical_target_set.size() <= 1) {
        return;
    }

    // Build global_mesh_groups in one pass: one group per host for single-host meshes, singleton for multi-host.
    std::vector<std::set<MeshId>> global_mesh_groups;
    std::map<std::string, std::size_t> host_group_index;
    for (const auto& [phys_mesh_id, adj] : physical_graph.mesh_adjacency_graphs_) {
        if (bound_physical_mesh_ids.contains(phys_mesh_id)) {
            continue;
        }
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
    if (!preferred_globals.empty()) {
        if (!single_group_fits) {
            log_debug(
                tt::LogFabric,
                "Inter-mesh host alignment: target count {} exceeds largest single partition; preferring minimal host "
                "cover ({} preferred globals)",
                logical_target_set.size(),
                preferred_globals.size());
        }
        for (const MeshId& target : logical_target_set) {
            inter_mesh_constraints.add_preferred_constraint(target, preferred_globals);
        }
    }
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

// Helper function to build inter-mesh constraints
// Maps logical meshes to physical meshes based on matching mesh host ranks
// A logical mesh maps to a physical mesh if the ASICs in that physical mesh have matching ranks
::tt::tt_fabric::MappingConstraints<MeshId, MeshId> build_inter_mesh_constraints(
    const TopologyMappingConfig& config,
    const PhysicalMultiMeshGraph& physical_graph,
    const AdjacencyGraph<MeshId>& mesh_logical_level_graph,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    ::tt::tt_fabric::MappingConstraints<MeshId, MeshId> inter_mesh_constraints;

    // Skip if rank bindings are disabled
    if (config.disable_rank_bindings) {
        add_inter_mesh_minimal_host_cover_from_hostname_map(
            config, physical_graph, mesh_logical_level_graph, inter_mesh_constraints, {});
        return inter_mesh_constraints;
    }

    std::map<MeshId, std::set<MeshId>> mesh_level_pinnings;
    for (const auto& [pos, fabric_node] : config.pinnings) {
        for (const auto& [physical_mesh_id, physical_mesh_graph] : physical_graph.mesh_adjacency_graphs_) {
            auto asic_position_map = build_asic_positions_map(physical_mesh_graph, config);
            if (asic_position_map.contains(pos)) {
                mesh_level_pinnings[fabric_node.mesh_id].insert(physical_mesh_id);
            }
        }
    }
    for (const auto& [mesh_id, physical_meshes] : mesh_level_pinnings) {
        if (!physical_meshes.empty()) {
            inter_mesh_constraints.add_required_constraint(mesh_id, physical_meshes);
        }
    }

    // Find the Physical graph mesh ID to asic id to mesh rank mapping based on the asics in the physical graph
    // Map: mesh_id_from_rank_map -> physical_mesh_id (from asic_id_to_mesh_rank)
    std::map<MeshId, MeshId> real_mesh_to_physical_mesh_id;
    for (const auto& [physical_mesh_id, physical_mesh_graph] : physical_graph.mesh_adjacency_graphs_) {
        const auto& asic_nodes = physical_mesh_graph.get_nodes();
        // Check that every asic node in physical mesh has a rank in asic_id_to_mesh_rank
        for (const auto& asic_id : asic_nodes) {
            for (const auto& [real_mesh_id, asic_ranks] : asic_id_to_mesh_rank) {
                if (asic_ranks.contains(asic_id)) {
                    auto [it, inserted] = real_mesh_to_physical_mesh_id.try_emplace(real_mesh_id, physical_mesh_id);
                    if (!inserted && it->second != physical_mesh_id) {
                        TT_THROW(
                            "Internal Error: Inter-mesh rank binding conflict: logical mesh {} is associated with "
                            "physical mesh {}, "
                            "but ASIC {} in physical mesh {} is also listed under that logical mesh in "
                            "asic_id_to_mesh_rank. Each logical mesh must map to a single physical mesh.",
                            real_mesh_id.get(),
                            it->second.get(),
                            asic_id.get(),
                            physical_mesh_id.get());
                    }
                    break;
                }
            }
        }
    }

    std::map<MeshId, MeshId> rank_bound_logical_to_physical;
    for (const auto& [logical_mesh_id, _] : fabric_node_id_to_mesh_rank) {
        const auto physical_it = real_mesh_to_physical_mesh_id.find(logical_mesh_id);
        if (physical_it == real_mesh_to_physical_mesh_id.end()) {
            continue;
        }
        rank_bound_logical_to_physical.emplace(logical_mesh_id, physical_it->second);
        inter_mesh_constraints.add_required_constraint(logical_mesh_id, physical_it->second);
    }

    add_inter_mesh_minimal_host_cover_from_hostname_map(
        config, physical_graph, mesh_logical_level_graph, inter_mesh_constraints, rank_bound_logical_to_physical);
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
            // One physical UNSET PSD host (e.g. mock discovery with a single MPI rank) but the MGD has multiple
            // mesh_host_ranks: we still need one "global partition" slot per same-rank target group for
            // set_same_rank_groups_constraint (injective matching requires nt <= ng). Duplicate the sole
            // partition; the solver assigns disjoint ASICs from that pool across groups.
            if (global_groups.size() == 1 && target_groups.size() > global_groups.size()) {
                const std::set<tt::tt_metal::AsicID> sole_partition = global_groups.front();
                global_groups.assign(target_groups.size(), sole_partition);
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

// Helper function to build pinning constraints
void add_pinning_constraints(
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

    bool success = true;

    // Apply pinning constraints
    for (const auto& [fabric_node, positions] : fabric_node_to_positions) {
        std::set<tt::tt_metal::AsicID> asic_ids;

        // Convert the ASIC positions to ASIC IDs
        for (const auto& position : positions) {
            auto it = asic_positions_to_asic_ids.find(position);
            if (it == asic_positions_to_asic_ids.end()) {
                log_critical(
                    tt::LogFabric,
                    "Pinned ASIC position (tray_id: {}, asic_location: {}) to fabric node id (mesh_id: {}, chip_id: "
                    "{}) from MGD not found in physical topology",
                    position.first.get(),
                    position.second.get(),
                    fabric_node.mesh_id.get(),
                    fabric_node.chip_id);
                success = false;
                continue;
            }
            asic_ids.insert(it->second.begin(), it->second.end());
        }

        if (!asic_ids.empty()) {
            if (!intra_mesh_constraints.add_required_constraint(fabric_node, asic_ids)) {
                TT_THROW(
                    "Failed to add required constraint for fabric node (mesh={}, chip={})",
                    fabric_node.mesh_id.get(),
                    fabric_node.chip_id);
            }
        }
    }
    TT_FATAL(success, "Failed to add pinning constraints");
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
    auto inter_mesh_constraints = build_inter_mesh_constraints(
        config, adjacency_map_physical, mesh_logical_graph, fabric_node_id_to_mesh_rank, asic_id_to_mesh_rank);
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
    log_info(
        tt::LogFabric,
        "Starting multi-mesh mapping: {} logical mesh(es) to {} physical mesh(es)",
        logical_meshes.size(),
        physical_meshes.size());

    bool success = false;

    TopologyMappingResult result;

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

        log_info(
            tt::LogFabric,
            "Multi-mesh mapping attempt {}/{}: Trying inter-mesh mapping",
            retry_attempt,
            max_retry_attempts);
        if (!failed_mesh_pairs.empty()) {
            log_debug(tt::LogFabric, "Failed mesh pairs from previous attempts: {}", failed_mesh_pairs.size());
        }

        // Perform inter-mesh mapping
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
        log_info(
            tt::LogFabric,
            "Attempt {}: Inter-mesh mapping succeeded, found {} mesh pair(s)",
            retry_attempt,
            solver_result.target_to_global.size());

        unsigned int mapped_mesh_pairs = 0;
        std::vector<std::pair<MeshId, MeshId>> current_attempt_failed_pairs;

        // Step 2: For each mesh mapping, do the sub mapping for fabric node id to asic id
        std::unordered_map<MeshId, MeshId> mesh_mappings(
            solver_result.target_to_global.begin(), solver_result.target_to_global.end());
        log_info(tt::LogFabric, "Attempt {}: Mapping {} mesh pair(s)", retry_attempt, mesh_mappings.size());
        for (const auto& [logical_mesh_id, physical_mesh_id] : mesh_mappings) {
            log_info(
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

                // If exit node constraints cannot be satisfied (no valid physical exit nodes or over-constrained),
                // treat this as a mapping failure and try next combination
                if (!exit_node_constraints_success) {
                    log_info(
                        tt::LogFabric,
                        "Attempt {}: Exit node constraints cannot be satisfied for mesh {} -> {}",
                        retry_attempt,
                        logical_mesh_id.get(),
                        physical_mesh_id.get());
                    if (!handle_forbidden_constraint(
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
                            "Exit node constraint error")) {
                        return result;  // Mapping should return early
                    }
                    continue;  // Skip to next physical mesh
                }
            }

            // Build ASIC positions map and add pinning constraints
            auto asic_positions_to_asic_ids = build_asic_positions_map(physical_graph, config);
            add_pinning_constraints(intra_mesh_constraints, asic_positions_to_asic_ids, config, logical_mesh_id);

            // Determine validation mode
            auto validation_mode = determine_intra_mesh_validation_mode(config, logical_mesh_id);

            // Perform the sub mapping for the fabric node id to the asic id
            auto sub_mapping = ::tt::tt_fabric::solve_topology_mapping(
                logical_graph, physical_graph, intra_mesh_constraints, validation_mode, quiet_mode);

            // If the intra-mesh mapping fails, add a forbidden constraint so it doesn't try to map this pair again
            if (!sub_mapping.success) {
                log_info(
                    tt::LogFabric,
                    "Attempt {}: Intra-mesh mapping failed for mesh {} -> {}",
                    retry_attempt,
                    logical_mesh_id.get(),
                    physical_mesh_id.get());
                if (!handle_forbidden_constraint(
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
                        "Constraint error")) {
                    return result;  // Mapping should return early
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
            log_info(
                tt::LogFabric,
                "Multi-mesh mapping succeeded after {} attempt(s): {} mesh pair(s) mapped",
                retry_attempt,
                mapped_mesh_pairs);
        } else {
            // Remove all the results that were added so far and start over
            log_info(
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

}  // namespace tt::tt_metal::experimental::tt_fabric
