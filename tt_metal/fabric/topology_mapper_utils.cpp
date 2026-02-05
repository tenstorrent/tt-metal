// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include "tt_metal/fabric/topology_solver_internal.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
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
    // Catch exceptions from constraint validation and convert to failure result
    try {
        constraints.add_required_trait_constraint(node_to_host_rank, asic_to_host_rank);
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
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
            bool found = logical_adjacency.find(fabric_node) != logical_adjacency.end();
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
            if (physical_node_set.find(asic_id) != physical_node_set.end()) {
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
            constraints.add_required_trait_constraint(fabric_node_traits, asic_traits);

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

    // Determine connection validation mode
    ConnectionValidationMode validation_mode =
        config.strict_mode ? ConnectionValidationMode::STRICT : ConnectionValidationMode::RELAXED;

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

LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    // TODO: Add support for mixing STRICT and RELAXED policies in the same graph.
    // Currently, MGD validation prevents mixing policies, but when this feature is added,
    // this function will need to handle both requested_intermesh_ports (strict) and
    // requested_intermesh_connections (relaxed) simultaneously, ensuring exit nodes are
    // only tracked for strict mode connections.

    // Build logical adjacency graphs for each mesh using topology solver's function
    auto mesh_adjacency_graphs = ::tt::tt_fabric::build_adjacency_graph_logical(mesh_graph);

    // Build logical multi-mesh adjacency graph
    LogicalMultiMeshGraph logical_multi_mesh_graph;

    // Store mesh adjacency graphs once (no duplication)
    for (const auto& [mesh_id, adjacency_graph] : mesh_adjacency_graphs) {
        logical_multi_mesh_graph.mesh_adjacency_graphs_[mesh_id] = adjacency_graph;
    }

    // Build mesh-level adjacency map using MeshIds (lightweight)
    ::tt::tt_fabric::AdjacencyGraph<MeshId>::AdjacencyMap mesh_level_adjacency_map;

    // Build exit node adjacency maps (only for strict mode)
    std::map<MeshId, AdjacencyGraph<FabricNodeId>::AdjacencyMap> exit_node_adjacency_maps;

    // Get requested inter-mesh connections (relaxed mode) and ports (strict mode)
    const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
    const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();

    // Process requested_intermesh_ports (strict mode) if it exists
    // Mapping: src_mesh -> dst_mesh -> list of (src_device, dst_device, num_channels)
    if (!requested_intermesh_ports.empty()) {
        for (const auto& [src_mesh_id_val, dst_mesh_map] : requested_intermesh_ports) {
            MeshId src_mesh_id(src_mesh_id_val);

            // Initialize exit node adjacency map for this mesh if needed
            if (exit_node_adjacency_maps.find(src_mesh_id) == exit_node_adjacency_maps.end()) {
                exit_node_adjacency_maps[src_mesh_id] = AdjacencyGraph<FabricNodeId>::AdjacencyMap();
            }

            for (const auto& [dst_mesh_id_val, port_list] : dst_mesh_map) {
                MeshId dst_mesh_id(dst_mesh_id_val);
                // Skip self-connections
                if (dst_mesh_id != src_mesh_id) {
                    // Initialize exit node adjacency map for destination mesh if needed
                    if (exit_node_adjacency_maps.find(dst_mesh_id) == exit_node_adjacency_maps.end()) {
                        exit_node_adjacency_maps[dst_mesh_id] = AdjacencyGraph<FabricNodeId>::AdjacencyMap();
                    }

                    // Add connections based on num_channels from each port entry
                    // Each tuple is (src_device, dst_device, num_channels)
                    for (const auto& port_entry : port_list) {
                        uint32_t src_device = std::get<0>(port_entry);
                        uint32_t dst_device = std::get<1>(port_entry);
                        uint32_t num_channels = std::get<2>(port_entry);

                        // Create FabricNodeIds for exit nodes
                        FabricNodeId src_exit_node(src_mesh_id, src_device);
                        FabricNodeId dst_exit_node(dst_mesh_id, dst_device);

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

    // Process requested_intermesh_connections (relaxed mode) if it exists
    // Mapping: src_mesh -> dst_mesh -> num_channels
    if (!requested_intermesh_connections.empty()) {
        for (const auto& [src_mesh_id_val, dst_mesh_map] : requested_intermesh_connections) {
            MeshId src_mesh_id(src_mesh_id_val);

            for (const auto& [dst_mesh_id_val, num_channels] : dst_mesh_map) {
                MeshId dst_mesh_id(dst_mesh_id_val);
                // Skip self-connections
                if (dst_mesh_id != src_mesh_id) {
                    // Add connections based on num_channels (multiple connections between same meshes)
                    for (uint32_t i = 0; i < num_channels; ++i) {
                        mesh_level_adjacency_map[src_mesh_id].push_back(dst_mesh_id);
                    }
                }
            }
        }
    }

    // Ensure all meshes are represented as nodes in the mesh-level graph, even if they have no connections
    // This is important for single-mesh scenarios where there are no inter-mesh connections
    for (const auto& [mesh_id, _] : mesh_adjacency_graphs) {
        if (mesh_level_adjacency_map.find(mesh_id) == mesh_level_adjacency_map.end()) {
            mesh_level_adjacency_map[mesh_id] = std::vector<MeshId>();
        }
    }

    // Build mesh-level graph from adjacency map
    logical_multi_mesh_graph.mesh_level_graph_ = ::tt::tt_fabric::AdjacencyGraph<MeshId>(mesh_level_adjacency_map);

    // Convert exit node adjacency maps to graphs (only populated in strict mode)
    for (const auto& [mesh_id, adj_map] : exit_node_adjacency_maps) {
        if (!adj_map.empty()) {
            logical_multi_mesh_graph.mesh_exit_node_graphs_[mesh_id] = AdjacencyGraph<FabricNodeId>(adj_map);
        }
    }

    return logical_multi_mesh_graph;
}

namespace {
/**
 * @brief Build a flat PhysicalAdjacencyMap from PhysicalSystemDescriptor
 *
 * Private helper that builds a complete flat adjacency map including all connections
 * (both intra-mesh and intermesh), with multiple entries per channel.
 */
PhysicalAdjacencyMap build_flat_adjacency_map_from_psd(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    PhysicalAdjacencyMap flat_adj;

    // Build a set of all ASIC IDs for quick lookup
    std::unordered_set<tt::tt_metal::AsicID> all_asics;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            all_asics.insert(asic_id);
        }
    }

    // Go through all connections in the physical system descriptor
    for (const auto& host_name : physical_system_descriptor.get_all_hostnames()) {
        for (const auto& [src_asic_id, asic_connections] : physical_system_descriptor.get_asic_topology(host_name)) {
            // Skip ASICs not in any mesh assignment
            if (!all_asics.contains(src_asic_id)) {
                continue;
            }

            for (const auto& asic_connection : asic_connections) {
                auto dst_asic_id = asic_connection.first;
                // Skip ASICs not in any mesh assignment
                if (!all_asics.contains(dst_asic_id)) {
                    continue;
                }

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

}  // namespace

PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // Build flat adjacency map from PhysicalSystemDescriptor
    PhysicalAdjacencyMap flat_adj = build_flat_adjacency_map_from_psd(physical_system_descriptor, asic_id_to_mesh_rank);

    // Convert to AdjacencyGraph and use the common algorithm
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    return build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);
}

PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // Build a map from AsicID to MeshId for quick lookup
    std::unordered_map<tt::tt_metal::AsicID, MeshId> asic_id_to_mesh_id;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            asic_id_to_mesh_id[asic_id] = mesh_id;
        }
    }

    // Build per-mesh adjacency maps (only intra-mesh connections)
    std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap> mesh_adjacency_maps;
    std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap> exit_node_adjacency_maps;
    AdjacencyGraph<MeshId>::AdjacencyMap mesh_level_adjacency_map;

    // Initialize adjacency maps for all meshes and ensure all ASICs are included
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        mesh_adjacency_maps[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap();
        exit_node_adjacency_maps[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap();
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
                exit_node_adjacency_maps[src_mesh_id][src_asic_id].push_back(dst_asic_id);
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
    for (const auto& [mesh_id, adj_map] : exit_node_adjacency_maps) {
        if (!adj_map.empty()) {
            physical_multi_mesh_graph.mesh_exit_node_graphs_[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>(adj_map);
        } else {
            // Initialize empty graph for meshes with no exit nodes
            physical_multi_mesh_graph.mesh_exit_node_graphs_[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>();
        }
    }

    // Ensure all meshes are represented in mesh-level graph, even if they have no connections
    for (const auto& [mesh_id, _] : asic_id_to_mesh_rank) {
        if (mesh_level_adjacency_map.find(mesh_id) == mesh_level_adjacency_map.end()) {
            mesh_level_adjacency_map[mesh_id] = std::vector<MeshId>();
        }
    }

    // Build mesh-level graph from adjacency map
    physical_multi_mesh_graph.mesh_level_graph_ = ::tt::tt_fabric::AdjacencyGraph<MeshId>(mesh_level_adjacency_map);

    return physical_multi_mesh_graph;
}

namespace {
// Helper function to build inter-mesh constraints
::tt::tt_fabric::MappingConstraints<MeshId, MeshId> build_inter_mesh_constraints(
    const ::tt::tt_fabric::AdjacencyGraph<MeshId>& mesh_physical_graph, const TopologyMappingConfig& config) {
    ::tt::tt_fabric::MappingConstraints<MeshId, MeshId> inter_mesh_constraints;
    // TODO: Remove this once rank bindings file is removed from multi-host systems
    // Use placeholder mesh id 1:1 mapping for physical to logical constraints for now
    if (!config.disable_rank_bindings) {
        for (const auto& mesh_id : mesh_physical_graph.get_nodes()) {
            inter_mesh_constraints.add_required_constraint(mesh_id, mesh_id);
        }
    }
    return inter_mesh_constraints;
}

// Helper function to determine inter-mesh validation mode
::tt::tt_fabric::ConnectionValidationMode determine_inter_mesh_validation_mode(const TopologyMappingConfig& config) {
    if (config.inter_mesh_validation_mode.has_value()) {
        return config.inter_mesh_validation_mode.value();
    } else if (config.strict_mode) {
        // Fallback for backward compatibility
        return ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }
    return ::tt::tt_fabric::ConnectionValidationMode::RELAXED;
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

// Helper function to add rank binding constraints
void add_rank_binding_constraints(
    ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>& intra_mesh_constraints,
    const TopologyMappingConfig& config,
    MeshId logical_mesh_id,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // TODO: Remove this once rank bindings file is removed from multi-host systems
    // Build Rank bindings constraints (only if rank bindings are enabled)
    if (!config.disable_rank_bindings) {
        // Check that rank mappings are provided
        if (fabric_node_id_to_mesh_rank.find(logical_mesh_id) != fabric_node_id_to_mesh_rank.end() &&
            asic_id_to_mesh_rank.find(logical_mesh_id) != asic_id_to_mesh_rank.end()) {
            intra_mesh_constraints.add_required_trait_constraint(
                fabric_node_id_to_mesh_rank.at(logical_mesh_id), asic_id_to_mesh_rank.at(logical_mesh_id));
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

    // Apply pinning constraints
    for (const auto& [fabric_node, positions] : fabric_node_to_positions) {
        std::set<tt::tt_metal::AsicID> asic_ids;

        // Convert the ASIC positions to ASIC IDs
        for (const auto& position : positions) {
            auto it = asic_positions_to_asic_ids.find(position);
            if (it == asic_positions_to_asic_ids.end()) {
                continue;
            }
            asic_ids.insert(it->second.begin(), it->second.end());
        }

        if (!asic_ids.empty()) {
            intra_mesh_constraints.add_required_constraint(fabric_node, asic_ids);
        }
    }
}

// Helper function to determine intra-mesh validation mode
::tt::tt_fabric::ConnectionValidationMode determine_intra_mesh_validation_mode(
    const TopologyMappingConfig& config, MeshId logical_mesh_id) {
    auto config_mode_it = config.mesh_validation_modes.find(logical_mesh_id);
    if (config_mode_it != config.mesh_validation_modes.end()) {
        return config_mode_it->second;
    } else if (config.strict_mode) {
        // Fallback for backward compatibility
        return ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }
    return ::tt::tt_fabric::ConnectionValidationMode::RELAXED;
}

// Helper function to perform intra-mesh mapping for a single mesh
TopologyMappingResult perform_intra_mesh_mapping(
    const ::tt::tt_fabric::AdjacencyGraph<FabricNodeId>& logical_graph,
    const ::tt::tt_fabric::AdjacencyGraph<tt::tt_metal::AsicID>& physical_graph,
    const ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>& intra_mesh_constraints,
    ::tt::tt_fabric::ConnectionValidationMode validation_mode,
    MeshId logical_mesh_id,
    bool quiet_mode = false) {
    TopologyMappingResult result;

    // Perform the sub mapping for the fabric node id to the asic id
    auto sub_mapping = ::tt::tt_fabric::solve_topology_mapping(
        logical_graph, physical_graph, intra_mesh_constraints, validation_mode, quiet_mode);

    // Populate result
    result.success = sub_mapping.success;
    if (!sub_mapping.success) {
        result.error_message = fmt::format(
            "Intra-mesh mapping failed for logical mesh {}: {}", logical_mesh_id.get(), sub_mapping.error_message);
    } else {
        // Build bidirectional mappings
        for (const auto& [fabric_node, asic] : sub_mapping.target_to_global) {
            result.fabric_node_to_asic.insert({fabric_node, asic});
            result.asic_to_fabric_node.insert({asic, fabric_node});
        }
    }

    return result;
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

}  // anonymous namespace

TopologyMappingResult map_multi_mesh_to_physical(
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const TopologyMappingConfig& config,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    using namespace ::tt::tt_fabric;

    // Step 1: Run Mesh to Mesh mapping algorithm
    auto& mesh_logical_graph = adjacency_map_logical.mesh_level_graph_;
    auto& mesh_physical_graph = adjacency_map_physical.mesh_level_graph_;

    // Build inter-mesh constraints and determine validation mode
    auto inter_mesh_constraints = build_inter_mesh_constraints(mesh_physical_graph, config);
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

    // If rank bindings are disabled, initialize valid mappings for all logical meshes
    // to all physical meshes so that forbidden constraints can be applied
    if (config.disable_rank_bindings) {
        std::set<MeshId> physical_mesh_set(physical_meshes.begin(), physical_meshes.end());
        for (const auto& logical_mesh_id : logical_meshes) {
            inter_mesh_constraints.add_required_constraint(logical_mesh_id, physical_mesh_set);
        }
        log_debug(tt::LogFabric, "Rank bindings disabled - all logical meshes can map to any physical mesh");
    }

    bool success = false;

    TopologyMappingResult result;

    // Maximum retry attempts to prevent infinite loops
    // This should be sufficient for most cases: if we have N logical meshes and M physical meshes,
    // worst case is N*M attempts (trying each logical mesh with each physical mesh)
    const unsigned int max_retry_attempts = logical_meshes.size() * physical_meshes.size() + 1;
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
        auto& mesh_mappings = solver_result.target_to_global;
        for (const auto& [logical_mesh_id, physical_mesh_id] : mesh_mappings) {
            // Get the logical graph and the physical graph
            const auto& logical_graph = adjacency_map_logical.mesh_adjacency_graphs_.at(logical_mesh_id);
            const auto& physical_graph = adjacency_map_physical.mesh_adjacency_graphs_.at(physical_mesh_id);

            // Build intra-mesh constraints
            ::tt::tt_fabric::MappingConstraints<FabricNodeId, tt::tt_metal::AsicID> intra_mesh_constraints;

            // Add rank binding constraints
            add_rank_binding_constraints(
                intra_mesh_constraints, config, logical_mesh_id, fabric_node_id_to_mesh_rank, asic_id_to_mesh_rank);

            // Build ASIC positions map and add pinning constraints
            auto asic_positions_to_asic_ids = build_asic_positions_map(physical_graph, config);
            add_pinning_constraints(intra_mesh_constraints, asic_positions_to_asic_ids, config, logical_mesh_id);

            // Determine validation mode
            auto validation_mode = determine_intra_mesh_validation_mode(config, logical_mesh_id);

            // Perform intra-mesh mapping
            // Use quiet mode for retry attempts (failures are expected during retries)
            auto intra_mesh_result = perform_intra_mesh_mapping(
                logical_graph, physical_graph, intra_mesh_constraints, validation_mode, logical_mesh_id, quiet_mode);

            // If the intra-mesh mapping fails, add a forbidden constraint so it doesn't try to map this pair again
            if (!intra_mesh_result.success) {
                log_info(
                    tt::LogFabric,
                    "Attempt {}: Intra-mesh mapping failed for mesh {} -> {}",
                    retry_attempt,
                    logical_mesh_id.get(),
                    physical_mesh_id.get());
                try {
                    inter_mesh_constraints.add_forbidden_constraint(logical_mesh_id, physical_mesh_id);
                    current_attempt_failed_pairs.emplace_back(logical_mesh_id, physical_mesh_id);
                } catch (const std::exception& e) {
                    // If adding forbidden constraint causes overconstrained nodes (no valid mappings left),
                    // this means we've exhausted all possibilities for this logical mesh.
                    // Treat this as a failure and return with an appropriate error message.
                    // Update failed pairs to include the current one that caused the exception
                    failed_mesh_pairs.insert(
                        failed_mesh_pairs.end(),
                        current_attempt_failed_pairs.begin(),
                        current_attempt_failed_pairs.end());
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
                    result.success = false;
                    result.error_message = build_inter_mesh_mapping_error_message(
                        retry_attempt,
                        logical_meshes,
                        physical_meshes,
                        inter_mesh_validation_mode,
                        fmt::format(
                            "All mapping possibilities exhausted for logical mesh {} after trying {} different mesh "
                            "configurations. "
                            "Constraint error: {}",
                            logical_mesh_id.get(),
                            failed_mesh_pairs.size(),
                            e.what()),
                        failed_mesh_pairs);
                    return result;
                }
            } else {
                mapped_mesh_pairs++;
                // Add the mapping to the result
                result.fabric_node_to_asic.insert(
                    intra_mesh_result.fabric_node_to_asic.begin(), intra_mesh_result.fabric_node_to_asic.end());
                result.asic_to_fabric_node.insert(
                    intra_mesh_result.asic_to_fabric_node.begin(), intra_mesh_result.asic_to_fabric_node.end());
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
