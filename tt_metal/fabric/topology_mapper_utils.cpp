// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal::experimental::tt_fabric {

TopologyMappingResult map_mesh_to_physical(
    MeshId mesh_id,
    const ::tt::tt_fabric::AdjacencyGraph<FabricNodeId>& logical_adjacency,
    const ::tt::tt_fabric::AdjacencyGraph<tt::tt_metal::AsicID>& physical_adjacency,
    const std::map<FabricNodeId, MeshHostRankId>& node_to_host_rank,
    const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_to_host_rank,
    const TopologyMappingConfig& config) {
    TopologyMappingResult result;

    using namespace ::tt::tt_fabric;

    // Use the adjacency graphs directly
    const AdjacencyGraph<FabricNodeId>& target_graph = logical_adjacency;
    const AdjacencyGraph<tt::tt_metal::AsicID>& global_graph = physical_adjacency;

    // Build constraints
    MappingConstraints<FabricNodeId, tt::tt_metal::AsicID> constraints;

    // Add mesh host rank constraints (trait-based constraint)
    constraints.add_required_trait_constraint(node_to_host_rank, asic_to_host_rank);

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
            const auto& logical_nodes = logical_adjacency.get_nodes();
            bool found = false;
            for (const auto& node : logical_nodes) {
                if (node == fabric_node) {
                    found = true;
                    break;
                }
            }
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
        const auto& physical_nodes = physical_adjacency.get_nodes();
        std::unordered_set<tt::tt_metal::AsicID> physical_node_set(physical_nodes.begin(), physical_nodes.end());
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
    auto solver_result = solve_topology_mapping(target_graph, global_graph, constraints, validation_mode);

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

    // Get requested inter-mesh connections (relaxed mode) and ports (strict mode)
    const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
    const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();

    // Process requested_intermesh_ports (strict mode) if it exists
    // Mapping: src_mesh -> dst_mesh -> list of (src_device, dst_device, num_channels)
    if (!requested_intermesh_ports.empty()) {
        for (const auto& [src_mesh_id_val, dst_mesh_map] : requested_intermesh_ports) {
            MeshId src_mesh_id(src_mesh_id_val);

            for (const auto& [dst_mesh_id_val, port_list] : dst_mesh_map) {
                MeshId dst_mesh_id(dst_mesh_id_val);
                // Skip self-connections
                if (dst_mesh_id != src_mesh_id) {
                    // Add connections based on num_channels from each port entry
                    // Each tuple is (src_device, dst_device, num_channels)
                    for (const auto& port_entry : port_list) {
                        uint32_t num_channels = std::get<2>(port_entry);
                        for (uint32_t i = 0; i < num_channels; ++i) {
                            mesh_level_adjacency_map[src_mesh_id].push_back(dst_mesh_id);
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

    return logical_multi_mesh_graph;
}

PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // create a unordered map of mesh ids to asic ids
    std::unordered_map<AsicID, MeshId> asic_id_to_mesh_id;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            asic_id_to_mesh_id[asic_id] = mesh_id;
        }
    }

    // Build physical adjacency graphs for each mesh
    auto mesh_adjacency_graphs =
        ::tt::tt_fabric::build_adjacency_graph_physical(physical_system_descriptor, asic_id_to_mesh_rank);

    // Build physical multi-mesh adjacency graph
    PhysicalMultiMeshGraph physical_multi_mesh_graph;

    // Store mesh adjacency graphs once (no duplication)
    for (const auto& [mesh_id, adjacency_graph] : mesh_adjacency_graphs) {
        physical_multi_mesh_graph.mesh_adjacency_graphs_[mesh_id] = adjacency_graph;
    }

    // Build mesh-level adjacency map using MeshIds (lightweight)
    ::tt::tt_fabric::AdjacencyGraph<MeshId>::AdjacencyMap mesh_level_adjacency_map;

    // NOTE: can't make assumption that all cross mesh connections are cross host yet, since we do implement some
    // multi-mesh per host

    // NOTE: asic_topology currently includes host to host connections, so
    // the following host to host connection logic is commented out. Need to check if this is a bug
    // Go through all host to host connections first
    // for (const auto& [_, host_connections] : physical_system_descriptor.get_host_topology()) {
    //    for (const auto& [_, exit_node_connections] : host_connections) {
    //        for (const auto& connection : exit_node_connections) {
    //            auto src_mesh_id = asic_id_to_mesh_id[connection.src_exit_node];
    //            auto dst_mesh_id = asic_id_to_mesh_id[connection.dst_exit_node];
    //            if (src_mesh_id != dst_mesh_id) {
    //                mesh_level_adjacency_map[src_mesh_id].push_back(dst_mesh_id);
    //            }
    //        }
    //    }
    //}

    // Go through all local connections
    for (const auto& host_name : physical_system_descriptor.get_all_hostnames()) {
        for (const auto& [src_asic_id, asic_connections] : physical_system_descriptor.get_asic_topology(host_name)) {
            for (const auto& asic_connection : asic_connections) {
                auto dst_asic_id = asic_connection.first;
                auto src_mesh_id = asic_id_to_mesh_id[src_asic_id];
                auto dst_mesh_id = asic_id_to_mesh_id[dst_asic_id];
                if (src_mesh_id != dst_mesh_id) {
                    const auto& eth_connections = asic_connection.second;
                    // Add one entry per channel (EthConnection) in this edge
                    for ([[maybe_unused]] const auto& eth_conn : eth_connections) {
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
    physical_multi_mesh_graph.mesh_level_graph_ = ::tt::tt_fabric::AdjacencyGraph<MeshId>(mesh_level_adjacency_map);

    return physical_multi_mesh_graph;
}

std::map<MeshId, TopologyMappingResult> map_multi_mesh_to_physical(
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank,
    const TopologyMappingConfig& config) {
    using namespace ::tt::tt_fabric;

    std::map<MeshId, TopologyMappingResult> results;

    // Step 1: Run Mesh to Mesh mapping algorithm
    auto& mesh_logical_graph = adjacency_map_logical.mesh_level_graph_;
    auto& mesh_physical_graph = adjacency_map_physical.mesh_level_graph_;

    // TODO: Remove this once rank bindings file is removed from multi-host systems
    // Use placeholder mesh id 1:1 mapping for physical to logical constraints for now
    MappingConstraints<MeshId, MeshId> inter_mesh_constraints;
    for (const auto& mesh_id : mesh_physical_graph.get_nodes()) {
        inter_mesh_constraints.add_required_constraint(mesh_id, mesh_id);
    }

    // Determine inter-mesh validation mode from config
    ConnectionValidationMode inter_mesh_validation_mode = ConnectionValidationMode::RELAXED;
    if (config.inter_mesh_validation_mode.has_value()) {
        inter_mesh_validation_mode = config.inter_mesh_validation_mode.value();
    } else if (config.strict_mode) {
        // Fallback for backward compatibility
        inter_mesh_validation_mode = ConnectionValidationMode::STRICT;
    }
    auto solver_result = solve_topology_mapping(
        mesh_logical_graph, mesh_physical_graph, inter_mesh_constraints, inter_mesh_validation_mode);

    // If the solver fails, return error results for all meshes
    if (!solver_result.success) {
        for (const auto& mesh_id : mesh_logical_graph.get_nodes()) {
            TopologyMappingResult result;
            result.success = false;
            result.error_message = fmt::format("Inter-mesh mapping failed: {}", solver_result.error_message);
            results[mesh_id] = result;
        }
        return results;
    }

    // Step 2: For each mesh mapping, do the sub mapping for fabric node id to asic id
    auto& mesh_mappings = solver_result.target_to_global;
    for (const auto& [logical_mesh_id, physical_mesh_id] : mesh_mappings) {
        TopologyMappingResult result;

        // Get the logical graph and the physical graph
        const auto& logical_graph = adjacency_map_logical.mesh_adjacency_graphs_.at(logical_mesh_id);
        const auto& physical_graph = adjacency_map_physical.mesh_adjacency_graphs_.at(physical_mesh_id);

        // Build intra-mesh constraints
        MappingConstraints<FabricNodeId, tt::tt_metal::AsicID> intra_mesh_constraints;

        // Build Rank bindings constraints
        intra_mesh_constraints.add_required_trait_constraint(
            fabric_node_id_to_mesh_rank.at(logical_mesh_id), asic_id_to_mesh_rank.at(logical_mesh_id));

        // Map of asic positions to asic ids (built from config.asic_positions)
        std::map<AsicPosition, std::set<tt::tt_metal::AsicID>> asic_positions_to_asic_ids;
        if (!config.asic_positions.empty()) {
            for (const auto& asic_id : physical_graph.get_nodes()) {
                auto pos_it = config.asic_positions.find(asic_id);
                if (pos_it != config.asic_positions.end()) {
                    asic_positions_to_asic_ids[pos_it->second].insert(asic_id);
                }
            }
        }

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

        // Do the sub mapping for the fabric node id to the asic id
        // This should be called once per mesh, after all pinning constraints are added
        // Determine validation mode from config
        ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED;
        auto config_mode_it = config.mesh_validation_modes.find(logical_mesh_id);
        if (config_mode_it != config.mesh_validation_modes.end()) {
            validation_mode = config_mode_it->second;
        } else if (config.strict_mode) {
            // Fallback for backward compatibility
            validation_mode = ConnectionValidationMode::STRICT;
        }
        auto sub_mapping =
            solve_topology_mapping(logical_graph, physical_graph, intra_mesh_constraints, validation_mode);

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

        results[logical_mesh_id] = result;
    }

    return results;
}

}  // namespace tt::tt_metal::experimental::tt_fabric
