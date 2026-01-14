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

}  // namespace tt::tt_metal::experimental::tt_fabric
