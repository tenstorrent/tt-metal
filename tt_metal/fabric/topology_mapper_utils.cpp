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
    if (!config.pinnings.empty()) {
        const auto& logical_nodes = logical_adjacency.get_nodes();
        const auto& physical_nodes = physical_adjacency.get_nodes();
        std::unordered_set<tt::tt_metal::AsicID> physical_node_set(physical_nodes.begin(), physical_nodes.end());

        std::map<FabricNodeId, AsicPosition> fabric_node_traits;
        std::map<tt::tt_metal::AsicID, AsicPosition> asic_traits;

        // Build fabric node trait map from pinnings with validation
        for (const auto& [pos, fabric_node] : config.pinnings) {
            if (fabric_node.mesh_id != mesh_id) {
                continue;
            }
            if (std::find(logical_nodes.begin(), logical_nodes.end(), fabric_node) == logical_nodes.end()) {
                result.success = false;
                result.error_message =
                    fmt::format("Pinned fabric node {} not found in logical mesh {}", fabric_node, mesh_id.get());
                return result;
            }
            auto [it, inserted] = fabric_node_traits.try_emplace(fabric_node, pos);
            if (!inserted) {
                result.success = false;
                result.error_message = fmt::format(
                    "Fabric node {} in mesh {} is pinned to multiple ASIC positions: (tray {}, loc {}) and (tray {}, "
                    "loc {})",
                    fabric_node,
                    mesh_id.get(),
                    *it->second.first,
                    *it->second.second,
                    *pos.first,
                    *pos.second);
                return result;
            }
        }

        // Build ASIC trait map from config.asic_positions, filtered to physical graph ASICs
        for (const auto& [asic_id, pos] : config.asic_positions) {
            if (physical_node_set.contains(asic_id)) {
                asic_traits[asic_id] = pos;
            }
        }

        // Add position-based trait constraint
        if (!fabric_node_traits.empty() && !asic_traits.empty()) {
            constraints.add_required_trait_constraint<AsicPosition>(fabric_node_traits, asic_traits);

            // Log pinnings
            std::string pinnings_str;
            for (auto it = fabric_node_traits.begin(); it != fabric_node_traits.end(); ++it) {
                if (it != fabric_node_traits.begin()) {
                    pinnings_str += ", ";
                }
                const auto& [fabric_node, pos] = *it;
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
                fabric_node_traits.size(),
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

std::map<MeshId, ::tt::tt_fabric::AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(
    const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    // Use topology solver version directly - fully qualify to avoid ambiguity
    return ::tt::tt_fabric::build_adjacency_map_logical(mesh_graph);
}

std::map<MeshId, ::tt::tt_fabric::AdjacencyGraph<tt::tt_metal::AsicID>> build_adjacency_map_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    // Use topology solver version directly - fully qualify to avoid ambiguity
    // Note: The topology solver version doesn't filter Z channels, but that filtering
    // should be handled at a higher level if needed (e.g., in the mesh graph generation)
    return ::tt::tt_fabric::build_adjacency_map_physical(physical_system_descriptor, asic_id_to_mesh_rank);
}

}  // namespace tt::tt_metal::experimental::tt_fabric
