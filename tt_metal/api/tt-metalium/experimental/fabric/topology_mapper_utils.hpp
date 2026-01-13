// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental::tt_fabric {

// Import types from tt::tt_fabric for use in this API
using ::tt::tt_fabric::FabricNodeId;
using ::tt::tt_fabric::MeshHostRankId;
using ::tt::tt_fabric::MeshId;

// Type aliases for adjacency maps used in topology mapping
using LogicalAdjacencyMap = std::map<FabricNodeId, std::vector<FabricNodeId>>;
using PhysicalAdjacencyMap = std::map<tt::tt_metal::AsicID, std::vector<tt::tt_metal::AsicID>>;

// Use ASICPosition from tt::tt_metal namespace
using AsicPosition = tt::tt_metal::ASICPosition;

// Map from AsicID to its physical position (TrayID, ASICLocation)
// Required only when using pinning constraints
using AsicPositionMap = std::map<tt::tt_metal::AsicID, AsicPosition>;

// Pinning constraint: maps an ASIC position to a FabricNodeId
// This constrains which physical ASIC a logical node can be mapped to
using PinningConstraint = std::pair<AsicPosition, FabricNodeId>;

/**
 * @brief Configuration options for topology mapping
 */
struct TopologyMappingConfig {
    // When true, validates that physical connections have at least as many
    // channels as required by logical connections. When false, only checks
    // that connections exist (relaxed mode).
    bool strict_mode = false;

    // Optional pinning constraints that restrict which physical ASICs
    // specific logical nodes can be mapped to
    std::vector<PinningConstraint> pinnings;

    // Map from AsicID to (TrayID, ASICLocation) - required if pinnings is non-empty.
    // Used to validate pinning constraints against the physical topology.
    AsicPositionMap asic_positions;
};

/**
 * @brief Result of topology mapping operation
 */
struct TopologyMappingResult {
    bool success = false;
    std::string error_message;

    // Bidirectional mappings between logical fabric nodes and physical ASICs
    std::map<FabricNodeId, tt::tt_metal::AsicID> fabric_node_to_asic;
    std::map<tt::tt_metal::AsicID, FabricNodeId> asic_to_fabric_node;
};

/**
 * @brief Run CSP algorithm to map logical nodes to physical ASICs
 *
 * This function implements the core topology mapping algorithm extracted from TopologyMapper.
 * It uses a constraint satisfaction approach with backtracking to find a valid mapping
 * that preserves the logical connectivity structure in the physical topology.
 *
 * The algorithm ensures:
 * - Every logical edge has a corresponding physical edge
 * - In strict mode, physical edges have at least as many channels as logical edges require
 * - Mesh host rank constraints are respected (logical nodes map to ASICs on the correct host)
 * - Optional pinning constraints are satisfied
 *
 * This function does NOT require MPI or any tt-metal runtime context. It operates purely
 * on the provided adjacency graphs and rank mappings.
 *
 * @param mesh_id              The mesh ID being mapped
 * @param logical_adjacency    AdjacencyGraph for logical topology
 * @param physical_adjacency   AdjacencyGraph for physical topology
 * @param node_to_host_rank    Map from FabricNodeId to the host rank that owns it
 * @param asic_to_host_rank    Map from AsicID to the host rank that owns it
 * @param config               Optional configuration (strict mode, pinning constraints)
 *
 * @return TopologyMappingResult containing success status and bidirectional mappings
 *
 * @note If the mapping fails (no valid assignment exists), success will be false
 *       and error_message will contain diagnostic information.
 */
TopologyMappingResult map_mesh_to_physical(
    MeshId mesh_id,
    const ::tt::tt_fabric::AdjacencyGraph<FabricNodeId>& logical_adjacency,
    const ::tt::tt_fabric::AdjacencyGraph<tt::tt_metal::AsicID>& physical_adjacency,
    const std::map<FabricNodeId, MeshHostRankId>& node_to_host_rank,
    const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_to_host_rank,
    const TopologyMappingConfig& config = {});

/**
 * @brief Build logical adjacency graphs from mesh graph connectivity
 *
 * Creates adjacency graphs for each mesh based on the logical connectivity defined in the mesh graph.
 * For each fabric node in a mesh, this function identifies its logical neighbors by examining
 * the intra-mesh connectivity from the mesh graph and creates an AdjacencyGraph.
 *
 * @param mesh_graph Reference to the mesh graph object containing fabric topology
 * @return std::map<MeshId, AdjacencyGraph<FabricNodeId>> Map from mesh ID to logical adjacency graph
 */
std::map<MeshId, ::tt::tt_fabric::AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(
    const ::tt::tt_fabric::MeshGraph& mesh_graph);

/**
 * @brief Build physical adjacency graphs from system descriptor connectivity
 *
 * Creates adjacency graphs for each mesh based on the physical connectivity defined in the physical system
 * descriptor. For each ASIC in a mesh, this function identifies its physical neighbors by examining the ASIC
 * neighbors from the physical system descriptor and filters them to only include neighbors that are also part of
 * the same mesh. The resulting graph contains ASIC IDs and their adjacent ASIC IDs within the mesh.
 *
 * @param cluster_type The type of the cluster
 * @param physical_system_descriptor Reference to the physical system descriptor containing ASIC topology
 * @param asic_id_to_mesh_rank Mapping of mesh IDs to ASIC IDs to mesh host ranks
 * @return std::map<MeshId, AdjacencyGraph<AsicID>> Map from mesh ID to physical adjacency graph
 */
std::map<MeshId, PhysicalAdjacencyMap> build_adjacency_map_physical(
    tt::tt_metal::ClusterType cluster_type,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);

}  // namespace tt::tt_metal::experimental::tt_fabric
