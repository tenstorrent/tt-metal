// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <optional>
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
using ::tt::tt_fabric::AdjacencyGraph;
using ::tt::tt_fabric::ConnectionValidationMode;
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

    // Per-mesh validation modes for intra-mesh mapping (fabric node to ASIC).
    // If empty, falls back to strict_mode for backward compatibility.
    std::map<MeshId, ConnectionValidationMode> mesh_validation_modes;

    // Validation mode for inter-mesh mapping (mesh to mesh).
    // Defaults to RELAXED for backward compatibility if not set.
    std::optional<ConnectionValidationMode> inter_mesh_validation_mode;

    // When true, disables rank binding constraints. Rank mappings will be ignored
    // and any mapping that satisfies connectivity constraints will be valid.
    bool disable_rank_bindings = false;
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
 * on the provided adjacency maps and rank mappings.
 *
 * @param mesh_id              The mesh ID being mapped
 * @param logical_adjacency    Map from FabricNodeId to list of neighbor FabricNodeIds
 * @param physical_adjacency   Map from AsicID to list of neighbor AsicIDs
 * @param node_to_host_rank    Map from FabricNodeId to the host rank that owns it
 * @param asic_to_host_rank    Map from AsicID to the host rank that owns it
 * @param config               Optional configuration (strict mode, pinning constraints)
 *
 * @return TopologyMappingResult containing success status and bidirectional mappings
 *
 * @note If the mapping fails (no valid assignment exists), success will be false
 *       and error_message will contain diagnostic information.
 *
 * @example
 * @code
 * LogicalAdjacencyMap logical_adj;
 * PhysicalAdjacencyMap physical_adj;
 * std::map<FabricNodeId, MeshHostRankId> node_to_rank;
 * std::map<AsicID, MeshHostRankId> asic_to_rank;
 *
 * // Populate adjacency maps from your topology data...
 *
 * TopologyMappingConfig config;
 * config.strict_mode = true;  // Validate channel counts
 *
 * auto result = map_mesh_to_physical(
 *     MeshId{0}, logical_adj, physical_adj, node_to_rank, asic_to_rank, config);
 *
 * if (result.success) {
 *     for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
 *         // Use the mapping...
 *     }
 * } else {
 *     std::cerr << "Mapping failed: " << result.error_message << std::endl;
 * }
 * @endcode
 */
TopologyMappingResult map_mesh_to_physical(
    MeshId mesh_id,
    const LogicalAdjacencyMap& logical_adjacency,
    const PhysicalAdjacencyMap& physical_adjacency,
    const std::map<FabricNodeId, MeshHostRankId>& node_to_host_rank,
    const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_to_host_rank,
    const TopologyMappingConfig& config = {});

/**
 * @brief Build logical adjacency maps from mesh graph connectivity
 *
 * Creates adjacency maps for each mesh based on the logical connectivity defined in the mesh graph.
 * For each fabric node in a mesh, this function identifies its logical neighbors by examining
 * the intra-mesh connectivity from the mesh graph and creates a mapping of FabricNodeId to
 * its vector of adjacent FabricNodeIds.
 *
 * @param mesh_graph Reference to the mesh graph object containing fabric topology
 * @return std::map<MeshId, LogicalAdjacencyMap> Map from mesh ID to logical adjacency map
 */
std::map<MeshId, LogicalAdjacencyMap> build_adjacency_map_logical(const ::tt::tt_fabric::MeshGraph& mesh_graph);

/**
 * @brief Build physical adjacency maps from system descriptor connectivity
 *
 * Creates adjacency maps for each mesh based on the physical connectivity defined in the physical system
 * descriptor. For each ASIC in a mesh, this function identifies its physical neighbors by examining the ASIC
 * neighbors from the physical system descriptor and filters them to only include neighbors that are also part of
 * the same mesh. The resulting map contains ASIC IDs mapped to their vectors of adjacent ASIC IDs within the mesh.
 *
 * @param cluster_type The type of the cluster
 * @param physical_system_descriptor Reference to the physical system descriptor containing ASIC topology
 * @param asic_id_to_mesh_rank Mapping of mesh IDs to ASIC IDs to mesh host ranks
 * @return std::map<MeshId, PhysicalAdjacencyMap> Map from mesh ID to physical adjacency map
 */
std::map<MeshId, PhysicalAdjacencyMap> build_adjacency_map_physical(
    tt::tt_metal::ClusterType cluster_type,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);

/**
 * @brief Represents a mesh node in a 2-layer adjacency graph
 *
 * Simplified to just be a MeshId. The internal adjacency graph is accessed via
 * LogicalMultiMeshGraph::get_internal_graph().
 */
using LogicalMeshNode = MeshId;

/**
 * @brief Multi-mesh adjacency graph where meshes are nodes
 *
 * Efficient representation that avoids duplicating adjacency graphs:
 * - Stores each mesh's internal adjacency graph once in a map
 * - Stores mesh-level connectivity as lightweight AdjacencyGraph<MeshId>
 *
 * This type represents a hierarchical adjacency graph:
 * - Top layer: adjacency graph of mesh IDs (which meshes connect to which meshes)
 * - Bottom layer: for each mesh, its internal adjacency graph (which fabric nodes connect within the mesh)
 */
struct LogicalMultiMeshGraph {
    // Map from MeshId to its internal adjacency graph (stored once, no duplication)
    std::map<MeshId, AdjacencyGraph<FabricNodeId>> mesh_adjacency_graphs_;

    // Mesh-level adjacency graph using MeshIds (lightweight, no graph duplication)
    AdjacencyGraph<MeshId> mesh_level_graph_;
};

/**
 * @brief Build a logical multi-mesh adjacency graph from a mesh graph
 *
 * Creates a LogicalMultiMeshGraph with:
 * - Mesh-level adjacency graph (AdjacencyGraph<MeshId>) representing inter-mesh connectivity
 * - Map of mesh IDs to their internal adjacency graphs (AdjacencyGraph<FabricNodeId>)
 *
 * The top layer represents inter-mesh connectivity (which meshes connect to which meshes),
 * while the internal graphs represent intra-mesh connectivity (which fabric nodes connect within each mesh).
 *
 * @param mesh_graph Reference to the mesh graph object containing fabric topology
 * @return LogicalMultiMeshGraph containing mesh-level graph and internal mesh graphs
 */
LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(const ::tt::tt_fabric::MeshGraph& mesh_graph);

/**
 * @brief Represents a physical mesh node in a 2-layer adjacency graph
 *
 * Simplified to just be a MeshId. The internal adjacency graph is accessed via
 * PhysicalMultiMeshGraph::mesh_adjacency_graphs_.
 */
using PhysicalMeshNode = MeshId;

/**
 * @brief Multi-mesh adjacency graph for physical ASICs where meshes are nodes
 *
 * Efficient representation that avoids duplicating adjacency graphs:
 * - Stores each mesh's internal adjacency graph once in a map
 * - Stores mesh-level connectivity as lightweight AdjacencyGraph<MeshId>
 *
 * This type represents a hierarchical adjacency graph:
 * - Top layer: adjacency graph of mesh IDs (which meshes connect to which meshes)
 * - Bottom layer: for each mesh, its internal adjacency graph (which ASICs connect within the mesh)
 */
struct PhysicalMultiMeshGraph {
    // Map from MeshId to its interkj/nal adjacency graph (stored once, no duplication)
    std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> mesh_adjacency_graphs_;

    // Mesh-level adjacency graph using MeshIds (lightweight, no graph duplication)
    AdjacencyGraph<MeshId> mesh_level_graph_;
};

/**
 * @brief Build a physical multi-mesh adjacency graph from physical system descriptor
 *
 * Creates a PhysicalMultiMeshGraph with:
 * - Mesh-level adjacency graph (AdjacencyGraph<MeshId>) representing inter-mesh connectivity
 * - Map of mesh IDs to their internal adjacency graphs (AdjacencyGraph<AsicID>)
 *
 * The top layer represents inter-mesh connectivity (which meshes connect to which meshes),
 * determined by checking if ASICs in one mesh connect to ASICs in another mesh.
 * The internal graphs represent intra-mesh connectivity (which ASICs connect within each mesh).
 *
 * @param physical_system_descriptor Reference to the physical system descriptor containing ASIC topology
 * @param asic_id_to_mesh_rank Mapping of mesh IDs to ASIC IDs to mesh host ranks
 * @return PhysicalMultiMeshGraph containing mesh-level graph and internal mesh nodes
 */
PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);

/**
 * @brief Map logical multi-mesh topology to physical multi-mesh topology
 *
 * This function performs a two-level mapping:
 * 1. Inter-mesh mapping: Maps logical meshes to physical meshes
 * 2. Intra-mesh mapping: For each mapped mesh pair, maps logical fabric nodes to physical ASICs
 *
 * The function respects:
 * - Mesh host rank constraints (logical nodes map to ASICs on the correct host)
 * - Optional pinning constraints that restrict which physical ASICs specific logical nodes can map to
 * - Inter-mesh connectivity constraints
 *
 * @param adjacency_map_logical Logical multi-mesh adjacency graph
 * @param adjacency_map_physical Physical multi-mesh adjacency graph
 * @param config Configuration options including pinning constraints, ASIC positions, and validation modes.
 *               config.mesh_validation_modes and config.inter_mesh_validation_mode should be set for proper
 *               validation. If not set, defaults to RELAXED mode. If config.strict_mode is true, it will be
 *               used as a fallback for backward compatibility.
 *               If config.disable_rank_bindings is true, rank mappings are ignored and can be omitted.
 * @param asic_id_to_mesh_rank Optional mapping of mesh IDs to ASIC IDs to mesh host ranks.
 *                             Required if config.disable_rank_bindings is false.
 * @param fabric_node_id_to_mesh_rank Optional mapping of mesh IDs to fabric node IDs to mesh host ranks.
 *                                    Required if config.disable_rank_bindings is false.
 *
 * @return TopologyMappingResult containing the overall mapping result with bidirectional mappings
 *         for all successfully mapped meshes
 *
 * @note If inter-mesh mapping fails, result.success will be false and error_message will contain details
 * @note If intra-mesh mapping fails for a specific mesh, the mapping will be retried with different
 *       inter-mesh pairings. If all attempts fail, result.success will be false
 * @note If config.disable_rank_bindings is true, rank constraints are ignored and any valid connectivity
 *       mapping is allowed
 */
TopologyMappingResult map_multi_mesh_to_physical(
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const TopologyMappingConfig& config,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {});

}  // namespace tt::tt_metal::experimental::tt_fabric
