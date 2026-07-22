// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {
class PhysicalGroupingDescriptor;
struct PsdPlacement;
}  // namespace tt::tt_fabric

namespace tt::tt_metal::experimental::tt_fabric {

// Import types from tt::tt_fabric for use in this API
using ::tt::tt_fabric::AdjacencyGraph;
using ::tt::tt_fabric::ConnectionValidationMode;
using ::tt::tt_fabric::FabricNodeId;
using ::tt::tt_fabric::LogicalChipId;
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

// Galaxy corner pinnings for a single mesh, ensuring QSFP links align with the fabric mesh corner nodes
// and the mesh is not folded. Pins all four logical corners to the four tray corners (with hard_pin_node_0
// fixing the NW corner to tray 1 / asic 1); nw_corner_only pins ONLY the NW corner to any tray-corner ASIC
// (asic_location==1 on trays 1..4) for sub-galaxy slices. Shared by
// generate_rank_bindings (Phase 1) and ControlPlane (Phase 2) so both apply identical placement.
std::vector<std::pair<FabricNodeId, std::vector<AsicPosition>>> get_galaxy_fixed_asic_position_pinnings_for_mesh(
    MeshId mesh_id,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    bool hard_pin_node_0 = false,
    bool nw_corner_only = false);

/**
 * @brief Configuration options for topology mapping
 */
struct TopologyMappingConfig {
    // Deprecated: ignored by topology mapping. Use mesh_validation_modes and inter_mesh_validation_mode
    // with ConnectionValidationMode::STRICT / RELAXED instead. Kept for backward compatibility with callers
    // that still set the field.
    bool strict_mode = false;

    // Optional pinning constraints that restrict which physical ASICs
    // specific logical nodes can be mapped to
    std::vector<PinningConstraint> pinnings;

    // Map from AsicID to (TrayID, ASICLocation) - required if pinnings is non-empty.
    // Used to validate pinning constraints against the physical topology.
    AsicPositionMap asic_positions;

    // Per-mesh validation modes for intra-mesh mapping (fabric node to ASIC).
    // If a logical mesh ID is missing, intra-mesh mapping uses RELAXED for that mesh.
    std::map<MeshId, ConnectionValidationMode> mesh_validation_modes;

    // Validation mode for inter-mesh mapping (mesh to mesh).
    // Defaults to RELAXED for backward compatibility if not set.
    std::optional<ConnectionValidationMode> inter_mesh_validation_mode;

    // When true, disables rank binding constraints. Rank mappings will be ignored
    // and any mapping that satisfies connectivity constraints will be valid.
    bool disable_rank_bindings = false;

    // Optional: Map from hostname to ASIC IDs on that host. When non-empty, enforces that each host
    // has exactly one rank binding (all ASICs on the same host map to fabric nodes with the same rank).
    // Used even when some ASICs have UNSET rank. Default empty.
    std::map<std::string, std::set<tt::tt_metal::AsicID>> hostname_to_asics;
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
 * @param config               Optional configuration (validation modes, pinning constraints)
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
 * config.mesh_validation_modes[MeshId{0}] = ConnectionValidationMode::STRICT;  // validate channel counts
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
 * @brief Represents a logical exit node that can be at either the mesh level or fabric node level
 *
 * Logical exit nodes can represent:
 * - Mesh-level exit nodes: mesh_id is set, fabric_node_id is empty (represents the entire mesh as an exit point)
 * - Fabric node-level exit nodes: both mesh_id and fabric_node_id are set (represents a specific fabric node as an exit
 * point)
 */
struct LogicalExitNode {
    MeshId mesh_id;
    std::optional<FabricNodeId> fabric_node_id;

    bool operator<(const LogicalExitNode& other) const {
        if (mesh_id < other.mesh_id) {
            return true;
        }
        if (other.mesh_id < mesh_id) {
            return false;
        }
        // If mesh_ids are equal, compare fabric_node_ids
        if (!fabric_node_id && !other.fabric_node_id) {
            return false;  // Both empty, equal
        }
        if (!fabric_node_id) {
            return true;  // This is empty, other is not, so this < other
        }
        if (!other.fabric_node_id) {
            return false;  // Other is empty, this is not, so other < this
        }
        return *fabric_node_id < *other.fabric_node_id;
    }

    bool operator==(const LogicalExitNode& other) const {
        return mesh_id == other.mesh_id && fabric_node_id == other.fabric_node_id;
    }
};

/**
 * @brief Represents a physical exit node (ASIC that connects to other meshes)
 *
 * Physical exit nodes represent ASICs that have intermesh connections.
 * Each physical exit node has a mesh_id (which mesh it belongs to) and an asic_id (the ASIC identifier).
 */
struct PhysicalExitNode {
    MeshId mesh_id;
    tt::tt_metal::AsicID asic_id;

    bool operator<(const PhysicalExitNode& other) const {
        if (mesh_id < other.mesh_id) {
            return true;
        }
        if (other.mesh_id < mesh_id) {
            return false;
        }
        return asic_id < other.asic_id;
    }

    bool operator==(const PhysicalExitNode& other) const {
        return mesh_id == other.mesh_id && asic_id == other.asic_id;
    }
};

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

    // Map from MeshId to exit node adjacency graph for that mesh (optional, only populated if specified)
    // Contains exit nodes (LogicalExitNode structs) that can represent either:
    // - Mesh-level exit nodes (mesh_id set, fabric_node_id empty) - the entire mesh serves as an exit point
    // - Fabric node-level exit nodes (both mesh_id and fabric_node_id set) - specific fabric nodes serve as exit points
    // and their connections to exit nodes in other meshes as edges.
    // Multiple channels between the same pair are represented by duplicate entries.
    // Only populated when strict mode intermesh ports are specified.
    std::map<MeshId, AdjacencyGraph<LogicalExitNode>> mesh_exit_node_graphs_;
};

/**
 * @brief Build a logical multi-mesh adjacency graph from a mesh graph
 *
 * Creates a LogicalMultiMeshGraph with:
 * - Mesh-level adjacency graph (AdjacencyGraph<MeshId>) representing inter-mesh connectivity
 * - Map of mesh IDs to their internal adjacency graphs (AdjacencyGraph<FabricNodeId>)
 * - Map of mesh IDs to exit node adjacency graphs (AdjacencyGraph<LogicalExitNode>), optional
 *   - Populated for both strict mode (requested_intermesh_ports) and relaxed mode (requested_intermesh_connections)
 *   - Strict mode: Creates fabric node-level exit nodes (LogicalExitNode with mesh_id and fabric_node_id set)
 *   - Relaxed mode: Creates mesh-level exit nodes (LogicalExitNode with mesh_id only, fabric_node_id is nullopt)
 *   - Exit nodes and their intermesh connections to other exit nodes
 *
 * The top layer represents inter-mesh connectivity (which meshes connect to which meshes),
 * while the internal graphs represent intra-mesh connectivity (which fabric nodes connect within each mesh).
 * Exit node graphs track which logical nodes (at mesh or fabric node level) serve as intermesh connection points.
 *
 * @param mesh_graph Reference to the mesh graph object containing fabric topology
 * @return LogicalMultiMeshGraph containing mesh-level graph, internal mesh graphs, and optional exit node graphs
 */
LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(const ::tt::tt_fabric::MeshGraph& mesh_graph);

/**
 * @brief Build logical multi-mesh adjacency graph from MeshGraphDescriptor
 *
 * Same as above but takes MeshGraphDescriptor instead of MeshGraph. Prefer this overload when
 * the descriptor is already available to avoid constructing a MeshGraph.
 */
LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor);

/**
 * @brief Merge logical multi-mesh graphs into one with automatic MeshId renumbering
 *
 * Inputs are processed in order. For each part, all distinct MeshIds in that part (in fabric
 * adjacency, mesh-level graph, and exit maps) are collected, sorted, and assigned consecutive
 * global ids starting at a running base. If \p per_part_local_to_global_mesh_ids is set, it is filled
 * with one map per input part: MGD-local mesh id -> merged global mesh id (for pinnings, validation, rank
 * bindings, etc.).
 *
 * A single input is returned unchanged; the optional vector contains one identity map for that graph.
 */
LogicalMultiMeshGraph merge_logical_multi_mesh_adjacency_graphs(
    const std::vector<LogicalMultiMeshGraph>& logical_multi_mesh_graphs,
    std::vector<std::map<MeshId, MeshId>>* per_part_local_to_global_mesh_ids = nullptr);

/**
 * @brief Represents a physical mesh node in a 2-layer adjacency graph
 *
 * Simplified to just be a MeshId. The internal adjacency graph is accessed via
 * PhysicalMultiMeshGraph::mesh_adjacency_graphs_.
 */
using PhysicalMeshNode = MeshId;

// Note: Exit node information is now stored as an AdjacencyGraph in PhysicalMultiMeshGraph
// No separate MeshExitNodeInfo struct needed - the adjacency graph itself represents exit nodes

/**
 * @brief Multi-mesh adjacency graph for physical ASICs where meshes are nodes
 *
 * Efficient representation that avoids duplicating adjacency graphs:
 * - Stores each mesh's internal adjacency graph once in a map
 * - Stores mesh-level connectivity as lightweight AdjacencyGraph<MeshId>
 * - Tracks exit node information as an adjacency graph (only exit nodes and their intermesh connections)
 *
 * This type represents a hierarchical adjacency graph:
 * - Top layer: adjacency graph of mesh IDs (which meshes connect to which meshes)
 * - Bottom layer: for each mesh, its internal adjacency graph (which ASICs connect within the mesh)
 * - Exit nodes: adjacency graph containing only exit nodes (ASICs that connect to other meshes)
 *   and their connections to ASICs in other meshes. Multiple connections are represented by
 *   duplicate entries in the neighbor vector (matching AdjacencyGraph's channel representation).
 */
struct PhysicalMultiMeshGraph {
    // Map from MeshId to its internal adjacency graph (stored once, no duplication)
    std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> mesh_adjacency_graphs_;

    // Mesh-level adjacency graph using MeshIds (lightweight, no graph duplication)
    AdjacencyGraph<MeshId> mesh_level_graph_;

    // Map from MeshId to exit node adjacency graph for that mesh
    // Contains only exit nodes (PhysicalExitNode structs representing ASICs that connect to ASICs in other meshes) as
    // nodes, and their connections to PhysicalExitNodes in other meshes as edges. Each PhysicalExitNode includes the
    // mesh_id (which mesh it belongs to) and asic_id (the ASIC identifier). Multiple channels between the same pair are
    // represented by duplicate entries.
    std::map<MeshId, AdjacencyGraph<PhysicalExitNode>> mesh_exit_node_graphs_;

    // PGD-derived intra-mesh pinning: physical MeshId (this graph's own mesh index, same key space as
    // mesh_adjacency_graphs_) -> (row-major logical chip id -> AsicPosition). Captured from the PGD<->MGD match
    // during grouping selection and carried through PSD placement, so later intra-mesh mapping can follow the PGD
    // layout instead of re-solving it. The inner resolution is purely logical-chip-id -> physical ASIC position
    // (TrayID + ASICLocation), NOT a specific hardware AsicID; the layout is expressed in stable physical
    // positions and resolved back to ASIC(s) at consume time. It deliberately does NOT bake a logical mesh
    // assignment into the key (that decision is made later during the multi-mesh solve). Populated when the graph
    // was built from a PhysicalGroupingDescriptor, or by the rank-bound PGD pinning fast path; empty otherwise.
    std::map<MeshId, std::map<LogicalChipId, AsicPosition>> mesh_pgd_pinnings_;
};

/**
 * Per-mesh ASIC footprint plus optional PGD-derived chip-id -> ASIC-position pinning.
 * Keyed by MeshId when building a PhysicalMultiMeshGraph from PSD placements.
 */
struct MeshPhysicalLayout {
    std::unordered_set<tt::tt_metal::AsicID> asics;
    // Empty when the placement did not carry a PGD<->MGD pinning (callers assume row-major identity).
    std::map<LogicalChipId, AsicPosition> mesh_node_to_asic_position;
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
 * @brief Rank-bound physical graph plus PGD preferred pinnings (ControlPlane / Phase 2 fast path).
 *
 * Builds mesh partitions from asic_id_to_mesh_rank, then for each mesh on that graph copies the
 * committed PGD<->MGD MESH grouping pinning for the mesh's MGD type onto mesh_pgd_pinnings_.
 */
PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const std::optional<std::vector<PinningConstraint>>& pinnings = std::nullopt);

/**
 * @brief Build a physical multi-mesh adjacency graph from physical system descriptor and physical grouping descriptor
 *
 * Creates a PhysicalMultiMeshGraph with:
 * - Mesh-level adjacency graph (AdjacencyGraph<MeshId>) representing inter-mesh connectivity
 * - Map of mesh IDs to their internal adjacency graphs (AdjacencyGraph<AsicID>)
 *
 * @param physical_system_descriptor Reference to the physical system descriptor containing ASIC topology
 * @param physical_grouping_descriptor Reference to the physical grouping descriptor containing mesh grouping
 * information
 * @param mesh_graph_descriptor Reference to the mesh graph descriptor containing logical mesh topology
 * @param pinnings Optional fabric-node pinning constraints used to restrict logical-to-physical mesh placement
 * during multi-shape physical graph construction
 * @return PhysicalMultiMeshGraph containing mesh-level graph and internal mesh nodes
 */
PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const std::optional<std::vector<PinningConstraint>>& pinnings = std::nullopt);

/**
 * @brief Build a physical multi-mesh adjacency graph using multiple MGDs (one PSD, one PGD)
 *
 * For each MGD, collects valid MESH groupings (same as the single-MGD build), then merges results.
 * With multiple MGD files in one process, ensure PGD/MGD keys remain consistent (each descriptor may need distinct
 * instance names when \c DistributedContext::subcontext_id() uniquifies names per split rank).
 *
 * @param mesh_graph_descriptors  Const reference to the caller's `std::vector` (the container is not copied;
 *                                only a reference is passed). Elements are the loaded MGDs in order.
 */
PhysicalMultiMeshGraph build_physical_multi_mesh_adjacency_graph(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const std::vector<tt::tt_fabric::MeshGraphDescriptor>& mesh_graph_descriptors);

/**
 * @brief Build a flat PhysicalAdjacencyMap from PhysicalSystemDescriptor
 *
 * Builds a complete flat adjacency map including all connections
 * (both intra-mesh and intermesh), with multiple entries per channel.
 *
 * @param physical_system_descriptor Reference to the physical system descriptor containing ASIC topology
 * @return PhysicalAdjacencyMap Map from AsicID to vector of neighbor AsicIDs (with multiple entries per channel)
 */
PhysicalAdjacencyMap build_flat_adjacency_map_from_psd(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor);

/**
 * @brief Build hierarchical multi-mesh graph from a flattened adjacency graph
 *
 * Takes a flat adjacency graph (all ASICs and their neighbors) and splits it into a multi-mesh graph
 * based on mesh layouts. This is useful when you have a pre-built adjacency graph and need to
 * organize it by mesh.
 *
 * The function:
 * - Splits the flat adjacency graph into per-mesh adjacency graphs (only intra-mesh connections)
 * - Builds the mesh-level graph based on intermesh connections
 * - Builds exit node graphs for each mesh
 * - Stores non-empty PGD pinnings under PhysicalMultiMeshGraph::mesh_pgd_pinnings_
 *
 * @param flat_adjacency_graph Flat adjacency graph containing all ASICs and their neighbors
 * @param mesh_layouts Map from MeshId to per-mesh ASIC footprint and optional PGD pinning. MeshIds are used
 *                     as-is in the resulting PhysicalMultiMeshGraph (no index remapping).
 * @return PhysicalMultiMeshGraph containing mesh-level graph, per-mesh adjacency graphs, and exit node graphs
 */
PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::map<MeshId, MeshPhysicalLayout>& mesh_layouts);

/**
 * @brief Build hierarchical multi-mesh graph from ASIC groupings (and optional PGD pinnings)
 *
 * Convenience overload: MeshIds are used as-is. Prefer std::map<MeshId, MeshPhysicalLayout> when both footprint and
 * pinning are available together.
 *
 * @param flat_adjacency_graph Flat adjacency graph containing all ASICs and their neighbors
 * @param mesh_groupings Map from MeshId to the set of ASIC IDs belonging to that mesh
 * @param mesh_pgd_pinnings Optional per-mesh logical-chip-id -> ASIC position layouts, keyed by the same MeshId
 *                          as mesh_groupings. Non-empty entries are stored under that MeshId.
 * @return PhysicalMultiMeshGraph containing mesh-level graph, per-mesh adjacency graphs, and exit node graphs
 */
PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::map<MeshId, std::unordered_set<tt::tt_metal::AsicID>>& mesh_groupings,
    const std::map<MeshId, std::map<LogicalChipId, tt::tt_metal::ASICPosition>>& mesh_pgd_pinnings = {});

/**
 * @brief Build hierarchical multi-mesh graph from PSD placements
 *
 * Each placement's ASIC footprint is keyed by MeshId{placement_index}; pinning is read from
 * placement.mesh_node_to_asic_position.
 */
PhysicalMultiMeshGraph build_hierarchical_from_flat_graph(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_adjacency_graph,
    const std::vector<::tt::tt_fabric::PsdPlacement>& placements);

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
 *               config.mesh_validation_modes and config.inter_mesh_validation_mode select STRICT vs RELAXED.
 *               If unset, mapping defaults to RELAXED for that scope.
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
 * @note When config.hostname_to_asics is non-empty, inter-mesh solving applies the same minimal host-cover bias as PGD
 *       (same-rank / preferred globals) so mesh-to-mesh mapping tends to use fewer hosts when possible.
 */
TopologyMappingResult map_multi_mesh_to_physical(
    const LogicalMultiMeshGraph& adjacency_map_logical,
    const PhysicalMultiMeshGraph& adjacency_map_physical,
    const TopologyMappingConfig& config,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {});

/** Log inter-mesh and per-mesh intra-mesh degree histograms at INFO (one line each). */
void log_logical_multi_mesh_adjacency_histograms(const LogicalMultiMeshGraph& multi_mesh_graph);
void log_physical_multi_mesh_adjacency_histograms(const PhysicalMultiMeshGraph& multi_mesh_graph);

// Choose one (exit, peer) FabricNodeId pair per candidate set ("hop") such that no FabricNodeId is
// reused across sets. `candidates[i]` are the candidate pairs for position i; returns the chosen pairs
// in order, or std::nullopt if no collision-free assignment exists (any set empty, or overconstrained).
//
// A backtracking solver for a system of distinct representatives (most-constrained set first). The blitz
// decode pipeline builder uses it to lay out inter-mesh ring hops, where per-hop greedy first-fit can
// strand a mid-chain hop on tight rings; kept here so it is reusable and unit-testable without a control
// plane.
std::optional<std::vector<std::pair<FabricNodeId, FabricNodeId>>> assign_non_colliding_hops(
    const std::vector<std::vector<std::pair<FabricNodeId, FabricNodeId>>>& candidates);

}  // namespace tt::tt_metal::experimental::tt_fabric

// Formatter for LogicalExitNode to enable fmt::format debugging
template <>
struct fmt::formatter<tt::tt_metal::experimental::tt_fabric::LogicalExitNode> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::experimental::tt_fabric::LogicalExitNode& exit_node, format_context& ctx) const
        -> format_context::iterator {
        if (exit_node.fabric_node_id.has_value()) {
            return fmt::format_to(
                ctx.out(),
                "LogicalExitNode(mesh_id={}, fabric_node_id={})",
                exit_node.mesh_id.get(),
                *exit_node.fabric_node_id);
        }
        return fmt::format_to(ctx.out(), "LogicalExitNode(mesh_id={}, fabric_node_id=None)", exit_node.mesh_id.get());
    }
};

// Formatter for PhysicalExitNode to enable fmt::format debugging
template <>
struct fmt::formatter<tt::tt_metal::experimental::tt_fabric::PhysicalExitNode> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::experimental::tt_fabric::PhysicalExitNode& exit_node, format_context& ctx) const
        -> format_context::iterator {
        return fmt::format_to(
            ctx.out(), "PhysicalExitNode(mesh_id={}, asic_id={})", exit_node.mesh_id.get(), exit_node.asic_id.get());
    }
};
