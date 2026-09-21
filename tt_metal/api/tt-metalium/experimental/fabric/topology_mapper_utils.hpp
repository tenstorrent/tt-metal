// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {
class PhysicalGroupingDescriptor;
struct PlacedMesh;
class SatPlacementEnumerationSession;
}  // namespace tt::tt_fabric

namespace tt::tt_metal::experimental::tt_fabric {

// Import types from tt::tt_fabric for use in this API
using ::tt::tt_fabric::AdjacencyGraph;
using ::tt::tt_fabric::ConnectionValidationMode;
using ::tt::tt_fabric::FabricNodeId;
using ::tt::tt_fabric::LogicalChipId;
using ::tt::tt_fabric::MeshHostRankId;
using ::tt::tt_fabric::MeshId;

// Type alias for the flat ASIC adjacency map used by hierarchical graph builders
using PhysicalAdjacencyMap = std::map<tt::tt_metal::AsicID, std::vector<tt::tt_metal::AsicID>>;

// Use ASICPosition from tt::tt_metal namespace
using AsicPosition = tt::tt_metal::ASICPosition;

// Map from AsicID to its physical position (TrayID, ASICLocation)
// Required only when using pinning constraints
using AsicPositionMap = std::map<tt::tt_metal::AsicID, AsicPosition>;

// MGD many-to-many pinning group (same type as MeshGraphDescriptor::get_pinnings() values).
using PinningConstraint = ::tt::tt_fabric::AsicPinningGroup;

// Pinning groups keyed by local mesh id (same shape as MeshGraphDescriptor::get_pinnings()).
using PinningsByMesh = std::map<::tt::tt_fabric::MeshId, std::vector<PinningConstraint>>;

inline void merge_pinnings_by_mesh(PinningsByMesh& dest, const std::vector<PinningConstraint>& groups) {
    for (const auto& group : groups) {
        if (!group.fabric_nodes.empty()) {
            dest[group.fabric_nodes.front().mesh_id].push_back(group);
        }
    }
}

// Galaxy corner pinnings for a single mesh, ensuring QSFP links align with the fabric mesh corner nodes
// and the mesh is not folded. Pins all four logical corners to the four tray corners (with hard_pin_node_0
// fixing the NW corner to tray 1 / asic 1); nw_corner_only pins ONLY the NW corner to any tray-corner ASIC
// (asic_location==1 on trays 1..4) for sub-galaxy slices. Shared by
// generate_rank_bindings (Phase 1) and ControlPlane (Phase 2) so both apply identical placement.
// Each returned group is 1:many (single corner node, multiple allowed tray positions).
std::vector<PinningConstraint> get_galaxy_fixed_asic_position_pinnings_for_mesh(
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

    // Optional many-to-many pinning groups restricting which physical ASIC positions
    // listed logical nodes may map to
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

using LogicalMeshNode = MeshId;
using PhysicalMeshNode = MeshId;

/**
 * @brief Represents a logical exit node that can be at either the mesh level or fabric node level
 *
 * Logical exit nodes can represent:
 * - Mesh-level exit nodes: mesh_id is set, fabric_node_id is empty (represents the entire mesh as an exit point)
 * - Fabric node-level exit nodes: both mesh_id and fabric_node_id are set (represents a specific fabric node as an exit
 * point)
 */
struct LogicalExitNode {
    LogicalMeshNode mesh_id;
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
    PhysicalMeshNode mesh_id;
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
 * - Stores mesh-level connectivity as lightweight AdjacencyGraph<LogicalMeshNode>
 *
 * This type represents a hierarchical adjacency graph:
 * - Top layer: adjacency graph of mesh IDs (which meshes connect to which meshes)
 * - Bottom layer: for each mesh, its internal adjacency graph (which fabric nodes connect within the mesh)
 */
struct LogicalMultiMeshGraph {
    // Map from LogicalMeshNode to its internal adjacency graph (stored once, no duplication)
    std::map<LogicalMeshNode, AdjacencyGraph<FabricNodeId>> mesh_adjacency_graphs_;

    // Mesh-level adjacency graph using LogicalMeshNodes (lightweight, no graph duplication)
    AdjacencyGraph<LogicalMeshNode> mesh_level_graph_;

    // Map from LogicalMeshNode to exit node adjacency graph for that mesh (optional, only populated if specified)
    // Contains exit nodes (LogicalExitNode structs) that can represent either:
    // - Mesh-level exit nodes (mesh_id set, fabric_node_id empty) - the entire mesh serves as an exit point
    // - Fabric node-level exit nodes (both mesh_id and fabric_node_id set) - specific fabric nodes serve as exit points
    // and their connections to exit nodes in other meshes as edges.
    // Multiple channels between the same pair are represented by duplicate entries.
    // Only populated when strict mode intermesh ports are specified.
    std::map<LogicalMeshNode, AdjacencyGraph<LogicalExitNode>> mesh_exit_node_graphs_;
};

LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor);

LogicalMultiMeshGraph merge_logical_multi_mesh_adjacency_graphs(
    const std::vector<LogicalMultiMeshGraph>& logical_multi_mesh_graphs,
    std::vector<std::map<MeshId, MeshId>>* per_part_local_to_global_mesh_ids = nullptr);

PhysicalAdjacencyMap build_flat_adjacency_map_from_psd(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor);

/**
 * @brief Multi-mesh adjacency graph for physical ASICs where meshes are nodes
 *
 * Efficient representation that avoids duplicating adjacency graphs:
 * - Stores each mesh's internal adjacency graph once in a map
 * - Stores mesh-level connectivity as lightweight AdjacencyGraph<PhysicalMeshNode>
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
    // Map from PhysicalMeshNode to its internal adjacency graph (stored once, no duplication)
    std::map<PhysicalMeshNode, AdjacencyGraph<tt::tt_metal::AsicID>> mesh_adjacency_graphs_;

    // Mesh-level adjacency graph using PhysicalMeshNodes (lightweight, no graph duplication)
    AdjacencyGraph<PhysicalMeshNode> mesh_level_graph_;

    // Map from MeshId to exit node adjacency graph for that mesh
    // Contains only exit nodes (PhysicalExitNode structs representing ASICs that connect to ASICs in other meshes) as
    // nodes, and their connections to PhysicalExitNodes in other meshes as edges. Each PhysicalExitNode includes the
    // mesh_id (which mesh it belongs to) and asic_id (the ASIC identifier). Multiple channels between the same pair are
    // represented by duplicate entries.
    std::map<PhysicalMeshNode, AdjacencyGraph<PhysicalExitNode>> mesh_exit_node_graphs_;

    // PGD-derived intra-mesh pinning: physical mesh (this graph's own mesh index, same key space as
    // mesh_adjacency_graphs_) -> (row-major logical chip id -> AsicPosition). Captured from the PGD<->MGD match
    // during grouping selection and carried through PSD placement, so later intra-mesh mapping can follow the PGD
    // layout instead of re-solving it. The inner resolution is purely logical-chip-id -> physical ASIC position
    // (TrayID + ASICLocation), NOT a specific hardware AsicID; the layout is expressed in stable physical
    // positions and resolved back to ASIC(s) at consume time. It deliberately does NOT bake a logical mesh
    // assignment into the key (that decision is made later during the multi-mesh solve). Populated when the graph
    // was built from a PhysicalGroupingDescriptor, or by the rank-bound PGD pinning fast path; empty otherwise.
    std::map<PhysicalMeshNode, std::map<LogicalChipId, AsicPosition>> mesh_pgd_pinnings_;
};

/**
 * @brief Map logical meshes onto the PSD using PGD SAT seating and identity intra-mesh.
 *
 * Unbound PGD path: joint placement assigns each logical mesh instance a footprint; intra-mesh is
 * completed on that identity mapping. A failed intra-mesh candidate is excluded from later seatings.
 */
TopologyMappingResult map_multi_mesh_to_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
    const TopologyMappingConfig& config,
    const std::optional<PinningsByMesh>& pinnings = {},
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {});

TopologyMappingResult map_multi_mesh_to_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const std::vector<const ::tt::tt_fabric::MeshGraphDescriptor*>& mesh_graph_descriptors,
    const TopologyMappingConfig& config,
    const std::vector<std::optional<PinningsByMesh>>& per_mgd_pinnings = {},
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {});

/**
 * @brief Pull-based enumerator: SAT seating, then a PhysicalMultiMeshGraph built for that seating,
 *        then identity intra-mesh.
 *
 *   MultiMeshSolutionEnumerator e(psd, pgd, mgd, config);
 *   while (auto solution = e.next()) { ... }
 *
 * Each next() takes one seating from SatPlacementEnumerationSession (MeshIds kept), builds the
 * physical graph from that seating, and completes fabric-node → ASIC on the identity mapping.
 * If intra-mesh fails against a seated candidate, that (mesh, footprint) is forbidden and the next
 * seating is tried.
 *
 * Lifetime: the PSD, PGD, and MeshGraphDescriptor(s) passed to the constructor must outlive
 * the enumerator.
 */
class MultiMeshSolutionEnumerator {
public:
    MultiMeshSolutionEnumerator(
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
        const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor,
        const TopologyMappingConfig& config,
        bool unique_shapes = false,
        const std::optional<PinningsByMesh>& pinnings = {},
        const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
        const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {});

    MultiMeshSolutionEnumerator(
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const tt::tt_fabric::PhysicalGroupingDescriptor& physical_grouping_descriptor,
        const std::vector<const ::tt::tt_fabric::MeshGraphDescriptor*>& mesh_graph_descriptors,
        const TopologyMappingConfig& config,
        bool unique_shapes = false,
        const std::vector<std::optional<PinningsByMesh>>& per_mgd_pinnings = {},
        const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
        const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {});

    MultiMeshSolutionEnumerator(const MultiMeshSolutionEnumerator&) = delete;
    MultiMeshSolutionEnumerator& operator=(const MultiMeshSolutionEnumerator&) = delete;
    MultiMeshSolutionEnumerator(MultiMeshSolutionEnumerator&&) noexcept;
    MultiMeshSolutionEnumerator& operator=(MultiMeshSolutionEnumerator&&) noexcept;
    ~MultiMeshSolutionEnumerator();

    /**
     * @brief Next seating whose identity intra-mesh completed, or nullopt when placement is exhausted.
     */
    std::optional<TopologyMappingResult> next();

    std::size_t solutions_returned() const { return emitted_; }

    const std::vector<std::map<MeshId, MeshId>>& per_part_local_to_global_mesh_ids() const {
        return per_part_local_to_global_mesh_ids_;
    }

private:
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor_ = nullptr;
    TopologyMappingConfig config_;
    LogicalMultiMeshGraph logical_;
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph_;
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank_;
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_node_id_to_mesh_rank_;
    std::vector<std::map<MeshId, MeshId>> per_part_local_to_global_mesh_ids_;
    ::tt::tt_fabric::ConnectionValidationMode inter_mesh_validation_mode_ =
        ::tt::tt_fabric::ConnectionValidationMode::RELAXED;
    std::unique_ptr<::tt::tt_fabric::SatPlacementEnumerationSession> placement_session_;
    std::vector<std::pair<MeshId, std::unordered_set<tt::tt_metal::AsicID>>> failed_mesh_candidates_;
    std::size_t attempts_ = 0;
    std::size_t emitted_ = 0;

    void fill_host_and_asic_positions_from_psd();
};

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
