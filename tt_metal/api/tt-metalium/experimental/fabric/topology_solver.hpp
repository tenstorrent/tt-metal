// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

/**
 * @brief Generic graph representation with minimal query interface
 *
 * AdjacencyGraph provides a generic graph representation that works with any node type.
 * It provides a minimal interface for querying graph structure: getting all nodes and
 * getting neighbors of a specific node.
 *
 * @tparam NodeId The type used to identify nodes in the graph
 */
template <typename NodeId>
class AdjacencyGraph {
public:
    using NodeType = NodeId;
    using AdjacencyMap = std::map<NodeId, std::vector<NodeId>>;

    /**
     * @brief Construct empty adjacency graph
     */
    AdjacencyGraph() = default;

    /**
     * @brief Construct adjacency graph from a MeshGraph
     *
     * @param mesh_graph The mesh graph to construct the adjacency graph from
     */
    explicit AdjacencyGraph(const AdjacencyMap& adjacency_map);

    /**
     * @brief Get all nodes in the graph
     *
     * @return const std::vector<NodeId>& Vector of all node IDs in the graph
     */
    const std::vector<NodeId>& get_nodes() const;

    /**
     * @brief Get neighbors of a specific node
     *
     * @param node The node ID to get neighbors for
     * @return const std::vector<NodeId>& Vector of neighbor node IDs
     */
    const std::vector<NodeId>& get_neighbors(const NodeId& node) const;

    /**
     * @brief Print adjacency map for debugging
     *
     * Prints the graph structure showing each node and its neighbors.
     * Useful for debugging mapping failures.
     *
     * @param graph_name Name to identify this graph in the output
     */
    void print_adjacency_map(const std::string& graph_name = "Graph") const;

private:
    AdjacencyMap adj_map_;
    std::vector<NodeId> nodes_cache_;
};

std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(const MeshGraph& mesh_graph);

std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> build_adjacency_map_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);

/**
 * @brief Unified constraint system for topology mapping
 *
 * MappingConstraints represents all constraints internally as trait maps. Both trait-based
 * constraints (one-to-many) and explicit pair constraints (one-to-one) are unified into a
 * single intersection-based representation.
 *
 * @tparam TargetNode The type of nodes in the target graph
 * @tparam GlobalNode The type of nodes in the global graph
 */
template <typename TargetNode, typename GlobalNode>
class MappingConstraints {
public:
    /**
     * @brief Construct empty constraints
     */
    MappingConstraints() = default;

    /**
     * @brief Constructor from sets of constraint pairs
     *
     * Converts pairs into the internal mapping representation.
     *
     * @param required_constraints Set of required constraint pairs (target, global)
     * @param preferred_constraints Set of preferred constraint pairs (target, global)
     */
    MappingConstraints(
        const std::set<std::pair<TargetNode, GlobalNode>>& required_constraints,
        const std::set<std::pair<TargetNode, GlobalNode>>& preferred_constraints = {});

    /**
     * @brief Add required trait-based constraint (one-to-many)
     *
     * Constrains target nodes with trait value T to only map to global nodes with same trait value T.
     * All constraints are intersected - a target node must satisfy ALL required constraints simultaneously.
     * Throws TT_THROW if constraint causes conflicts (empty valid mappings).
     *
     * @tparam TraitType The type of the trait value (must be explicitly specified)
     * @param target_traits Map from target nodes to their trait values
     * @param global_traits Map from global nodes to their trait values
     * @throws std::runtime_error If constraint causes empty valid mappings for any target node
     */
    template <typename TraitType>
    void add_required_trait_constraint(
        const std::map<TargetNode, TraitType>& target_traits, const std::map<GlobalNode, TraitType>& global_traits);

    /**
     * @brief Add preferred trait-based constraint (one-to-many)
     *
     * Constrains target nodes with trait value T to prefer mapping to global nodes with same trait value T.
     * Preferred constraints guide the solver but don't restrict valid mappings.
     *
     * @tparam TraitType The type of the trait value (must be explicitly specified)
     * @param target_traits Map from target nodes to their trait values
     * @param global_traits Map from global nodes to their trait values
     */
    template <typename TraitType>
    void add_preferred_trait_constraint(
        const std::map<TargetNode, TraitType>& target_traits, const std::map<GlobalNode, TraitType>& global_traits);

    /**
     * @brief Add explicit required constraint (one-to-one)
     *
     * Pins a specific target node to a specific global node.
     * Intersects with existing constraints. Throws TT_THROW if constraint causes conflicts.
     *
     * @param target_node The target node to constrain
     * @param global_node The global node it must map to
     * @throws std::runtime_error If constraint causes empty valid mappings
     */
    void add_required_constraint(TargetNode target_node, GlobalNode global_node);

    /**
     * @brief Add explicit preferred constraint (one-to-one)
     *
     * Suggests a mapping but doesn't restrict valid mappings.
     * The solver can still choose other nodes if needed.
     *
     * @param target_node The target node
     * @param global_node The preferred global node to map to
     */
    void add_preferred_constraint(TargetNode target_node, GlobalNode global_node);

    /**
     * @brief Get valid mappings for a specific target node
     *
     * @param target The target node
     * @return const std::set<GlobalNode>& Set of global nodes this target can map to
     */
    const std::set<GlobalNode>& get_valid_mappings(TargetNode target) const;

    /**
     * @brief Get preferred mappings for a specific target node
     *
     * @param target The target node
     * @return const std::set<GlobalNode>& Set of preferred global nodes for this target
     */
    const std::set<GlobalNode>& get_preferred_mappings(TargetNode target) const;

    /**
     * @brief Check if a specific mapping is valid
     *
     * @param target The target node
     * @param global The global node
     * @return true if the mapping satisfies all required constraints, false otherwise
     */
    bool is_valid_mapping(TargetNode target, GlobalNode global) const;

    /**
     * @brief Get all valid mappings (for solver access)
     *
     * @return const std::map<TargetNode, std::set<GlobalNode>>& Map of all valid mappings
     */
    const std::map<TargetNode, std::set<GlobalNode>>& get_valid_mappings() const;

    /**
     * @brief Get all preferred mappings (for solver access)
     *
     * @return const std::map<TargetNode, std::set<GlobalNode>>& Map of all preferred mappings
     */
    const std::map<TargetNode, std::set<GlobalNode>>& get_preferred_mappings() const;

private:
    // Internal representation: intersection of all constraints
    std::map<TargetNode, std::set<GlobalNode>> valid_mappings_;      // Required constraints
    std::map<TargetNode, std::set<GlobalNode>> preferred_mappings_;  // Preferred constraints

    // Helper to intersect two sets
    static std::set<GlobalNode> intersect_sets(const std::set<GlobalNode>& set1, const std::set<GlobalNode>& set2);

    // Internal validation - throws if invalid
    void validate_and_throw() const;
};

}  // namespace tt::tt_fabric

// Include template implementations
#include <tt-metalium/experimental/fabric/topology_solver.tpp>
