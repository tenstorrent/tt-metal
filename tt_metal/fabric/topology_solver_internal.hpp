// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::detail {

/**
 * @brief Indexed graph representation for efficient lookups
 *
 * Converts AdjacencyGraph into index-based representation for O(1) lookups.
 * Stores deduplicated, sorted adjacency lists and connection counts.
 */
template <typename TargetNode, typename GlobalNode>
struct GraphIndexData {
    // Node vectors
    std::vector<TargetNode> target_nodes;
    std::vector<GlobalNode> global_nodes;

    // Index mappings
    std::map<TargetNode, size_t> target_to_idx;
    std::map<GlobalNode, size_t> global_to_idx;

    // Adjacency index vectors (deduplicated, sorted)
    std::vector<std::vector<size_t>> target_adj_idx;
    std::vector<std::vector<size_t>> global_adj_idx;

    // Connection count maps (for strict mode / multi-edge support)
    // conn_count[i][j] = number of channels from node i to node j
    std::vector<std::map<size_t, size_t>> target_conn_count;
    std::vector<std::map<size_t, size_t>> global_conn_count;

    // Degree vectors
    std::vector<size_t> target_deg;
    std::vector<size_t> global_deg;

    size_t n_target = 0;
    size_t n_global = 0;
};

/**
 * @brief Build GraphIndexData from AdjacencyGraph inputs
 */
template <typename TargetNode, typename GlobalNode>
GraphIndexData<TargetNode, GlobalNode> build_graph_index_data(
    const AdjacencyGraph<TargetNode>& target_graph, const AdjacencyGraph<GlobalNode>& global_graph);

/**
 * @brief Indexed constraint representation for efficient lookups
 *
 * Converts MappingConstraints into index-based representation for O(1) constraint checks.
 * Stores restricted and preferred mappings as index vectors.
 */
template <typename TargetNode, typename GlobalNode>
struct ConstraintIndexData {
    // Restricted mappings: target_idx -> vector of valid global_indices
    // If empty for a target_idx, that target can map to any global node
    std::vector<std::vector<size_t>> restricted_global_indices;

    // Preferred mappings: target_idx -> vector of preferred global_indices
    // Used for optimization, doesn't restrict valid mappings
    std::vector<std::vector<size_t>> preferred_global_indices;

    // Helper: check if mapping is valid
    bool is_valid_mapping(size_t target_idx, size_t global_idx) const;

    // Helper: get candidates for target node
    // Returns restricted candidates if available, otherwise returns empty vector (meaning all are valid)
    const std::vector<size_t>& get_candidates(size_t target_idx) const;
};

/**
 * @brief Build ConstraintIndexData from MappingConstraints and GraphIndexData
 */
template <typename TargetNode, typename GlobalNode>
ConstraintIndexData<TargetNode, GlobalNode> build_constraint_index_data(
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data);

template <typename TargetNode, typename GlobalNode>
struct NodeSelector;

template <typename TargetNode, typename GlobalNode>
struct CandidateGenerator;

template <typename TargetNode, typename GlobalNode>
struct ConsistencyChecker;

template <typename TargetNode, typename GlobalNode>
struct PathGraphDetector;

template <typename TargetNode, typename GlobalNode>
class DFSSearchEngine;

template <typename TargetNode, typename GlobalNode>
struct MappingValidator;

}  // namespace tt::tt_fabric::detail

// Include template implementations
#include "tt_metal/fabric/topology_solver_internal.tpp"
