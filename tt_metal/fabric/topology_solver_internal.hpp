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

// Forward declarations
template <typename TargetNode, typename GlobalNode>
struct ConstraintIndexData;

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
