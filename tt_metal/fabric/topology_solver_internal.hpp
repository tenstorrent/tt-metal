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

    /**
     * @brief Construct GraphIndexData from AdjacencyGraph inputs
     *
     * Builds indexed representation from target and global graphs.
     */
    GraphIndexData(
        const AdjacencyGraph<TargetNode>& target_graph, const AdjacencyGraph<GlobalNode>& global_graph);
};

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

    /**
     * @brief Construct ConstraintIndexData from MappingConstraints and GraphIndexData
     *
     * Builds indexed constraint representation from constraints and graph data.
     */
    ConstraintIndexData(
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data);

    // Helper: check if mapping is valid
    bool is_valid_mapping(size_t target_idx, size_t global_idx) const;

    // Helper: get candidates for target node
    // Returns restricted candidates if available, otherwise returns empty vector (meaning all are valid)
    const std::vector<size_t>& get_candidates(size_t target_idx) const;
};

/**
 * @brief Unified heuristic for node selection and candidate generation
 *
 * Combines node selection and candidate generation with explicit priority:
 * 1. Hard constraints (must satisfy)
 * 2. Soft constraints (optimize for)
 * 3. Runtime optimization (minimize search tree)
 *
 * Non-templated class - types are deduced from GraphIndexData at method call time.
 */
class SearchHeuristic {
public:
    /**
     * @brief Result of node selection and candidate generation
     */
    struct SelectionResult {
        size_t target_idx = SIZE_MAX;         // Selected target node index (SIZE_MAX if none)
        std::vector<size_t> candidates;      // Valid candidates (ordered by cost, lower = better)
    };

    /**
     * @brief Select next target node and generate ordered candidates
     *
     * Uses integer cost scoring (lower cost = higher priority):
     * - Node cost combines: hard constraints → soft constraints → runtime optimization
     * - Candidate cost: filters by hard constraints, orders by soft + runtime
     *
     * Uses ConstraintIndexData for fast index-based lookups (no node type conversions needed).
     *
     * @param graph_data Graph index data
     * @param constraint_data Constraint index data (for fast lookups)
     * @param mapping Current partial mapping (mapping[i] = global_idx or -1)
     * @param used Which global nodes are already used (used[i] = true if assigned)
     * @param validation_mode Connection validation mode
     * @return SelectionResult with selected node and ordered candidates
     */
    template <typename TargetNode, typename GlobalNode>
    static SelectionResult select_and_generate_candidates(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Check if candidate satisfies all hard constraints
     *
     * @return true if candidate should be included, false if should be filtered out
     *
     * Public so ConsistencyChecker can use it for forward consistency checking.
     */
    template <typename TargetNode, typename GlobalNode>
    static bool check_hard_constraints(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);

private:
    /**
     * @brief Compute cost for selecting a target node (lower = better)
     *
     * cost = (candidate_count * HARD_WEIGHT)
     *      - (preferred_count * SOFT_WEIGHT)
     *      - (mapped_neighbors * RUNTIME_WEIGHT)
     */
    template <typename TargetNode, typename GlobalNode>
    static int compute_node_cost(
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Compute cost for a valid candidate (lower = better)
     *
     * Only called for candidates that passed hard constraint checks.
     * cost = -is_preferred * SOFT_WEIGHT
     *      - channel_match_count * SOFT_WEIGHT
     *      + degree_gap * RUNTIME_WEIGHT
     */
    template <typename TargetNode, typename GlobalNode>
    static int compute_candidate_cost(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Generate and order candidates for a target node
     *
     * Filters by hard constraints first, then orders by cost (lower = better)
     */
    template <typename TargetNode, typename GlobalNode>
    static std::vector<size_t> generate_ordered_candidates(
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    // Cost weights (ensure hard >> soft >> runtime)
    static constexpr int HARD_WEIGHT = 1000000;
    static constexpr int SOFT_WEIGHT = 1000;
    static constexpr int RUNTIME_WEIGHT = 1;
};

/**
 * @brief ConsistencyChecker validates partial mappings during DFS to prune invalid branches early
 *
 * Non-templated struct with templated methods - template types are deduced from GraphIndexData arguments.
 * This allows usage without explicit template parameters: ConsistencyChecker::check_local_consistency(...)
 */
struct ConsistencyChecker {
    /**
     * @brief Check local consistency: verify assignment is consistent with already-assigned neighbors
     *
     * Checks that if target node A is mapped to global X, and target node B (neighbor of A)
     * is mapped to global Y, then X and Y must be connected.
     * In STRICT mode: also checks channel counts are sufficient.
     *
     * @param target_idx Index of target node being assigned
     * @param global_idx Index of global node being assigned to
     * @param graph_data Indexed graph data
     * @param mapping Current partial mapping (target_idx -> global_idx)
     * @param validation_mode Channel validation mode
     * @return true if assignment is locally consistent
     */
    template <typename TargetNode, typename GlobalNode>
    static bool check_local_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Check forward consistency: ensure assignment leaves viable options for future neighbors
     *
     * Verifies that each unassigned neighbor of the target node has at least one viable candidate
     * among the unused neighbors of the global node.
     *
     * @param target_idx Index of target node being assigned
     * @param global_idx Index of global node being assigned to
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data
     * @param mapping Current partial mapping
     * @param used Which global nodes are already used
     * @param validation_mode Channel validation mode
     * @return true if assignment leaves viable options for future neighbors
     */
    template <typename TargetNode, typename GlobalNode>
    static bool check_forward_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Count unused global nodes reachable from a starting point
     *
     * Used for path graph fast path optimization to verify there are enough
     * unused nodes for remaining target nodes.
     *
     * @param start_global_idx Starting global node index
     * @param graph_data Indexed graph data (only uses global graph)
     * @param used Which global nodes are already used
     * @return Number of unused nodes reachable from start_global_idx
     */
    template <typename TargetNode, typename GlobalNode>
    static size_t count_reachable_unused(
        size_t start_global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const std::vector<bool>& used);
};

template <typename TargetNode, typename GlobalNode>
struct PathGraphDetector;

template <typename TargetNode, typename GlobalNode>
class DFSSearchEngine;

template <typename TargetNode, typename GlobalNode>
struct MappingValidator;

}  // namespace tt::tt_fabric::detail

// Include template implementations
#include "tt_metal/fabric/topology_solver_internal.tpp"
