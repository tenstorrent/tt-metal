// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <unordered_set>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::detail {

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
        const std::vector<int>& mapping,
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

}  // namespace tt::tt_fabric::detail

// Include template implementations
#include "tt_metal/fabric/topology_solver_internal.tpp"
