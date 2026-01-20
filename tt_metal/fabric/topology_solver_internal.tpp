// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Template implementation file - included at the end of topology_solver_internal.hpp
// This allows template implementations to be separated from declarations while
// still enabling implicit instantiation for any type.

#ifndef TOPOLOGY_SOLVER_INTERNAL_TPP
#define TOPOLOGY_SOLVER_INTERNAL_TPP

#include <algorithm>
#include <climits>   // For INT_MAX
#include <cstddef>   // For SIZE_MAX
#include <sstream>
#include <unordered_set>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
GraphIndexData<TargetNode, GlobalNode>::GraphIndexData(
    const AdjacencyGraph<TargetNode>& target_graph, const AdjacencyGraph<GlobalNode>& global_graph)
    : n_target(target_graph.get_nodes().size()),
      n_global(global_graph.get_nodes().size()) {

    // Build node vectors
    const auto& target_nodes_vec = target_graph.get_nodes();
    const auto& global_nodes_vec = global_graph.get_nodes();

    target_nodes.reserve(target_nodes_vec.size());
    for (const auto& node : target_nodes_vec) {
        target_nodes.push_back(node);
    }

    global_nodes.reserve(global_nodes_vec.size());
    for (const auto& node : global_nodes_vec) {
        global_nodes.push_back(node);
    }

    // Build index mappings
    for (size_t i = 0; i < n_target; ++i) {
        target_to_idx[target_nodes[i]] = i;
    }

    for (size_t i = 0; i < n_global; ++i) {
        global_to_idx[global_nodes[i]] = i;
    }

    // Build connection count maps and deduplicated adjacency index vectors
    target_conn_count.resize(n_target);
    target_adj_idx.resize(n_target);
    target_deg.resize(n_target, 0);

    for (size_t i = 0; i < n_target; ++i) {
        const auto& node = target_nodes[i];
        const auto& neighbors = target_graph.get_neighbors(node);
        std::unordered_set<size_t> seen_indices;

        for (const auto& neigh : neighbors) {
            // Skip self-connections
            if (neigh == node) {
                continue;
            }
            auto it = target_to_idx.find(neigh);
            if (it != target_to_idx.end()) {
                size_t idx = it->second;
                target_conn_count[i][idx]++;
                if (seen_indices.insert(idx).second) {
                    target_adj_idx[i].push_back(idx);
                }
            }
        }
        std::sort(target_adj_idx[i].begin(), target_adj_idx[i].end());
        // Degree is the number of unique neighbors (not counting multi-edges or self-connections)
        target_deg[i] = target_adj_idx[i].size();
    }

    // Build connection count maps and deduplicated adjacency index vectors for global graph
    global_conn_count.resize(n_global);
    global_adj_idx.resize(n_global);
    global_deg.resize(n_global, 0);

    for (size_t i = 0; i < n_global; ++i) {
        const auto& node = global_nodes[i];
        const auto& neighbors = global_graph.get_neighbors(node);
        std::unordered_set<size_t> seen_indices;

        for (const auto& neigh : neighbors) {
            // Skip self-connections
            if (neigh == node) {
                continue;
            }
            auto it = global_to_idx.find(neigh);
            if (it != global_to_idx.end()) {
                size_t idx = it->second;
                global_conn_count[i][idx]++;
                if (seen_indices.insert(idx).second) {
                    global_adj_idx[i].push_back(idx);
                }
            }
        }
        std::sort(global_adj_idx[i].begin(), global_adj_idx[i].end());
        // Degree is the number of unique neighbors (not counting multi-edges or self-connections)
        global_deg[i] = global_adj_idx[i].size();
    }
}

template <typename TargetNode, typename GlobalNode>
bool ConstraintIndexData<TargetNode, GlobalNode>::is_valid_mapping(size_t target_idx, size_t global_idx) const {
    // If no restrictions for this target, all mappings are valid
    if (target_idx >= restricted_global_indices.size() || restricted_global_indices[target_idx].empty()) {
        return true;
    }
    // Check if global_idx is in the restricted list (vectors are sorted, so use binary_search)
    const auto& candidates = restricted_global_indices[target_idx];
    return std::binary_search(candidates.begin(), candidates.end(), global_idx);
}

template <typename TargetNode, typename GlobalNode>
const std::vector<size_t>& ConstraintIndexData<TargetNode, GlobalNode>::get_candidates(size_t target_idx) const {
    // Return restricted candidates if available
    if (target_idx < restricted_global_indices.size() && !restricted_global_indices[target_idx].empty()) {
        return restricted_global_indices[target_idx];
    }
    // Return empty vector to indicate all nodes are valid candidates
    static const std::vector<size_t> empty_vec;
    return empty_vec;
}

template <typename TargetNode, typename GlobalNode>
ConstraintIndexData<TargetNode, GlobalNode>::ConstraintIndexData(
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data) {
    // Initialize vectors for all target nodes
    restricted_global_indices.resize(graph_data.n_target);
    preferred_global_indices.resize(graph_data.n_target);

    // Get valid and preferred mappings from constraints
    const auto& valid_mappings = constraints.get_valid_mappings();
    const auto& preferred_mappings = constraints.get_preferred_mappings();

    // Convert node-based mappings to index-based mappings
    for (size_t i = 0; i < graph_data.n_target; ++i) {
        const TargetNode& target_node = graph_data.target_nodes[i];

        // Process required constraints (restricted mappings)
        auto valid_it = valid_mappings.find(target_node);
        if (valid_it != valid_mappings.end() && !valid_it->second.empty()) {
            // Convert GlobalNode set to index vector
            std::vector<size_t> restricted_indices;
            restricted_indices.reserve(valid_it->second.size());
            std::vector<GlobalNode> missing_nodes;
            for (const auto& global_node : valid_it->second) {
                auto idx_it = graph_data.global_to_idx.find(global_node);
                if (idx_it != graph_data.global_to_idx.end()) {
                    restricted_indices.push_back(idx_it->second);
                } else {
                    missing_nodes.push_back(global_node);
                }
            }
            std::sort(restricted_indices.begin(), restricted_indices.end());

            // Log warning if constraint nodes are missing from the graph
            if (!missing_nodes.empty()) {
                std::stringstream missing_nodes_str;
                bool first = true;
                for (const auto& node : missing_nodes) {
                    if (!first) {
                        missing_nodes_str << ", ";
                    }
                    first = false;
                    missing_nodes_str << node;
                }

                log_warning(
                    tt::LogFabric,
                    "Topology solver: {} constraint node(s) for target node {} are not present in the global graph. "
                    "These nodes will be ignored. Missing nodes: {}",
                    missing_nodes.size(),
                    i,
                    missing_nodes_str.str());

                // Warn if all constraint nodes are missing (empty restricted_indices)
                if (restricted_indices.empty()) {
                    log_warning(
                        tt::LogFabric,
                        "Topology solver: All constraint nodes for target node {} are missing from the global graph. "
                        "This target node will have no restrictions.",
                        i);
                }
            }

            restricted_global_indices[i] = std::move(restricted_indices);
        }

        // Process preferred constraints
        auto preferred_it = preferred_mappings.find(target_node);
        if (preferred_it != preferred_mappings.end() && !preferred_it->second.empty()) {
            // Convert GlobalNode set to index vector
            std::vector<size_t> preferred_indices;
            preferred_indices.reserve(preferred_it->second.size());
            std::vector<GlobalNode> missing_nodes;
            for (const auto& global_node : preferred_it->second) {
                auto idx_it = graph_data.global_to_idx.find(global_node);
                if (idx_it != graph_data.global_to_idx.end()) {
                    preferred_indices.push_back(idx_it->second);
                } else {
                    missing_nodes.push_back(global_node);
                }
            }
            std::sort(preferred_indices.begin(), preferred_indices.end());

            // Log warning if preferred constraint nodes are missing from the graph
            if (!missing_nodes.empty()) {
                std::stringstream missing_nodes_str;
                bool first = true;
                for (const auto& node : missing_nodes) {
                    if (!first) {
                        missing_nodes_str << ", ";
                    }
                    first = false;
                    missing_nodes_str << node;
                }

                log_warning(
                    tt::LogFabric,
                    "Topology solver: {} preferred constraint node(s) for target node {} are not present in the global "
                    "graph. These nodes will be ignored. Missing nodes: {}",
                    missing_nodes.size(),
                    i,
                    missing_nodes_str.str());
            }

            preferred_global_indices[i] = std::move(preferred_indices);
        }
    }
}

// ============================================================================
// SearchHeuristic Implementation
// ============================================================================

template <typename TargetNode, typename GlobalNode>
bool SearchHeuristic::check_hard_constraints(
    size_t target_idx,
    size_t global_idx,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    const std::vector<int>& mapping,
    ConnectionValidationMode validation_mode) {
    // 1. Check required constraints (fast index-based lookup)
    if (!constraint_data.is_valid_mapping(target_idx, global_idx)) {
        return false;
    }

    // 2. Check edges to mapped neighbors
    for (size_t neighbor : graph_data.target_adj_idx[target_idx]) {
        int mapped_global = mapping[neighbor];
        if (mapped_global != -1) {
            size_t mapped_global_idx = static_cast<size_t>(mapped_global);
            // Check if edge exists
            bool edge_exists = std::binary_search(
                graph_data.global_adj_idx[global_idx].begin(),
                graph_data.global_adj_idx[global_idx].end(),
                mapped_global_idx);
            if (!edge_exists) {
                return false;
            }
        }
    }

    // 3. Check degree
    if (graph_data.global_deg[global_idx] < graph_data.target_deg[target_idx]) {
        return false;
    }

    // 4. Check channel counts (strict mode)
    if (validation_mode == ConnectionValidationMode::STRICT) {
        for (size_t neighbor : graph_data.target_adj_idx[target_idx]) {
            int mapped_global = mapping[neighbor];
            if (mapped_global != -1) {
                size_t mapped_global_idx = static_cast<size_t>(mapped_global);
                size_t required = graph_data.target_conn_count[target_idx].at(neighbor);
                auto it = graph_data.global_conn_count[global_idx].find(mapped_global_idx);
                if (it == graph_data.global_conn_count[global_idx].end() || it->second < required) {
                    return false;
                }
            }
        }
    }

    return true;
}

template <typename TargetNode, typename GlobalNode>
int SearchHeuristic::compute_candidate_cost(
    size_t target_idx,
    size_t global_idx,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    const std::vector<int>& mapping,
    ConnectionValidationMode validation_mode) {
    // Cost formula (lower cost = better candidate):
    // cost = -is_preferred * SOFT_WEIGHT
    //      - channel_match_score (in relaxed mode only)
    //      + degree_gap * RUNTIME_WEIGHT
    //
    // Where:
    // - is_preferred: whether this candidate satisfies preferred constraints
    // - channel_match_score: bonus for exact channel matches, penalty for mismatches (relaxed mode)
    // - degree_gap: difference between global and target node degrees (runtime optimization)

    // Check if preferred (fast index-based lookup)
    bool is_preferred = false;
    if (target_idx < constraint_data.preferred_global_indices.size()) {
        const auto& preferred = constraint_data.preferred_global_indices[target_idx];
        is_preferred = std::binary_search(preferred.begin(), preferred.end(), global_idx);
    }

    // Compute channel match score (relaxed mode only)
    // Prefer connections closer to required count (exact match = best, then closest above, then closest below)
    int channel_match_score = 0;
    if (validation_mode == ConnectionValidationMode::RELAXED) {
        for (size_t neighbor : graph_data.target_adj_idx[target_idx]) {
            int mapped_global = mapping[neighbor];
            if (mapped_global != -1) {
                size_t mapped_global_idx = static_cast<size_t>(mapped_global);
                size_t required = graph_data.target_conn_count[target_idx].at(neighbor);
                auto it = graph_data.global_conn_count[global_idx].find(mapped_global_idx);
                if (it != graph_data.global_conn_count[global_idx].end()) {
                    size_t actual = it->second;
                    if (actual >= required) {
                        // Perfect match or close above: prefer exact match, then closest above
                        size_t gap = actual - required;
                        // Score: higher for exact match (gap=0), lower for larger gaps
                        // Use SOFT_WEIGHT for exact match, SOFT_WEIGHT/2 for gap=1, SOFT_WEIGHT/4 for gap=2, etc.
                        float raw_bonus = static_cast<float>(SOFT_WEIGHT) / (1.0f + static_cast<float>(gap));
                        int match_bonus = static_cast<int>(std::max(1.0f, raw_bonus));  // Ensure minimum of 1 for large gaps
                        channel_match_score += match_bonus;
                    } else {
                        // Below required: still allow but penalize based on how far below
                        // Penalty increases with gap (but less than perfect match bonus)
                        size_t gap = required - actual;
                        float raw_penalty = static_cast<float>(SOFT_WEIGHT) / (10.0f + static_cast<float>(gap));
                        int penalty = static_cast<int>(std::max(1.0f, raw_penalty));  // Small penalty, decreases with gap, minimum of 1
                        channel_match_score -= penalty;
                    }
                }
            }
        }
    }

    // Compute degree gap (runtime optimization)
    size_t target_deg = graph_data.target_deg[target_idx];
    size_t global_deg = graph_data.global_deg[global_idx];
    int degree_gap_cost;
    if (global_deg >= target_deg) {
        // Clamp to INT_MAX to avoid overflow when casting size_t to int
        size_t degree_gap = global_deg - target_deg;
        degree_gap_cost = static_cast<int>(std::min(degree_gap, static_cast<size_t>(INT_MAX)));
    } else {
        // Shouldn't happen if check_hard_constraints was called, but handle gracefully
        degree_gap_cost = INT_MAX;
    }

    // Cost = -is_preferred * SOFT_WEIGHT
    //      - channel_match_score
    //      + degree_gap * RUNTIME_WEIGHT
    // Lower cost = better candidate
    int preferred_cost;
    if (is_preferred) {
        preferred_cost = SOFT_WEIGHT;
    } else {
        preferred_cost = 0;
    }
    return static_cast<int>(-preferred_cost - channel_match_score +
                            degree_gap_cost);
}

template <typename TargetNode, typename GlobalNode>
std::vector<size_t> SearchHeuristic::generate_ordered_candidates(
    size_t target_idx,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    const std::vector<int>& mapping,
    const std::vector<bool>& used,
    ConnectionValidationMode validation_mode) {
    std::vector<std::pair<size_t, int>> candidates_with_cost;

    // Generate candidates and filter by hard constraints
    for (size_t j = 0; j < graph_data.n_global; ++j) {
        if (used[j]) {
            continue;
        }

        // Check hard constraints - filter out invalid candidates
        if (!check_hard_constraints(target_idx, j, graph_data, constraint_data, mapping, validation_mode)) {
            continue;
        }

        // Valid candidate - compute cost
        int cost = compute_candidate_cost(target_idx, j, graph_data, constraint_data, mapping, validation_mode);
        candidates_with_cost.push_back({j, cost});
    }

    // Sort by cost (lower = better)
    std::sort(candidates_with_cost.begin(), candidates_with_cost.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Extract ordered candidate indices
    std::vector<size_t> candidates;
    candidates.reserve(candidates_with_cost.size());
    for (const auto& [idx, cost] : candidates_with_cost) {
        candidates.push_back(idx);
    }

    return candidates;
}

template <typename TargetNode, typename GlobalNode>
int SearchHeuristic::compute_node_cost(
    size_t target_idx,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    const std::vector<int>& mapping,
    const std::vector<bool>& used,
    ConnectionValidationMode validation_mode) {
    // Count valid candidates (hard constraint filtering)
    size_t candidate_count = 0;
    size_t preferred_count = 0;

    for (size_t j = 0; j < graph_data.n_global; ++j) {
        if (used[j]) {
            continue;
        }

        // Check hard constraints (in relaxed mode, we still filter by basic constraints but allow channel mismatches)
        if (check_hard_constraints(target_idx, j, graph_data, constraint_data, mapping, validation_mode)) {
            candidate_count++;

            // Check if preferred (fast index-based lookup)
            if (target_idx < constraint_data.preferred_global_indices.size()) {
                const auto& preferred = constraint_data.preferred_global_indices[target_idx];
                if (std::binary_search(preferred.begin(), preferred.end(), j)) {
                    preferred_count++;
                }
            }
        }
    }

    // Count mapped neighbors
    size_t mapped_neighbors = 0;
    for (size_t neighbor : graph_data.target_adj_idx[target_idx]) {
        if (mapping[neighbor] != -1) {
            mapped_neighbors++;
        }
    }

    // Cost = (candidate_count * HARD_WEIGHT)
    //      - (preferred_count * SOFT_WEIGHT)
    //      - (mapped_neighbors * RUNTIME_WEIGHT)
    // Lower cost = more constrained = higher priority
    return static_cast<int>(candidate_count * HARD_WEIGHT - preferred_count * SOFT_WEIGHT -
                            mapped_neighbors * RUNTIME_WEIGHT);
}

template <typename TargetNode, typename GlobalNode>
SearchHeuristic::SelectionResult SearchHeuristic::select_and_generate_candidates(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    const std::vector<int>& mapping,
    const std::vector<bool>& used,
    ConnectionValidationMode validation_mode) {
    // Find unassigned target node with lowest cost (most constrained)
    // Break ties deterministically by selecting the node with the lowest index
    size_t best_target = SIZE_MAX;
    int best_cost = INT_MAX;

    for (size_t i = 0; i < graph_data.n_target; ++i) {
        if (mapping[i] != -1) {
            continue;  // Already assigned
        }

        int cost = compute_node_cost(i, graph_data, constraint_data, mapping, used, validation_mode);
        if (cost < best_cost || (cost == best_cost && i < best_target)) {
            best_cost = cost;
            best_target = i;
        }
    }

    // If all target nodes are already assigned, no candidates can be generated.
    if (best_target == SIZE_MAX) {
        return {best_target, {}};
    }

    // Generate ordered candidates for selected node
    std::vector<size_t> candidates =
        generate_ordered_candidates(best_target, graph_data, constraint_data, mapping, used, validation_mode);

    return {best_target, candidates};
}

// ============================================================================
// ConsistencyChecker Implementation
// ============================================================================

template <typename TargetNode, typename GlobalNode>
bool ConsistencyChecker::check_local_consistency(
    size_t target_idx,
    size_t global_idx,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const std::vector<int>& mapping,
    ConnectionValidationMode validation_mode) {
    // Check all neighbors of target_idx that are already mapped
    for (size_t neighbor : graph_data.target_adj_idx[target_idx]) {
        int mapped_global = mapping[neighbor];
        if (mapped_global == -1) {
            continue;  // Neighbor not yet mapped, skip
        }

        size_t mapped_global_idx = static_cast<size_t>(mapped_global);

        // Check if edge exists between global_idx and mapped_global_idx
        bool edge_exists = std::binary_search(
            graph_data.global_adj_idx[global_idx].begin(),
            graph_data.global_adj_idx[global_idx].end(),
            mapped_global_idx);

        if (!edge_exists) {
            return false;  // Edge doesn't exist, inconsistent
        }

        // In STRICT mode, also check channel counts
        if (validation_mode == ConnectionValidationMode::STRICT) {
            size_t required = graph_data.target_conn_count[target_idx].at(neighbor);
            auto it = graph_data.global_conn_count[global_idx].find(mapped_global_idx);
            if (it == graph_data.global_conn_count[global_idx].end() || it->second < required) {
                return false;  // Insufficient channels in strict mode
            }
        }
    }

    return true;  // All checks passed
}

template <typename TargetNode, typename GlobalNode>
bool ConsistencyChecker::check_forward_consistency(
    size_t target_idx,
    size_t global_idx,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    std::vector<int>& mapping,
    const std::vector<bool>& used,
    ConnectionValidationMode validation_mode) {
    // For each unassigned neighbor of target_idx, check if there's at least one viable candidate
    for (size_t neighbor : graph_data.target_adj_idx[target_idx]) {
        if (mapping[neighbor] != -1) {
            continue;  // Already assigned, skip
        }

        // Check if neighbor has at least one viable candidate among unused neighbors of global_idx
        bool has_viable_candidate = false;

        for (size_t candidate_global : graph_data.global_adj_idx[global_idx]) {
            if (used[candidate_global]) {
                continue;  // Already used, skip
            }

            // Check if candidate satisfies hard constraints for neighbor
            if (!SearchHeuristic::check_hard_constraints(
                    neighbor, candidate_global, graph_data, constraint_data, mapping, validation_mode)) {
                continue;  // Doesn't satisfy hard constraints
            }

            // Check local consistency for neighbor -> candidate_global
            // Modify mapping in-place, check, then restore to avoid O(n) copy per candidate
            int old_mapping_value = mapping[neighbor];
            mapping[neighbor] = static_cast<int>(candidate_global);

            if (ConsistencyChecker::check_local_consistency(neighbor, candidate_global, graph_data, mapping, validation_mode)) {
                mapping[neighbor] = old_mapping_value;  // Restore before returning
                has_viable_candidate = true;
                break;  // Found at least one viable candidate
            }

            mapping[neighbor] = old_mapping_value;  // Restore before trying next candidate
        }

        if (!has_viable_candidate) {
            return false;  // Neighbor has no viable candidates, forward check fails
        }
    }

    return true;  // All unassigned neighbors have viable candidates
}

template <typename TargetNode, typename GlobalNode>
size_t ConsistencyChecker::count_reachable_unused(
    size_t start_global_idx,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const std::vector<bool>& used) {
    // BFS to count all unused nodes reachable from start_global_idx
    std::vector<bool> visited(graph_data.n_global, false);
    std::vector<size_t> queue;
    queue.push_back(start_global_idx);
    visited[start_global_idx] = true;

    size_t count = 0;

    while (!queue.empty()) {
        size_t current = queue.back();
        queue.pop_back();

        // Count if unused
        if (!used[current]) {
            count++;
        }

        // Add neighbors to queue
        for (size_t neighbor : graph_data.global_adj_idx[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    return count;
}

}  // namespace tt::tt_fabric::detail

#endif  // TOPOLOGY_SOLVER_INTERNAL_TPP
