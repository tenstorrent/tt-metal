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
    const AdjacencyGraph<TargetNode>& target_graph, const AdjacencyGraph<GlobalNode>& global_graph) {

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

    n_target = target_nodes.size();
    n_global = global_nodes.size();

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
void GraphIndexData<TargetNode, GlobalNode>::print_node_degrees() const {
    std::stringstream ss;
    ss << "\n=== Node Degrees ===" << std::endl;
    ss << "Target graph (" << n_target << " nodes):" << std::endl;
    for (size_t i = 0; i < n_target; ++i) {
        ss << "  Node " << target_nodes[i] << ": degree " << target_deg[i] << std::endl;
    }
    ss << "Global graph (" << n_global << " nodes):" << std::endl;
    for (size_t i = 0; i < n_global; ++i) {
        ss << "  Node " << global_nodes[i] << ": degree " << global_deg[i] << std::endl;
    }
    ss << "====================" << std::endl;
    log_info(tt::LogFabric, "{}", ss.str());
}

template <typename TargetNode, typename GlobalNode>
void GraphIndexData<TargetNode, GlobalNode>::print_adjacency_maps() const {
    std::stringstream ss;
    ss << "\n=== Target Graph Adjacency Map ===" << std::endl;
    ss << "Total nodes: " << n_target << std::endl;
    for (size_t i = 0; i < n_target; ++i) {
        ss << "  Node " << target_nodes[i] << " (degree " << target_deg[i] << "): ";
        if (target_adj_idx[i].empty()) {
            ss << "no neighbors";
        } else {
            bool first = true;
            for (size_t neighbor_idx : target_adj_idx[i]) {
                if (!first) {
                    ss << ", ";
                }
                first = false;
                ss << target_nodes[neighbor_idx];
                // Show connection count if multi-edge
                auto it = target_conn_count[i].find(neighbor_idx);
                if (it != target_conn_count[i].end() && it->second > 1) {
                    ss << "(" << it->second << " channels)";
                }
            }
        }
        ss << std::endl;
    }
    ss << "\n=== Global Graph Adjacency Map ===" << std::endl;
    ss << "Total nodes: " << n_global << std::endl;
    for (size_t i = 0; i < n_global; ++i) {
        ss << "  Node " << global_nodes[i] << " (degree " << global_deg[i] << "): ";
        if (global_adj_idx[i].empty()) {
            ss << "no neighbors";
        } else {
            bool first = true;
            for (size_t neighbor_idx : global_adj_idx[i]) {
                if (!first) {
                    ss << ", ";
                }
                first = false;
                ss << global_nodes[neighbor_idx];
                // Show connection count if multi-edge
                auto it = global_conn_count[i].find(neighbor_idx);
                if (it != global_conn_count[i].end() && it->second > 1) {
                    ss << "(" << it->second << " channels)";
                }
            }
        }
        ss << std::endl;
    }
    ss << "===================================" << std::endl;
    log_info(tt::LogFabric, "{}", ss.str());
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
            for (const auto& global_node : valid_it->second) {
                auto idx_it = graph_data.global_to_idx.find(global_node);
                if (idx_it != graph_data.global_to_idx.end()) {
                    restricted_indices.push_back(idx_it->second);
                }
            }
            std::sort(restricted_indices.begin(), restricted_indices.end());
            restricted_global_indices[i] = std::move(restricted_indices);
        }

        // Process preferred constraints
        auto preferred_it = preferred_mappings.find(target_node);
        if (preferred_it != preferred_mappings.end() && !preferred_it->second.empty()) {
            // Convert GlobalNode set to index vector
            std::vector<size_t> preferred_indices;
            preferred_indices.reserve(preferred_it->second.size());
            for (const auto& global_node : preferred_it->second) {
                auto idx_it = graph_data.global_to_idx.find(global_node);
                if (idx_it != graph_data.global_to_idx.end()) {
                    preferred_indices.push_back(idx_it->second);
                }
            }
            std::sort(preferred_indices.begin(), preferred_indices.end());
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
    int cost = 0;

    // Soft constraint factors (subtract from cost = lower cost = better)
    // Preferred constraint (fast index-based lookup)
    bool is_preferred = false;
    if (target_idx < constraint_data.preferred_global_indices.size()) {
        const auto& preferred = constraint_data.preferred_global_indices[target_idx];
        is_preferred = std::binary_search(preferred.begin(), preferred.end(), global_idx);
    }
    if (is_preferred) {
        cost -= SOFT_WEIGHT;
    }

    // Channel count matching (relaxed mode)
    // Prefer connections closer to required count (exact match = best, then closest above, then closest below)
    if (validation_mode == ConnectionValidationMode::RELAXED) {
        int channel_match_score = 0;
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
                        int match_bonus = static_cast<int>(SOFT_WEIGHT / (1 + gap));
                        channel_match_score += match_bonus;
                    } else {
                        // Below required: still allow but penalize based on how far below
                        // Penalty increases with gap (but less than perfect match bonus)
                        size_t gap = required - actual;
                        int penalty = static_cast<int>(SOFT_WEIGHT / (10 + gap));  // Small penalty, decreases with gap
                        channel_match_score -= penalty;
                    }
                }
            }
        }
        cost -= channel_match_score;  // Subtract score (negative cost = better)
    }

    // Runtime optimization (add to cost = higher cost = worse)
    // Note: global_deg >= target_deg is guaranteed by check_hard_constraints
    size_t target_deg = graph_data.target_deg[target_idx];
    size_t global_deg = graph_data.global_deg[global_idx];
    size_t degree_gap = global_deg - target_deg;
    cost += static_cast<int>(degree_gap * RUNTIME_WEIGHT);

    return cost;
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
    size_t best_target = SIZE_MAX;
    int best_cost = INT_MAX;

    for (size_t i = 0; i < graph_data.n_target; ++i) {
        if (mapping[i] != -1) {
            continue;  // Already assigned
        }

        int cost = compute_node_cost(i, graph_data, constraint_data, mapping, used, validation_mode);
        if (cost < best_cost) {
            best_cost = cost;
            best_target = i;
        }
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
    const std::vector<int>& mapping,
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
            // Create temporary mapping to check
            std::vector<int> temp_mapping = mapping;
            temp_mapping[neighbor] = static_cast<int>(candidate_global);

            if (ConsistencyChecker::check_local_consistency(neighbor, candidate_global, graph_data, temp_mapping, validation_mode)) {
                has_viable_candidate = true;
                break;  // Found at least one viable candidate
            }
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

// ============================================================================
// DFSSearchEngine Implementation
// ============================================================================

template <typename TargetNode, typename GlobalNode>
uint64_t DFSSearchEngine<TargetNode, GlobalNode>::hash_state(const std::vector<int>& mapping) const {
    // FNV-1a hash function
    const uint64_t fnv_offset = 1469598103934665603ull;
    const uint64_t fnv_prime = 1099511628211ull;
    uint64_t h = fnv_offset;

    for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] != -1) {
            // Hash (target_idx, global_idx) pair
            uint64_t pairh = (static_cast<uint64_t>(i) << 32) ^ static_cast<uint64_t>(mapping[i] + 1);
            h ^= pairh;
            h *= fnv_prime;
        }
    }

    return h;
}

template <typename TargetNode, typename GlobalNode>
bool DFSSearchEngine<TargetNode, GlobalNode>::dfs_recursive(
    size_t pos,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    ConnectionValidationMode validation_mode) {
    // Base case: all target nodes assigned
    if (pos >= graph_data.n_target) {
        return true;
    }

    // Periodic progress logging (similar to topology_mapper_utils.cpp)
    state_.dfs_calls++;
    if ((state_.dfs_calls & ((1u << 18) - 1)) == 0) {
        size_t assigned = 0;
        for (auto v : state_.mapping) {
            assigned += (v != -1);
        }
        log_info(
            tt::LogFabric,
            "TopologySolver DFS progress: calls={}, assigned={}/{}, failed_states={}",
            state_.dfs_calls,
            assigned,
            graph_data.n_target,
            state_.failed_states.size());
    }

    // Check memoization cache
    uint64_t state_hash = hash_state(state_.mapping);
    if (state_.failed_states.find(state_hash) != state_.failed_states.end()) {
        return false;
    }

    // Select next target node and generate ordered candidates
    auto selection = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, state_.mapping, state_.used, validation_mode);

    // Check if no unassigned nodes found (shouldn't happen if pos < n_target, but handle gracefully)
    if (selection.target_idx == SIZE_MAX || selection.candidates.empty()) {
        // No valid candidates - explain why mapping failed
        if (selection.target_idx != SIZE_MAX) {
            // Found a target node but no valid candidates in global graph
            size_t remaining_targets = graph_data.n_target - pos;
            size_t remaining_global = graph_data.n_global;
            for (size_t i = 0; i < graph_data.n_global; ++i) {
                if (state_.used[i]) {
                    remaining_global--;
                }
            }
            std::string error_msg = fmt::format(
                "Cannot place target node {} in global graph: no valid candidates found. "
                "Remaining: {} target nodes to place, {} unused nodes in global graph",
                graph_data.target_nodes[selection.target_idx],
                remaining_targets,
                remaining_global);
            log_error(tt::LogFabric, "{}", error_msg);
            if (state_.error_message.empty()) {
                state_.error_message = error_msg;
            }
        } else {
            // No unassigned target nodes found (shouldn't happen)
            std::string error_msg = fmt::format(
                "Search error: no unassigned target nodes found, but {} nodes still need to be placed",
                graph_data.n_target - pos);
            log_error(tt::LogFabric, "{}", error_msg);
            if (state_.error_message.empty()) {
                state_.error_message = error_msg;
            }
        }
        state_.failed_states.insert(state_hash);
        return false;
    }

    size_t target_idx = selection.target_idx;

    // Try each candidate in order (best first)
    for (size_t global_idx : selection.candidates) {
        // Check local consistency (edges to already-mapped neighbors)
        if (!ConsistencyChecker::check_local_consistency(
                target_idx, global_idx, graph_data, state_.mapping, validation_mode)) {
            continue;  // Skip invalid candidate
        }

        // Check forward consistency (future neighbors have viable options)
        if (!ConsistencyChecker::check_forward_consistency(
                target_idx, global_idx, graph_data, constraint_data, state_.mapping, state_.used, validation_mode)) {
            continue;  // Skip candidate that leaves no options
        }

        // Assign candidate
        state_.mapping[target_idx] = static_cast<int>(global_idx);
        state_.used[global_idx] = true;

        // Recursive call
        if (dfs_recursive(pos + 1, graph_data, constraint_data, validation_mode)) {
            return true;  // Success!
        }

        // Backtrack
        state_.mapping[target_idx] = -1;
        state_.used[global_idx] = false;
        state_.backtrack_count++;
    }

    // All candidates failed - mark state as failed
    state_.failed_states.insert(state_hash);
    return false;
}

template <typename TargetNode, typename GlobalNode>
bool DFSSearchEngine<TargetNode, GlobalNode>::search(
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode validation_mode) {
    // Reset internal state
    state_ = SearchState();
    state_.mapping.resize(graph_data.n_target, -1);
    state_.used.resize(graph_data.n_global, false);

    // Log node degrees and degree histograms at the start of mapping
    // Build degree histograms for more descriptive logging
    auto build_degree_histogram = [](const std::vector<size_t>& degrees) -> std::string {
        std::map<size_t, size_t> hist;
        for (auto d : degrees) {
            hist[d]++;
        }
        std::string s = "{";
        bool first = true;
        for (const auto& [degree, count] : hist) {
            if (!first) {
                s += ", ";
            }
            first = false;
            s += std::to_string(degree) + ":" + std::to_string(count);
        }
        s += "}";
        return s;
    };

    std::string target_deg_hist = build_degree_histogram(graph_data.target_deg);
    std::string global_deg_hist = build_degree_histogram(graph_data.global_deg);

    log_info(
        tt::LogFabric,
        "Topology mapping search starting: target_graph_nodes={}, global_graph_nodes={}, "
        "target_degree_histogram={}, global_degree_histogram={}",
        graph_data.n_target,
        graph_data.n_global,
        target_deg_hist,
        global_deg_hist);

    // Check if global graph has enough nodes
    if (graph_data.n_global < graph_data.n_target) {
        std::string error_msg = fmt::format(
            "Cannot map target graph to global graph: target graph has {} nodes, but global graph only has {} nodes",
            graph_data.n_target,
            graph_data.n_global);
        log_error(tt::LogFabric, "{}", error_msg);
        state_.error_message = error_msg;
        return false;
    }

    // Pre-assign nodes from required constraints (pinnings)
    size_t assigned_count = 0;
    const auto& valid_mappings = constraints.get_valid_mappings();
    for (size_t i = 0; i < graph_data.n_target; ++i) {
        const auto& target_node = graph_data.target_nodes[i];
        auto it = valid_mappings.find(target_node);
        if (it != valid_mappings.end() && it->second.size() == 1) {
            // This target node has exactly one required constraint (pinning)
            const GlobalNode& required_global = *it->second.begin();
            auto global_it = graph_data.global_to_idx.find(required_global);
            if (global_it != graph_data.global_to_idx.end()) {
                size_t global_idx = global_it->second;

                // Validate that this pre-assignment is consistent with already-assigned neighbors
                for (size_t neighbor : graph_data.target_adj_idx[i]) {
                    if (state_.mapping[neighbor] != -1) {
                        size_t neighbor_global_idx = static_cast<size_t>(state_.mapping[neighbor]);
                        // Check if edge exists between global_idx and neighbor_global_idx
                        bool edge_exists = std::binary_search(
                            graph_data.global_adj_idx[global_idx].begin(),
                            graph_data.global_adj_idx[global_idx].end(),
                            neighbor_global_idx);
                        if (!edge_exists) {
                            std::string error_msg = fmt::format(
                                "Pre-assignment conflict: target node {} must map to global node {} (required constraint), "
                                "but target node {} (adjacent to {}) is already mapped to global node {}, "
                                "and global nodes {} and {} are not adjacent. This violates graph isomorphism requirements.",
                                target_node,
                                required_global,
                                graph_data.target_nodes[neighbor],
                                target_node,
                                graph_data.global_nodes[neighbor_global_idx],
                                required_global,
                                graph_data.global_nodes[neighbor_global_idx]);
                            log_error(tt::LogFabric, "{}", error_msg);
                            state_.error_message = error_msg;
                            return false;
                        }
                    }
                }

                state_.mapping[i] = static_cast<int>(global_idx);
                state_.used[global_idx] = true;
                assigned_count++;
            }
        }
    }

    // Check if enough unused nodes remain in global graph
    size_t unused_count = 0;
    for (size_t i = 0; i < graph_data.n_global; ++i) {
        if (!state_.used[i]) {
            unused_count++;
        }
    }
    size_t remaining_targets = graph_data.n_target - assigned_count;
    if (unused_count < remaining_targets) {
        std::string error_msg = fmt::format(
            "Cannot complete mapping: {} target nodes still need to be placed, but only {} unused nodes remain in global graph",
            remaining_targets,
            unused_count);
        log_error(tt::LogFabric, "{}", error_msg);
        state_.error_message = error_msg;
        return false;
    }

    // Start DFS from current position
    bool found = dfs_recursive(assigned_count, graph_data, constraint_data, validation_mode);

    if (!found && assigned_count == 0) {
        // Search failed from the beginning - provide summary
        std::string error_msg = fmt::format(
            "Failed to find mapping: target graph with {} nodes cannot be placed in global graph with {} nodes under given constraints",
            graph_data.n_target,
            graph_data.n_global);
        log_error(tt::LogFabric, "{}", error_msg);
        if (state_.error_message.empty()) {
            state_.error_message = error_msg;
        }
    }

    return found;
}

// ============================================================================
// MappingValidator Implementation
// ============================================================================

template <typename TargetNode, typename GlobalNode>
void MappingValidator<TargetNode, GlobalNode>::validate_connection_counts(
    const std::vector<int>& mapping,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    ConnectionValidationMode validation_mode,
    std::vector<std::string>* warnings) {
    // Check all edges in the mapping
    for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] == -1) {
            continue;  // Not mapped
        }

        size_t global_idx = static_cast<size_t>(mapping[i]);
        const TargetNode& target_node = graph_data.target_nodes[i];
        const GlobalNode& global_node = graph_data.global_nodes[global_idx];

        // Check all neighbors of this target node
        for (size_t neighbor : graph_data.target_adj_idx[i]) {
            if (mapping[neighbor] == -1) {
                continue;  // Neighbor not mapped
            }

            size_t neighbor_global_idx = static_cast<size_t>(mapping[neighbor]);
            const TargetNode& neighbor_target = graph_data.target_nodes[neighbor];
            const GlobalNode& neighbor_global = graph_data.global_nodes[neighbor_global_idx];

            // Get required channel count
            size_t required = graph_data.target_conn_count[i].at(neighbor);

            // Get actual channel count
            auto it = graph_data.global_conn_count[global_idx].find(neighbor_global_idx);
            if (it == graph_data.global_conn_count[global_idx].end()) {
                // Edge doesn't exist - this should have been caught earlier, but handle gracefully
                if (validation_mode == ConnectionValidationMode::STRICT) {
                    std::string error_msg = fmt::format(
                        "Strict mode validation failed: target graph edge from node {} to {} exists, "
                        "but physical edge from {} to {} does not exist in global graph. "
                        "This indicates a mapping inconsistency.",
                        target_node,
                        neighbor_target,
                        global_node,
                        neighbor_global);
                    log_error(tt::LogFabric, "{}", error_msg);
                    warnings->push_back(error_msg);
                }
                continue;
            }

            size_t actual = it->second;

            // Check if sufficient
            if (actual < required) {
                if (validation_mode == ConnectionValidationMode::STRICT) {
                    std::string error_msg = fmt::format(
                        "Strict mode validation failed: target graph edge from node {} to {} requires {} channels, "
                        "but physical edge from {} to {} only has {} channels. "
                        "Strict mode requires sufficient channel capacity for all edges.",
                        target_node,
                        neighbor_target,
                        required,
                        global_node,
                        neighbor_global,
                        actual);
                    log_error(tt::LogFabric, "{}", error_msg);
                    warnings->push_back(error_msg);
                } else if (validation_mode == ConnectionValidationMode::RELAXED) {
                    std::string warning_msg = fmt::format(
                        "Relaxed mode: target graph edge from node {} to {} requires {} channels, "
                        "but physical edge from {} to {} only has {} channels. "
                        "Mapping will proceed but may have insufficient bandwidth.",
                        target_node,
                        neighbor_target,
                        required,
                        global_node,
                        neighbor_global,
                        actual);
                    log_info(tt::LogFabric, "{}", warning_msg);
                    warnings->push_back(warning_msg);
                }
            }
        }
    }
}

template <typename TargetNode, typename GlobalNode>
bool MappingValidator<TargetNode, GlobalNode>::validate_mapping(
    const std::vector<int>& mapping,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    ConnectionValidationMode validation_mode,
    std::vector<std::string>* warnings) {
    // First, validate that all target nodes are mapped
    std::vector<size_t> unmapped_targets;
    for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] == -1) {
            unmapped_targets.push_back(i);
        }
    }

    if (!unmapped_targets.empty()) {
        std::string unmapped_list;
        for (size_t idx : unmapped_targets) {
            if (!unmapped_list.empty()) {
                unmapped_list += ", ";
            }
            unmapped_list += fmt::format("{}", graph_data.target_nodes[idx]);
        }
        std::string error_msg = fmt::format(
            "Mapping validation failed: {} target node(s) are not mapped to any global node: {}",
            unmapped_targets.size(),
            unmapped_list);
        log_error(tt::LogFabric, "{}", error_msg);
        if (warnings != nullptr) {
            warnings->push_back(error_msg);
        }
        return false;
    }

    // Validate that all edges exist in the global graph
    for (size_t i = 0; i < mapping.size(); ++i) {
        size_t global_idx = static_cast<size_t>(mapping[i]);
        const TargetNode& target_node = graph_data.target_nodes[i];
        const GlobalNode& global_node = graph_data.global_nodes[global_idx];

        // Check all neighbors
        for (size_t neighbor : graph_data.target_adj_idx[i]) {
            size_t neighbor_global_idx = static_cast<size_t>(mapping[neighbor]);
            const TargetNode& neighbor_target = graph_data.target_nodes[neighbor];
            const GlobalNode& neighbor_global = graph_data.global_nodes[neighbor_global_idx];

            // Check if edge exists in global graph
            bool edge_exists = std::binary_search(
                graph_data.global_adj_idx[global_idx].begin(),
                graph_data.global_adj_idx[global_idx].end(),
                neighbor_global_idx);

            if (!edge_exists) {
                std::string error_msg = fmt::format(
                    "Mapping validation failed: target graph has edge from node {} to {}, "
                    "but global graph does not have corresponding edge from {} to {}. "
                    "This indicates the mapping violates graph isomorphism requirements.",
                    target_node,
                    neighbor_target,
                    global_node,
                    neighbor_global);
                log_error(tt::LogFabric, "{}", error_msg);
                if (warnings != nullptr) {
                    warnings->push_back(error_msg);
                }
                return false;
            }
        }
    }

    // Validate connection counts (collects warnings/errors)
    if (warnings != nullptr) {
        validate_connection_counts(mapping, graph_data, validation_mode, warnings);
    }

    // In STRICT mode, fail if any warnings were added (they're actually errors)
    if (validation_mode == ConnectionValidationMode::STRICT && warnings != nullptr && !warnings->empty()) {
        log_error(
            tt::LogFabric,
            "Mapping validation failed in strict mode: {} validation error(s) found",
            warnings->size());
        return false;
    }

    if (validation_mode == ConnectionValidationMode::RELAXED && warnings != nullptr && !warnings->empty()) {
        log_info(
            tt::LogFabric,
            "Mapping validation completed in relaxed mode: {} warning(s) about channel count mismatches",
            warnings->size());
    }

    return true;
}

template <typename TargetNode, typename GlobalNode>
void MappingValidator<TargetNode, GlobalNode>::print_mapping(
    const std::vector<int>& mapping,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data) {
    std::stringstream ss;
    ss << "\n=== Current Mapping ===" << std::endl;
    size_t mapped_count = 0;
    for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] != -1) {
            size_t global_idx = static_cast<size_t>(mapping[i]);
            ss << "  Target node " << graph_data.target_nodes[i] << " -> Global node "
               << graph_data.global_nodes[global_idx] << std::endl;
            mapped_count++;
        } else {
            ss << "  Target node " << graph_data.target_nodes[i] << " -> UNMAPPED" << std::endl;
        }
    }
    ss << "Total mapped: " << mapped_count << " of " << mapping.size() << " target nodes" << std::endl;
    ss << "========================" << std::endl;
    log_info(tt::LogFabric, "{}", ss.str());
}

template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> MappingValidator<TargetNode, GlobalNode>::build_result(
    const std::vector<int>& mapping,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const DFSSearchEngine<TargetNode, GlobalNode>::SearchState& state,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode validation_mode) {
    MappingResult<TargetNode, GlobalNode> result;

    // Always build bidirectional mappings, even if validation fails
    // This allows users to see the closest/best mapping found for debugging
    size_t mapped_count = 0;
    for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] != -1) {
            size_t global_idx = static_cast<size_t>(mapping[i]);
            const TargetNode& target_node = graph_data.target_nodes[i];
            const GlobalNode& global_node = graph_data.global_nodes[global_idx];

            result.target_to_global[target_node] = global_node;
            result.global_to_target[global_node] = target_node;
            mapped_count++;
        }
    }

    // Validate mapping and collect detailed error messages
    std::vector<std::string> validation_warnings;
    bool valid = validate_mapping(mapping, graph_data, validation_mode, &validation_warnings);

    if (!valid) {
        result.success = false;
        if (!validation_warnings.empty()) {
            // Use first validation error as main error message
            result.error_message = validation_warnings[0];
            log_error(
                tt::LogFabric,
                "Mapping validation failed: {} validation error(s) detected. First error: {}. "
                "Saving partial mapping: {} of {} target nodes mapped",
                validation_warnings.size(),
                validation_warnings[0],
                mapped_count,
                graph_data.n_target);
            // Add remaining as warnings
            for (size_t i = 1; i < validation_warnings.size(); ++i) {
                result.warnings.push_back(validation_warnings[i]);
                log_error(tt::LogFabric, "Additional validation error: {}", validation_warnings[i]);
            }
        } else if (!state.error_message.empty()) {
            // Fall back to search state error message
            result.error_message = state.error_message;
            log_error(
                tt::LogFabric,
                "Mapping failed during search: {}. Saving partial mapping: {} of {} target nodes mapped",
                state.error_message,
                mapped_count,
                graph_data.n_target);
        } else {
            result.error_message = fmt::format(
                "Mapping validation failed: incomplete or invalid mapping. {} of {} target nodes mapped",
                mapped_count,
                graph_data.n_target);
            log_error(tt::LogFabric, "{}", result.error_message);
        }

        // Still compute statistics and copy search stats even if validation failed
        // This helps users understand what was found
        // Compute constraint statistics
        const auto& valid_mappings = constraints.get_valid_mappings();
        const auto& preferred_mappings = constraints.get_preferred_mappings();

        size_t required_satisfied = 0;
        size_t preferred_satisfied = 0;
        size_t preferred_total = 0;

        for (const auto& [target_node, global_node] : result.target_to_global) {
            // Check required constraints
            auto valid_it = valid_mappings.find(target_node);
            if (valid_it != valid_mappings.end() && !valid_it->second.empty()) {
                // This target node has required constraints
                if (valid_it->second.find(global_node) != valid_it->second.end()) {
                    required_satisfied++;
                }
            }

            // Check preferred constraints
            auto preferred_it = preferred_mappings.find(target_node);
            if (preferred_it != preferred_mappings.end() && !preferred_it->second.empty()) {
                preferred_total += preferred_it->second.size();
                if (preferred_it->second.find(global_node) != preferred_it->second.end()) {
                    preferred_satisfied++;
                }
            }
        }

        result.constraint_stats.required_satisfied = required_satisfied;
        result.constraint_stats.preferred_satisfied = preferred_satisfied;
        result.constraint_stats.preferred_total = preferred_total;

        result.stats.dfs_calls = state.dfs_calls;
        result.stats.backtrack_count = state.backtrack_count;
        result.warnings = std::move(validation_warnings);

        return result;
    }

    // Compute constraint statistics
    const auto& valid_mappings = constraints.get_valid_mappings();
    const auto& preferred_mappings = constraints.get_preferred_mappings();

    size_t required_satisfied = 0;
    size_t preferred_satisfied = 0;
    size_t preferred_total = 0;

    for (const auto& [target_node, global_node] : result.target_to_global) {
        // Check required constraints
        auto valid_it = valid_mappings.find(target_node);
        if (valid_it != valid_mappings.end() && !valid_it->second.empty()) {
            // This target node has required constraints
            if (valid_it->second.find(global_node) != valid_it->second.end()) {
                required_satisfied++;
            }
        }

        // Check preferred constraints
        auto preferred_it = preferred_mappings.find(target_node);
        if (preferred_it != preferred_mappings.end() && !preferred_it->second.empty()) {
            preferred_total += preferred_it->second.size();
            if (preferred_it->second.find(global_node) != preferred_it->second.end()) {
                preferred_satisfied++;
            }
        }
    }

    result.constraint_stats.required_satisfied = required_satisfied;
    result.constraint_stats.preferred_satisfied = preferred_satisfied;
    result.constraint_stats.preferred_total = preferred_total;

    // Copy warnings (from relaxed mode channel count mismatches)
    result.warnings = std::move(validation_warnings);

    // Copy statistics
    result.stats.dfs_calls = state.dfs_calls;
    result.stats.backtrack_count = state.backtrack_count;

    // Log success with statistics
    log_info(
        tt::LogFabric,
        "Mapping validation succeeded: {} target nodes mapped to {} global nodes. "
        "DFS calls: {}, backtracks: {}. Required constraints satisfied: {}, preferred constraints satisfied: {}/{}",
        graph_data.n_target,
        graph_data.n_global,
        state.dfs_calls,
        state.backtrack_count,
        result.constraint_stats.required_satisfied,
        result.constraint_stats.preferred_satisfied,
        result.constraint_stats.preferred_total);

    if (!result.warnings.empty()) {
        log_info(
            tt::LogFabric,
            "Mapping completed with {} warning(s) about channel count mismatches (relaxed mode)",
            result.warnings.size());
    }

    // Success!
    result.success = true;
    return result;
}

}  // namespace tt::tt_fabric::detail

#endif  // TOPOLOGY_SOLVER_INTERNAL_TPP
