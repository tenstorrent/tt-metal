// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Template implementation file - included at the end of topology_solver_internal.hpp
// This allows template implementations to be separated from declarations while
// still enabling implicit instantiation for any type.

#ifndef TOPOLOGY_SOLVER_INTERNAL_TPP
#define TOPOLOGY_SOLVER_INTERNAL_TPP

#include <algorithm>
#include <sstream>
#include <unordered_set>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
GraphIndexData<TargetNode, GlobalNode> build_graph_index_data(
    const AdjacencyGraph<TargetNode>& target_graph, const AdjacencyGraph<GlobalNode>& global_graph) {
    GraphIndexData<TargetNode, GlobalNode> data;

    // Build node vectors
    const auto& target_nodes_vec = target_graph.get_nodes();
    const auto& global_nodes_vec = global_graph.get_nodes();

    data.target_nodes = target_nodes_vec;
    data.global_nodes = global_nodes_vec;
    data.n_target = data.target_nodes.size();
    data.n_global = data.global_nodes.size();

    // Build index mappings
    for (size_t i = 0; i < data.n_target; ++i) {
        data.target_to_idx[data.target_nodes[i]] = i;
    }

    for (size_t i = 0; i < data.n_global; ++i) {
        data.global_to_idx[data.global_nodes[i]] = i;
    }

    // Build connection count maps and deduplicated adjacency index vectors
    data.target_conn_count.resize(data.n_target);
    data.target_adj_idx.resize(data.n_target);
    data.target_deg.resize(data.n_target, 0);

    for (size_t i = 0; i < data.n_target; ++i) {
        const auto& node = data.target_nodes[i];
        const auto& neighbors = target_graph.get_neighbors(node);
        std::unordered_set<size_t> seen_indices;

        for (const auto& neigh : neighbors) {
            // Skip self-connections
            if (neigh == node) {
                continue;
            }
            auto it = data.target_to_idx.find(neigh);
            if (it != data.target_to_idx.end()) {
                size_t idx = it->second;
                data.target_conn_count[i][idx]++;
                if (seen_indices.insert(idx).second) {
                    data.target_adj_idx[i].push_back(idx);
                }
            }
        }
        std::sort(data.target_adj_idx[i].begin(), data.target_adj_idx[i].end());
        // Degree is the number of unique neighbors (not counting multi-edges or self-connections)
        data.target_deg[i] = data.target_adj_idx[i].size();
    }

    // Build connection count maps and deduplicated adjacency index vectors for global graph
    data.global_conn_count.resize(data.n_global);
    data.global_adj_idx.resize(data.n_global);
    data.global_deg.resize(data.n_global, 0);

    for (size_t i = 0; i < data.n_global; ++i) {
        const auto& node = data.global_nodes[i];
        const auto& neighbors = global_graph.get_neighbors(node);
        std::unordered_set<size_t> seen_indices;

        for (const auto& neigh : neighbors) {
            // Skip self-connections
            if (neigh == node) {
                continue;
            }
            auto it = data.global_to_idx.find(neigh);
            if (it != data.global_to_idx.end()) {
                size_t idx = it->second;
                data.global_conn_count[i][idx]++;
                if (seen_indices.insert(idx).second) {
                    data.global_adj_idx[i].push_back(idx);
                }
            }
        }
        std::sort(data.global_adj_idx[i].begin(), data.global_adj_idx[i].end());
        // Degree is the number of unique neighbors (not counting multi-edges or self-connections)
        data.global_deg[i] = data.global_adj_idx[i].size();
    }

    return data;
}

template <typename TargetNode, typename GlobalNode>
bool ConstraintIndexData<TargetNode, GlobalNode>::is_valid_mapping(size_t target_idx, size_t global_idx) const {
    // If no restrictions for this target, all mappings are valid
    if (target_idx >= restricted_global_indices.size() || restricted_global_indices[target_idx].empty()) {
        return true;
    }
    // Check if global_idx is in the restricted list
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
ConstraintIndexData<TargetNode, GlobalNode> build_constraint_index_data(
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data) {
    ConstraintIndexData<TargetNode, GlobalNode> constraint_data;

    // Initialize vectors for all target nodes
    constraint_data.restricted_global_indices.resize(graph_data.n_target);
    constraint_data.preferred_global_indices.resize(graph_data.n_target);

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

            constraint_data.restricted_global_indices[i] = std::move(restricted_indices);
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

            constraint_data.preferred_global_indices[i] = std::move(preferred_indices);
        }
    }

    return constraint_data;
}

}  // namespace tt::tt_fabric::detail

#endif  // TOPOLOGY_SOLVER_INTERNAL_TPP
