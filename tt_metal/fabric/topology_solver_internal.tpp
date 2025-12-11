// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Template implementation file - included at the end of topology_solver_internal.hpp
// This allows template implementations to be separated from declarations while
// still enabling implicit instantiation for any type.

#ifndef TOPOLOGY_SOLVER_INTERNAL_TPP
#define TOPOLOGY_SOLVER_INTERNAL_TPP

#include <algorithm>
#include <unordered_set>

namespace tt::tt_fabric::detail {

template <typename TargetNode, typename GlobalNode>
GraphIndexData<TargetNode, GlobalNode> build_graph_index_data(
    const AdjacencyGraph<TargetNode>& target_graph, const AdjacencyGraph<GlobalNode>& global_graph) {
    GraphIndexData<TargetNode, GlobalNode> data;

    // Build node vectors
    const auto& target_nodes_vec = target_graph.get_nodes();
    const auto& global_nodes_vec = global_graph.get_nodes();

    data.target_nodes.reserve(target_nodes_vec.size());
    for (const auto& node : target_nodes_vec) {
        data.target_nodes.push_back(node);
    }

    data.global_nodes.reserve(global_nodes_vec.size());
    for (const auto& node : global_nodes_vec) {
        data.global_nodes.push_back(node);
    }

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

}  // namespace tt::tt_fabric::detail

#endif  // TOPOLOGY_SOLVER_INTERNAL_TPP
