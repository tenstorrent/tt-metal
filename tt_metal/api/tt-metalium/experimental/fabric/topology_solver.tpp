// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Template implementation file - included at the end of topology_solver.hpp
// This allows template implementations to be separated from declarations while
// still enabling implicit instantiation for any type.

#ifndef TOPOLOGY_SOLVER_TPP
#define TOPOLOGY_SOLVER_TPP

#include <sstream>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric {

// AdjacencyGraph template method implementations
template <typename NodeId>
AdjacencyGraph<NodeId>::AdjacencyGraph(const AdjacencyMap& adjacency_map) : adj_map_(adjacency_map) {
    for (const auto& [node, neighbors] : adjacency_map) {
        nodes_cache_.push_back(node);
    }
}

template <typename NodeId>
const std::vector<NodeId>& AdjacencyGraph<NodeId>::get_nodes() const {
    return nodes_cache_;
}

template <typename NodeId>
const std::vector<NodeId>& AdjacencyGraph<NodeId>::get_neighbors(NodeId node) const {
    auto it = adj_map_.find(node);
    if (it != adj_map_.end()) {
        return it->second;
    }
    // Return empty vector if node not found
    static const std::vector<NodeId> empty_vec;
    return empty_vec;
}

template <typename NodeId>
void AdjacencyGraph<NodeId>::print_adjacency_map(const std::string& graph_name) const {
    std::stringstream ss;
    ss << "\n=== " << graph_name << " Adjacency Map ===" << std::endl;
    ss << "Total nodes: " << nodes_cache_.size() << std::endl;

    for (const auto& node : nodes_cache_) {
        const auto& neighbors = get_neighbors(node);
        ss << "  Node " << node << " (degree " << neighbors.size() << "): ";
        if (neighbors.empty()) {
            ss << "no neighbors";
        } else {
            bool first = true;
            for (const auto& neighbor : neighbors) {
                if (!first) {
                    ss << ", ";
                }
                first = false;
                ss << neighbor;
            }
        }
        ss << std::endl;
    }
    ss << "========================================" << std::endl;
    log_info(tt::LogFabric, "{}", ss.str());
}

}  // namespace tt::tt_fabric

#endif  // TOPOLOGY_SOLVER_TPP
