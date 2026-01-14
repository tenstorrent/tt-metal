// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Template implementation file - included at the end of topology_solver.hpp
// This allows template implementations to be separated from declarations while
// still enabling implicit instantiation for any type.

#ifndef TOPOLOGY_SOLVER_TPP
#define TOPOLOGY_SOLVER_TPP

#include <algorithm>
#include <sstream>
#include <chrono>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

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
const std::vector<NodeId>& AdjacencyGraph<NodeId>::get_neighbors(const NodeId& node) const {
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

// MappingConstraints trait constraint template method implementations
template <typename TargetNode, typename GlobalNode>
template <typename TraitType>
void MappingConstraints<TargetNode, GlobalNode>::add_required_trait_constraint(
    const std::map<TargetNode, TraitType>& target_traits, const std::map<GlobalNode, TraitType>& global_traits) {
    // Build reverse map: trait value -> set of global nodes with that trait
    std::map<TraitType, std::set<GlobalNode>> trait_to_globals;
    for (const auto& [global_node, trait_value] : global_traits) {
        trait_to_globals[trait_value].insert(global_node);
    }

    std::vector<TargetNode> conflicted_targets;

    // Apply trait constraint to valid_mappings_ (required)
    for (const auto& [target_node, trait_value] : target_traits) {
        auto trait_it = trait_to_globals.find(trait_value);
        if (trait_it == trait_to_globals.end()) {
            // No global nodes with this trait value - clear valid mappings
            valid_mappings_[target_node].clear();
            conflicted_targets.push_back(target_node);
        } else {
            // Intersect with existing valid mappings
            if (valid_mappings_[target_node].empty()) {
                // First constraint: initialize with all nodes with matching trait
                valid_mappings_[target_node] = trait_it->second;
            } else {
                // Intersect with existing constraints
                auto old_size = valid_mappings_[target_node].size();
                valid_mappings_[target_node] = intersect_sets(valid_mappings_[target_node], trait_it->second);
                if (valid_mappings_[target_node].empty() && old_size > 0) {
                    conflicted_targets.push_back(target_node);
                }
            }
        }
    }

    // Validate automatically and throw if invalid
    validate_and_throw();
}

template <typename TargetNode, typename GlobalNode>
template <typename TraitType>
void MappingConstraints<TargetNode, GlobalNode>::add_preferred_trait_constraint(
    const std::map<TargetNode, TraitType>& target_traits, const std::map<GlobalNode, TraitType>& global_traits) {
    // Build reverse map: trait value -> set of global nodes with that trait
    std::map<TraitType, std::set<GlobalNode>> trait_to_globals;
    for (const auto& [global_node, trait_value] : global_traits) {
        trait_to_globals[trait_value].insert(global_node);
    }

    // Apply trait constraint to preferred_mappings_ (preferred)
    for (const auto& [target_node, trait_value] : target_traits) {
        auto trait_it = trait_to_globals.find(trait_value);
        if (trait_it != trait_to_globals.end()) {
            // Intersect with existing preferred mappings
            if (preferred_mappings_[target_node].empty()) {
                // First preferred constraint: initialize with all nodes with matching trait
                preferred_mappings_[target_node] = trait_it->second;
            } else {
                // Intersect with existing preferred constraints
                preferred_mappings_[target_node] = intersect_sets(preferred_mappings_[target_node], trait_it->second);
            }
        }
    }
}

// MappingConstraints non-template method implementations
// These are non-template methods of a template class, so they're implemented here
// to allow implicit instantiation without explicit instantiations
template <typename TargetNode, typename GlobalNode>
MappingConstraints<TargetNode, GlobalNode>::MappingConstraints(
    const std::set<std::pair<TargetNode, GlobalNode>>& required_constraints,
    const std::set<std::pair<TargetNode, GlobalNode>>& preferred_constraints) {
    // Convert required pairs into mapping format
    for (const auto& [target, global] : required_constraints) {
        if (valid_mappings_[target].empty()) {
            valid_mappings_[target].insert(global);
        } else {
            std::set<GlobalNode> singleton{global};
            valid_mappings_[target] = intersect_sets(valid_mappings_[target], singleton);
        }
    }

    // Convert preferred pairs into mapping format
    for (const auto& [target, global] : preferred_constraints) {
        preferred_mappings_[target].insert(global);
    }

    // Validate automatically and throw if invalid
    validate_and_throw();
}

template <typename TargetNode, typename GlobalNode>
void MappingConstraints<TargetNode, GlobalNode>::add_required_constraint(
    TargetNode target_node, GlobalNode global_node) {
    // Intersect valid_mappings_[target] with {global_node}
    if (valid_mappings_[target_node].empty()) {
        // First constraint: initialize with this single node
        valid_mappings_[target_node].insert(global_node);
    } else {
        // Intersect with existing constraints
        std::set<GlobalNode> singleton{global_node};
        valid_mappings_[target_node] = intersect_sets(valid_mappings_[target_node], singleton);
    }

    // Validate automatically and throw if invalid
    validate_and_throw();
}

template <typename TargetNode, typename GlobalNode>
void MappingConstraints<TargetNode, GlobalNode>::add_preferred_constraint(
    TargetNode target_node, GlobalNode global_node) {
    // Add to preferred mappings (doesn't restrict valid mappings)
    preferred_mappings_[target_node].insert(global_node);
}

template <typename TargetNode, typename GlobalNode>
void MappingConstraints<TargetNode, GlobalNode>::validate_and_throw() const {
    // Check if any target node has an empty valid_mappings_ set
    std::vector<TargetNode> conflicted_targets;
    for (const auto& [target, valid_set] : valid_mappings_) {
        if (valid_set.empty()) {
            conflicted_targets.push_back(target);
        }
    }

    if (!conflicted_targets.empty()) {
        std::ostringstream oss;
        oss << "Constraint validation failed: " << conflicted_targets.size()
            << " target node(s) have no valid mappings (overconstrained).";
        TT_THROW("{}", oss.str());
    }
}

template <typename TargetNode, typename GlobalNode>
const std::set<GlobalNode>& MappingConstraints<TargetNode, GlobalNode>::get_valid_mappings(TargetNode target) const {
    static const std::set<GlobalNode> empty_set;
    auto it = valid_mappings_.find(target);
    if (it != valid_mappings_.end()) {
        return it->second;
    }
    return empty_set;
}

template <typename TargetNode, typename GlobalNode>
const std::set<GlobalNode>& MappingConstraints<TargetNode, GlobalNode>::get_preferred_mappings(
    TargetNode target) const {
    static const std::set<GlobalNode> empty_set;
    auto it = preferred_mappings_.find(target);
    if (it != preferred_mappings_.end()) {
        return it->second;
    }
    return empty_set;
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::is_valid_mapping(TargetNode target, GlobalNode global) const {
    // If target is not in the dictionary, assume mapping is valid
    auto it = valid_mappings_.find(target);
    if (it == valid_mappings_.end()) {
        return true;
    }
    // Otherwise, check if global node is in the valid mappings set
    return it->second.find(global) != it->second.end();
}

template <typename TargetNode, typename GlobalNode>
const std::map<TargetNode, std::set<GlobalNode>>& MappingConstraints<TargetNode, GlobalNode>::get_valid_mappings()
    const {
    return valid_mappings_;
}

template <typename TargetNode, typename GlobalNode>
const std::map<TargetNode, std::set<GlobalNode>>& MappingConstraints<TargetNode, GlobalNode>::get_preferred_mappings()
    const {
    return preferred_mappings_;
}

template <typename TargetNode, typename GlobalNode>
std::set<GlobalNode> MappingConstraints<TargetNode, GlobalNode>::intersect_sets(
    const std::set<GlobalNode>& set1, const std::set<GlobalNode>& set2) {
    std::set<GlobalNode> result;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
    return result;
}

}  // namespace tt::tt_fabric

#endif  // TOPOLOGY_SOLVER_TPP
