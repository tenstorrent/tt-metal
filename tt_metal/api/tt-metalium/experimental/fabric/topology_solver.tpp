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

#include <fmt/format.h>
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
        ss << fmt::format("  Node {} (degree {}): ", node, neighbors.size());
        if (neighbors.empty()) {
            ss << "no neighbors";
        } else {
            bool first = true;
            for (const auto& neighbor : neighbors) {
                if (!first) {
                    ss << ", ";
                }
                first = false;
                ss << fmt::format("{}", neighbor);
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
bool MappingConstraints<TargetNode, GlobalNode>::add_required_trait_constraint(
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

    // Validate automatically and return false if invalid
    return validate();
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

    // Note: Validation is not performed in constructor - it will be validated when constraints are added
    // via add_required_constraint() or other methods that return bool
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_required_constraint(
    TargetNode target_node, GlobalNode global_node) {
    // Save current state before modifying (for rollback if validation fails)
    std::map<TargetNode, std::set<GlobalNode>> saved_state;
    auto it = valid_mappings_.find(target_node);
    if (it != valid_mappings_.end()) {
        saved_state[target_node] = it->second;
    }

    // If this global node is already reserved, add target_node to the reserved set
    auto reserved_it = reserved_global_nodes_.find(global_node);
    if (reserved_it != reserved_global_nodes_.end()) {
        reserved_it->second.insert(target_node);
    }

    // Intersect valid_mappings_[target] with {global_node}
    if (valid_mappings_[target_node].empty()) {
        // First constraint: initialize with this single node
        valid_mappings_[target_node].insert(global_node);
    } else {
        // Intersect with existing constraints
        std::set<GlobalNode> singleton{global_node};
        valid_mappings_[target_node] = intersect_sets(valid_mappings_[target_node], singleton);
    }

    // Validate automatically and return false if invalid (will restore saved_state on failure)
    return validate(&saved_state);
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_required_constraint(
    TargetNode target_node, const std::set<GlobalNode>& global_nodes) {
    // Save current state before modifying (for rollback if validation fails)
    std::map<TargetNode, std::set<GlobalNode>> saved_state;
    auto it = valid_mappings_.find(target_node);
    if (it != valid_mappings_.end()) {
        saved_state[target_node] = it->second;
    }

    // If any of these global nodes are already reserved, add target_node to the reserved set
    for (const auto& global_node : global_nodes) {
        auto reserved_it = reserved_global_nodes_.find(global_node);
        if (reserved_it != reserved_global_nodes_.end()) {
            reserved_it->second.insert(target_node);
        }
    }

    // Intersect valid_mappings_[target] with global_nodes
    if (valid_mappings_[target_node].empty()) {
        // First constraint: initialize with the provided set of nodes
        valid_mappings_[target_node] = global_nodes;
    } else {
        // Intersect with existing constraints
        valid_mappings_[target_node] = intersect_sets(valid_mappings_[target_node], global_nodes);
    }

    // Validate automatically and return false if invalid (will restore saved_state on failure)
    return validate(&saved_state);
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_required_constraint(
    const std::set<TargetNode>& target_nodes, GlobalNode global_node) {
    // Save current state before modifying (for rollback if validation fails)
    std::map<TargetNode, std::set<GlobalNode>> saved_state;
    for (const auto& target_node : target_nodes) {
        auto it = valid_mappings_.find(target_node);
        if (it != valid_mappings_.end()) {
            saved_state[target_node] = it->second;
        }
    }

    // If this global node is already reserved, add all target_nodes to the reserved set
    auto reserved_it = reserved_global_nodes_.find(global_node);
    if (reserved_it != reserved_global_nodes_.end()) {
        reserved_it->second.insert(target_nodes.begin(), target_nodes.end());
    }

    // For each target node, intersect valid_mappings_[target] with {global_node}
    for (const auto& target_node : target_nodes) {
        if (valid_mappings_[target_node].empty()) {
            // First constraint: initialize with this single node
            valid_mappings_[target_node].insert(global_node);
        } else {
            // Intersect with existing constraints
            std::set<GlobalNode> singleton{global_node};
            valid_mappings_[target_node] = intersect_sets(valid_mappings_[target_node], singleton);
        }
    }

    // Validate automatically and return false if invalid (will restore saved_state on failure)
    return validate(&saved_state);
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_required_constraint(
    const std::set<TargetNode>& target_nodes, const std::set<GlobalNode>& global_nodes) {
    // Save current state before modifying (for rollback if validation fails)
    std::map<TargetNode, std::set<GlobalNode>> saved_state;
    for (const auto& target_node : target_nodes) {
        auto it = valid_mappings_.find(target_node);
        if (it != valid_mappings_.end()) {
            saved_state[target_node] = it->second;
        }
    }

    // For each target node, ensure it can map to any of the global nodes
    // This creates a many-to-many relationship: any target can map to any global
    for (const auto& target_node : target_nodes) {
        if (valid_mappings_[target_node].empty()) {
            // First constraint: initialize with the provided set of global nodes
            valid_mappings_[target_node] = global_nodes;
        } else {
            // Intersect with existing constraints to ensure compatibility
            // This allows the target to map to any global node that satisfies both
            // the existing constraints and the new many-to-many constraint
            valid_mappings_[target_node] = intersect_sets(valid_mappings_[target_node], global_nodes);
        }
    }

    // Track that these global nodes are reserved for these target nodes via many-to-many constraint
    // This allows us to enforce that nodes not in the constraint cannot map to these global nodes
    for (const auto& global_node : global_nodes) {
        reserved_global_nodes_[global_node].insert(target_nodes.begin(), target_nodes.end());
    }

    // Validate automatically and return false if invalid (will restore saved_state on failure)
    return validate(&saved_state);
}

template <typename TargetNode, typename GlobalNode>
void MappingConstraints<TargetNode, GlobalNode>::add_preferred_constraint(
    TargetNode target_node, GlobalNode global_node) {
    // Add to preferred mappings (doesn't restrict valid mappings)
    preferred_mappings_[target_node].insert(global_node);
}

template <typename TargetNode, typename GlobalNode>
void MappingConstraints<TargetNode, GlobalNode>::add_preferred_constraint(
    TargetNode target_node, const std::set<GlobalNode>& global_nodes) {
    // Intersect preferred_mappings_[target] with global_nodes
    if (preferred_mappings_[target_node].empty()) {
        // First preferred constraint: initialize with the provided set of nodes
        preferred_mappings_[target_node] = global_nodes;
    } else {
        // Intersect with existing preferred constraints
        preferred_mappings_[target_node] = intersect_sets(preferred_mappings_[target_node], global_nodes);
    }
}

template <typename TargetNode, typename GlobalNode>
void MappingConstraints<TargetNode, GlobalNode>::add_preferred_constraint(
    const std::set<TargetNode>& target_nodes, GlobalNode global_node) {
    // For each target node, intersect preferred_mappings_[target] with {global_node}
    for (const auto& target_node : target_nodes) {
        if (preferred_mappings_[target_node].empty()) {
            // First preferred constraint: initialize with this single node
            preferred_mappings_[target_node].insert(global_node);
        } else {
            // Intersect with existing preferred constraints
            std::set<GlobalNode> singleton{global_node};
            preferred_mappings_[target_node] = intersect_sets(preferred_mappings_[target_node], singleton);
        }
    }
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::validate(
    const std::map<TargetNode, std::set<GlobalNode>>* saved_state) {
    // Filter out invalid mappings (e.g., those that conflict with reserved global nodes)
    // This ensures that valid_mappings_ only contains mappings that pass is_valid_mapping
    // We need to check against reserved nodes, so we check each mapping
    for (auto& [target, valid_set] : valid_mappings_) {
        std::set<GlobalNode> filtered_set;
        for (const auto& global : valid_set) {
            // Check if this global node is reserved and if target is allowed
            auto reserved_it = reserved_global_nodes_.find(global);
            if (reserved_it != reserved_global_nodes_.end()) {
                // This global node is reserved - check if target is allowed
                if (reserved_it->second.find(target) != reserved_it->second.end()) {
                    filtered_set.insert(global);
                }
            } else {
                // Not reserved, so it's valid
                filtered_set.insert(global);
            }
        }
        valid_set = std::move(filtered_set);
    }

    // Check if any target node has an empty valid_mappings_ set
    std::vector<TargetNode> conflicted_targets;
    for (const auto& [target, valid_set] : valid_mappings_) {
        if (valid_set.empty()) {
            conflicted_targets.push_back(target);
        }
    }

    if (!conflicted_targets.empty()) {
        // Restore saved state if provided
        if (saved_state != nullptr) {
            // Restore only the affected nodes from saved_state
            for (const auto& [target, saved_valid_set] : *saved_state) {
                valid_mappings_[target] = saved_valid_set;
            }
        }

        std::ostringstream oss;
        oss << "Constraint validation failed: " << conflicted_targets.size()
            << " target node(s) have no valid mappings (overconstrained).\n";
        oss << "Overconstrained target nodes:\n";
        for (const auto& target : conflicted_targets) {
            oss << "  - " << target << "\n";
        }

        // Show summary of all constraints for context
        oss << "\nConstraint summary:\n";
        oss << "  Total target nodes with constraints: " << valid_mappings_.size() << "\n";
        oss << "  Target nodes with valid mappings: " << (valid_mappings_.size() - conflicted_targets.size()) << "\n";
        oss << "  Overconstrained target nodes: " << conflicted_targets.size() << "\n";

        // Log info message instead of throwing
        log_info(tt::LogFabric, "{}", oss.str());
        return false;
    }

    // Validate cardinality constraints are still satisfiable with current required constraints
    // (only if we didn't restore saved state, as that means validation passed before)
    if (saved_state == nullptr) {
        if (!validate_cardinality_constraints()) {
            return false;
        }
    }

    return true;
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
    // Check if this global node is reserved by a many-to-many constraint
    auto reserved_it = reserved_global_nodes_.find(global);
    if (reserved_it != reserved_global_nodes_.end()) {
        // This global node is reserved - check if the target node is allowed to map to it
        if (reserved_it->second.find(target) == reserved_it->second.end()) {
            // Target node is not in the allowed set for this reserved global node
            return false;
        }
    }

    // If target is not in the dictionary, assume mapping is valid (unless it's reserved above)
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
const std::vector<std::pair<std::set<std::pair<TargetNode, GlobalNode>>, size_t>>&
MappingConstraints<TargetNode, GlobalNode>::get_cardinality_constraints() const {
    return cardinality_constraints_;
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_forbidden_constraint(
    TargetNode target_node, GlobalNode global_node) {
    auto it = valid_mappings_.find(target_node);
    if (it == valid_mappings_.end()) {
        return true;  // No constraints exist for this target, nothing to forbid
    }

    std::map<TargetNode, std::set<GlobalNode>> saved_state{{target_node, it->second}};
    it->second.erase(global_node);
    return validate(&saved_state);
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_forbidden_constraint(
    TargetNode target_node, const std::set<GlobalNode>& global_nodes) {
    auto it = valid_mappings_.find(target_node);
    if (it == valid_mappings_.end()) {
        return true;  // No constraints exist for this target, nothing to forbid
    }

    std::map<TargetNode, std::set<GlobalNode>> saved_state{{target_node, it->second}};
    for (const auto& global_node : global_nodes) {
        it->second.erase(global_node);
    }
    return validate(&saved_state);
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_forbidden_constraint(
    const std::set<TargetNode>& target_nodes, GlobalNode global_node) {
    std::map<TargetNode, std::set<GlobalNode>> saved_state;
    for (const auto& target_node : target_nodes) {
        auto it = valid_mappings_.find(target_node);
        if (it != valid_mappings_.end()) {
            saved_state[target_node] = it->second;
            it->second.erase(global_node);
        }
    }

    if (!saved_state.empty()) {
        return validate(&saved_state);
    }
    return true;  // No constraints exist for any target, nothing to forbid
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_cardinality_constraint(
    const std::set<std::pair<TargetNode, GlobalNode>>& mapping_pairs, size_t min_count) {
    if (mapping_pairs.empty()) {
        log_info(tt::LogFabric, "Cardinality constraint requires at least one mapping pair");
        return false;
    }
    if (min_count > mapping_pairs.size()) {
        log_info(tt::LogFabric, "Cardinality constraint min_count ({}) cannot be greater than number of pairs ({})", min_count, mapping_pairs.size());
        return false;
    }
    if (min_count == 0) {
        log_info(tt::LogFabric, "Cardinality constraint min_count must be at least 1");
        return false;
    }

    // Validate compatibility with existing required constraints
    std::set<std::pair<TargetNode, GlobalNode>> valid_pairs;
    std::vector<std::pair<TargetNode, GlobalNode>> invalid_pairs;

    for (const auto& [target_node, global_node] : mapping_pairs) {
        // Check if this pair is compatible with existing required constraints
        if (is_valid_mapping(target_node, global_node)) {
            valid_pairs.insert({target_node, global_node});
        } else {
            invalid_pairs.push_back({target_node, global_node});
        }
    }

    // Check if we have enough valid pairs to satisfy min_count
    if (valid_pairs.size() < min_count) {
        std::ostringstream oss;
        oss << "Cardinality constraint incompatible with existing required constraints.\n";
        oss << "  Required: at least " << min_count << " pair(s) must be satisfied\n";
        oss << "  Valid pairs (compatible with required constraints): " << valid_pairs.size() << "\n";
        oss << "  Invalid pairs (conflict with required constraints): " << invalid_pairs.size() << "\n";

        if (!invalid_pairs.empty()) {
            oss << "  Invalid pairs:\n";
            for (const auto& [target, global] : invalid_pairs) {
                auto valid_it = valid_mappings_.find(target);
                if (valid_it != valid_mappings_.end() && !valid_it->second.empty()) {
                    std::string valid_list;
                    bool first = true;
                    for (const auto& valid_global : valid_it->second) {
                        if (!first) valid_list += ", ";
                        first = false;
                        valid_list += fmt::format("{}", valid_global);
                    }
                    oss << fmt::format("    - ({}, {}): {} is not in valid mappings for {} (valid: {})\n",
                        fmt::format("{}", target), fmt::format("{}", global),
                        fmt::format("{}", global), fmt::format("{}", target), valid_list);
                } else if (valid_it != valid_mappings_.end()) {
                    oss << fmt::format("    - ({}, {}): {} has no valid mappings (overconstrained)\n",
                        fmt::format("{}", target), fmt::format("{}", global), fmt::format("{}", target));
                } else {
                    oss << fmt::format("    - ({}, {}): {} has required constraints that exclude {}\n",
                        fmt::format("{}", target), fmt::format("{}", global),
                        fmt::format("{}", target), fmt::format("{}", global));
                }
            }
        }

        log_info(tt::LogFabric, "{}", oss.str());
        return false;
    }

    // Warn if some pairs were filtered but constraint is still satisfiable
    if (!invalid_pairs.empty() && valid_pairs.size() >= min_count) {
        log_warning(
            tt::LogFabric,
            "Cardinality constraint: {} pair(s) were filtered out due to conflicts with required constraints, "
            "but constraint is still satisfiable with {} remaining valid pair(s) (min_count: {})",
            invalid_pairs.size(),
            valid_pairs.size(),
            min_count);
    }

    // Validate that all cardinality constraints together are satisfiable BEFORE adding
    // Create a temporary constraint to test validation
    cardinality_constraints_.emplace_back(valid_pairs, min_count);
    if (!validate_cardinality_constraints()) {
        // Validation failed - remove the constraint we just added
        cardinality_constraints_.pop_back();
        return false;
    }

    return true;
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::add_cardinality_constraint(
    const std::set<TargetNode>& target_nodes,
    const std::set<GlobalNode>& global_nodes,
    size_t min_count) {
    if (target_nodes.empty()) {
        log_info(tt::LogFabric, "Cardinality constraint requires at least one target node");
        return false;
    }
    if (global_nodes.empty()) {
        log_info(tt::LogFabric, "Cardinality constraint requires at least one global node");
        return false;
    }

    // Generate all pairs from the Cartesian product of target_nodes × global_nodes
    std::set<std::pair<TargetNode, GlobalNode>> mapping_pairs;
    for (const auto& target_node : target_nodes) {
        for (const auto& global_node : global_nodes) {
            mapping_pairs.insert({target_node, global_node});
        }
    }

    // Delegate to the existing implementation
    return add_cardinality_constraint(mapping_pairs, min_count);
}

template <typename TargetNode, typename GlobalNode>
std::set<GlobalNode> MappingConstraints<TargetNode, GlobalNode>::intersect_sets(
    const std::set<GlobalNode>& set1, const std::set<GlobalNode>& set2) {
    std::set<GlobalNode> result;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
    return result;
}

template <typename TargetNode, typename GlobalNode>
bool MappingConstraints<TargetNode, GlobalNode>::validate_cardinality_constraints() const {
    // Validate cardinality constraints are satisfiable with current required constraints
    if (cardinality_constraints_.empty()) {
        return true;
    }

    // Check each cardinality constraint has enough valid pairs
    for (size_t i = 0; i < cardinality_constraints_.size(); ++i) {
        const auto& [mapping_pairs, min_count] = cardinality_constraints_[i];

        size_t valid_count = 0;
        std::vector<std::pair<TargetNode, GlobalNode>> invalid_pairs;

        for (const auto& [target_node, global_node] : mapping_pairs) {
            if (is_valid_mapping(target_node, global_node)) {
                valid_count++;
            } else {
                invalid_pairs.push_back({target_node, global_node});
            }
        }

        if (valid_count < min_count) {
            std::ostringstream oss;
            oss << "Cardinality constraint " << (i + 1) << " is unsatisfiable with current required constraints.\n";
            oss << "  Required: at least " << min_count << " pair(s) must be satisfied\n";
            oss << "  Valid pairs (compatible with required constraints): " << valid_count << "\n";
            oss << "  Invalid pairs: " << invalid_pairs.size() << "\n";

            if (!invalid_pairs.empty()) {
                oss << "  Invalid pairs:\n";
                for (const auto& [target, global] : invalid_pairs) {
                    auto valid_it = valid_mappings_.find(target);
                    if (valid_it != valid_mappings_.end() && !valid_it->second.empty()) {
                        oss << fmt::format("    - ({}, {}): {} is not in valid mappings for {}\n",
                            fmt::format("{}", target), fmt::format("{}", global),
                            fmt::format("{}", global), fmt::format("{}", target));
                    } else {
                        oss << fmt::format("    - ({}, {}): {} has no valid mappings\n",
                            fmt::format("{}", target), fmt::format("{}", global),
                            fmt::format("{}", target));
                    }
                }
            }

            // Log info message instead of throwing
            log_info(tt::LogFabric, "{}", oss.str());
            return false;
        }
    }

    return true;
}

// solve_topology_mapping template implementation
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode connection_validation_mode,
    bool quiet_mode) {
    using namespace tt::tt_fabric::detail;

    auto start_time = std::chrono::steady_clock::now();

    // Build indexed graph representation
    GraphIndexData<TargetNode, GlobalNode> graph_data(target_graph, global_graph);

    // Build indexed constraint representation
    ConstraintIndexData<TargetNode, GlobalNode> constraint_data(constraints, graph_data);

    // Run DFS search (state is now internal to the engine)
    DFSSearchEngine<TargetNode, GlobalNode> search_engine;
    search_engine.search(graph_data, constraint_data, connection_validation_mode, quiet_mode);

    // Calculate elapsed time
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Get state from engine and build result using validator
    const auto& state = search_engine.get_state();
    auto result = MappingValidator<TargetNode, GlobalNode>::build_result(
        state.mapping, graph_data, constraint_data, state, connection_validation_mode, quiet_mode);

    // Set elapsed time
    result.stats.elapsed_time = elapsed_ms;

    return result;
}

template <typename TargetNode, typename GlobalNode>
void print_mapping_result(const MappingResult<TargetNode, GlobalNode>& result) {
    std::stringstream ss;
    ss << "\n=== Mapping Result ===" << std::endl;
    ss << "Success: " << (result.success ? "true" : "false") << std::endl;
    if (!result.error_message.empty()) {
        ss << "Error: " << result.error_message << std::endl;
    }

    ss << "\nMappings:" << std::endl;
    for (const auto& [target_node, global_node] : result.target_to_global) {
        ss << "  Target node " << target_node << " -> Global node " << global_node << std::endl;
    }
    ss << "Total mapped: " << result.target_to_global.size() << " target nodes" << std::endl;

    if (!result.warnings.empty()) {
        ss << "\nWarnings (" << result.warnings.size() << "):" << std::endl;
        for (const auto& warning : result.warnings) {
            ss << "  - " << warning << std::endl;
        }
    }

    ss << "\nStatistics:" << std::endl;
    ss << "  DFS calls: " << result.stats.dfs_calls << std::endl;
    ss << "  Backtracks: " << result.stats.backtrack_count << std::endl;
    ss << "  Memoization hits: " << result.stats.memoization_hits << std::endl;
    ss << "  Elapsed time: " << result.stats.elapsed_time.count() << " ms" << std::endl;
    ss << "  Required constraints satisfied: " << result.constraint_stats.required_satisfied << std::endl;
    ss << "  Preferred constraints satisfied: " << result.constraint_stats.preferred_satisfied << "/"
       << result.constraint_stats.preferred_total << std::endl;

    ss << "======================" << std::endl;
    log_info(tt::LogFabric, "{}", ss.str());
}

}  // namespace tt::tt_fabric

#endif  // TOPOLOGY_SOLVER_TPP
