// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <sstream>
#include <unordered_set>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/assert.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_fabric {

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

// Explicit template instantiations for common types
template class AdjacencyGraph<FabricNodeId>;
template class AdjacencyGraph<tt::tt_metal::AsicID>;

std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(const MeshGraph& mesh_graph) {
    std::map<MeshId, AdjacencyGraph<FabricNodeId>> adjacency_map;

    auto get_local_adjacents = [&](FabricNodeId fabric_node_id, MeshId mesh_id) {
        auto adjacent_map = mesh_graph.get_intra_mesh_connectivity()[*mesh_id][fabric_node_id.chip_id];

        std::vector<FabricNodeId> adjacents;
        bool relaxed = mesh_graph.is_intra_mesh_policy_relaxed(mesh_id);
        for (const auto& [neighbor_chip_id, edge] : adjacent_map) {
            // Skip self-connections
            if (neighbor_chip_id == fabric_node_id.chip_id) {
                continue;
            }
            size_t repeat_count = relaxed ? 1 : edge.connected_chip_ids.size();
            for (size_t i = 0; i < repeat_count; ++i) {
                adjacents.push_back(FabricNodeId(mesh_id, neighbor_chip_id));
            }
        }
        return adjacents;
    };

    // Iterate over all mesh IDs from the mesh graph
    for (const auto& mesh_id : mesh_graph.get_mesh_ids()) {
        AdjacencyGraph<FabricNodeId>::AdjacencyMap logical_adjacency_map;
        for (const auto& [_, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            auto fabric_node_id = FabricNodeId(mesh_id, chip_id);
            logical_adjacency_map[fabric_node_id] = get_local_adjacents(fabric_node_id, mesh_id);
        }
        adjacency_map[mesh_id] = AdjacencyGraph<FabricNodeId>(logical_adjacency_map);
    }

    return adjacency_map;
}

std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> build_adjacency_map_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> adjacency_map;

    // Build a set of ASIC IDs for each mesh based on mesh rank mapping
    std::map<MeshId, std::unordered_set<tt::tt_metal::AsicID>> mesh_asic_ids;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            mesh_asic_ids[mesh_id].insert(asic_id);
        }
    }

    for (const auto& [mesh_id, mesh_asics] : mesh_asic_ids) {
        auto z_channels = std::unordered_set<uint8_t>{8, 9};
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

        auto get_local_adjacents = [&](tt::tt_metal::AsicID asic_id,
                                       const std::unordered_set<tt::tt_metal::AsicID>& mesh_asics) {
            std::vector<tt::tt_metal::AsicID> adjacents;

            for (const auto& neighbor : physical_system_descriptor.get_asic_neighbors(asic_id)) {
                // Skip self-connections
                if (neighbor == asic_id) {
                    continue;
                }
                // Make sure that the neighbor is in the mesh
                if (mesh_asics.contains(neighbor)) {
                    // Add each neighbor multiple times based on number of ethernet connections
                    auto eth_connections = physical_system_descriptor.get_eth_connections(asic_id, neighbor);
                    for (const auto& eth_connection : eth_connections) {
                        // NOTE: IGNORE Z channels for Blackhole galaxy in intra mesh connectivity for now since
                        // they cause issues with uniform mesh mapping since topology mapper algorithm does not prefer
                        // taking the full connectivity path vs downgrading through z channels for intramesh
                        // connectivity https://github.com/tenstorrent/tt-metal/issues/31846
                        if (cluster_type == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY &&
                            (z_channels.contains(eth_connection.src_chan) ||
                             z_channels.contains(eth_connection.dst_chan))) {
                            continue;
                        }
                        adjacents.push_back(neighbor);
                    }
                }
            }
            return adjacents;
        };

        AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap physical_adjacency_map;
        for (const auto& asic_id : mesh_asics) {
            physical_adjacency_map[asic_id] = get_local_adjacents(asic_id, mesh_asics);
        }
        adjacency_map[mesh_id] = AdjacencyGraph<tt::tt_metal::AsicID>(physical_adjacency_map);
    }

    return adjacency_map;
}

// MappingConstraints implementation
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

// Explicit template instantiations for common types
// These instantiate non-template methods (like intersect_sets, validate_and_throw, etc.)
template class MappingConstraints<uint32_t, uint64_t>;  // Test types
template class MappingConstraints<FabricNodeId, tt::tt_metal::AsicID>;

// Explicit instantiations for trait constraint methods with various trait types
// These are needed because the template methods are implemented in the .cpp file
// and require explicit instantiation for each trait type used.
template void MappingConstraints<uint32_t, uint64_t>::add_required_trait_constraint<std::string>(
    const std::map<uint32_t, std::string>&, const std::map<uint64_t, std::string>&);
template void MappingConstraints<uint32_t, uint64_t>::add_preferred_trait_constraint<int>(
    const std::map<uint32_t, int>&, const std::map<uint64_t, int>&);
template void MappingConstraints<uint32_t, uint64_t>::add_required_trait_constraint<uint8_t>(
    const std::map<uint32_t, uint8_t>&, const std::map<uint64_t, uint8_t>&);
template void MappingConstraints<uint32_t, uint64_t>::add_preferred_trait_constraint<uint32_t>(
    const std::map<uint32_t, uint32_t>&, const std::map<uint64_t, uint32_t>&);
template void MappingConstraints<uint32_t, uint64_t>::add_required_trait_constraint<size_t>(
    const std::map<uint32_t, size_t>&, const std::map<uint64_t, size_t>&);

}  // namespace tt::tt_fabric
