// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <ostream>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <queue>
#include <memory>
#include <cctype>
#include <functional>
#include <optional>
#include <tt_stl/fmt.hpp>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>

#include "protobuf/physical_grouping_descriptor.pb.h"
#include "protobuf/mesh_graph_descriptor.pb.h"
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-logger/tt-logger.hpp>
#include <map>

#include <google/protobuf/text_format.h>

using namespace tt::tt_fabric;

namespace {

// Helper function to build adjacency graph from row-major mesh connection
// Always uses LINE connectivity (no wrap-around) with configurable connections per edge
AdjacencyGraph<uint32_t> build_row_major_mesh_graph(
    const std::vector<uint32_t>& instance_ids,
    const std::vector<int32_t>& dims,
    const std::string& grouping_name,
    uint32_t connections_per_edge) {
    std::map<uint32_t, std::vector<uint32_t>> adj_map;

    if (instance_ids.empty() || dims.empty()) {
        return AdjacencyGraph<uint32_t>(adj_map);
    }

    // Calculate total size
    int32_t total_size = 1;
    for (int32_t dim : dims) {
        total_size *= dim;
    }

    if (static_cast<size_t>(total_size) != instance_ids.size()) {
        std::string dims_str = "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) {
                dims_str += ", ";
            }
            dims_str += std::to_string(dims[i]);
        }
        dims_str += "]";

        std::string error_msg = fmt::format(
            "Invalid row_major_mesh configuration in grouping '{}': "
            "dimensions {} multiply to {} (expected {} instances), but grouping has {} instance(s). "
            "The product of row_major_mesh dimensions must equal the number of instances in the grouping. "
            "If this is a mistake in the Physical Grouping Descriptor file, please file an error with the scaleout "
            "team.",
            grouping_name.empty() ? "<unknown>" : grouping_name,
            dims_str,
            total_size,
            total_size,
            instance_ids.size());
        TT_THROW("{}", error_msg);
    }

    // Build coordinate system helpers
    auto get_coords = [&](uint32_t idx) -> std::vector<int32_t> {
        std::vector<int32_t> coords(dims.size());
        int32_t remaining = static_cast<int32_t>(idx);
        for (int32_t i = static_cast<int32_t>(dims.size()) - 1; i >= 0; --i) {
            coords[i] = remaining % dims[i];
            remaining /= dims[i];
        }
        return coords;
    };

    auto get_index = [&](const std::vector<int32_t>& coords) -> uint32_t {
        uint32_t idx = 0;
        uint32_t multiplier = 1;
        for (int32_t i = static_cast<int32_t>(dims.size()) - 1; i >= 0; --i) {
            idx += static_cast<uint32_t>(coords[i]) * multiplier;
            multiplier *= static_cast<uint32_t>(dims[i]);
        }
        return idx;
    };

    // Build adjacency: connect neighbors in each dimension
    for (uint32_t node_idx = 0; node_idx < instance_ids.size(); ++node_idx) {
        uint32_t node_id = instance_ids[node_idx];
        std::vector<int32_t> coords = get_coords(node_idx);

        // For each dimension, connect to neighbor
        for (int32_t dim_idx = 0; dim_idx < static_cast<int32_t>(dims.size()); ++dim_idx) {
            // Connect to neighbor in positive direction
            if (coords[dim_idx] < dims[dim_idx] - 1) {
                std::vector<int32_t> neighbor_coords = coords;
                neighbor_coords[dim_idx]++;
                uint32_t neighbor_idx = get_index(neighbor_coords);
                uint32_t neighbor_id = instance_ids[neighbor_idx];

                // Add connections_per_edge edges (bidirectional)
                for (uint32_t conn = 0; conn < connections_per_edge; ++conn) {
                    adj_map[node_id].push_back(neighbor_id);
                    adj_map[neighbor_id].push_back(node_id);
                }
            }
        }
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

// Helper function to build adjacency graph from MGD mesh instance's device topology
// Builds a row-major mesh graph based on the mesh's device_topology dims
// This represents the topology at the ASIC level, which matches the flattened physical grouping graphs
AdjacencyGraph<uint32_t> build_mgd_mesh_instance_adjacency(
    const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId mesh_instance_id) {
    const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_instance_id);
    TT_FATAL(mesh_instance.kind == NodeKind::Mesh, "build_mgd_mesh_instance_adjacency called on non-mesh instance");

    const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(mesh_instance.desc);
    TT_FATAL(mesh_desc != nullptr, "Mesh descriptor is null");

    // Get device topology dimensions (represents ASIC-level layout)
    const auto& device_topology = mesh_desc->device_topology();
    std::vector<int32_t> device_dims(device_topology.dims().begin(), device_topology.dims().end());

    if (device_dims.empty()) {
        // No device topology - return empty graph
        return AdjacencyGraph<uint32_t>();
    }

    // Calculate number of ASICs
    int32_t num_asics = 1;
    for (int32_t dim : device_dims) {
        num_asics *= dim;
    }

    // Create abstract ASIC node IDs (0, 1, 2, ..., num_asics-1)
    std::vector<uint32_t> asic_ids;
    asic_ids.reserve(num_asics);
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_asics); ++i) {
        asic_ids.push_back(i);
    }

    // Build row-major mesh graph representing ASIC-level topology
    // Always uses LINE connectivity (no wrap-around) and 1 connection per edge
    auto result = build_row_major_mesh_graph(asic_ids, device_dims, "", 1);

    return result;
}

// Helper function to build adjacency graph from MGD switch instance
// Similar to build_mgd_mesh_instance_adjacency - builds row-major mesh graph from device_topology
AdjacencyGraph<uint32_t> build_mgd_switch_instance_adjacency(
    const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId switch_instance_id) {
    const auto& switch_instance = mesh_graph_descriptor.get_instance(switch_instance_id);
    TT_FATAL(
        switch_instance.kind == NodeKind::Switch, "build_mgd_switch_instance_adjacency called on non-switch instance");

    const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(switch_instance.desc);
    TT_FATAL(switch_desc != nullptr, "Switch descriptor is null");

    // Get device topology dimensions (represents ASIC-level layout)
    const auto& device_topology = switch_desc->device_topology();
    std::vector<int32_t> device_dims(device_topology.dims().begin(), device_topology.dims().end());

    if (device_dims.empty()) {
        // No device topology - return empty graph
        return AdjacencyGraph<uint32_t>();
    }

    // Calculate number of ASICs
    int32_t num_asics = 1;
    for (int32_t dim : device_dims) {
        num_asics *= dim;
    }

    // Create abstract ASIC node IDs (0, 1, 2, ..., num_asics-1)
    std::vector<uint32_t> asic_ids;
    asic_ids.reserve(num_asics);
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_asics); ++i) {
        asic_ids.push_back(i);
    }

    // Build row-major mesh graph representing ASIC-level topology
    // Always uses LINE connectivity (no wrap-around) and 1 connection per edge
    auto result = build_row_major_mesh_graph(asic_ids, device_dims, "", 1);

    return result;
}

// Helper function to build adjacency graph from MGD graph instance
// The graph instance's sub_instances become nodes, and connections between them become edges
// Ensures no duplicate connections and all connections are bidirectional
AdjacencyGraph<uint32_t> build_mgd_graph_instance_adjacency(
    const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId graph_instance_id) {
    const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_instance_id);

    // Get all sub-instances (these will be the nodes in our adjacency graph)
    std::vector<uint32_t> sub_instance_ids(graph_instance.sub_instances.begin(), graph_instance.sub_instances.end());

    // Build adjacency map from connections
    std::map<uint32_t, std::vector<uint32_t>> adj_map;

    // Initialize adjacency map for all sub-instances
    for (uint32_t sub_id : sub_instance_ids) {
        adj_map[sub_id] = std::vector<uint32_t>();
    }

    // Use a set to track processed edges to avoid duplicates
    std::set<std::pair<uint32_t, uint32_t>> processed_edges;

    // Get all connections for this graph instance
    const auto& connection_ids = mesh_graph_descriptor.connections_by_instance_id(graph_instance_id);

    // Build adjacency from connections
    for (ConnectionId conn_id : connection_ids) {
        const auto& conn = mesh_graph_descriptor.get_connection(conn_id);

        // Connections have nodes array: [src, dst]
        if (conn.nodes.size() >= 2) {
            uint32_t src = conn.nodes[0];
            uint32_t dst = conn.nodes[1];

            // Only add edges if both nodes are sub-instances of this graph
            if (graph_instance.sub_instances.contains(src) && graph_instance.sub_instances.contains(dst)) {
                // Skip self-loops
                if (src == dst) {
                    continue;
                }

                // Normalize edge pair to avoid duplicates (treat (A,B) and (B,A) as the same)
                auto edge_pair = std::minmax(src, dst);

                // Only add edge if not already processed (prevents duplicates)
                if (processed_edges.insert(edge_pair).second) {
                    // Add bidirectional edge (undirected graph)
                    adj_map[src].push_back(dst);
                    adj_map[dst].push_back(src);
                }
            }
        }
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

}  // namespace

namespace tt::tt_fabric {

// Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts)
// Calculates required ASIC counts bottom-up and builds adjacency graphs
// Returns map: (type, name) -> GroupingInfo
std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>>
PhysicalGroupingDescriptor::build_mgd_to_grouping_info_map(const MeshGraphDescriptor& mesh_graph_descriptor) {
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos;

    // ===== Step 1: Calculate required ASIC counts bottom-up =====
    // Map: (type, name) -> required_asics
    std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> required_asics_map;

    // Step 1a: Calculate required ASICs for all mesh instances (bottom level)
    for (GlobalNodeId mesh_id : mesh_graph_descriptor.all_meshes()) {
        const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_id);
        uint32_t required_chips = mesh_graph_descriptor.get_chip_count(mesh_id);
        required_asics_map[mesh_instance.type][mesh_instance.name] = required_chips;
    }

    // Step 1a: Calculate required ASICs for all switch instances (bottom level)
    // Switches are treated as MESH type for grouping purposes
    for (GlobalNodeId switch_id : mesh_graph_descriptor.all_switches()) {
        const auto& switch_instance = mesh_graph_descriptor.get_instance(switch_id);
        uint32_t required_chips = mesh_graph_descriptor.get_switch_chip_count(switch_id);
        // Store switches under MESH type (switches are treated as MESH type)
        required_asics_map["MESH"][switch_instance.name] = required_chips;
    }

    // Step 1b: Calculate required ASICs for graph instances bottom-up (children before parents)
    // Process graphs in topological order by iterating until all are processed
    std::unordered_set<GlobalNodeId> processed_graphs;
    bool progress_made = true;

    while (progress_made) {
        progress_made = false;

        for (GlobalNodeId graph_id : mesh_graph_descriptor.all_graphs()) {
            if (processed_graphs.contains(graph_id)) {
                continue;  // Already processed
            }

            const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_id);
            const std::string& graph_type = graph_instance.type;
            const std::string& graph_name = graph_instance.name;

            // Check if all sub-instances have been processed (have required_asics calculated)
            bool all_sub_instances_ready = true;
            uint32_t required_asics = 0;

            for (GlobalNodeId sub_id : graph_instance.sub_instances) {
                const auto& sub_instance = mesh_graph_descriptor.get_instance(sub_id);

                // Switches are treated as MESH type for grouping purposes
                // Use "MESH" type for switches, otherwise use the sub_instance's actual type
                std::string lookup_type = (sub_instance.kind == NodeKind::Switch) ? "MESH" : sub_instance.type;

                // Check if this sub-instance's required_asics is already calculated
                auto sub_type_it = required_asics_map.find(lookup_type);
                if (sub_type_it == required_asics_map.end()) {
                    all_sub_instances_ready = false;
                    break;
                }

                auto sub_name_it = sub_type_it->second.find(sub_instance.name);
                if (sub_name_it == sub_type_it->second.end()) {
                    all_sub_instances_ready = false;
                    break;
                }

                required_asics += sub_name_it->second;
            }

            // If all sub-instances are ready, calculate and store this graph's required_asics
            if (all_sub_instances_ready) {
                required_asics_map[graph_type][graph_name] = required_asics;
                processed_graphs.insert(graph_id);
                progress_made = true;
            }
        }
    }

    // Verify all graphs were processed (should not have cycles, but check for safety)
    for (GlobalNodeId graph_id : mesh_graph_descriptor.all_graphs()) {
        const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_id);
        auto type_it = required_asics_map.find(graph_instance.type);
        if (type_it == required_asics_map.end() || !type_it->second.contains(graph_instance.name)) {
            TT_THROW(
                "Failed to calculate required ASIC count for graph instance '{}' (type '{}'). "
                "This may indicate a circular dependency in the MGD.",
                graph_instance.name,
                graph_instance.type);
        }
    }

    // ===== Step 2: Build GroupingInfo objects with adjacency graphs and ASIC counts =====

    // Process mesh instances
    // Store only one entry per mesh definition name (M0, M1), not per instance (M0_0, M0_1, etc.)
    std::set<std::string> processed_mesh_definitions;
    for (GlobalNodeId mesh_id : mesh_graph_descriptor.all_meshes()) {
        const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_id);
        const std::string& mesh_type = mesh_instance.type;
        const std::string& mesh_name = mesh_instance.name;

        // Skip if we've already processed this mesh definition
        if (processed_mesh_definitions.contains(mesh_name)) {
            continue;
        }
        processed_mesh_definitions.insert(mesh_name);

        // Build adjacency graph for this mesh instance (use first instance of this mesh definition)
        AdjacencyGraph<uint32_t> adjacency_graph = build_mgd_mesh_instance_adjacency(mesh_graph_descriptor, mesh_id);

        // Get required ASIC count (calculated above)
        uint32_t asic_count = required_asics_map.at(mesh_type).at(mesh_name);

        // Get device topology dimensions for corner orientation assignment
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(mesh_instance.desc);
        TT_FATAL(mesh_desc != nullptr, "Mesh descriptor is null");
        const auto& device_topology = mesh_desc->device_topology();
        std::vector<int32_t> device_dims(device_topology.dims().begin(), device_topology.dims().end());

        // Create GroupingInfo
        GroupingInfo grouping_info;
        grouping_info.name = mesh_name;  // Keep original name for matching
        grouping_info.type = mesh_type;
        grouping_info.asic_count = asic_count;
        grouping_info.adjacency_graph = std::move(adjacency_graph);

        // Create a single item representing the mesh (for corner orientation assignment)
        // The item represents the entire mesh as a single unit
        GroupingItemInfo mesh_item;
        mesh_item.type = GroupingItemInfo::ItemType::GROUPING_REF;
        mesh_item.grouping_name = mesh_name;
        grouping_info.items.push_back(std::move(mesh_item));

        // Assign corner orientations based on mesh dimensions
        // For mesh instances with a single item, the helper function will assign corners appropriately
        PhysicalGroupingDescriptor::assign_corner_orientations_to_grouping(grouping_info, device_dims);

        // Store keyed by mesh definition name (not instance key)
        mgd_grouping_infos[mesh_type][mesh_name] = std::move(grouping_info);
    }

    // Process switch instances
    // Switches are treated as MESH type for grouping purposes
    // Store only one entry per switch definition name (SW0, SW1), not per instance (SW0_0, SW0_1, etc.)
    std::set<std::string> processed_switch_definitions;
    for (GlobalNodeId switch_id : mesh_graph_descriptor.all_switches()) {
        const auto& switch_instance = mesh_graph_descriptor.get_instance(switch_id);
        const std::string& switch_name = switch_instance.name;

        // Skip if we've already processed this switch definition
        if (processed_switch_definitions.contains(switch_name)) {
            continue;
        }
        processed_switch_definitions.insert(switch_name);

        // Build adjacency graph for this switch instance (use first instance of this switch definition)
        AdjacencyGraph<uint32_t> adjacency_graph =
            build_mgd_switch_instance_adjacency(mesh_graph_descriptor, switch_id);

        // Get required ASIC count (calculated above, stored under MESH type)
        uint32_t asic_count = required_asics_map.at("MESH").at(switch_name);

        // Get device topology dimensions for corner orientation assignment
        const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(switch_instance.desc);
        TT_FATAL(switch_desc != nullptr, "Switch descriptor is null");
        const auto& device_topology = switch_desc->device_topology();
        std::vector<int32_t> device_dims(device_topology.dims().begin(), device_topology.dims().end());

        // Create GroupingInfo
        GroupingInfo grouping_info;
        grouping_info.name = switch_name;  // Keep original name for matching
        grouping_info.type = "MESH";       // Switches are treated as MESH type
        grouping_info.asic_count = asic_count;
        grouping_info.adjacency_graph = std::move(adjacency_graph);

        // Create a single item representing the switch (for corner orientation assignment)
        // The item represents the entire switch as a single unit
        GroupingItemInfo switch_item;
        switch_item.type = GroupingItemInfo::ItemType::GROUPING_REF;
        switch_item.grouping_name = switch_name;
        grouping_info.items.push_back(std::move(switch_item));

        // Assign corner orientations based on switch dimensions
        // For switch instances with a single item, the helper function will assign corners appropriately
        PhysicalGroupingDescriptor::assign_corner_orientations_to_grouping(grouping_info, device_dims);

        // Store keyed by MESH type (switches are treated as MESH type)
        mgd_grouping_infos["MESH"][switch_name] = std::move(grouping_info);
    }

    // Process graph instances
    for (GlobalNodeId graph_id : mesh_graph_descriptor.all_graphs()) {
        const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_id);
        const std::string& graph_type = graph_instance.type;
        const std::string& graph_name = graph_instance.name;

        // Skip if already processed (same name/type)
        if (mgd_grouping_infos.contains(graph_type) && mgd_grouping_infos.at(graph_type).contains(graph_name)) {
            continue;
        }

        // Build adjacency graph for this graph instance
        AdjacencyGraph<uint32_t> adjacency_graph = build_mgd_graph_instance_adjacency(mesh_graph_descriptor, graph_id);

        // Get required ASIC count (calculated above)
        uint32_t asic_count = required_asics_map.at(graph_type).at(graph_name);

        // Create GroupingInfo
        GroupingInfo grouping_info;
        grouping_info.name = graph_name;
        grouping_info.type = graph_type;
        grouping_info.asic_count = asic_count;
        grouping_info.adjacency_graph = std::move(adjacency_graph);
        // items left empty - not needed for matching

        mgd_grouping_infos[graph_type][graph_name] = std::move(grouping_info);
    }

    return mgd_grouping_infos;
}

}  // namespace tt::tt_fabric

namespace {

// -----------------------------------------------------------------------------
// Phase 3: Higher-layer graph matching helpers
// -----------------------------------------------------------------------------

bool is_mgd_graph_ready(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const std::string& graph_name,
    const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<GroupingInfo>>>& result,
    const std::unordered_map<std::string, std::string>& known_mappings) {
    const auto& instance_ids = mesh_graph_descriptor.instances_by_name(graph_name);
    if (instance_ids.empty()) {
        return false;
    }
    const auto& graph_instance = mesh_graph_descriptor.get_instance(instance_ids[0]);
    for (GlobalNodeId sub_id : graph_instance.sub_instances) {
        const auto& sub_instance = mesh_graph_descriptor.get_instance(sub_id);
        if (!result.contains(sub_instance.type) && !known_mappings.contains(sub_instance.type)) {
            return false;
        }
    }
    return true;
}

bool mgd_graph_depends_on(
    const MeshGraphDescriptor& mesh_graph_descriptor, const std::string& dep_graph_name, const std::string& on_type) {
    const auto& instance_ids = mesh_graph_descriptor.instances_by_name(dep_graph_name);
    if (instance_ids.empty()) {
        return false;
    }
    const auto& graph_instance = mesh_graph_descriptor.get_instance(instance_ids[0]);
    for (GlobalNodeId sub_id : graph_instance.sub_instances) {
        const auto& sub_instance = mesh_graph_descriptor.get_instance(sub_id);
        if (sub_instance.type == on_type) {
            return true;
        }
    }
    return false;
}

bool pgd_grouping_depends_on(const GroupingInfo& pgd_grouping, const std::string& on_type) {
    for (const auto& item : pgd_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == on_type) {
            return true;
        }
    }
    return false;
}

void process_higher_layer_and_recurse(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>>& mgd_grouping_infos,
    const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<GroupingInfo>>>&
        resolved_groupings_cache_,
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<GroupingInfo>>>& result,
    std::unordered_map<std::string, std::string>& known_mappings,
    const std::string& mgd_type,
    const std::string& graph_name) {
    if (result.contains(mgd_type) && result.at(mgd_type).contains(graph_name)) {
        return;
    }

    const auto& instance_ids = mesh_graph_descriptor.instances_by_name(graph_name);
    if (instance_ids.empty()) {
        return;
    }
    GlobalNodeId repr_graph_id = instance_ids[0];
    if (!is_mgd_graph_ready(mesh_graph_descriptor, graph_name, result, known_mappings)) {
        return;
    }

    const auto& graph_instance = mesh_graph_descriptor.get_instance(repr_graph_id);

    std::unordered_set<std::string> allowed_pgd_child_types;
    for (GlobalNodeId sub_id : graph_instance.sub_instances) {
        const auto& sub_instance = mesh_graph_descriptor.get_instance(sub_id);
        auto it = known_mappings.find(sub_instance.type);
        if (it != known_mappings.end()) {
            allowed_pgd_child_types.insert(it->second);
        }
    }

    AdjacencyGraph<uint32_t> mgd_adjacency = build_mgd_graph_instance_adjacency(mesh_graph_descriptor, repr_graph_id);

    size_t mgd_nodes = mgd_adjacency.get_nodes().size();
    if (mgd_nodes == 0) {
        return;
    }

    std::vector<GroupingInfo> matches;
    for (const auto& [pgd_name, type_map] : resolved_groupings_cache_) {
        for (const auto& [pgd_type, pgd_groupings] : type_map) {
            if (pgd_type == "MESH") {
                continue;
            }
            for (const auto& pgd_grouping : pgd_groupings) {
                // PGD grouping must depend on one of the allowed child types
                bool depends_on_allowed = false;
                for (const std::string& allowed_type : allowed_pgd_child_types) {
                    if (pgd_grouping_depends_on(pgd_grouping, allowed_type)) {
                        depends_on_allowed = true;
                        break;
                    }
                }
                if (!depends_on_allowed) {
                    continue;
                }

                size_t pgd_nodes = pgd_grouping.adjacency_graph.get_nodes().size();
                if (pgd_nodes < mgd_nodes) {
                    continue;
                }

                auto mapping_result = solve_topology_mapping<uint32_t, uint32_t>(
                    mgd_adjacency, pgd_grouping.adjacency_graph, {}, ConnectionValidationMode::STRICT, true);

                if (mapping_result.success) {
                    matches.push_back(pgd_grouping);
                }
            }
        }
    }

    if (!matches.empty()) {
        const GroupingInfo* best = matches.data();
        for (const auto& m : matches) {
            if (m.adjacency_graph.get_nodes().size() == mgd_nodes) {
                best = &m;
                break;
            }
        }
        result[mgd_type][graph_name].push_back(*best);
        known_mappings[mgd_type] = best->type;
    } else {
        // No matches found - use the MGD grouping info itself
        auto mgd_it = mgd_grouping_infos.find(mgd_type);
        if (mgd_it != mgd_grouping_infos.end()) {
            auto instance_it = mgd_it->second.find(graph_name);
            if (instance_it != mgd_it->second.end()) {
                result[mgd_type][graph_name].push_back(instance_it->second);
            }
        }
    }

    for (const auto& [dep_mgd_type, dep_instances] : mgd_grouping_infos) {
        if (dep_mgd_type == "MESH") {
            continue;
        }
        for (const auto& [dep_graph_name, _] : dep_instances) {
            if (!mgd_graph_depends_on(mesh_graph_descriptor, dep_graph_name, mgd_type)) {
                continue;
            }
            if (!is_mgd_graph_ready(mesh_graph_descriptor, dep_graph_name, result, known_mappings)) {
                continue;
            }
            if (result.contains(dep_mgd_type) && result.at(dep_mgd_type).contains(dep_graph_name)) {
                continue;
            }
            process_higher_layer_and_recurse(
                mesh_graph_descriptor,
                mgd_grouping_infos,
                resolved_groupings_cache_,
                result,
                known_mappings,
                dep_mgd_type,
                dep_graph_name);
        }
    }
}

}  // namespace

namespace tt::tt_fabric {

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const {
    return get_valid_groupings_for_mgd(mesh_graph_descriptor, &physical_system_descriptor);
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) const {
    ValidGroupingsMap result;

    // ===== PHASE 0: Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts) =====
    // This step calculates required ASIC counts bottom-up and builds adjacency graphs
    log_info(tt::LogFabric, "Building MGD to GroupingInfo map");
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos =
        PhysicalGroupingDescriptor::build_mgd_to_grouping_info_map(mesh_graph_descriptor);

    // ===== PHASE 1: Build flattened adjacency graphs for all mesh group infos =====
    // Map from grouping name to vector of flattened GroupingInfo (supports multiple definitions with same name)
    std::unordered_map<std::string, std::vector<GroupingInfo>> mesh_flat_groupings;
    // Find MESH type groupings across all names
    log_info(tt::LogFabric, "Finding MESH type groupings across all names");
    bool found_mesh = false;
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        auto mesh_it = type_map.find("MESH");
        if (mesh_it != type_map.end()) {
            found_mesh = true;
            for (const auto& mesh_group_info : mesh_it->second) {
                auto meshes = build_flattened_adjacency_mesh(mesh_group_info, physical_system_descriptor);
                for (auto& meshe : meshes) {
                    mesh_flat_groupings[mesh_group_info.name].push_back(std::move(meshe));
                }
            }
        }
    }
    if (!found_mesh) {
        TT_THROW("Internal error: MESH grouping not found in resolved_groupings_cache_");
    }

    // ===== PHASE 2: Match MESH mgd groupings to MESH groupings =====
    // For each MGD mesh instance, find all valid PGD mesh groupings that can contain it
    log_info(tt::LogFabric, "Matching MESH mgd groupings to MESH groupings");
    for (const auto& [mgd_instance_key, mgd_mesh_grouping] : mgd_grouping_infos["MESH"]) {
        const std::string& instance_name = mgd_instance_key;  // Use unique instance key (includes mesh_id)
        const GroupingInfo& mgd_grouping_info = mgd_mesh_grouping;
        const std::string& instance_type = mgd_grouping_info.type;  // Should be "MESH"

        // Required nodes from MGD adjacency graph (this represents the topology pattern to match)
        size_t required_nodes = mgd_grouping_info.adjacency_graph.get_nodes().size();

        // Group valid candidates by node difference (map is ordered by key ascending)
        // Store (name, index) pairs to handle multiple groupings with same name
        log_info(tt::LogFabric, "Grouping valid candidates by node difference");
        std::map<size_t, std::vector<std::pair<std::string, size_t>>> candidates_by_diff;
        for (const auto& [name, grouping_infos] : mesh_flat_groupings) {
            for (size_t idx = 0; idx < grouping_infos.size(); ++idx) {
                const auto& grouping_info = grouping_infos[idx];
                size_t n = grouping_info.adjacency_graph.get_nodes().size();
                if (n >= required_nodes) {
                    candidates_by_diff[n - required_nodes].emplace_back(name, idx);
                }
            }
        }
        log_info(tt::LogFabric, "Found {} valid candidates by node difference", candidates_by_diff.size());

        // Process difference levels from closest to farthest; stop at first level with any match
        log_info(tt::LogFabric, "Processing difference levels from closest to farthest");
        std::vector<std::pair<std::string, size_t>> best_matches;
        for (const auto& [node_diff, name_idx_pairs] : candidates_by_diff) {
            (void)node_diff;
            for (const auto& [name, idx] : name_idx_pairs) {
                const auto& grouping_info = mesh_flat_groupings.at(name)[idx];
                // NOTE: If we ever want to support mixed type topologies, we need to add constraints to match the types
                MappingConstraints<uint32_t, uint32_t> constraints;
                constraints.add_required_constraint(0, 0);
                log_info(tt::LogFabric, "Solving topology mapping for {} and {}", mgd_grouping_info.name, name);
                auto mapping_result = solve_topology_mapping<uint32_t, uint32_t>(
                    mgd_grouping_info.adjacency_graph,
                    grouping_info.adjacency_graph,
                    constraints,
                    ConnectionValidationMode::STRICT,
                    true);
                if (mapping_result.success) {
                    log_info(
                        tt::LogFabric,
                        "Successfully solved topology mapping for {} and {}",
                        mgd_grouping_info.name,
                        name);
                    best_matches.emplace_back(name, idx);
                } else {
                    log_info(
                        tt::LogFabric,
                        "Failed to solve topology mapping for {} and {}, with error: {}",
                        mgd_grouping_info.name,
                        name,
                        mapping_result.error_message);
                }
            }
            if (!best_matches.empty()) {
                break;  // Found matches at this (best) level
            }
        }
        log_info(tt::LogFabric, "Found {} best matches", best_matches.size());

        // Store all best matches (add all entries that are possible)
        if (best_matches.empty()) {
            // No match found - use the MGD grouping info itself
            result[instance_type][instance_name].push_back(mgd_grouping_info);
        } else {
            for (const auto& [match_name, match_idx] : best_matches) {
                // Look up the flattened GroupingInfo from lookup map (already contains flattened adjacency graphs)
                auto lookup_it = mesh_flat_groupings.find(match_name);
                if (lookup_it != mesh_flat_groupings.end() && match_idx < lookup_it->second.size()) {
                    // Return the flattened GroupingInfo (not the original)
                    const GroupingInfo& flattened_grouping = lookup_it->second[match_idx];
                    result[instance_type][instance_name].push_back(flattened_grouping);
                }
            }
        }
    }

    // =============================================================================
    // Phase 3: Higher-layer graph matching (FABRIC, SUPER_FABRIC, etc.)
    // =============================================================================

    std::unordered_map<std::string, std::string> known_mappings;
    known_mappings["MESH"] = "MESH";

    log_info(tt::LogFabric, "Matching higher-layer graph mgd groupings to higher-layer groupings");
    for (const auto& [mgd_type, mgd_instances] : mgd_grouping_infos) {
        if (mgd_type == "MESH") {
            continue;
        }
        for (const auto& [graph_name, _] : mgd_instances) {
            if (!is_mgd_graph_ready(mesh_graph_descriptor, graph_name, result, known_mappings)) {
                continue;
            }
            if (!mgd_graph_depends_on(mesh_graph_descriptor, graph_name, "MESH")) {
                continue;
            }
            process_higher_layer_and_recurse(
                mesh_graph_descriptor,
                mgd_grouping_infos,
                resolved_groupings_cache_,
                result,
                known_mappings,
                mgd_type,
                graph_name);
        }
    }

    // Ensure all types and instances from MGD have entries in result
    // Use MGD grouping info if no matches were found
    log_info(tt::LogFabric, "Ensuring all types and instances from MGD have entries in result");
    for (const auto& [mgd_type, mgd_instances] : mgd_grouping_infos) {
        for (const auto& [instance_name, mgd_grouping_info] : mgd_instances) {
            // If not already present, use the MGD grouping info
            if (!result[mgd_type].contains(instance_name)) {
                result[mgd_type][instance_name].push_back(mgd_grouping_info);
            }
        }
    }

    log_info(tt::LogFabric, "Returning valid groupings map");
    return result;
}

}  // namespace tt::tt_fabric

namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::ASICLocation;
using tt::tt_metal::TrayID;
using tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd;
using tt::tt_metal::experimental::tt_fabric::PhysicalAdjacencyMap;

// Host boundaries come only from the PSD (get_host_name_for_asic). Global groups are one set per host (variable
// size). One PGD mesh target group: hard same-rank if some host has >= mesh ASICs; otherwise preferred ASICs on a
// greedy minimal set of largest hosts (by ASIC count) to bias toward fewer cross-host hops.
void configure_pgd_psd_host_alignment_constraints(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<uint32_t, AsicID>& constraints) {
    // Collect hostname map for all asics in physical graph
    std::map<std::string, std::set<AsicID>> host_to_asics;
    for (const AsicID& asic_id : physical_graph.get_nodes()) {
        host_to_asics[physical_system_descriptor.get_host_name_for_asic(asic_id)].insert(asic_id);
    }

    // Collect all targets from PGD grouping info
    std::set<uint32_t> all_targets;
    for (uint32_t node_id : grouping_info.adjacency_graph.get_nodes()) {
        if (node_id >= grouping_info.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = grouping_info.items[node_id];
        if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
            continue;
        }
        all_targets.insert(node_id);
    }

    if (all_targets.empty()) {
        return;
    }
    if (host_to_asics.size() <= 1) {
        return;
    }

    std::vector<std::set<AsicID>> global_groups;
    global_groups.reserve(host_to_asics.size());
    for (auto& [_, asics] : host_to_asics) {
        if (!asics.empty()) {
            global_groups.push_back(std::move(asics));
        }
    }

    const auto [single_group_fits, preferred_globals] =
        ::tt::tt_fabric::PhysicalGroupingDescriptor::find_minimum_coverage_group(all_targets, global_groups);
    if (single_group_fits) {
        std::vector<std::set<uint32_t>> target_groups;
        target_groups.push_back(all_targets);
        if (constraints.set_same_rank_groups_constraint(target_groups, global_groups)) {
            return;
        }
        log_warning(
            tt::LogFabric,
            "PGD host alignment: failed to set same-rank groups constraint; falling back to preferred globals");
    }
    if (!preferred_globals.empty()) {
        if (!single_group_fits) {
            log_debug(
                tt::LogFabric,
                "PGD host alignment: target count {} exceeds largest single partition; preferring minimal host cover "
                "({} preferred globals)",
                all_targets.size(),
                preferred_globals.size());
        }
        for (const uint32_t& target : all_targets) {
            constraints.add_preferred_constraint(target, preferred_globals);
        }
    }
}

std::string build_pgd_mapping_failure_message(
    const std::string& grouping_name,
    const GroupingInfo& grouping_info,
    const MappingResult<uint32_t, AsicID>& result) {
    size_t total = grouping_info.adjacency_graph.get_nodes().size();
    size_t mapped_count = result.target_to_global.size();
    size_t unmapped_count = total - mapped_count;

    return fmt::format(
        "PGD grouping '{}' could not be mapped to PSD: {}/{} nodes mapped, {} unmatched",
        grouping_name,
        mapped_count,
        total,
        unmapped_count);
}

// Helper function to solve the topology mapping with pinning constraints from GroupingInfo
MappingResult<uint32_t, AsicID> solve_for_one_grouping_to_psd(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<uint32_t, AsicID> initial_constraints = {}) {
    MappingConstraints<uint32_t, AsicID> constraints = std::move(initial_constraints);

    // Set quiet mode to suppress verbose constraint validation messages during PGD solving
    constraints.set_quiet_mode(true);

    // Build trait maps: graph nodes are 0..n-1, items[i] is the item for node i
    std::map<uint32_t, TrayID> target_tray_traits;
    std::map<uint32_t, ASICLocation> target_location_traits;

    for (uint32_t node_id : grouping_info.adjacency_graph.get_nodes()) {
        if (node_id >= grouping_info.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = grouping_info.items[node_id];
        if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
            continue;
        }
        if (*item.tray_id > 0) {
            target_tray_traits[node_id] = item.tray_id;
        }
        // Skip ASIC_LOCATION_UNSPECIFIED (256) - it means "any ASIC ID" (no constraint)
        // Only add constraint for specified ASIC locations (0-8)
        if (*item.asic_location <= 8) {
            target_location_traits[node_id] = item.asic_location;
        }
    }
    // Build trait maps for global nodes (from physical graph)
    std::map<AsicID, TrayID> global_tray_traits;
    std::map<AsicID, ASICLocation> global_location_traits;

    for (const auto& asic_id : physical_graph.get_nodes()) {
        TrayID tray_id = physical_system_descriptor.get_tray_id(asic_id);
        ASICLocation asic_location = physical_system_descriptor.get_asic_location(asic_id);
        global_tray_traits[asic_id] = tray_id;
        global_location_traits[asic_id] = asic_location;
    }

    // Add trait constraints for tray_id and asic_location
    if (!target_tray_traits.empty() && !global_tray_traits.empty()) {
        if (!constraints.add_required_trait_constraint<TrayID>(target_tray_traits, global_tray_traits)) {
            MappingResult<uint32_t, AsicID> failure;
            failure.success = false;
            failure.error_message = "Failed to add required trait constraint for tray_id";
            return failure;
        }
    }
    if (!target_location_traits.empty() && !global_location_traits.empty()) {
        if (!constraints.add_required_trait_constraint<ASICLocation>(target_location_traits, global_location_traits)) {
            MappingResult<uint32_t, AsicID> failure;
            failure.success = false;
            failure.error_message = "Failed to add required trait constraint for asic_location";
            return failure;
        }
    }

    // PSD-only host partition (ASIC -> hostname): same-rank when the full mesh fits on one host, else unconstrained.
    configure_pgd_psd_host_alignment_constraints(
        grouping_info, physical_graph, physical_system_descriptor, constraints);

    return solve_topology_mapping(
        grouping_info.adjacency_graph, physical_graph, constraints, ConnectionValidationMode::RELAXED, true);
}

bool is_flattened(const GroupingInfo& grouping) {
    return grouping.asic_count == grouping.adjacency_graph.get_nodes().size();
}

}  // namespace

namespace tt::tt_fabric {

// Maximum placements to find before stopping (safeguard against infinite loops).
constexpr size_t kMaxPlacementsPerRun = 10000;
// TODO: Optimize constraints for maximum usage
// https://github.com/tenstorrent/tt-metal/issues/40639
std::vector<MappingResult<uint32_t, AsicID>> solve_for_many_groupings_to_psd(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    const AdjacencyGraph<uint32_t>& flat_mesh = grouping_info.adjacency_graph;

    // Check if mesh is empty
    size_t flat_mesh_size = flat_mesh.get_nodes().size();
    if (flat_mesh_size == 0) {
        return {};
    }

    // Iteratively solve for copies until no more can be found
    std::vector<MappingResult<uint32_t, AsicID>> results;
    MappingConstraints<uint32_t, AsicID> current_constraints;
    std::set<std::set<AsicID>> seen_asic_sets;  // Guard against infinite loop

    while (results.size() < kMaxPlacementsPerRun) {
        MappingResult<uint32_t, AsicID> result = solve_for_one_grouping_to_psd(
            grouping_info, physical_graph, physical_system_descriptor, current_constraints);

        if (!result.success) {
            break;
        }

        std::set<AsicID> used_asic_ids;
        for (const auto& [_, asic_id] : result.target_to_global) {
            used_asic_ids.insert(asic_id);
        }

        results.push_back(result);

        std::set<uint32_t> all_target_nodes(flat_mesh.get_nodes().begin(), flat_mesh.get_nodes().end());
        TT_FATAL(
            current_constraints.add_forbidden_constraint(all_target_nodes, used_asic_ids),
            "Homogeneous solver: failed to add forbidden constraints to all groupings");
    }

    return results;
}

// Heterogeneous version: pack multiple different grouping types onto the physical graph.
// Each grouping can have a different topology. ASICs are shared globally - no overlap between any mappings.
// Returns map from each GroupingInfo to its vector of mapping results.
// Uses the same constraint pattern as homogeneous: add forbidden constraints after each success.
// TODO: Look into changing this algoirthm to use simulated annealing
// https://github.com/tenstorrent/tt-metal/issues/40639
std::unordered_map<const GroupingInfo*, std::vector<MappingResult<uint32_t, AsicID>>>
solve_for_many_groupings_to_psd_heterogeneous(
    const std::vector<GroupingInfo>& groupings,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    std::vector<std::vector<MappingResult<uint32_t, AsicID>>> results(groupings.size());
    MappingConstraints<uint32_t, AsicID> current_constraints;

    std::set<uint32_t> all_target_nodes_union;
    for (const auto& grouping : groupings) {
        for (uint32_t n : grouping.adjacency_graph.get_nodes()) {
            all_target_nodes_union.insert(n);
        }
    }

    while (true) {
        size_t total_results = 0;
        for (const auto& r : results) {
            total_results += r.size();
        }
        if (total_results >= kMaxPlacementsPerRun) {
            log_warning(
                tt::LogFabric,
                "Heterogeneous solver: hit max placements limit ({}) - stopping to prevent infinite loop",
                kMaxPlacementsPerRun);
            break;
        }
        bool found_any = false;
        bool overconstrained = false;
        for (size_t i = 0; const auto& grouping : groupings) {
            const AdjacencyGraph<uint32_t>& flat_mesh = grouping.adjacency_graph;

            if (flat_mesh.get_nodes().empty()) {
                ++i;
                continue;
            }

            MappingResult<uint32_t, AsicID> result = solve_for_one_grouping_to_psd(
                grouping, physical_graph, physical_system_descriptor, current_constraints);

            if (!result.success) {
                ++i;
                continue;
            }

            std::set<AsicID> used_asic_ids;
            for (const auto& [_, asic_id] : result.target_to_global) {
                used_asic_ids.insert(asic_id);
            }

            results[i].push_back(result);
            found_any = true;

            TT_FATAL(
                current_constraints.add_forbidden_constraint(all_target_nodes_union, used_asic_ids),
                "Internal Error: Heterogeneous solver: failed to add forbidden constraints to all groupings");
        }
        if (!found_any || overconstrained) {
            break;
        }
    }

    std::unordered_map<const GroupingInfo*, std::vector<MappingResult<uint32_t, AsicID>>> map_result;
    for (size_t i = 0; i < groupings.size(); ++i) {
        map_result.emplace(&groupings[i], std::move(results[i]));
    }
    return map_result;
}
}  // namespace tt::tt_fabric

std::unordered_set<tt::tt_metal::AsicID> PhysicalGroupingDescriptor::find_any_in_psd(
    const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const {
    std::vector<std::string> errors;
    return find_any_in_psd(grouping, physical_system_descriptor, errors);
}

// NOTE this only works on flattenable meshes right now
// TODO: Expand Find any to non-flattenable meshes by doing recursive mapping
std::unordered_set<tt::tt_metal::AsicID> PhysicalGroupingDescriptor::find_any_in_psd(
    const GroupingInfo& grouping,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::vector<std::string>& errors_out) const {
    // Build physical adjacency map from PSD (empty map means include all ASICs)
    PhysicalAdjacencyMap physical_adj_map = build_flat_adjacency_map_from_psd(physical_system_descriptor);
    // Convert to AdjacencyGraph
    AdjacencyGraph<AsicID> physical_graph(physical_adj_map);

    // Detect if its flattened or not, if it its not then flatten it
    std::vector<GroupingInfo> flat_meshes;
    if (!is_flattened(grouping)) {
        flat_meshes = build_flattened_adjacency_mesh(grouping, physical_system_descriptor);
    } else {
        flat_meshes.push_back(grouping);
    }

    std::unordered_set<tt::tt_metal::AsicID> asic_ids;
    const GroupingInfo* last_mesh_tried = nullptr;
    MappingResult<uint32_t, AsicID> last_result;

    // Use the first flat mesh that actually fits
    for (const auto& flat_mesh : flat_meshes) {
        if (flat_mesh.adjacency_graph.get_nodes().empty()) {
            continue;
        }

        last_mesh_tried = &flat_mesh;
        // solve_for_one_grouping_to_psd uses items[node_id] for trait constraints
        auto result = solve_for_one_grouping_to_psd(flat_mesh, physical_graph, physical_system_descriptor);
        last_result = result;

        if (result.success) {
            for (const auto& [target_node, asic_id] : result.target_to_global) {
                asic_ids.insert(asic_id);
            }
            return asic_ids;
        }
    }

    // If flat_meshes is empty, it means PSD filtering removed all possibilities
    // This is a valid case (grouping can't be mapped to this PSD), not an internal error
    if (flat_meshes.empty()) {
        // Return empty set - grouping cannot be mapped to this PSD
        return asic_ids;
    }

    // Check if there's an actual internal error (all meshes have empty graphs)
    bool all_empty = true;
    for (const auto& flat_mesh : flat_meshes) {
        if (!flat_mesh.adjacency_graph.get_nodes().empty()) {
            all_empty = false;
            break;
        }
    }

    if (all_empty) {
        TT_THROW("Internal error: grouping produced empty graph");
    }

    if (last_mesh_tried != nullptr) {
        errors_out.push_back(build_pgd_mapping_failure_message(grouping.name, *last_mesh_tried, last_result));
    }

    return asic_ids;
}

std::vector<std::unordered_set<tt::tt_metal::AsicID>> PhysicalGroupingDescriptor::find_all_in_psd(
    const std::vector<GroupingInfo>& groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const {
    std::vector<std::string> errors;
    return find_all_in_psd(groupings, physical_system_descriptor, errors);
}

std::vector<std::unordered_set<tt::tt_metal::AsicID>> PhysicalGroupingDescriptor::find_all_in_psd(
    const std::vector<GroupingInfo>& groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const AdjacencyGraph<AsicID>& physical_graph) const {
    std::vector<std::string> errors;
    return find_all_in_psd(groupings, physical_system_descriptor, physical_graph, errors);
}

std::vector<std::unordered_set<tt::tt_metal::AsicID>> PhysicalGroupingDescriptor::find_all_in_psd(
    const std::vector<GroupingInfo>& groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::vector<std::string>& errors_out) const {
    PhysicalAdjacencyMap physical_adj_map = build_flat_adjacency_map_from_psd(physical_system_descriptor);
    AdjacencyGraph<AsicID> physical_graph(physical_adj_map);
    return find_all_in_psd(groupings, physical_system_descriptor, physical_graph, errors_out);
}

// NOTE this only works on flattenable meshes right now
std::vector<std::unordered_set<tt::tt_metal::AsicID>> PhysicalGroupingDescriptor::find_all_in_psd(
    const std::vector<GroupingInfo>& groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const AdjacencyGraph<AsicID>& physical_graph,
    std::vector<std::string>& errors_out) const {
    // Flatten each grouping and collect all non-empty flat meshes
    std::vector<GroupingInfo> flat_meshes;
    for (const auto& grouping : groupings) {
        auto flattened = is_flattened(grouping) ? std::vector<GroupingInfo>{grouping}
                                                : build_flattened_adjacency_mesh(grouping, physical_system_descriptor);
        for (const auto& f : flattened) {
            if (!f.adjacency_graph.get_nodes().empty()) {
                flat_meshes.push_back(f);
            }
        }
    }

    std::vector<std::unordered_set<tt::tt_metal::AsicID>> all_asic_id_sets;
    if (!flat_meshes.empty()) {
        auto heterogeneous_results =
            solve_for_many_groupings_to_psd_heterogeneous(flat_meshes, physical_graph, physical_system_descriptor);

        for (const auto& grouping : flat_meshes) {
            auto it = heterogeneous_results.find(&grouping);
            if (it == heterogeneous_results.end()) {
                continue;
            }
            for (const auto& result : it->second) {
                if (result.success) {
                    std::unordered_set<tt::tt_metal::AsicID> asic_set;
                    for (const auto& [target_node, asic_id] : result.target_to_global) {
                        asic_set.insert(asic_id);
                    }
                    all_asic_id_sets.push_back(std::move(asic_set));
                }
            }
        }
    }

    // If no mappings found, populate errors
    if (all_asic_id_sets.empty()) {
        if (flat_meshes.empty()) {
            errors_out.push_back("No valid groupings found for PSD");
        } else {
            const GroupingInfo& mesh_to_use = flat_meshes.back();
            auto result = solve_for_one_grouping_to_psd(mesh_to_use, physical_graph, physical_system_descriptor);
            errors_out.push_back(build_pgd_mapping_failure_message(mesh_to_use.name, mesh_to_use, result));
        }
    }

    return all_asic_id_sets;
}
