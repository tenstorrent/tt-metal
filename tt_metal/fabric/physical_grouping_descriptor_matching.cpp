// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <ostream>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <queue>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <span>
#include <vector>
#include <tt_stl/fmt.hpp>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "protobuf/physical_grouping_descriptor.pb.h"
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-logger/tt-logger.hpp>
#include <map>

#include <google/protobuf/text_format.h>

using namespace tt::tt_fabric;

namespace {

// TEMPORARY(2x2-4x1-cross): true iff one of {MGD mesh, grouping} is a 2x2 and the other a 4x1 (by
// non-trivial declared dims). A [2,2] ring and a [4,1] ring are the same 4-cycle, so the matcher would
// otherwise swap them; a 4x1 strip for a [2,2] mesh straddles the tray boundary (TestGalaxyLayoutCheck).
// TODO(2x2-4x1-cross): remove once shape/topology disambiguation is stable.
bool is_2x2_4x1_cross(const std::optional<DeclaredTopology>& device_topo, const GroupingInfo& grouping) {
    auto shape = [](const std::vector<int32_t>& dims) {
        std::vector<int32_t> v;
        for (int32_t d : dims) {
            if (d > 1) {
                v.push_back(d);
            }
        }
        std::sort(v.begin(), v.end());
        return v;
    };
    const std::vector<int32_t> mgd = device_topo ? shape(device_topo->dims) : std::vector<int32_t>{};
    const std::vector<int32_t> grp = shape(grouping.flattened_node_grid_dims);
    const std::vector<int32_t> s2x2 = {2, 2};
    const std::vector<int32_t> s4x1 = {4};
    return (mgd == s2x2 && grp == s4x1) || (mgd == s4x1 && grp == s2x2);
}

GroupingInfo finalize_mesh_grouping_with_device_topology(
    const GroupingInfo& grouping,
    const DeclaredTopology& device_topo,
    const std::map<LogicalChipId, GroupingChipId>* mgd_to_pgd_nodes = nullptr) {
    const bool has_ring =
        std::any_of(device_topo.ring_dims.begin(), device_topo.ring_dims.end(), [](bool is_ring) { return is_ring; });
    if (!has_ring) {
        return grouping;
    }

    int64_t num_nodes = 1;
    for (int32_t dim : device_topo.dims) {
        num_nodes *= dim;
    }

    std::vector<GroupingChipId> node_ids;
    node_ids.reserve(static_cast<size_t>(num_nodes));
    if (mgd_to_pgd_nodes == nullptr) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(num_nodes); ++i) {
            node_ids.push_back(i);
        }
    } else {
        for (uint32_t mgd_id = 0; mgd_id < static_cast<uint32_t>(num_nodes); ++mgd_id) {
            auto it = mgd_to_pgd_nodes->find(mgd_id);
            TT_FATAL(
                it != mgd_to_pgd_nodes->end(),
                "Grouping '{}' is missing topology mapping for MGD node {}",
                grouping.name,
                mgd_id);
            node_ids.push_back(it->second);
        }
    }

    GroupingInfo result = grouping;
    result.adjacency_graph =
        build_row_major_mesh_graph(node_ids, device_topo.dims, grouping.name, 1, device_topo.ring_dims);
    // The finalized grouping represents exactly the device-topology nodes. When the source PGD grouping is
    // larger than the MGD mesh (node_diff > 0, e.g. a 4x8 PGD candidate matched to a 4x4 mesh), it carries a
    // larger asic_count; reset it to the node count so the grouping stays self-consistent.
    result.asic_count = static_cast<uint32_t>(num_nodes);
    return result;
}

struct MeshTopologyMatch {
    std::string name;
    size_t idx = 0;
    MappingResult<LogicalChipId, GroupingChipId> mapping;
};

// Helper function to build adjacency graph from MGD mesh instance's device topology
// Builds a row-major mesh graph based on the mesh's device_topology dims
// This represents the topology at the ASIC level, which matches the flattened physical grouping graphs
AdjacencyGraph<GroupingChipId> build_mgd_mesh_instance_adjacency(
    const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId mesh_instance_id) {
    const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_instance_id);
    TT_FATAL(mesh_instance.kind == NodeKind::Mesh, "build_mgd_mesh_instance_adjacency called on non-mesh instance");

    // Device topology dimensions represent the ASIC-level layout
    const auto declared = mesh_graph_descriptor.get_declared_topology(mesh_instance);
    const std::vector<int32_t>& device_dims = declared.dims;

    if (device_dims.empty()) {
        // No device topology - return empty graph
        return AdjacencyGraph<GroupingChipId>();
    }

    // Calculate number of ASICs
    int32_t num_asics = 1;
    for (int32_t dim : device_dims) {
        num_asics *= dim;
    }

    // Create abstract ASIC node IDs (0, 1, 2, ..., num_asics-1)
    std::vector<GroupingChipId> asic_ids;
    asic_ids.reserve(num_asics);
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_asics); ++i) {
        asic_ids.push_back(i);
    }

    // Build the graph with the MGD's declared per-dimension topology (RING vs LINE). Using the real wrap
    // edges is what restricts the match to the correct PGD topology variant: a RING/RING MGD is a torus, so
    // only the TORUSXY variant contains all its wrap edges and matches, while MESH/TORUSX/TORUSY (missing
    // some wraps) correctly fail to match. (A LINE-only graph here would embed in every variant.)
    return build_row_major_mesh_graph(asic_ids, device_dims, "", 1, declared.ring_dims);
}

// Helper function to build adjacency graph from MGD switch instance
// Similar to build_mgd_mesh_instance_adjacency - builds row-major mesh graph from device_topology
AdjacencyGraph<GroupingChipId> build_mgd_switch_instance_adjacency(
    const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId switch_instance_id) {
    const auto& switch_instance = mesh_graph_descriptor.get_instance(switch_instance_id);
    TT_FATAL(
        switch_instance.kind == NodeKind::Switch, "build_mgd_switch_instance_adjacency called on non-switch instance");

    // Device topology dimensions represent the ASIC-level layout
    const std::vector<int32_t> device_dims = mesh_graph_descriptor.get_declared_topology(switch_instance).dims;

    if (device_dims.empty()) {
        // No device topology - return empty graph
        return AdjacencyGraph<GroupingChipId>();
    }

    // Calculate number of ASICs
    int32_t num_asics = 1;
    for (int32_t dim : device_dims) {
        num_asics *= dim;
    }

    // Create abstract ASIC node IDs (0, 1, 2, ..., num_asics-1)
    std::vector<GroupingChipId> asic_ids;
    asic_ids.reserve(num_asics);
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_asics); ++i) {
        asic_ids.push_back(i);
    }

    // LINE-only graph for topology matching (RING edges are added when groupings are committed).
    return build_row_major_mesh_graph(asic_ids, device_dims, "", 1);
}

// Helper function to build adjacency graph from MGD graph instance
// The graph instance's sub_instances become nodes, and connections between them become edges
// Ensures no duplicate connections and all connections are bidirectional
AdjacencyGraph<GroupingChipId> build_mgd_graph_instance_adjacency(
    const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId graph_instance_id) {
    const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_instance_id);

    // Get all sub-instances (these will be the nodes in our adjacency graph)
    std::vector<GroupingChipId> sub_instance_ids(
        graph_instance.sub_instances.begin(), graph_instance.sub_instances.end());

    // Build adjacency map from connections
    std::map<GroupingChipId, std::vector<GroupingChipId>> adj_map;

    // Initialize adjacency map for all sub-instances
    for (GroupingChipId sub_id : sub_instance_ids) {
        adj_map[sub_id] = std::vector<GroupingChipId>();
    }

    // Use a set to track processed edges to avoid duplicates
    std::set<std::pair<GroupingChipId, GroupingChipId>> processed_edges;

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

    return AdjacencyGraph<GroupingChipId>(adj_map);
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

    // Step 1b: Calculate required ASICs for all switch instances (bottom level)
    // Switches are treated as MESH type for grouping purposes
    for (GlobalNodeId switch_id : mesh_graph_descriptor.all_switches()) {
        const auto& switch_instance = mesh_graph_descriptor.get_instance(switch_id);
        uint32_t required_chips = mesh_graph_descriptor.get_switch_chip_count(switch_id);
        // Store switches under MESH type (switches are treated as MESH type)
        required_asics_map["MESH"][switch_instance.name] = required_chips;
    }

    // Step 1c: Calculate required ASICs for graph instances bottom-up (children before parents)
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
        AdjacencyGraph<GroupingChipId> adjacency_graph =
            build_mgd_mesh_instance_adjacency(mesh_graph_descriptor, mesh_id);

        // Get required ASIC count (calculated above)
        uint32_t asic_count = required_asics_map.at(mesh_type).at(mesh_name);

        // Get device topology dimensions for corner orientation assignment
        const std::vector<int32_t> device_dims = mesh_graph_descriptor.get_declared_topology(mesh_instance).dims;

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
        AdjacencyGraph<GroupingChipId> adjacency_graph =
            build_mgd_switch_instance_adjacency(mesh_graph_descriptor, switch_id);

        // Get required ASIC count (calculated above, stored under MESH type)
        uint32_t asic_count = required_asics_map.at("MESH").at(switch_name);

        // Get device topology dimensions for corner orientation assignment
        const std::vector<int32_t> device_dims = mesh_graph_descriptor.get_declared_topology(switch_instance).dims;

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
        AdjacencyGraph<GroupingChipId> adjacency_graph =
            build_mgd_graph_instance_adjacency(mesh_graph_descriptor, graph_id);

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

    AdjacencyGraph<GroupingChipId> mgd_adjacency =
        build_mgd_graph_instance_adjacency(mesh_graph_descriptor, repr_graph_id);

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

                auto mapping_result = solve_topology_mapping<GroupingChipId, GroupingChipId>(
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
                log_info(
                    tt::LogFabric,
                    "Physical groupings: Mesh graph descriptor {} '{}': 0 topology match(es), fallback to Mesh graph "
                    "descriptor: {} ({})",
                    mgd_type,
                    graph_name,
                    instance_it->second.name,
                    instance_it->second.type);
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

namespace {

std::set<uint32_t> get_mesh_ids_for_mgd_instance_name(
    const MeshGraphDescriptor& mesh_graph_descriptor, const std::string& instance_name) {
    std::set<uint32_t> mesh_ids;
    const auto& instance_ids = mesh_graph_descriptor.instances_by_name(instance_name);
    for (const GlobalNodeId global_id : instance_ids) {
        const auto& instance = mesh_graph_descriptor.get_instance(global_id);
        if (instance.kind == NodeKind::Mesh || instance.kind == NodeKind::Switch) {
            mesh_ids.insert(instance.local_id);
        }
    }
    return mesh_ids;
}

std::set<GroupingChipId> find_pgd_nodes_at_asic_position(
    const GroupingInfo& pgd_grouping, const tt::tt_metal::ASICPosition& position) {
    std::set<GroupingChipId> pgd_nodes;
    for (const GroupingChipId node_id : pgd_grouping.adjacency_graph.get_nodes()) {
        if (node_id >= pgd_grouping.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = pgd_grouping.items[node_id];
        if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
            continue;
        }
        if (item.tray_id == position.first && item.asic_location == position.second) {
            pgd_nodes.insert(node_id);
        }
    }
    return pgd_nodes;
}

// Compose logical chip_id -> PGD slot (TrayID + ASICLocation) from an MGD<->PGD topology match and the PGD
// grouping's per-node item labels. Called at PGD<->MGD commit time in get_valid_groupings_for_mgd.
std::map<LogicalChipId, tt::tt_metal::ASICPosition> compose_mesh_node_to_asic_position_from_pgd_match(
    const GroupingInfo& grouping, const std::map<LogicalChipId, GroupingChipId>& mgd_node_to_grouping_node) {
    std::map<LogicalChipId, tt::tt_metal::ASICPosition> node_to_position;
    for (const auto& [mgd_node, grouping_node] : mgd_node_to_grouping_node) {
        if (grouping_node >= grouping.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = grouping.items[grouping_node];
        if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
            continue;
        }
        node_to_position.emplace(mgd_node, tt::tt_metal::ASICPosition{item.tray_id, item.asic_location});
    }
    return node_to_position;
}

// Compose logical chip id -> MeshGraph host partition from the MGD host_topology. Empty only when the MGD
// declared no host topology at all, so an empty result means "nothing was declared" rather than "one host".
// Called at PGD<->MGD commit time in get_valid_groupings_for_mgd.
//
// The split is pulled directly from the MeshGraph host ranks (fabric_node_id_to_mesh_rank), the authoritative
// per-chip host_topology tiling built by MeshGraph::get_host_rank_for_chip. The PGD match pairing is not
// applied: rank-binding yaml groups follow those MeshGraph tiles. Fitting each tile inside a PGD HOST is
// enforced separately by configure_mgd_pgd_host_alignment_constraints.
std::map<LogicalChipId, uint32_t> compose_mesh_node_to_host_group_from_mgd_match(
    const std::optional<DeclaredTopology>& mgd_topo,
    const std::map<LogicalChipId, GroupingChipId>& mgd_node_to_grouping_node,
    MeshId mesh_id,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    std::map<LogicalChipId, uint32_t> node_to_host_group;
    if (!mgd_topo.has_value() || mgd_topo->host_dims.empty() || mgd_topo->dims.empty()) {
        return node_to_host_group;
    }

    // host_topology [1,1] is not "no opinion": it declares one rank owning the whole mesh. The MeshGraph rank
    // map already tiles it (every chip -> host_rank 0 for [1,1]), giving a single group that must land inside a
    // single host. When the MeshGraph produced no ranks for this mesh there is nothing to enforce -> empty.
    const auto rank_it = fabric_node_id_to_mesh_rank.find(mesh_id);
    if (rank_it == fabric_node_id_to_mesh_rank.end()) {
        return node_to_host_group;
    }
    for (const auto& [mgd_node, unused_grouping_node] : mgd_node_to_grouping_node) {
        (void)unused_grouping_node;
        const auto chip_it = rank_it->second.find(FabricNodeId{mesh_id, mgd_node});
        if (chip_it != rank_it->second.end()) {
            node_to_host_group.emplace(mgd_node, chip_it->second.get());
        }
    }
    return node_to_host_group;
}

// The MGD<->PGD half of the host contract: the MGD's declared host ranks held against the host division this PGD
// variant sits on, before the match is chosen. Hard, and only in this direction -- each declared rank must be
// carvable inside one of the descriptor's hosts, while ranks are free to share one, so a host_topology finer than
// the descriptor's hosts stays legal and what is refused is a rank whose chips would come from two hosts. That is
// the same rule configure_pgd_psd_host_alignment_constraints applies against the PSD's hosts, moved to where the
// orientation is still open, so a rank straddling a boundary is steered away from rather than found afterwards.
//
// Returns false when no orientation can satisfy it, which retires this variant. Inert unless both sides have
// spoken: a variant with no declared hosts under it, or an MGD with no declared split, constrains nothing.
bool configure_mgd_pgd_host_alignment_constraints(
    const GroupingInfo& mgd_grouping_info,
    const GroupingInfo& grouping_info,
    const std::optional<DeclaredTopology>& mgd_topo,
    MappingConstraints<LogicalChipId, GroupingChipId>& constraints,
    MeshId mesh_id,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    const bool nothing_declared = grouping_info.mesh_node_to_pgd_host_group.empty() || !mgd_topo.has_value() ||
                                  mgd_topo->dims.empty() || mgd_topo->host_dims.empty();
    if (nothing_declared) {
        return true;
    }

    // MGD host ranks are the MeshGraph host_topology tiling (fabric_node_id_to_mesh_rank). No mesh-graph ranks
    // for this mesh -> nothing to align against, so this direction of the contract is inert.
    const auto rank_it = fabric_node_id_to_mesh_rank.find(mesh_id);
    if (rank_it == fabric_node_id_to_mesh_rank.end()) {
        return true;
    }
    std::map<uint32_t, std::set<LogicalChipId>> mgd_nodes_by_rank;
    for (LogicalChipId mgd_node : mgd_grouping_info.adjacency_graph.get_nodes()) {
        const auto chip_it = rank_it->second.find(FabricNodeId{mesh_id, mgd_node});
        if (chip_it != rank_it->second.end()) {
            mgd_nodes_by_rank[chip_it->second.get()].insert(mgd_node);
        }
    }
    std::map<uint32_t, std::set<GroupingChipId>> pgd_nodes_by_host;
    for (const auto& [pgd_node, pgd_host] : grouping_info.mesh_node_to_pgd_host_group) {
        pgd_nodes_by_host[pgd_host].insert(pgd_node);
    }

    std::vector<std::set<LogicalChipId>> target_groups;
    target_groups.reserve(mgd_nodes_by_rank.size());
    for (auto& [_, rank_nodes] : mgd_nodes_by_rank) {
        target_groups.push_back(std::move(rank_nodes));
    }
    std::vector<std::set<GroupingChipId>> global_groups;
    global_groups.reserve(pgd_nodes_by_host.size());
    for (auto& [_, host_nodes] : pgd_nodes_by_host) {
        global_groups.push_back(std::move(host_nodes));
    }

    if (!constraints.set_same_rank_groups_constraint(target_groups, global_groups)) {
        log_debug(
            tt::LogFabric,
            "Host alignment retires '{}' for '{}': its {} declared rank(s) do not fit the {} host(s) this variant "
            "sits on",
            grouping_info.name,
            mgd_grouping_info.name,
            target_groups.size(),
            global_groups.size());
        return false;
    }
    return true;
}

// Applies MGD pinning groups as required constraints. `resolve_globals_at_position` maps each
// pinned ASIC position to the globals that occupy it (PGD grouping nodes, or PSD AsicIDs).
template <typename GlobalNode, typename ResolveGlobals>
std::size_t add_mgd_asic_position_pinning_constraints(
    MappingConstraints<LogicalChipId, GlobalNode>& constraints,
    const std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>& pinnings,
    const ResolveGlobals& resolve_globals_at_position) {
    std::size_t constraints_added = 0;
    for (const auto& group : pinnings) {
        std::set<LogicalChipId> mgd_nodes;
        std::set<GlobalNode> globals;
        for (const auto& fabric_node : group.fabric_nodes) {
            mgd_nodes.insert(fabric_node.chip_id);
        }
        for (const auto& position : group.asic_positions) {
            const auto found = resolve_globals_at_position(position);
            globals.insert(found.begin(), found.end());
        }
        if (!mgd_nodes.empty() && !globals.empty()) {
            if (!constraints.add_required_constraint(mgd_nodes, globals)) {
                return 0;
            }
            ++constraints_added;
        }
    }
    return constraints_added;
}

// One match/commit pass per distinct per-mesh pin set, taken straight from the MGD. With no pins at all, a
// single empty variant is returned so the caller still makes one pass and falls back to its (0,0) anchor.
std::vector<std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>> enumerate_pin_set_variants(
    const tt::tt_metal::experimental::tt_fabric::PinningsByMesh& pinnings_by_mesh) {
    std::vector<std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>> pin_set_variants;
    std::set<std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>> seen_pin_sets;
    for (const auto& [mesh_id, pin_set] : pinnings_by_mesh) {
        // Only chip_id and the ASIC positions reach the solver, so compare with the mesh ids zeroed out:
        // one mesh_id_regex entry expanded over many meshes is the same work and collapses to one pass.
        auto mesh_agnostic = pin_set;
        for (auto& group : mesh_agnostic) {
            for (auto& fabric_node : group.fabric_nodes) {
                fabric_node.mesh_id = MeshId{0};
            }
        }
        if (seen_pin_sets.insert(std::move(mesh_agnostic)).second) {
            pin_set_variants.push_back(pin_set);
        }
    }
    if (pin_set_variants.empty()) {
        pin_set_variants.emplace_back();
    }
    return pin_set_variants;
}

using tt::tt_metal::AsicID;
using tt::tt_metal::ASICLocation;
using tt::tt_metal::TrayID;

std::vector<std::set<AsicID>> collect_psd_host_groups(
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    std::map<std::string, std::set<AsicID>> host_to_asics;
    for (const AsicID& asic_id : physical_graph.get_nodes()) {
        host_to_asics[physical_system_descriptor.get_host_name_for_asic(asic_id)].insert(asic_id);
    }
    std::vector<std::set<AsicID>> global_groups;
    global_groups.reserve(host_to_asics.size());
    for (auto& [_, asics] : host_to_asics) {
        if (!asics.empty()) {
            global_groups.push_back(std::move(asics));
        }
    }
    return global_groups;
}

std::set<LogicalChipId> collect_pgd_asic_targets(const GroupingInfo& grouping_info) {
    std::set<LogicalChipId> all_targets;
    for (LogicalChipId node_id : grouping_info.adjacency_graph.get_nodes()) {
        if (node_id >= grouping_info.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = grouping_info.items[node_id];
        if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
            continue;
        }
        all_targets.insert(node_id);
    }
    return all_targets;
}

// Align the MGD's declared host_topology with the PSD's host partitions.
//
// The contract is asymmetric: the physical host boundaries constrain the logical ones, never the reverse. Each
// mesh host rank the MGD declares must land inside a single PSD host, because a rank is one process on one host
// and cannot own chips on two. The logical side may subdivide further -- several mesh host ranks sharing one
// physical host is fine -- which is why this is a same-host requirement per declared group rather than a demand
// for one distinct host per group.
//
// A host_topology of [1,1] is one declared rank covering the whole mesh, so it goes through the same path and
// is held to the same rule: the mesh must fit inside one host. A torus that can only close through inter-host
// links does not earn an exception here -- it has to declare the hosts it spans.
//
// The soft same-host preference is left for groupings that reach this with no declared host topology at all,
// where there is no contract to enforce and one host is merely the better tie-break.
//
// mesh_node_to_host_group is keyed by MeshGraph / MGD chip id. Placement targets are those chips
// (MGD fallback) or the PGD nodes they pin to (committed match). The host split is applied either way.
bool configure_pgd_psd_host_alignment_constraints(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<LogicalChipId, AsicID>& constraints) {
    // The chips this grouping names. The MGD fallback names none, being the MGD's own topology rather than a
    // PGD layout, so its declared split names them instead: the nodes the split covers are exactly the ones it
    // has to hold together, and without this the fallback would be the one variant exempt from the rule.
    std::set<LogicalChipId> all_targets = collect_pgd_asic_targets(grouping_info);
    if (all_targets.empty()) {
        for (const auto& [node_id, _] : grouping_info.mesh_node_to_host_group) {
            all_targets.insert(node_id);
        }
    }
    if (all_targets.empty()) {
        return true;
    }

    const std::vector<std::set<AsicID>> global_groups =
        collect_psd_host_groups(physical_graph, physical_system_descriptor);
    if (global_groups.size() <= 1) {
        return true;
    }

    // Bias `targets` toward the fewest host partitions that can cover them. Preferences never make the solve
    // infeasible, so this only breaks ties the hard constraints leave open.
    const auto prefer_minimal_host_cover = [&](const std::set<LogicalChipId>& targets) {
        const auto [fits_one_host, preferred_globals] =
            ::tt::tt_fabric::PhysicalGroupingDescriptor::find_minimum_coverage_group(targets, global_groups);
        if (!preferred_globals.empty()) {
            for (const LogicalChipId& target : targets) {
                constraints.add_preferred_constraint(target, preferred_globals);
            }
        }
        return fits_one_host;
    };

    if (!grouping_info.mesh_node_to_host_group.empty()) {
        // MeshGraph host_topology split: chip -> rank from compose_mesh_node_to_host_group_from_mgd_match.
        // The constraint graph may still number nodes as PGD grouping chips; if this chip has a committed
        // slot pinning, use the PGD node at that slot, otherwise the chip is already the placement target
        // (MGD fallback). The rank key is always the MeshGraph partition, not the PGD host index.
        std::map<uint32_t, std::set<LogicalChipId>> targets_by_mesh_host_rank;
        for (const auto& [mgd_chip, mesh_host_rank] : grouping_info.mesh_node_to_host_group) {
            LogicalChipId placement_node = mgd_chip;
            const auto pos_it = grouping_info.mesh_node_to_asic_position.find(mgd_chip);
            if (pos_it != grouping_info.mesh_node_to_asic_position.end()) {
                for (LogicalChipId node_id : grouping_info.adjacency_graph.get_nodes()) {
                    if (node_id >= grouping_info.items.size()) {
                        continue;
                    }
                    const GroupingItemInfo& item = grouping_info.items[node_id];
                    if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
                        continue;
                    }
                    if (tt::tt_metal::ASICPosition{item.tray_id, item.asic_location} == pos_it->second) {
                        placement_node = node_id;
                        break;
                    }
                }
            }
            targets_by_mesh_host_rank[mesh_host_rank].insert(placement_node);
        }
        std::vector<std::set<LogicalChipId>> target_groups;
        target_groups.reserve(targets_by_mesh_host_rank.size());
        for (auto& [_, group_targets] : targets_by_mesh_host_rank) {
            target_groups.push_back(std::move(group_targets));
        }

        // Hard, and only in this direction: no PSD host boundary may cut through a declared mesh host rank. Each
        // target group must therefore be carvable inside one PSD host. Groups are free to share a host, so a
        // host_topology finer than the physical hosts stays legal; what is rejected is a single declared rank
        // whose chips would have to come from two different hosts.
        if (!constraints.set_same_rank_groups_constraint(target_groups, global_groups)) {
            return false;
        }
        // Cap the hosts used at the number of declared groups: the split may collapse onto fewer hosts, never
        // spread onto more than it declared.
        constraints.set_max_same_rank_groups_used(target_groups.size());
        return true;
    }

    prefer_minimal_host_cover(all_targets);
    return true;
}

// Add the PGD→PSD embedding constraints (trait + host alignment) to `constraints`, in place and on top
// of whatever the caller already put there. That is how a caller anchors the embedding: seed the object
// with forbidden chips and an adjacency cardinality constraint, hand it here, and the grouping's own
// constraints are layered on without disturbing them.
// Returns false if a required trait constraint cannot be satisfied (e.g. slot count mismatch);
// `error_out` is set when that happens.
bool add_pgd_to_psd_constraints(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<LogicalChipId, AsicID>& constraints,
    std::string* error_out = nullptr) {
    // Set quiet mode to suppress verbose constraint validation messages during PGD solving
    constraints.set_quiet_mode(true);

    // Build trait maps: graph nodes are LogicalChipIds, items[i] is the item for node i
    std::map<LogicalChipId, TrayID> target_tray_traits;
    std::map<LogicalChipId, ASICLocation> target_location_traits;

    for (LogicalChipId node_id : grouping_info.adjacency_graph.get_nodes()) {
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

    // When set to 1, do not require PGD (tray_id, asic_location) on logical nodes to match UMD-reported ASIC
    // positions. Use only when slot counts already match but the labeled graph has no embedding (e.g. host / tray
    // order differs from PGD row-major). Host-alignment constraints below still apply. Bring-up only.
    const char* relax_env = std::getenv("TT_METAL_RELAX_PGD_SLOT_CONSTRAINTS");
    const bool relax_pgd_slot_traits = (relax_env != nullptr && relax_env[0] == '1');
    if (relax_pgd_slot_traits) {
        log_warning(
            tt::LogFabric,
            "TT_METAL_RELAX_PGD_SLOT_CONSTRAINTS=1: skipping PGD tray / ASIC-location trait constraints for "
            "PGD→PSD embedding");
    }

    // Add trait constraints for tray_id and asic_location
    if (!relax_pgd_slot_traits && !target_tray_traits.empty() && !global_tray_traits.empty()) {
        if (!constraints.add_required_trait_constraint<TrayID>(target_tray_traits, global_tray_traits)) {
            if (error_out) {
                *error_out = "Failed to add required trait constraint for tray_id";
            }
            return false;
        }
    }
    if (!relax_pgd_slot_traits && !target_location_traits.empty() && !global_location_traits.empty()) {
        if (!constraints.add_required_trait_constraint<ASICLocation>(target_location_traits, global_location_traits)) {
            if (error_out) {
                *error_out = "Failed to add required trait constraint for asic_location";
            }
            return false;
        }
    }

    if (!configure_pgd_psd_host_alignment_constraints(
            grouping_info, physical_graph, physical_system_descriptor, constraints)) {
        if (error_out != nullptr) {
            *error_out =
                fmt::format("Failed to configure host alignment constraints for grouping '{}'", grouping_info.name);
        }
        return false;
    }

    return true;
}

// Enumerate up to `max_solutions` distinct image-set placements of `grouping_info` on `physical_graph`.
// Resumable enumeration of one grouping variant's placements. Bundles the grouping being enumerated with
// everything that must outlive a single call so the next one continues the same search instead of
// re-encoding it: the session, the grouping's constraints (encoded once), and the append-only list of
// mappings already returned. `solves` is the running inner-solve count; `exhausted` is the terminal "no
// further distinct mapping" signal, and the ONLY trustworthy "this list is complete" flag. Never moved
// once used -- the DFS engine keeps pointers into the session's own snapshots -- so keep it behind a
// unique_ptr (or in an object that is itself never moved) when it lives in a container.
struct GroupingVariantEnumeration {
    const GroupingInfo* variant = nullptr;  // the grouping being enumerated
    std::unique_ptr<TopologyMappingEnumerationSession<LogicalChipId, AsicID>> session;
    MappingConstraints<LogicalChipId, AsicID> constraints;  // trait / host-alignment, encoded once
    std::vector<std::map<LogicalChipId, AsicID>> excluded;  // mappings already returned
    std::size_t solves = 0;
    bool started = false;
    bool exhausted = false;
};

// Enumerate distinct placements (unique_shapes=true, so the solver skips permutations that reuse the same
// ASIC set) of one grouping against the physical graph, returning up to `max_solutions` of them
// (max_solutions == 0 means "up to the enumeration safety cap").
//
// One session drives every call. A one-shot caller (resume == nullptr) gets a fresh session on the stack
// that is discarded on return; a resuming caller supplies its own via `resume`, so repeated calls continue
// ONE enumeration -- each returns the next batch at one incremental solve apiece rather than re-deriving
// the earlier ones -- and passes resume->constraints as `constraints`. Everything between the two is
// identical, which is the point of sharing this body. `stats` is aggregate per-call accounting for the
// non-resuming callers; a resuming caller passes nullptr and does its own.
std::vector<MappingResult<LogicalChipId, AsicID>> enumerate_flat_grouping_embeddings(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    size_t max_solutions,
    MappingConstraints<LogicalChipId, AsicID>& constraints,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::STRICT,
    PlacementSolveStats* stats = nullptr,
    GroupingVariantEnumeration* resume = nullptr,
    bool unique_shapes = true) {
    // A one-shot call owns its enumeration state for the duration of the call; a resuming one borrows the
    // caller's. `local` is constructed only when there is no session to resume, so the resume path pays
    // nothing for it.
    std::optional<GroupingVariantEnumeration> local;
    if (resume == nullptr) {
        local.emplace();
    }
    GroupingVariantEnumeration& state = (resume != nullptr) ? *resume : *local;

    if (state.exhausted) {
        return {};
    }
    if (!state.started) {
        if (state.variant == nullptr) {
            state.variant = &grouping_info;
        }
        // Encode the grouping's trait / host-alignment constraints once; the session constructor snapshots them.
        // This is the "host match" phase.
        const bool encoded =
            add_pgd_to_psd_constraints(grouping_info, physical_graph, physical_system_descriptor, constraints, nullptr);
        if (!encoded) {
            state.exhausted = true;
            return {};
        }
        state.session = std::make_unique<TopologyMappingEnumerationSession<LogicalChipId, AsicID>>(
            grouping_info.adjacency_graph,
            physical_graph,
            constraints,
            validation_mode,
            /*quiet_mode=*/true,
            TopologyMappingSolverEngine::Auto,
            unique_shapes);
        if (state.session == nullptr || !state.session->started()) {
            state.exhausted = true;
            return {};
        }
        state.started = true;
    }
    if (max_solutions == 0 || max_solutions > kTopologyMappingEnumerateSolutionsHardCap) {
        max_solutions = kTopologyMappingEnumerateSolutionsHardCap;  // same clamp solve_topology_mapping_n applies
    }

    const auto solve_start = std::chrono::steady_clock::now();
    std::vector<MappingResult<LogicalChipId, AsicID>> mappings;
    for (size_t i = 0; i < max_solutions; ++i) {
        MappingResult<LogicalChipId, AsicID> mapping = state.session->next();
        ++state.solves;
        if (!mapping.success) {
            state.exhausted = true;
            break;
        }
        state.excluded.push_back(mapping.target_to_global);
        mappings.push_back(std::move(mapping));
    }

    // Aggregate per-call stats for the non-resuming callers. Inner per-solve counts are deliberately not
    // threaded through: a resuming caller passes stats=nullptr and tracks candidates its own way.
    if (stats != nullptr) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - solve_start);
        const std::size_t n_target = grouping_info.adjacency_graph.get_nodes().size();
        const std::size_t n_global = physical_graph.get_nodes().size();
        ++stats->inner_solver_calls;
        stats->inner_solver_elapsed += elapsed;
        stats->inner_solutions_found += mappings.size();
        const bool used_sat = tt::tt_fabric::detail::topology_mapping_should_use_sat_engine(
            TopologyMappingSolverEngine::Auto, n_target, n_global);
        if (used_sat) {
            ++stats->inner_solver_sat_calls;
            stats->sat_elapsed += elapsed;
        } else {
            ++stats->inner_solver_dfs_calls;
            stats->dfs_elapsed += elapsed;
            if (!mappings.empty()) {
                stats->inner_dfs_visits += mappings.front().stats.dfs_calls;
                stats->inner_dfs_backtracks += mappings.front().stats.backtrack_count;
                stats->inner_dfs_memoization_hits += mappings.front().stats.memoization_hits;
            }
        }
        if (elapsed >= stats->slowest_inner_elapsed) {
            stats->slowest_inner_elapsed = elapsed;
            stats->slowest_inner_n_target = n_target;
            stats->slowest_inner_n_global = n_global;
            stats->slowest_inner_used_sat = used_sat;
        }
    }
    return mappings;
}

}  // namespace

namespace tt::tt_fabric {

std::string PlacementSolveStats::to_string() const {
    return fmt::format(
        "PlacementSolveStats:\n"
        "  success: {}  meshes: {}/{}\n"
        "  inner solver calls: {} (SAT {}, DFS {})  ({} us)\n"
        "  inner DFS visits: {}  backtracks: {}  memo hits: {}\n"
        "  candidates generated: {}  inner solutions: {}\n"
        "  slowest inner: {} {}x{} in {} us\n"
        "  master (SAT joint placement): attempted {}  success {}  candidates {}  growth rounds {}  attempts {}  "
        "lists complete {}\n"
        "  master SAT: {} vars  {} clauses  enumerate {} us  encode {} us  solve {} us\n"
        "  total: {} us ({:.3f} ms)",
        success,
        meshes_placed,
        meshes_total,
        inner_solver_calls,
        inner_solver_sat_calls,
        inner_solver_dfs_calls,
        inner_solver_elapsed.count(),
        inner_dfs_visits,
        inner_dfs_backtracks,
        inner_dfs_memoization_hits,
        candidates_generated,
        inner_solutions_found,
        slowest_inner_used_sat ? "SAT" : "DFS",
        slowest_inner_n_target,
        slowest_inner_n_global,
        slowest_inner_elapsed.count(),
        master_solve_attempted,
        master_solve_success,
        master_candidates_enumerated,
        master_growth_rounds,
        master_sat_attempts,
        candidate_lists_complete,
        master_sat_vars,
        master_sat_clauses,
        master_enumeration_elapsed.count(),
        master_encode_elapsed.count(),
        master_solve_elapsed.count(),
        total_elapsed.count(),
        static_cast<double>(total_elapsed.count()) / 1000.0);
}

namespace {

// Whether the machine has this host: some host of it must hold every chip the flattened host names. The
// chips are read through the flattening's own nodes, since one grouping flattens to a host per place it sits
// and each holds only its share of the chips the grouping names, and a node that says no tray or ASIC
// location names no chip so it claims nothing.
//
// Containment rather than equality, so a descriptor may divide the machine more finely than the machine
// divides itself; what this turns away is a host whose chips are spread over two machine hosts, which no
// process could ever own. The flattening cannot answer this on its own: the only test it applies is
// can_map_to_psd, which asks whether the machine has enough chips at each slot, and on a machine built of
// identical hosts the tray labels repeat host to host, so a host claiming chips from two of them passes
// that and fails this.
// The machine's own hosts, as the placement stages partition them, read back as the slots each one holds.
// Collected once per call: the host level is checked against this for every way each of its hosts could sit,
// and the machine does not change while that happens.
std::vector<std::set<tt::tt_metal::ASICPosition>> collect_machine_host_slots(
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    std::vector<std::set<tt::tt_metal::ASICPosition>> machine_host_slots;
    for (const std::set<AsicID>& machine_host : collect_psd_host_groups(physical_graph, physical_system_descriptor)) {
        std::set<tt::tt_metal::ASICPosition> slots;
        for (const AsicID& asic_id : machine_host) {
            const auto& descriptor = physical_system_descriptor.get_asic_descriptors().at(asic_id);
            slots.emplace(descriptor.tray_id, descriptor.asic_location);
        }
        machine_host_slots.push_back(std::move(slots));
    }
    return machine_host_slots;
}

bool machine_has_this_host(
    const GroupingInfo& flattened_host, const std::vector<std::set<tt::tt_metal::ASICPosition>>& machine_host_slots) {
    std::set<tt::tt_metal::ASICPosition> claimed_slots;
    for (GroupingChipId node_id : flattened_host.adjacency_graph.get_nodes()) {
        if (node_id >= flattened_host.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = flattened_host.items[node_id];
        if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION && *item.tray_id != 0 && *item.asic_location != 0) {
            claimed_slots.emplace(item.tray_id, item.asic_location);
        }
    }
    if (claimed_slots.empty()) {
        return false;  // names no chips, so it holds none of the machine and speaks for no mesh.
    }
    return std::any_of(machine_host_slots.begin(), machine_host_slots.end(), [&](const auto& one_machine_host) {
        return std::includes(
            one_machine_host.begin(), one_machine_host.end(), claimed_slots.begin(), claimed_slots.end());
    });
}

// MGD mesh topology as a placement variant (host split stamped, embeddable on PSD under pin variants).
std::optional<GroupingInfo> build_mgd_mesh_placement_fallback(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const std::string& instance_name,
    const GroupingInfo& mgd_grouping_info,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const AdjacencyGraph<tt::tt_metal::AsicID>& psd_physical_graph,
    const tt::tt_metal::experimental::tt_fabric::PinningsByMesh& pinnings_by_mesh,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) {
    const auto device_topo = mesh_graph_descriptor.get_effective_declared_topology(instance_name);
    GroupingInfo mgd_fallback = device_topo.has_value()
                                    ? finalize_mesh_grouping_with_device_topology(mgd_grouping_info, *device_topo)
                                    : mgd_grouping_info;

    const std::set<uint32_t> instance_mesh_ids =
        get_mesh_ids_for_mgd_instance_name(mesh_graph_descriptor, instance_name);
    const MeshId mesh_id = MeshId{instance_mesh_ids.empty() ? 0 : *instance_mesh_ids.begin()};

    std::map<LogicalChipId, GroupingChipId> fallback_nodes_are_mgd_chips;
    for (LogicalChipId node_id : mgd_fallback.adjacency_graph.get_nodes()) {
        fallback_nodes_are_mgd_chips.emplace(node_id, node_id);
    }
    mgd_fallback.mesh_node_to_host_group = compose_mesh_node_to_host_group_from_mgd_match(
        device_topo, fallback_nodes_are_mgd_chips, mesh_id, fabric_node_id_to_mesh_rank);

    const std::vector<std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>> pin_set_variants =
        enumerate_pin_set_variants(pinnings_by_mesh);

    std::map<tt::tt_metal::ASICPosition, std::set<tt::tt_metal::AsicID>> asics_by_position;
    for (const tt::tt_metal::AsicID& asic_id : psd_physical_graph.get_nodes()) {
        asics_by_position[{physical_system_descriptor.get_tray_id(asic_id),
                           physical_system_descriptor.get_asic_location(asic_id)}]
            .insert(asic_id);
    }
    for (const auto& active_pinnings : pin_set_variants) {
        MappingConstraints<LogicalChipId, tt::tt_metal::AsicID> solve_constraints;
        if (!active_pinnings.empty() &&
            add_mgd_asic_position_pinning_constraints(
                solve_constraints, active_pinnings, [&](const tt::tt_metal::ASICPosition& position) {
                    auto it = asics_by_position.find(position);
                    return it == asics_by_position.end() ? std::set<tt::tt_metal::AsicID>{} : it->second;
                }) == 0) {
            continue;
        }
        if (!enumerate_flat_grouping_embeddings(
                 mgd_fallback,
                 psd_physical_graph,
                 physical_system_descriptor,
                 /*max_solutions=*/1,
                 solve_constraints)
                 .empty()) {
            return mgd_fallback;
        }
    }
    return std::nullopt;
}

// MeshGraph is the source of host ranks (get_host_rank_for_chip). Used when the caller did not pass a
// fabric_node_id_to_mesh_rank map, so get_valid_groupings / MGD fallbacks still enforce host_topology.
std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_node_id_to_mesh_rank_from_mesh_graph(
    const MeshGraphDescriptor& mesh_graph_descriptor) {
    const MeshGraph mesh_graph(mesh_graph_descriptor);
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> mapping;
    for (const MeshId mesh_id : mesh_graph.get_all_mesh_ids()) {
        for (const auto& [unused_coord, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            (void)unused_coord;
            const std::optional<MeshHostRankId> host_rank = mesh_graph.get_host_rank_for_chip(mesh_id, chip_id);
            if (host_rank.has_value()) {
                mapping[mesh_id][FabricNodeId(mesh_id, chip_id)] = *host_rank;
            }
        }
    }
    return mapping;
}

}  // namespace

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings,
    bool require_placement,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) const {
    return get_valid_groupings_for_mgd(
        mesh_graph_descriptor, &physical_system_descriptor, pinnings, require_placement, fabric_node_id_to_mesh_rank);
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings,
    bool require_placement,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) const {
    ValidGroupingsMap result;
    // Use the caller-supplied ranks when present; otherwise derive them from the mesh graph. Binding a local
    // const-ref to either the parameter or the local storage keeps this zero-copy (callers pass lvalues).
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> ranks_from_mesh_graph;
    if (fabric_node_id_to_mesh_rank.empty()) {
        ranks_from_mesh_graph = fabric_node_id_to_mesh_rank_from_mesh_graph(mesh_graph_descriptor);
    }
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& mesh_ranks =
        fabric_node_id_to_mesh_rank.empty() ? ranks_from_mesh_graph : fabric_node_id_to_mesh_rank;

    std::optional<AdjacencyGraph<tt::tt_metal::AsicID>> psd_physical_graph;
    if (physical_system_descriptor != nullptr) {
        psd_physical_graph.emplace(
            tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd(*physical_system_descriptor));
    }

    // 1-3. The descriptor's own host level: every way each HOSTS grouping can sit, flattened against the
    // machine the same way the meshes are, keeping the ones the machine actually has. This has to come after
    // the flattening, since a host is usually assembled from references to other groupings and names no chips
    // of its own until those are resolved, and one grouping flattens to a host per place it sits -- so there
    // is nothing to hold against the machine before the flattening, and one thing per place after it.
    //
    // A way of sitting that the machine does not have is skipped rather than fatal: the descriptor is
    // offering possibilities, and the answer to one the machine cannot hold is that it is not among the hosts
    // the meshes are held against. That leaves the hosts below exactly the ones some host of the machine
    // holds whole, which is what step 4 attributes each mesh's chips to and step 5 holds the declared ranks
    // against. Declaring hosts stays optional: a descriptor with no HOSTS grouping claims nothing, and when
    // none survive everything below runs as it did.
    const std::vector<std::set<tt::tt_metal::ASICPosition>> machine_host_slots =
        physical_system_descriptor != nullptr
            ? collect_machine_host_slots(*psd_physical_graph, *physical_system_descriptor)
            : std::vector<std::set<tt::tt_metal::ASICPosition>>{};
    std::vector<GroupingInfo> flattened_declared_hosts;
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        const auto hosts_it = type_map.find("HOSTS");
        if (hosts_it == type_map.end()) {
            continue;
        }
        for (const GroupingInfo& declared_host : hosts_it->second) {
            // A host wired in a way no mesh layout describes cannot be flattened into one, so it is left
            // alone rather than flattened and rejected.
            if (declared_host.items.size() > 1 && declared_host.instance_tile_layout_dims.empty()) {
                continue;
            }
            for (auto& variant : build_flattened_adjacency_mesh(declared_host, physical_system_descriptor)) {
                if (!machine_host_slots.empty() && !machine_has_this_host(variant, machine_host_slots)) {
                    continue;
                }
                flattened_declared_hosts.push_back(std::move(variant));
            }
        }
    }

    // ===== PHASE 0: Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts) =====
    // This step calculates required ASIC counts bottom-up and builds adjacency graphs
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos =
        PhysicalGroupingDescriptor::build_mgd_to_grouping_info_map(mesh_graph_descriptor);

    // Incoming pins are already keyed by local mesh id (MGD get_pinnings + caller-merged galaxy pins).
    const tt::tt_metal::experimental::tt_fabric::PinningsByMesh all_pinnings_by_mesh =
        pinnings.value_or(tt::tt_metal::experimental::tt_fabric::PinningsByMesh{});

    // ===== PHASE 1: Build flattened adjacency graphs for all mesh group infos =====
    // Map from grouping name to vector of flattened GroupingInfo (supports multiple definitions with same name)
    std::unordered_map<std::string, std::vector<GroupingInfo>> mesh_flat_groupings;
    // Find MESH type groupings across all names
    bool found_mesh = false;
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        auto mesh_it = type_map.find("MESH");
        if (mesh_it != type_map.end()) {
            found_mesh = true;
            for (const auto& mesh_group_info : mesh_it->second) {
                auto meshes = build_flattened_adjacency_mesh(mesh_group_info, physical_system_descriptor);
                for (auto& meshe : meshes) {
                    // 4. The host each of this variant's chips sits on, by comparing its slots with the
                    // declared hosts above. Done once per variant here rather than per MGD instance
                    // below, since it says something about the variant alone.
                    // A mesh no declared host holds gets no groups, and step 5 then has nothing to hold its
                    // ranks against and leaves it to the PSD's own hosts further down.
                    assign_pgd_host_groups(meshe, flattened_declared_hosts);

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
    // Deterministic processing order across MGD mesh instances (unordered_map iteration is unspecified)
    std::vector<std::string> mesh_mgd_instance_order;
    mesh_mgd_instance_order.reserve(mgd_grouping_infos.at("MESH").size());
    for (const auto& [k, _] : mgd_grouping_infos.at("MESH")) {
        mesh_mgd_instance_order.push_back(k);
    }
    std::sort(mesh_mgd_instance_order.begin(), mesh_mgd_instance_order.end());

    for (const std::string& mgd_instance_key : mesh_mgd_instance_order) {
        const GroupingInfo& mgd_mesh_grouping = mgd_grouping_infos.at("MESH").at(mgd_instance_key);
        const std::string& instance_name = mgd_instance_key;  // Use unique instance key (includes mesh_id)
        const GroupingInfo& mgd_grouping_info = mgd_mesh_grouping;
        const std::string& instance_type = mgd_grouping_info.type;  // Should be "MESH"

        // A single MGD descriptor may be instantiated as several meshes that are pinned differently. Pins
        // arrive keyed by mesh, so look up only this descriptor's mesh ids.
        tt::tt_metal::experimental::tt_fabric::PinningsByMesh pinnings_by_mesh;
        for (uint32_t mesh_id : get_mesh_ids_for_mgd_instance_name(mesh_graph_descriptor, instance_name)) {
            if (auto it = all_pinnings_by_mesh.find(MeshId{mesh_id}); it != all_pinnings_by_mesh.end()) {
                pinnings_by_mesh.emplace(it->first, it->second);
            }
        }

        const std::vector<std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>> pin_set_variants =
            enumerate_pin_set_variants(pinnings_by_mesh);

        // Required nodes from MGD adjacency graph (this represents the topology pattern to match)
        size_t required_nodes = mgd_grouping_info.adjacency_graph.get_nodes().size();

        // Cheap necessary-condition prefilter for the (expensive) topology solve. solve_topology_mapping
        // looks for an injective edge-preserving map of the MGD graph (target) into a PGD variant (global),
        // so every MGD edge must land on a distinct PGD edge -> |E(PGD)| >= |E(MGD)| is required. A RING/RING
        // MGD is a full torus (degree 4 everywhere, ~2*N edges) while the MESH/TORUSX/TORUSY variants of the
        // same grid drop wrap edges, so they have strictly fewer edges and can never contain it. Counting
        // edges is O(V); the SAT solve it skips is many orders of magnitude slower (seconds per 128-node
        // candidate), so this eliminates the provably-impossible variants up front instead of solving them.
        auto count_undirected_edges = [](const AdjacencyGraph<GroupingChipId>& g) -> size_t {
            size_t directed = 0;
            for (GroupingChipId node : g.get_nodes()) {
                directed += g.get_neighbors(node).size();
            }
            return directed / 2;  // each undirected edge is stored from both endpoints
        };
        const size_t required_edges = count_undirected_edges(mgd_grouping_info.adjacency_graph);

        const auto device_topo = mesh_graph_descriptor.get_effective_declared_topology(instance_name);

        // MeshGraph host ranks for this instance's mesh, used to source the host_topology split directly.
        const std::set<uint32_t> instance_mesh_ids =
            get_mesh_ids_for_mgd_instance_name(mesh_graph_descriptor, instance_name);
        const MeshId instance_mesh_id = MeshId{instance_mesh_ids.empty() ? 0 : *instance_mesh_ids.begin()};

        // Group valid candidates by node difference (map is ordered by key ascending)
        // Store (name, index) pairs to handle multiple groupings with same name.
        // Iterate PGD names in sorted order so candidate order within each diff bucket is stable.
        log_info(tt::LogFabric, "Grouping valid candidates by node difference");
        std::map<size_t, std::vector<std::pair<std::string, size_t>>> candidates_by_diff;
        std::vector<std::string> pgd_mesh_grouping_names;
        pgd_mesh_grouping_names.reserve(mesh_flat_groupings.size());
        for (const auto& [name, _] : mesh_flat_groupings) {
            pgd_mesh_grouping_names.push_back(name);
        }
        std::sort(pgd_mesh_grouping_names.begin(), pgd_mesh_grouping_names.end());
        for (const std::string& name : pgd_mesh_grouping_names) {
            const auto& grouping_infos = mesh_flat_groupings.at(name);
            for (size_t idx = 0; idx < grouping_infos.size(); ++idx) {
                const auto& grouping_info = grouping_infos[idx];
                size_t n = grouping_info.adjacency_graph.get_nodes().size();
                if (n >= required_nodes) {
                    candidates_by_diff[n - required_nodes].emplace_back(name, idx);
                }
            }
        }

        // Process difference levels from closest to farthest; commit only when embedding on PSD succeeds.
        // Each pin set gets its own match/commit pass, so a shared descriptor accumulates the groupings of
        // every column it is pinned to.
        std::vector<MeshTopologyMatch> best_matches_topology;
        std::vector<MeshTopologyMatch> best_matches_psd_placed;
        size_t last_topology_match_count = 0;

        for (const auto& active_pinnings : pin_set_variants) {
            for (const auto& [node_diff, name_idx_pairs] : candidates_by_diff) {
                best_matches_topology.clear();
                best_matches_psd_placed.clear();
                best_matches_topology.reserve(name_idx_pairs.size());

                for (const auto& [name, idx] : name_idx_pairs) {
                    const auto& grouping_info = mesh_flat_groupings.at(name)[idx];

                    // Necessary-condition prefilter: a variant with fewer edges than the MGD cannot contain it
                    // (every MGD edge needs a distinct variant edge). Skip without paying for the SAT solve.
                    const size_t variant_edges = count_undirected_edges(grouping_info.adjacency_graph);
                    if (variant_edges < required_edges) {
                        log_debug(
                            tt::LogFabric,
                            "Skipping {} for {}: {} edges < {} MGD edges (cannot contain the topology)",
                            name,
                            mgd_grouping_info.name,
                            variant_edges,
                            required_edges);
                        continue;
                    }

                    MappingConstraints<LogicalChipId, GroupingChipId> constraints;
                    if (!active_pinnings.empty()) {
                        // Keep only groupings that host at least one pin, with the pins that do apply
                        // required to hold together.
                        if (add_mgd_asic_position_pinning_constraints(
                                constraints, active_pinnings, [&](const tt::tt_metal::ASICPosition& position) {
                                    return find_pgd_nodes_at_asic_position(grouping_info, position);
                                }) == 0) {
                            continue;
                        }
                    }

                    // 5. The declared host split against the host division this variant sits on, so the match
                    // comes back in an orientation whose ranks each fall inside one of the descriptor's hosts.
                    if (!configure_mgd_pgd_host_alignment_constraints(
                            mgd_grouping_info, grouping_info, device_topo, constraints, instance_mesh_id, mesh_ranks)) {
                        continue;
                    }

                    // TEMPORARY(2x2-4x1-cross): a [2,2] ring and a [4,1] ring are the same 4-cycle, so the
                    // matcher can swap one for the other -- a 4x1 strip for a [2,2] mesh straddles the tray
                    // boundary and breaks the 2x2 tray-locality invariant (TestGalaxyLayoutCheck). Forbid the
                    // cross both ways. TODO(2x2-4x1-cross): remove once shape/topology disambiguation is stable.
                    if (is_2x2_4x1_cross(device_topo, grouping_info)) {
                        continue;
                    }

                    auto mapping_result = solve_topology_mapping<LogicalChipId, GroupingChipId>(
                        mgd_grouping_info.adjacency_graph,
                        grouping_info.adjacency_graph,
                        constraints,
                        ConnectionValidationMode::STRICT,
                        true);
                    if (mapping_result.success) {
                        best_matches_topology.push_back({name, idx, std::move(mapping_result)});
                    } else {
                        log_debug(
                            tt::LogFabric,
                            "Failed to solve topology mapping for {} and {}, with error: {}",
                            mgd_grouping_info.name,
                            name,
                            mapping_result.error_message);
                    }
                }

                if (best_matches_topology.empty()) {
                    continue;
                }
                last_topology_match_count = best_matches_topology.size();

                // The grouping committed for this MGD mesh is the matched PGD topology variant itself. Each variant
                // already encodes its own topology (the MESH grid, or RING wrap edges for TORUSX/TORUSY/TORUSXY) and
                // was pre-filtered by can_map_to_psd during flattening, so we PSD-validate and commit the variant's
                // own adjacency directly rather than rebuilding it from the MGD device topology. Keeping the PGD
                // (tray_id, asic_location) slot labels is intentional so adjacency-guided placement
                // places on the same graph.
                auto make_committed_grouping = [&](const MeshTopologyMatch& match) -> GroupingInfo {
                    GroupingInfo committed = mesh_flat_groupings.at(match.name)[match.idx];
                    // The topology solve used the MGD mesh adjacency as target and this PGD variant as global, so
                    // target_to_global is MGD-node -> PGD grouping-node. Compose logical chip_id -> PGD slot pinning
                    // now so downstream consumes it directly without re-deriving the intermediate node pairing.
                    committed.mesh_node_to_asic_position =
                        compose_mesh_node_to_asic_position_from_pgd_match(committed, match.mapping.target_to_global);
                    committed.mesh_node_to_host_group = compose_mesh_node_to_host_group_from_mgd_match(
                        device_topo, match.mapping.target_to_global, instance_mesh_id, mesh_ranks);
                    return committed;
                };

                // Prefer the simplest topology that fits: MESH, then whichever torus wraps remain after
                // dropping wraps on dims of size 2 or less (those axes keep ordinary MESH links).
                auto variant_priority = [&](const MeshTopologyMatch& m) -> int {
                    return effective_torus_variant_priority(mesh_flat_groupings.at(m.name)[m.idx]);
                };
                std::stable_sort(
                    best_matches_topology.begin(),
                    best_matches_topology.end(),
                    [&](const MeshTopologyMatch& a, const MeshTopologyMatch& b) {
                        return variant_priority(a) < variant_priority(b);
                    });

                // Check and only use the Groupings found that can actually be placed on the PSD.
                // The committed candidate is already one flattened variant, so this uses the enumerating
                // solve rather than flattening the hierarchical grouping again.
                // The MGD fallback is the same check, but it is not a PGD match, so it runs once
                // after this loop rather than once per candidate.
                if (physical_system_descriptor != nullptr) {
                    // The gate must validate under the same channel policy the actual solver
                    // (map_multi_mesh_to_physical) uses per mesh -- see the is_intra_mesh_policy_relaxed()
                    // gate there. Hardcoding STRICT here over-rejects a RELAXED mesh whose seams (e.g. the
                    // cross-host seams of a host-split mesh) intentionally carry fewer channels than the
                    // grouping's internal edges. NOTE: for a channel count of 1 the two modes encode
                    // identically (the STRICT channel check is guarded by required_channels > 1), so this
                    // only changes behaviour for meshes that declare count > 1 with a RELAXED policy.
                    // instance_name is a mesh *definition* name (e.g. "M0"), so it may resolve to several
                    // instance mesh ids when the definition is instantiated more than once. Every instance of
                    // one definition reads the same MeshDescriptor, so they must all carry the same policy;
                    // a disagreement means the lookups have been corrupted, which we fail loudly on.
                    // Defaults to STRICT (relaxed = false) when the definition has no mesh instances.
                    bool instance_relaxed = false;
                    const auto instance_mesh_ids =
                        get_mesh_ids_for_mgd_instance_name(mesh_graph_descriptor, instance_name);
                    if (!instance_mesh_ids.empty()) {
                        instance_relaxed =
                            mesh_graph_descriptor.is_intra_mesh_policy_relaxed(MeshId{*instance_mesh_ids.begin()});
                        for (uint32_t mesh_id : instance_mesh_ids) {
                            TT_FATAL(
                                mesh_graph_descriptor.is_intra_mesh_policy_relaxed(MeshId{mesh_id}) == instance_relaxed,
                                "Internal error: mesh definition '{}' has instances with disagreeing intra-mesh "
                                "channel policies; all instances of one definition must share a single policy",
                                instance_name);
                        }
                    }
                    const ConnectionValidationMode gate_validation_mode =
                        instance_relaxed ? ConnectionValidationMode::RELAXED : ConnectionValidationMode::STRICT;
                    for (const auto& match : best_matches_topology) {
                        const GroupingInfo committed_candidate = make_committed_grouping(match);
                        MappingConstraints<LogicalChipId, tt::tt_metal::AsicID> solve_constraints;
                        const auto placements = enumerate_flat_grouping_embeddings(
                            committed_candidate,
                            *psd_physical_graph,
                            *physical_system_descriptor,
                            /*max_solutions=*/1,
                            solve_constraints,
                            gate_validation_mode);
                        if (!placements.empty()) {
                            best_matches_psd_placed.push_back(match);
                        } else {
                            log_debug(
                                tt::LogFabric,
                                "PGD '{}' matched MGD '{}' topologically but could not be placed on PSD "
                                "(no ASIC embedding found)",
                                committed_candidate.name,
                                mgd_grouping_info.name);
                        }
                    }
                } else {
                    best_matches_psd_placed = best_matches_topology;
                }

                if (!best_matches_psd_placed.empty()) {
                    for (const auto& match : best_matches_psd_placed) {
                        auto lookup_it = mesh_flat_groupings.find(match.name);
                        if (lookup_it != mesh_flat_groupings.end() && match.idx < lookup_it->second.size()) {
                            result[instance_type][instance_name].push_back(make_committed_grouping(match));
                        }
                    }
                    std::string committed_summary;
                    for (size_t i = 0; i < best_matches_psd_placed.size(); ++i) {
                        const auto& match = best_matches_psd_placed[i];
                        const auto& grouping = mesh_flat_groupings.at(match.name)[match.idx];
                        if (i > 0) {
                            committed_summary += ", ";
                        }
                        committed_summary += fmt::format("{} ({})", grouping.name, grouping.type);
                    }
                    log_info(
                        tt::LogFabric,
                        "Physical groupings: Mesh graph descriptor '{}': {} topology match(es), committed: {}",
                        mgd_grouping_info.name,
                        best_matches_topology.size(),
                        committed_summary);
                    break;
                }
            }
        }  // end per-pin-set pass (pin_set_variants)

        // Callers that only want preferred pinnings on an already rank-bound graph pass require_placement=false:
        // for them an unplaceable mesh is a missing hint, not a broken system, and they degrade to no pinning.
        const auto& committed = result[instance_type][instance_name];
        if (committed.empty()) {
            TT_FATAL(
                !require_placement,
                "Physical groupings: Mesh graph descriptor '{}': no PGD grouping could be placed on the PSD "
                "({} topology match(es))",
                mgd_grouping_info.name,
                last_topology_match_count);
            log_warning(
                tt::LogFabric,
                "Physical groupings: Mesh graph descriptor '{}': no PGD grouping could be placed on the PSD "
                "({} topology match(es)); continuing without pinning hints for this mesh",
                mgd_grouping_info.name,
                last_topology_match_count);
        }
    }

    // =============================================================================
    // Phase 3: Higher-layer graph matching (FABRIC, SUPER_FABRIC, etc.)
    // =============================================================================

    std::unordered_map<std::string, std::string> known_mappings;
    known_mappings["MESH"] = "MESH";

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
    for (const auto& [mgd_type, mgd_instances] : mgd_grouping_infos) {
        for (const auto& [instance_name, mgd_grouping_info] : mgd_instances) {
            // If not already present, use the MGD grouping info
            if (!result[mgd_type].contains(instance_name)) {
                result[mgd_type][instance_name].push_back(mgd_grouping_info);
            }
        }
    }

    return result;
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_mgd_placement_fallbacks_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings) {
    ValidGroupingsMap result;
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> mesh_ranks =
        fabric_node_id_to_mesh_rank_from_mesh_graph(mesh_graph_descriptor);
    const AdjacencyGraph<tt::tt_metal::AsicID> psd_physical_graph(
        tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd(physical_system_descriptor));

    const std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos =
        build_mgd_to_grouping_info_map(mesh_graph_descriptor);
    const auto mesh_it = mgd_grouping_infos.find("MESH");
    if (mesh_it == mgd_grouping_infos.end()) {
        return result;
    }

    const tt::tt_metal::experimental::tt_fabric::PinningsByMesh all_pinnings_by_mesh =
        pinnings.value_or(tt::tt_metal::experimental::tt_fabric::PinningsByMesh{});

    std::vector<std::string> mesh_mgd_instance_order;
    mesh_mgd_instance_order.reserve(mesh_it->second.size());
    for (const auto& [k, _] : mesh_it->second) {
        mesh_mgd_instance_order.push_back(k);
    }
    std::sort(mesh_mgd_instance_order.begin(), mesh_mgd_instance_order.end());

    for (const std::string& instance_name : mesh_mgd_instance_order) {
        const GroupingInfo& mgd_grouping_info = mesh_it->second.at(instance_name);
        tt::tt_metal::experimental::tt_fabric::PinningsByMesh pinnings_by_mesh;
        for (uint32_t mesh_id : get_mesh_ids_for_mgd_instance_name(mesh_graph_descriptor, instance_name)) {
            if (auto it = all_pinnings_by_mesh.find(MeshId{mesh_id}); it != all_pinnings_by_mesh.end()) {
                pinnings_by_mesh.emplace(it->first, it->second);
            }
        }
        if (auto fallback = build_mgd_mesh_placement_fallback(
                mesh_graph_descriptor,
                instance_name,
                mgd_grouping_info,
                physical_system_descriptor,
                psd_physical_graph,
                pinnings_by_mesh,
                mesh_ranks)) {
            result["MESH"][instance_name].push_back(std::move(*fallback));
            log_info(
                tt::LogFabric,
                "Physical groupings: Mesh graph descriptor '{}': MGD placement fallback {} ({}) embeds on PSD",
                mgd_grouping_info.name,
                mgd_grouping_info.name,
                mgd_grouping_info.type);
        }
    }
    return result;
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_mgd_placement_fallbacks_for_mgds(
    const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::vector<std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>>& per_mgd_pinnings) const {
    ValidGroupingsMap out;
    for (size_t i = 0; i < mesh_graph_descriptors.size(); ++i) {
        std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh> pins;
        if (i < per_mgd_pinnings.size()) {
            pins = per_mgd_pinnings[i];
        }
        auto one = get_mgd_placement_fallbacks_for_mgd(mesh_graph_descriptors[i], physical_system_descriptor, pins);
        for (const auto& [type, by_name] : one) {
            for (const auto& [name, gvec] : by_name) {
                auto& dest = out[type][merged_instance_key(i, mesh_graph_descriptors.size(), name)];
                dest.insert(dest.end(), gvec.begin(), gvec.end());
            }
        }
    }
    return out;
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgds(
    const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::vector<std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>>& per_mgd_pinnings,
    bool require_placement) const {
    ValidGroupingsMap out;
    // Multi-MGD: instance names can collide across descriptors (e.g. both have "M0"). Prefix with "mgd{i}_"
    // so the merged map keeps them distinct; single-MGD stays unprefixed. SAT joint placement looks the
    // prefix up with merged_instance_key. Pins for MGD i stay in that descriptor's local mesh-id space.
    for (size_t i = 0; i < mesh_graph_descriptors.size(); ++i) {
        std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh> pins;
        if (i < per_mgd_pinnings.size()) {
            pins = per_mgd_pinnings[i];
        }
        auto one =
            get_valid_groupings_for_mgd(mesh_graph_descriptors[i], physical_system_descriptor, pins, require_placement);
        for (const auto& [type, by_name] : one) {
            for (const auto& [name, gvec] : by_name) {
                auto& dest = out[type][merged_instance_key(i, mesh_graph_descriptors.size(), name)];
                dest.insert(dest.end(), gvec.begin(), gvec.end());
            }
        }
    }
    return out;
}

}  // namespace tt::tt_fabric

namespace tt::tt_fabric {

bool PhysicalGroupingDescriptor::can_map_to_psd(
    const GroupingInfo& grouping_info, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    using tt::tt_metal::ASICPosition;

    // Build a multiset of ASICPosition slots available in the PSD.
    std::map<ASICPosition, size_t> psd_slot_counts;
    for (const auto& [_, desc] : physical_system_descriptor.get_asic_descriptors()) {
        if (*desc.tray_id > 0 && *desc.asic_location <= 8) {
            psd_slot_counts[{desc.tray_id, desc.asic_location}]++;
        }
    }

    // Count how many ASICs the grouping needs per ASICPosition slot.
    std::map<ASICPosition, size_t> required_slot_counts;
    for (GroupingChipId node_id : grouping_info.adjacency_graph.get_nodes()) {
        if (node_id >= grouping_info.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = grouping_info.items[node_id];
        if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
            continue;
        }
        if (*item.tray_id == 0 || *item.asic_location > 8) {
            continue;
        }
        required_slot_counts[{item.tray_id, item.asic_location}]++;
    }

    for (const auto& [slot, needed] : required_slot_counts) {
        auto it = psd_slot_counts.find(slot);
        if (it == psd_slot_counts.end() || it->second < needed) {
            return false;
        }
    }
    return true;
}

std::vector<MappingResult<LogicalChipId, AsicID>>
PhysicalGroupingDescriptor::enumerate_distinct_placements_for_grouping(
    const GroupingInfo& grouping,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t max_solutions) const {
    AdjacencyGraph<AsicID> physical_graph(
        tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd(physical_system_descriptor));
    MappingConstraints<LogicalChipId, AsicID> constraints;
    return enumerate_flat_grouping_embeddings(
        grouping, physical_graph, physical_system_descriptor, max_solutions, constraints);
}

// Joint placement types used by the SAT seating session.
using GlobalMeshId = MeshId;

namespace {

// =====================================================================================================
// Two-layer joint placement (TOPOLOGY_MAPPER_PLAN_4_SAT_JOINT_PLACEMENT.md).
//
// Layer 1 (geometry): enumerate each grouping variant's placements once against the FULL physical
// graph. Everything positional -- tray / ASIC-location pinning, host alignment, torus wraps, the mesh's
// own internal adjacency -- is discharged here, and the result is a list of ASIC footprints. Lists are
// keyed by mesh DEFINITION (the grouping-variant vector), so every instance of the same mesh shares one.
//
// Layer 2 (combinatorics): one SAT instance chooses one footprint per mesh INSTANCE such that no ASIC
// serves two meshes and every mesh-level edge is realised by enough fabric links. Only disjointness and
// seams depend on the joint assignment, and both are bitset operations over the footprints.
//
// Column generation: the lists are pulled in batches from resumable per-variant enumeration sessions,
// and grown only when the master problem comes back UNSAT. Truncation degrades in one direction only: a
// SAT model is always a real placement, and UNSAT is trustworthy only when every session is exhausted
// (CandidatePool::complete() on every mesh).
// =====================================================================================================

// Dense 0..N-1 numbering for the machine's physical graph (deterministic node order). Shared by every
// CandidatePool on the same graph so footprints and seam caches agree on bit indices.
struct DenseAsicIndex {
    explicit DenseAsicIndex(const AdjacencyGraph<AsicID>& physical_graph) {
        const auto& nodes = physical_graph.get_nodes();
        dense_to_asic_.reserve(nodes.size());
        asic_to_dense_.reserve(nodes.size());
        for (const AsicID& asic : nodes) {
            asic_to_dense_.emplace(asic, dense_to_asic_.size());
            dense_to_asic_.push_back(asic);
        }
    }

    std::size_t size() const { return dense_to_asic_.size(); }
    std::size_t dense(const AsicID& asic) const { return asic_to_dense_.at(asic); }

private:
    std::unordered_map<AsicID, std::size_t> asic_to_dense_;
    std::vector<AsicID> dense_to_asic_;
};

}  // namespace

// One legal seating of one grouping variant. Footprint-only: the per-node mapping from the inner solve is
// deliberately dropped -- downstream reconstructs positions
// from the variant's pinning map, not from the embedding.
//
// Seam geometry between two seatings is fabric_links_to (link count). Disjointness in the master SAT
// uses by_dense_asic on dense_asics(), not footprint bitsets. boundary_bitset_ / internal_footprint_bitset_
// gate and count links via the private Bitset helpers.
class SatPlacementEnumerationSession::Candidate {
public:
    const std::vector<uint32_t>& dense_asics() const { return dense_asics_; }

    const std::vector<AsicID>& asics() const { return asics_; }

    const GroupingInfo* variant() const { return variant_; }

    // One fabric eth link leaving this footprint toward exterior dense ASIC `dense` (parallel links kept).
    const std::vector<uint32_t>& boundary_fabric_links() const { return boundary_link_dense_; }

    // Fabric links from this seating into `other`'s chips (parallel eth links counted separately).
    std::size_t fabric_links_to(const Candidate& other) const {
        if (!boundary_bitset_.intersects(other.internal_footprint_bitset_)) {
            return 0;
        }
        std::size_t links = 0;
        for (const uint32_t dense : boundary_link_dense_) {
            if (other.internal_footprint_bitset_.test(dense)) {
                ++links;
            }
        }
        return links;
    }

    Candidate(
        const GroupingInfo* variant,
        std::size_t machine_asic_count,
        const std::map<LogicalChipId, AsicID>& placement,
        const DenseAsicIndex& asic_index,
        const std::vector<std::vector<uint32_t>>& neighbors_of_dense) :
        internal_footprint_bitset_(machine_asic_count), boundary_bitset_(machine_asic_count), variant_(variant) {
        for (const auto& [_, asic] : placement) {
            const std::size_t dense = asic_index.dense(asic);
            internal_footprint_bitset_.set(dense);
            dense_asics_.push_back(static_cast<uint32_t>(dense));
            asics_.push_back(asic);
        }
        for (const uint32_t chip : dense_asics_) {
            for (const uint32_t neighbor : neighbors_of_dense[chip]) {
                if (!internal_footprint_bitset_.test(neighbor)) {
                    boundary_link_dense_.push_back(neighbor);
                    boundary_bitset_.set(neighbor);
                }
            }
        }
    }

private:
    // Dense ASIC membership set. Each candidate footprint is keyed by 0..N-1 for the whole physical
    // fabric, not by AsicID, so disjointness and seam checks are word-wise ANDs instead of set lookups
    // over variable chip ids. The numbering lives on CandidatePool and is applied only at construction.
    //
    // std::bitset is not used because N is runtime (system size). Two instances per candidate:
    //   internal_footprint_bitset_ — chips occupied by this seating;
    //   boundary_bitset_ — exterior shell (neighbors not in the footprint); gates fabric_links_to().
    //
    // boundary_link_dense_ (outside Bitset) keeps per-link multiplicity for seam SAT clauses; the bitset
    // alone would collapse parallel links between the same chip pair.
    struct Bitset {
        explicit Bitset(std::size_t bit_count) : words_((bit_count + 63) / 64, 0) {}

        void clear() { std::fill(words_.begin(), words_.end(), 0); }
        void set(std::size_t bit) { words_[bit >> 6] |= (uint64_t{1} << (bit & 63)); }
        bool test(std::size_t bit) const { return ((words_[bit >> 6] >> (bit & 63)) & 1) != 0; }

        // Any shared bit => the two sets overlap (adjacent footprints or duplicate ASIC).
        bool intersects(const Bitset& other) const {
            for (std::size_t i = 0; i < words_.size(); ++i) {
                if ((words_[i] & other.words_[i]) != 0) {
                    return true;
                }
            }
            return false;
        }

        const std::vector<uint64_t>& words() const { return words_; }

    private:
        std::vector<uint64_t> words_;
    };

    Bitset internal_footprint_bitset_;
    Bitset boundary_bitset_;
    std::vector<uint32_t> boundary_link_dense_;
    std::vector<uint32_t> dense_asics_;
    std::vector<AsicID> asics_;
    // Borrowed, never owned: name, type and mesh_node_to_asic_position live on the grouping, which is
    // identical across every placement of the variant. global_mesh_groupings holds the vectors by value
    // and outlives the solve; it must not be mutated once these pointers are taken.
    const GroupingInfo* variant_;
};

// All grouping variants and enumerated candidates for one global mesh instance.
class SatPlacementEnumerationSession::CandidatePool {
public:
    CandidatePool(
        const std::vector<GroupingInfo>& groupings,
        const AdjacencyGraph<AsicID>& physical_graph,
        const tt::tt_metal::PhysicalSystemDescriptor& psd,
        ConnectionValidationMode validation_mode,
        std::set<AsicID> allowed_asics = {}) :
        physical_graph_(physical_graph),
        psd_(psd),
        validation_mode_(validation_mode),
        asic_index_(physical_graph),
        allowed_asics_(std::move(allowed_asics)) {
        neighbors_of_dense_.assign(asic_index_.size(), {});
        for (const AsicID& asic : physical_graph.get_nodes()) {
            auto& nbrs = neighbors_of_dense_[asic_index_.dense(asic)];
            for (const AsicID& neighbor : physical_graph.get_neighbors(asic)) {
                nbrs.push_back(static_cast<uint32_t>(asic_index_.dense(neighbor)));
            }
        }
        // TODO(preferred-meshes): Rework variant ranking for the master solve. Grouping priority from
        // get_valid_groupings_for_mgd is not a simple boolean (PGD vs MGD vs torus variant vs footprint
        // quality); encode it as a proper ranking/objective once the placement model is settled.
        for (const GroupingInfo& grouping : groupings) {
            if (grouping.adjacency_graph.get_nodes().empty()) {
                continue;
            }
            GroupingVariant variant;
            variant.grouping = &grouping;
            variant.enumeration.variant = &grouping;
            variants_.emplace(&grouping, std::move(variant));
        }
    }

    std::size_t asic_count() const { return asic_index_.size(); }

    // Seat order matches master SAT seat_lit_by_mesh and AdjacencyMatrix row/column indices.
    const std::vector<Candidate>& candidates() const { return candidates_; }

    // Index of `candidate` in candidates(); invalid if `candidate` is not an element of this pool.
    std::optional<std::size_t> seat_index(const Candidate& candidate) const {
        if (candidates_.empty()) {
            return std::nullopt;
        }
        const Candidate* begin = candidates_.data();
        const Candidate* end = begin + candidates_.size();
        const Candidate* seat = &candidate;
        if (seat < begin || seat >= end) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(seat - begin);
    }

    // Pull up to `batch_per_variant` more candidates from every non-exhausted variant in this pool.
    std::size_t grow(std::size_t batch_per_variant) {
        if (batch_per_variant == 0) {
            return 0;
        }
        std::size_t added = 0;
        for (auto& [_, variant] : variants_) {
            added += grow_variant(variant, batch_per_variant);
        }
        return added;
    }

    bool variants_exhausted() const {
        for (const auto& [_, variant] : variants_) {
            if (!variant.enumeration.exhausted) {
                return false;
            }
        }
        return true;
    }

    // Adds a grouping variant that was not in the initial PGD-only pool (e.g. MGD placement fallback).
    // Returns false if the variant was already present or has an empty graph.
    bool add_grouping(const GroupingInfo& grouping) {
        if (grouping.adjacency_graph.get_nodes().empty()) {
            return false;
        }
        if (variants_.contains(&grouping)) {
            return false;
        }
        GroupingVariant variant;
        variant.grouping = &grouping;
        variant.enumeration.variant = &grouping;
        variants_.emplace(&grouping, std::move(variant));
        return true;
    }

    bool has_grouping(const GroupingInfo& grouping) const { return variants_.contains(&grouping); }

private:
    // Resumable enumeration state for one grouping variant. Map nodes are never moved -- the session keeps
    // pointers into its own graph/constraint snapshots. Seatings live in candidates_ on the pool.
    struct GroupingVariant {
        const GroupingInfo* grouping = nullptr;
        GroupingVariantEnumeration enumeration;
        std::size_t candidates_found = 0;
        bool allowed_asics_encoded = false;
    };

    // If the session supplied an ASIC limit, every grouping chip may map only into that set.
    // Encoded once; skipped entirely when allowed_asics_ is empty.
    bool encode_allowed_asics(GroupingVariant& variant) {
        if (variant.allowed_asics_encoded) {
            return true;
        }
        variant.allowed_asics_encoded = true;
        if (allowed_asics_.empty()) {
            return true;
        }
        const std::set<LogicalChipId> grouping_chips(
            variant.grouping->adjacency_graph.get_nodes().begin(), variant.grouping->adjacency_graph.get_nodes().end());
        if (grouping_chips.empty() ||
            !variant.enumeration.constraints.add_required_constraint(grouping_chips, allowed_asics_)) {
            variant.enumeration.exhausted = true;
            return false;
        }
        return true;
    }

    // Pull up to `batch` more distinct placements from one variant. Returns how many were added. The
    // resuming enumeration session tracks excluded full mappings; unique_shapes=false keeps orientations
    // that share an ASIC set but differ in logical-to-physical connectivity.
    std::size_t grow_variant(GroupingVariant& variant, std::size_t batch) {
        GroupingVariantEnumeration& state = variant.enumeration;
        if (state.exhausted) {
            return 0;
        }

        std::size_t added = 0;
        while (added < batch && !state.exhausted) {
            if (!encode_allowed_asics(variant)) {
                break;
            }
            const auto mappings = enumerate_flat_grouping_embeddings(
                *variant.grouping,
                physical_graph_,
                psd_,
                batch - added,
                state.constraints,
                validation_mode_,
                nullptr,
                &state,
                /*unique_shapes=*/false);
            if (mappings.empty()) {
                break;
            }
            for (const auto& mapping : mappings) {
                if (!mapping.success) {
                    continue;
                }
                candidates_.emplace_back(
                    variant.grouping, asic_index_.size(), mapping.target_to_global, asic_index_, neighbors_of_dense_);
                ++variant.candidates_found;
                ++added;
                if (added >= batch) {
                    break;
                }
            }
        }
        return added;
    }

    const AdjacencyGraph<AsicID>& physical_graph_;
    const tt::tt_metal::PhysicalSystemDescriptor& psd_;
    ConnectionValidationMode validation_mode_;
    DenseAsicIndex asic_index_;
    std::vector<std::vector<uint32_t>> neighbors_of_dense_;
    std::map<const GroupingInfo*, GroupingVariant> variants_;
    std::vector<Candidate> candidates_;  // append-only; each Candidate points at its GroupingInfo variant
    std::set<AsicID> allowed_asics_;
};

namespace {

using Candidate = SatPlacementEnumerationSession::Candidate;
using CandidatePool = SatPlacementEnumerationSession::CandidatePool;

// Saturated fabric link counts between two mesh instances' candidate pools. Row `from_seat` is
// pools.at(from_mesh).candidates()[from_seat]; column `to_seat` is the peer mesh's candidate at that index.
class AdjacencyMatrix {
public:
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    std::size_t saturated_link_count(std::size_t from_seat, std::size_t to_seat) const {
        return data_[from_seat * cols_ + to_seat];
    }

    bool satisfies_channel(std::size_t from_seat, std::size_t to_seat, std::size_t required_links) const {
        return saturated_link_count(from_seat, to_seat) >= required_links;
    }

    std::span<const uint8_t> row(std::size_t from_seat) const { return {data_.data() + from_seat * cols_, cols_}; }

    std::size_t saturated_link_count(
        const CandidatePool& from_pool,
        const Candidate& from_seat,
        const CandidatePool& to_pool,
        const Candidate& to_seat) const {
        const std::optional<std::size_t> from_index = from_pool.seat_index(from_seat);
        const std::optional<std::size_t> to_index = to_pool.seat_index(to_seat);
        TT_ASSERT(from_index.has_value() && to_index.has_value());
        return saturated_link_count(*from_index, *to_index);
    }

private:
    friend class AdjacencyMatrixCache;

    // For each machine ASIC, which seat indices in `seats` occupy that ASIC (for seam matrix assembly).
    static std::vector<std::vector<std::size_t>> seat_indices_by_dense(
        const std::vector<Candidate>& seats, std::size_t dense_asic_count) {
        std::vector<std::vector<std::size_t>> by_dense(dense_asic_count);
        for (std::size_t seat = 0; seat < seats.size(); ++seat) {
            for (const uint32_t dense : seats[seat].dense_asics()) {
                by_dense[dense].push_back(seat);
            }
        }
        return by_dense;
    }

    // Each crossing link is one boundary step on `from` landing on a chip in `to`'s footprint. Index `to`
    // seats by dense ASIC once, then walk each `from` seat's boundary links — O(|from|*|boundary|* fans)
    // instead of O(|from|*|to|*|boundary|) pairwise bitset tests.
    static AdjacencyMatrix build(
        const std::vector<Candidate>& from_seats,
        const std::vector<Candidate>& to_seats,
        std::size_t dense_asic_count) {
        AdjacencyMatrix table;
        table.rows_ = from_seats.size();
        table.cols_ = to_seats.size();
        table.data_.assign(table.rows_ * table.cols_, 0);
        const std::vector<std::vector<std::size_t>> to_seats_at_dense =
            seat_indices_by_dense(to_seats, dense_asic_count);
        for (std::size_t from_seat = 0; from_seat < table.rows_; ++from_seat) {
            uint8_t* row = table.data_.data() + from_seat * table.cols_;
            for (const uint32_t dense : from_seats[from_seat].boundary_fabric_links()) {
                for (const std::size_t to_seat : to_seats_at_dense[dense]) {
                    if (row[to_seat] < 255) {
                        ++row[to_seat];
                    }
                }
            }
        }
        return table;
    }

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<uint8_t> data_;
};

// Lazy cache of AdjacencyMatrix tables keyed by (from_mesh, to_mesh). Built fresh each generation.
class AdjacencyMatrixCache {
public:
    const AdjacencyMatrix& adjacency_matrix(
        GlobalMeshId from_mesh, GlobalMeshId to_mesh, const std::map<GlobalMeshId, CandidatePool>& pools) {
        const auto key = std::make_pair(from_mesh, to_mesh);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        const CandidatePool& from_pool = pools.at(from_mesh);
        AdjacencyMatrix built =
            AdjacencyMatrix::build(from_pool.candidates(), pools.at(to_mesh).candidates(), from_pool.asic_count());
        return cache_.emplace(key, std::move(built)).first->second;
    }

private:
    std::map<std::pair<GlobalMeshId, GlobalMeshId>, AdjacencyMatrix> cache_;
};

std::map<GlobalMeshId, CandidatePool> create_sat_placement_pools(
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<GlobalMeshId, ConnectionValidationMode>& sat_intra_mesh_mode_by_mesh,
    const std::map<GlobalMeshId, std::set<AsicID>>& allowed_asics_by_mesh) {
    std::map<GlobalMeshId, CandidatePool> pools;
    for (const auto& [mesh_id, groupings] : global_mesh_groupings) {
        const auto mode_it = sat_intra_mesh_mode_by_mesh.find(mesh_id);
        TT_FATAL(
            mode_it != sat_intra_mesh_mode_by_mesh.end(),
            "Internal error: SAT joint placement: mesh {} has no intra-mesh validation mode",
            *mesh_id);
        std::set<AsicID> allowed_asics;
        if (const auto asic_it = allowed_asics_by_mesh.find(mesh_id); asic_it != allowed_asics_by_mesh.end()) {
            allowed_asics = asic_it->second;
        }
        pools.emplace(
            mesh_id,
            CandidatePool(
                groupings, physical_graph, physical_system_descriptor, mode_it->second, std::move(allowed_asics)));
    }
    return pools;
}

std::size_t grow_sat_placement_pools(
    std::map<GlobalMeshId, CandidatePool>& pools, std::size_t batch_per_variant, PlacementSolveStats* stats) {
    const auto start = std::chrono::steady_clock::now();
    std::size_t grown = 0;
    for (auto& [_, pool] : pools) {
        if (pool.variants_exhausted()) {
            continue;
        }
        grown += pool.grow(batch_per_variant);
    }
    if (stats != nullptr) {
        stats->master_enumeration_elapsed +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
    }
    return grown;
}

std::size_t count_sat_placement_candidates(const std::map<GlobalMeshId, CandidatePool>& pools) {
    std::size_t total = 0;
    for (const auto& [_, pool] : pools) {
        total += pool.candidates().size();
    }
    return total;
}

// Inject MGD fallback variants only when PGD seats are exhausted or the grow cap is hit.
bool inject_sat_placement_fallbacks(
    std::map<GlobalMeshId, CandidatePool>& pools, const std::map<GlobalMeshId, GroupingInfo>& mgd_fallback_by_mesh) {
    bool added = false;
    for (const auto& [mesh_id, fallback] : mgd_fallback_by_mesh) {
        auto it = pools.find(mesh_id);
        if (it == pools.end() || it->second.has_grouping(fallback)) {
            continue;
        }
        if (it->second.add_grouping(fallback)) {
            added = true;
        }
    }
    return added;
}

AdjacencyGraph<const Candidate*> build_sat_placement_seat_graph(
    const std::map<GlobalMeshId, CandidatePool>& pools, const AdjacencyGraph<GlobalMeshId>& mesh_level_graph) {
    AdjacencyMatrixCache adjacency_cache;
    std::map<const Candidate*, std::vector<const Candidate*>> seat_adj;
    std::map<GlobalMeshId, std::vector<const Candidate*>> mesh_seats;
    for (const auto& [mesh_id, pool] : pools) {
        auto& seats = mesh_seats[mesh_id];
        seats.reserve(pool.candidates().size());
        for (const Candidate& candidate : pool.candidates()) {
            const Candidate* seat = &candidate;
            seats.push_back(seat);
            seat_adj[seat] = {};
        }
    }

    std::set<std::pair<GlobalMeshId, GlobalMeshId>> seamed_pairs;
    for (const GlobalMeshId& m1 : mesh_level_graph.get_nodes()) {
        for (const GlobalMeshId& m2 : mesh_level_graph.get_neighbors(m1)) {
            if (m1 >= m2 || !seamed_pairs.insert({m1, m2}).second) {
                continue;
            }
            const AdjacencyMatrix& adjacency = adjacency_cache.adjacency_matrix(m1, m2, pools);
            const auto& s1 = mesh_seats.at(m1);
            const auto& s2 = mesh_seats.at(m2);
            TT_ASSERT(adjacency.rows() == s1.size() && adjacency.cols() == s2.size());
            for (std::size_t from_seat = 0; from_seat < s1.size(); ++from_seat) {
                for (std::size_t to_seat = 0; to_seat < s2.size(); ++to_seat) {
                    const std::size_t links = adjacency.saturated_link_count(from_seat, to_seat);
                    if (links == 0) {
                        continue;
                    }
                    for (std::size_t copy = 0; copy < links; ++copy) {
                        seat_adj[s1[from_seat]].push_back(s2[to_seat]);
                        seat_adj[s2[to_seat]].push_back(s1[from_seat]);
                    }
                }
            }
        }
    }
    return AdjacencyGraph<const Candidate*>(std::move(seat_adj));
}

bool build_sat_placement_constraints(
    const std::map<GlobalMeshId, CandidatePool>& pools,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<GlobalMeshId, const Candidate*>& constraints) {
    constraints = {};
    std::map<const Candidate*, std::vector<uint32_t>> seat_to_asics;
    for (const auto& [mesh_id, pool] : pools) {
        std::set<const Candidate*> seats;
        for (const Candidate& candidate : pool.candidates()) {
            const Candidate* seat = &candidate;
            seats.insert(seat);
            seat_to_asics[seat] = candidate.dense_asics();
        }
        if (seats.empty() || !constraints.add_required_constraint(mesh_id, seats)) {
            return false;
        }
    }
    // Chip disjointness: each ASIC is claimed by at most one chosen seat.
    if (!constraints.add_resource_constraint<uint32_t>(seat_to_asics)) {
        return false;
    }
    // Group the single-host seats by host so the host-count cap below can constrain how many distinct
    // hosts are occupied.
    std::map<std::string, std::set<const Candidate*>> seats_by_host;
    for (const auto& [seat, _] : seat_to_asics) {
        std::string host;
        bool single_host = true;
        for (const AsicID& asic : seat->asics()) {
            const std::string name = physical_system_descriptor.get_host_name_for_asic(asic);
            if (host.empty()) {
                host = name;
            } else if (name != host) {
                single_host = false;
                break;
            }
        }
        if (single_host && !host.empty()) {
            seats_by_host[host].insert(seat);
        }
    }
    std::vector<std::set<const Candidate*>> host_seat_groups;
    host_seat_groups.reserve(seats_by_host.size());
    for (auto& [_, seats] : seats_by_host) {
        if (!seats.empty()) {
            host_seat_groups.push_back(std::move(seats));
        }
    }
    if (host_seat_groups.size() > 1) {
        if (!constraints.set_same_rank_groups_constraint({}, host_seat_groups)) {
            return false;
        }
        // HARD host-count cap: force the placement into the minimum number of hosts. Chip disjointness
        // (add_resource_constraint above) already caps a host at floor(host_chips / mesh_chips) meshes, so
        // capping the number of occupied hosts at ceil(meshes / capacity) forces every used host to be
        // packed full. Unlike per-host fill-all, this is a GLOBAL constraint the solver cannot dodge by
        // spreading across more locally-"full" hosts. If the cap is infeasible at the current candidate
        // seats, next() grows pools and only drops the cap after growth is exhausted.
        std::size_t asics_per_mesh = 0;
        for (const auto& [seat, asics] : seat_to_asics) {
            (void)seat;
            asics_per_mesh = std::max(asics_per_mesh, asics.size());
        }
        std::map<std::string, std::size_t> host_asic_counts;
        for (const auto& [asic_id, desc] : physical_system_descriptor.get_asic_descriptors()) {
            (void)asic_id;
            ++host_asic_counts[desc.host_name];
        }
        std::size_t max_host_asics = 0;
        for (const auto& [host, count] : host_asic_counts) {
            (void)host;
            max_host_asics = std::max(max_host_asics, count);
        }
        const std::size_t capacity = (asics_per_mesh > 0) ? (max_host_asics / asics_per_mesh) : 0;
        if (capacity > 0) {
            const std::size_t k = (pools.size() + capacity - 1) / capacity;
            // Only cap when it can actually reduce the host count. If the meshes need every available host
            // anyway (k >= host groups -- e.g. a superpod that fully packs all its hosts), the cap constrains
            // nothing but bolts an expensive at-most-k CNF onto an already-large solve, so the grow loop stalls.
            // Skip it: the unconstrained solve already yields the only (all-hosts-full) packing.
            if (k < host_seat_groups.size()) {
                constraints.set_max_same_rank_groups_used(k);
            }
        }
    }
    return true;
}

// Stage 3: decode chosen seats into ASIC placements.
AssignedMeshes decode_sat_placement(const MappingResult<GlobalMeshId, const Candidate*>& result) {
    AssignedMeshes assignment;
    assignment.reserve(result.target_to_global.size());
    for (const auto& [mesh_id, seat] : result.target_to_global) {
        if (seat == nullptr) {
            continue;
        }
        PsdPlacement placement;
        placement.mesh_node_to_asic_position = seat->variant()->mesh_node_to_asic_position;
        placement.asics.insert(seat->asics().begin(), seat->asics().end());
        assignment.push_back(PlacedMesh{mesh_id, std::move(placement), seat->variant()->name, seat->variant()->type});
    }
    return assignment;
}

constexpr std::size_t kGrowBudgetPerVariant = 32;
constexpr std::size_t kMaxGrowthCycles = 4;

void apply_valid_groupings_map(
    const ValidGroupingsMap& valid_groupings,
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const std::vector<MeshId>& mesh_ids,
    std::map<MeshId, std::vector<GroupingInfo>>& primary) {
    const std::unordered_map<InstanceName, std::vector<GroupingInfo>>* mesh_groupings = nullptr;
    if (valid_groupings.contains("MESH")) {
        mesh_groupings = &valid_groupings.at("MESH");
    }
    const auto mesh_id_to_instance_name = mesh_graph_descriptor.mesh_id_to_instance_name();
    for (const MeshId mesh_id : mesh_ids) {
        const auto name_it = mesh_id_to_instance_name.find(mesh_id);
        TT_FATAL(
            name_it != mesh_id_to_instance_name.end(),
            "Internal error: SAT placement: mesh {} has no instance name, so its grouping variants cannot be looked up",
            *mesh_id);
        TT_FATAL(
            mesh_groupings != nullptr,
            "Internal error: SAT placement: mesh '{}' (mesh {}) has no grouping in the provided valid-groupings map",
            name_it->second,
            *mesh_id);
        const auto groupings_it = mesh_groupings->find(name_it->second);
        TT_FATAL(
            groupings_it != mesh_groupings->end() && !groupings_it->second.empty(),
            "Internal error: SAT placement: mesh '{}' (mesh {}) has no grouping in the provided valid-groupings map",
            name_it->second,
            *mesh_id);
        primary.emplace(mesh_id, groupings_it->second);
    }
}

}  // namespace

SatPlacementEnumerationSession::SatPlacementEnumerationSession(
    const PhysicalGroupingDescriptor& physical_grouping_descriptor,
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    PlacementSolveStats* stats,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    bool unique_shapes,
    const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank) :
    physical_system_descriptor_(&physical_system_descriptor), stats_(stats), unique_shapes_(unique_shapes) {
    using tt::tt_metal::experimental::tt_fabric::build_logical_multi_mesh_adjacency_graph;

    const ValidGroupingsMap valid_groupings = physical_grouping_descriptor.get_valid_groupings_for_mgd(
        mesh_graph_descriptor,
        physical_system_descriptor,
        pinnings,
        /*require_placement=*/false,
        fabric_node_id_to_mesh_rank);
    const auto mesh_it = valid_groupings.find("MESH");
    if (mesh_it == valid_groupings.end() || mesh_it->second.empty()) {
        return;
    }

    const auto logical = build_logical_multi_mesh_adjacency_graph(mesh_graph_descriptor);
    if (logical.mesh_adjacency_graphs_.empty()) {
        return;
    }
    mesh_level_graph_ = logical.mesh_level_graph_;

    const ValidGroupingsMap mgd_fallbacks_by_key = PhysicalGroupingDescriptor::get_mgd_placement_fallbacks_for_mgd(
        mesh_graph_descriptor, physical_system_descriptor, pinnings);
    const std::unordered_map<InstanceName, std::vector<GroupingInfo>>* mgd_fallback_mesh = nullptr;
    if (mgd_fallbacks_by_key.contains("MESH")) {
        mgd_fallback_mesh = &mgd_fallbacks_by_key.at("MESH");
    }

    const std::unordered_map<InstanceName, std::vector<GroupingInfo>>& mesh_groupings = mesh_it->second;
    const auto mesh_id_to_instance_name = mesh_graph_descriptor.mesh_id_to_instance_name();
    for (const auto& [mesh_id, unused_graph] : logical.mesh_adjacency_graphs_) {
        (void)unused_graph;
        const auto name_it = mesh_id_to_instance_name.find(mesh_id);
        TT_FATAL(
            name_it != mesh_id_to_instance_name.end(),
            "Internal error: SAT placement: mesh {} has no instance name, so its grouping variants cannot be looked up",
            *mesh_id);
        const InstanceName& grouping_key = name_it->second;
        const auto groupings_it = mesh_groupings.find(grouping_key);
        const bool has_pgd = groupings_it != mesh_groupings.end() && !groupings_it->second.empty();
        std::optional<GroupingInfo> mgd_fallback;
        if (mgd_fallback_mesh != nullptr) {
            const auto mgd_fb_it = mgd_fallback_mesh->find(grouping_key);
            if (mgd_fb_it != mgd_fallback_mesh->end() && !mgd_fb_it->second.empty()) {
                mgd_fallback = mgd_fb_it->second.front();
                mgd_fallback_by_mesh_.emplace(mesh_id, *mgd_fallback);
            }
        }
        TT_FATAL(
            has_pgd || mgd_fallback.has_value(),
            "Internal error: SAT placement: mesh '{}' (mesh {}) has no PGD grouping and no embeddable MGD fallback",
            grouping_key,
            *mesh_id);
        global_mesh_groupings_.emplace(
            mesh_id, has_pgd ? groupings_it->second : std::vector<GroupingInfo>{*mgd_fallback});
        sat_intra_mesh_mode_by_mesh_.emplace(
            mesh_id,
            mesh_graph_descriptor.is_intra_mesh_policy_relaxed(mesh_id) ? ConnectionValidationMode::RELAXED
                                                                        : ConnectionValidationMode::STRICT);
    }
    relaxed_inter_mesh_policy_ = mesh_graph_descriptor.is_inter_mesh_policy_relaxed();
    finish_init(asic_id_to_mesh_rank);
}

SatPlacementEnumerationSession::SatPlacementEnumerationSession(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    PlacementSolveStats* stats,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank,
    bool unique_shapes) :
    physical_system_descriptor_(&physical_system_descriptor), stats_(stats), unique_shapes_(unique_shapes) {
    using tt::tt_metal::experimental::tt_fabric::build_logical_multi_mesh_adjacency_graph;

    const auto logical = build_logical_multi_mesh_adjacency_graph(mesh_graph_descriptor);
    if (logical.mesh_adjacency_graphs_.empty()) {
        return;
    }
    mesh_level_graph_ = logical.mesh_level_graph_;

    std::vector<MeshId> mesh_ids;
    mesh_ids.reserve(logical.mesh_adjacency_graphs_.size());
    for (const auto& [mesh_id, unused_graph] : logical.mesh_adjacency_graphs_) {
        (void)unused_graph;
        mesh_ids.push_back(mesh_id);
        sat_intra_mesh_mode_by_mesh_.emplace(
            mesh_id,
            mesh_graph_descriptor.is_intra_mesh_policy_relaxed(mesh_id) ? ConnectionValidationMode::RELAXED
                                                                        : ConnectionValidationMode::STRICT);
    }
    apply_valid_groupings_map(
        PhysicalGroupingDescriptor::get_mgd_placement_fallbacks_for_mgd(
            mesh_graph_descriptor, physical_system_descriptor, pinnings),
        mesh_graph_descriptor,
        mesh_ids,
        global_mesh_groupings_);
    fallbacks_in_ = true;
    relaxed_inter_mesh_policy_ = mesh_graph_descriptor.is_inter_mesh_policy_relaxed();
    finish_init(asic_id_to_mesh_rank);
}

void SatPlacementEnumerationSession::finish_init(
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    if (stats_ != nullptr) {
        stats_->meshes_total = mesh_level_graph_.get_nodes().size();
        stats_->master_solve_attempted = true;
    }
    physical_graph_ = AdjacencyGraph<AsicID>(
        tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd(*physical_system_descriptor_));

    for (const GlobalMeshId& mesh_id : mesh_level_graph_.get_nodes()) {
        if (!global_mesh_groupings_.contains(mesh_id)) {
            log_warning(tt::LogFabric, "SAT joint placement: mesh {} has no grouping variants; falling back", *mesh_id);
            return;
        }
    }

    std::map<GlobalMeshId, std::set<AsicID>> allowed_asics_by_mesh;
    for (const auto& [mesh_id, unused_groupings] : global_mesh_groupings_) {
        (void)unused_groupings;
        const auto asic_it = asic_id_to_mesh_rank.find(mesh_id);
        if (asic_it == asic_id_to_mesh_rank.end() || asic_it->second.empty()) {
            continue;
        }
        std::set<AsicID> asics;
        std::unordered_set<AsicID> extra_asics;
        extra_asics.reserve(asic_it->second.size());
        for (const auto& [asic_id, unused_rank] : asic_it->second) {
            (void)unused_rank;
            asics.insert(asic_id);
            extra_asics.insert(asic_id);
        }
        extra_required_.emplace_back(mesh_id, std::move(extra_asics));
        allowed_asics_by_mesh.emplace(mesh_id, std::move(asics));
    }
    pools_ = std::make_unique<std::map<GlobalMeshId, CandidatePool>>(create_sat_placement_pools(
        global_mesh_groupings_,
        physical_graph_,
        *physical_system_descriptor_,
        sat_intra_mesh_mode_by_mesh_,
        allowed_asics_by_mesh));
    grow_sat_placement_pools(*pools_, kGrowBudgetPerVariant, stats_);
    master_solve_ = std::make_unique<MasterSolve>(this);
    ready_ = true;
}

bool SatPlacementEnumerationSession::MasterSolve::restart(bool relaxed_mode, bool drop_cap) {
    reset();
    AdjacencyGraph<const Candidate*> seat_graph =
        build_sat_placement_seat_graph(*owner_->pools_, owner_->mesh_level_graph_);
    if (!(build_sat_placement_constraints(
              *owner_->pools_, *owner_->physical_system_descriptor_, owner_->constraints_) &&
          owner_->apply_extra_constraints(owner_->constraints_))) {
        return false;
    }
    if (drop_cap) {
        owner_->constraints_.set_max_same_rank_groups_used(0);
    }
    const auto encode_start = std::chrono::steady_clock::now();
    session = std::make_unique<TopologyMappingEnumerationSession<MeshId, const Candidate*>>(
        owner_->mesh_level_graph_,
        seat_graph,
        owner_->constraints_,
        relaxed_mode ? ConnectionValidationMode::RELAXED : ConnectionValidationMode::STRICT,
        /*quiet_mode=*/true,
        TopologyMappingSolverEngine::Sat,
        owner_->unique_shapes_);
    for (const auto& mapping : owner_->excluded_seat_maps()) {
        session->exclude_mapping(mapping);
    }
    if (owner_->stats_ != nullptr) {
        owner_->stats_->master_sat_vars = seat_graph.get_nodes().size();
        owner_->stats_->master_sat_clauses = seat_graph.get_nodes().size();
        owner_->stats_->master_encode_elapsed +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - encode_start);
    }
    relaxed = relaxed_mode;
    drop_host_cap = drop_cap;
    return true;
}

MappingResult<MeshId, const Candidate*> SatPlacementEnumerationSession::MasterSolve::next(bool drop_cap) {
    MappingResult<MeshId, const Candidate*> failure;
    failure.success = false;
    const bool relaxed_mode = owner_->relaxed_inter_mesh_policy_;
    // Unchanged mode and host cap: this is another model from the session already encoded.
    if (!matches(relaxed_mode, drop_cap) && !restart(relaxed_mode, drop_cap)) {
        return failure;
    }
    ++owner_->attempts_;
    const auto solve_start = std::chrono::steady_clock::now();
    MappingResult<MeshId, const Candidate*> result = session->next();
    if (owner_->stats_ != nullptr) {
        owner_->stats_->master_solve_elapsed +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - solve_start);
        owner_->stats_->master_sat_attempts = owner_->attempts_;
    }
    return result;
}

void SatPlacementEnumerationSession::MasterSolve::block_assigned(const AssignedMeshes& assigned) {
    if (session == nullptr || !session->started()) {
        return;
    }
    std::map<MeshId, const Candidate*> seats;
    for (const PlacedMesh& placed : assigned) {
        const std::set<const Candidate*> matching = owner_->seats_matching(placed.mesh_id, placed.placement.asics);
        if (matching.empty()) {
            continue;
        }
        seats.emplace(placed.mesh_id, *matching.begin());
    }
    if (!seats.empty()) {
        session->exclude_mapping(seats);
    }
}

SatPlacementEnumerationSession::~SatPlacementEnumerationSession() = default;

void SatPlacementEnumerationSession::invalidate_pending_solve() {
    pending_.clear();
    pending_index_ = 0;
    solved_ = false;
    // Constraints changed: the live encoding no longer matches. The next placement builds a new session.
    if (master_solve_ != nullptr) {
        master_solve_->reset();
    }
}

std::set<const SatPlacementEnumerationSession::Candidate*> SatPlacementEnumerationSession::seats_matching(
    GlobalMeshId mesh_id, const std::unordered_set<AsicID>& asics) const {
    std::set<const Candidate*> seats;
    if (pools_ == nullptr) {
        return seats;
    }
    const auto pool_it = pools_->find(mesh_id);
    if (pool_it == pools_->end()) {
        return seats;
    }
    for (const Candidate& candidate : pool_it->second.candidates()) {
        if (candidate.asics().size() != asics.size()) {
            continue;
        }
        bool match = true;
        for (const AsicID& asic : candidate.asics()) {
            if (!asics.contains(asic)) {
                match = false;
                break;
            }
        }
        if (match) {
            seats.insert(&candidate);
        }
    }
    return seats;
}

bool SatPlacementEnumerationSession::apply_extra_constraints(
    MappingConstraints<GlobalMeshId, const Candidate*>& constraints) const {
    for (const auto& [mesh_id, asics] : extra_forbidden_) {
        const std::set<const Candidate*> seats = seats_matching(mesh_id, asics);
        if (seats.empty()) {
            continue;
        }
        if (!constraints.add_forbidden_constraint(mesh_id, seats)) {
            return false;
        }
    }
    for (const auto& [mesh_id, asics] : extra_required_) {
        const std::set<const Candidate*> seats = seats_matching(mesh_id, asics);
        if (seats.empty() || !constraints.add_required_constraint(mesh_id, seats)) {
            return false;
        }
    }
    return true;
}

std::vector<std::map<GlobalMeshId, const Candidate*>> SatPlacementEnumerationSession::excluded_seat_maps() const {
    std::vector<std::map<GlobalMeshId, const Candidate*>> excluded;
    excluded.reserve(yielded_footprints_.size());
    for (const auto& footprints : yielded_footprints_) {
        std::map<GlobalMeshId, const Candidate*> seats;
        for (const auto& [mesh_id, asics] : footprints) {
            const std::set<const Candidate*> matching = seats_matching(mesh_id, asics);
            if (matching.empty()) {
                continue;
            }
            seats.emplace(mesh_id, *matching.begin());
        }
        if (!seats.empty()) {
            excluded.push_back(std::move(seats));
        }
    }
    return excluded;
}

void SatPlacementEnumerationSession::remember_yielded(const AssignedMeshes& assigned) {
    std::map<GlobalMeshId, std::unordered_set<AsicID>> footprints;
    for (const PlacedMesh& placed : assigned) {
        footprints.emplace(placed.mesh_id, placed.placement.asics);
    }
    yielded_footprints_.push_back(std::move(footprints));
}

bool SatPlacementEnumerationSession::add_forbidden_constraint(MeshId mesh_id, const std::unordered_set<AsicID>& asics) {
    extra_forbidden_.emplace_back(mesh_id, asics);
    if (!ready_) {
        return true;
    }
    const std::set<const Candidate*> seats = seats_matching(mesh_id, asics);
    if (!seats.empty()) {
        MappingConstraints<GlobalMeshId, const Candidate*> trial;
        if (build_sat_placement_constraints(*pools_, *physical_system_descriptor_, trial) &&
            !apply_extra_constraints(trial)) {
            extra_forbidden_.pop_back();
            return false;
        }
    }
    invalidate_pending_solve();
    return true;
}

bool SatPlacementEnumerationSession::add_forbidden_constraint(const PlacedMesh& placed) {
    return add_forbidden_constraint(placed.mesh_id, placed.placement.asics);
}

bool SatPlacementEnumerationSession::add_required_constraint(MeshId mesh_id, const std::unordered_set<AsicID>& asics) {
    if (!ready_) {
        extra_required_.emplace_back(mesh_id, asics);
        return true;
    }
    extra_required_.emplace_back(mesh_id, asics);
    const std::set<const Candidate*> seats = seats_matching(mesh_id, asics);
    if (!seats.empty()) {
        MappingConstraints<GlobalMeshId, const Candidate*> trial;
        if (build_sat_placement_constraints(*pools_, *physical_system_descriptor_, trial) &&
            !apply_extra_constraints(trial)) {
            extra_required_.pop_back();
            return false;
        }
    }
    invalidate_pending_solve();
    return true;
}

bool SatPlacementEnumerationSession::add_required_constraint(const PlacedMesh& placed) {
    return add_required_constraint(placed.mesh_id, placed.placement.asics);
}

bool SatPlacementEnumerationSession::exclude_mapping(const AssignedMeshes& assigned) {
    if (!ready_ || assigned.empty()) {
        return false;
    }
    // Exclude the *combination* (this exact set of mesh footprints), not each mesh placement independently.
    // Forbidding each footprint on its own would also rule out other valid mappings that happen to reuse one
    // of these footprints. remember_yielded() records the combination so a later rebuilt solve excludes just
    // it. A live session is blocked in place so the next next() skips it without re-encoding.
    remember_yielded(assigned);
    if (master_solve_ != nullptr) {
        master_solve_->block_assigned(assigned);
    }
    return true;
}

AssignedMeshes SatPlacementEnumerationSession::next() {
    if (!ready_) {
        return {};
    }
    if (pending_index_ < pending_.size()) {
        remember_yielded(pending_[pending_index_]);
        AssignedMeshes assigned = pending_[pending_index_++];
        if (stats_ != nullptr) {
            stats_->meshes_placed = assigned.size();
            stats_->success = stats_->meshes_total != 0 && stats_->meshes_placed == stats_->meshes_total;
        }
        return assigned;
    }
    if (solved_) {
        return {};
    }

    // Same mode and host cap: next() is another model from the live session.
    MappingResult<MeshId, const Candidate*> result = master_solve_->next(/*drop_cap=*/false);
    while (!result.success && cycle_ < kMaxGrowthCycles) {
        ++cycle_;
        const bool at_cap = cycle_ >= kMaxGrowthCycles;
        std::size_t grown = grow_sat_placement_pools(*pools_, kGrowBudgetPerVariant, stats_);
        if ((grown == 0 || at_cap) && !fallbacks_in_ &&
            inject_sat_placement_fallbacks(*pools_, mgd_fallback_by_mesh_)) {
            fallbacks_in_ = true;
            grown += grow_sat_placement_pools(*pools_, kGrowBudgetPerVariant, stats_);
        }
        if (grown == 0) {
            break;
        }
        // Growth reallocates candidate pointers, so the live encoding cannot be reused.
        master_solve_->reset();
        result = master_solve_->next(/*drop_cap=*/false);
        if (at_cap) {
            break;
        }
    }
    // Growth is exhausted and the capped solve still failed. Dropping the cap changes the encoding.
    if (!result.success) {
        result = master_solve_->next(/*drop_cap=*/true);
    }
    if (!result.success) {
        solved_ = true;
        bool complete = true;
        for (const auto& [_, pool] : *pools_) {
            if (!pool.variants_exhausted()) {
                complete = false;
                break;
            }
        }
        const std::size_t candidates = count_sat_placement_candidates(*pools_);
        if (stats_ != nullptr) {
            stats_->master_growth_rounds = cycle_;
            stats_->master_candidates_enumerated = candidates;
            stats_->candidate_lists_complete = complete;
            stats_->master_sat_attempts = attempts_;
        }
        log_warning(
            tt::LogFabric,
            "SAT joint placement: no placement found after {} attempt(s) and {} growth cycle(s) over {} candidate(s); "
            "candidate lists {} -- the UNSAT verdict is {}",
            attempts_,
            cycle_,
            candidates,
            complete ? "COMPLETE" : "TRUNCATED",
            complete ? "trustworthy" : "NOT trustworthy");
        return {};
    }

    AssignedMeshes assigned = decode_sat_placement(result);
    remember_yielded(assigned);
    if (stats_ != nullptr) {
        bool lists_complete = true;
        for (const auto& [_, pool] : *pools_) {
            if (!pool.variants_exhausted()) {
                lists_complete = false;
                break;
            }
        }
        stats_->master_solve_success = true;
        stats_->master_growth_rounds = cycle_;
        stats_->master_candidates_enumerated = count_sat_placement_candidates(*pools_);
        stats_->candidate_lists_complete = lists_complete;
        stats_->meshes_placed = assigned.size();
        stats_->success = stats_->meshes_total != 0 && assigned.size() == stats_->meshes_total;
    }
    return assigned;
}

std::vector<AssignedMeshes> SatPlacementEnumerationSession::all() {
    std::vector<AssignedMeshes> solutions;
    while (true) {
        AssignedMeshes assigned = next();
        if (assigned.empty()) {
            break;
        }
        solutions.push_back(std::move(assigned));
    }
    return solutions;
}

}  // namespace tt::tt_fabric
