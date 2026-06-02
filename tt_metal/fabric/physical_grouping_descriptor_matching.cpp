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
#include <cctype>
#include <cstdlib>
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

// Helper function to build adjacency graph from row-major mesh connection.
// LINE neighbors are always included. When `ring_dims[d]` is true, also wrap both ends of dimension d.
// Missing `ring_dims` entries are treated as LINE (no wrap). RING wrap is skipped when dim < 3.
AdjacencyGraph<uint32_t> build_row_major_mesh_graph(
    const std::vector<uint32_t>& instance_ids,
    const std::vector<int32_t>& dims,
    const std::string& grouping_name,
    uint32_t connections_per_edge,
    const std::vector<bool>& ring_dims = {}) {
    std::map<uint32_t, std::vector<uint32_t>> adj_map;

    if (instance_ids.empty() || dims.empty()) {
        return AdjacencyGraph<uint32_t>(adj_map);
    }

    // Calculate total size
    int64_t total_size = 1;
    for (int32_t dim : dims) {
        if (dim <= 0) {
            break;
        }
        total_size *= dim;
        if (total_size > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            total_size = -1;  // signal overflow; comparison below will throw
            break;
        }
    }

    if (total_size < 0 || static_cast<size_t>(total_size) != instance_ids.size()) {
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

        for (int32_t dim_idx = 0; dim_idx < static_cast<int32_t>(dims.size()); ++dim_idx) {
            const int32_t dim_size = dims[dim_idx];
            const int32_t coord_val = coords[dim_idx];
            const bool is_ring =
                dim_idx < static_cast<int32_t>(ring_dims.size()) && ring_dims[static_cast<size_t>(dim_idx)];

            auto add_neighbor_coord = [&](int32_t neighbor_coord_val) {
                std::vector<int32_t> neighbor_coords = coords;
                neighbor_coords[dim_idx] = neighbor_coord_val;
                uint32_t neighbor_id = instance_ids[get_index(neighbor_coords)];
                for (uint32_t conn = 0; conn < connections_per_edge; ++conn) {
                    adj_map[node_id].push_back(neighbor_id);
                    adj_map[neighbor_id].push_back(node_id);
                }
            };

            // +direction LINE neighbor (one-sided, matches PGD flatten and main — do not also walk -direction
            // or each undirected edge is inserted twice and STRICT matching sees 2 channels per edge).
            if (coord_val < dim_size - 1) {
                add_neighbor_coord(coord_val + 1);
            }

            // RING wrap: connect coord 0 to dim-1 (skip dim < 3; bidirectional push covers both ends).
            if (is_ring && dim_size >= 3 && coord_val == 0) {
                add_neighbor_coord(dim_size - 1);
            }
        }
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

struct MgdDeviceTopology {
    std::vector<int32_t> dims;
    std::vector<bool> ring_dims;
};

std::optional<MgdDeviceTopology> get_mgd_instance_device_topology(
    const MeshGraphDescriptor& mesh_graph_descriptor, const std::string& instance_name) {
    const auto& instance_ids = mesh_graph_descriptor.instances_by_name(instance_name);
    if (instance_ids.empty()) {
        return std::nullopt;
    }
    const auto& instance = mesh_graph_descriptor.get_instance(instance_ids[0]);

    const proto::TorusTopology* device_topology = nullptr;
    if (instance.kind == NodeKind::Mesh) {
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
        if (mesh_desc != nullptr) {
            device_topology = &mesh_desc->device_topology();
        }
    } else if (instance.kind == NodeKind::Switch) {
        const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(instance.desc);
        if (switch_desc != nullptr) {
            device_topology = &switch_desc->device_topology();
        }
    }
    if (device_topology == nullptr || device_topology->dims().empty()) {
        return std::nullopt;
    }

    MgdDeviceTopology topo;
    topo.dims.assign(device_topology->dims().begin(), device_topology->dims().end());
    topo.ring_dims.reserve(device_topology->dim_types_size());
    for (int i = 0; i < device_topology->dim_types_size(); ++i) {
        topo.ring_dims.push_back(device_topology->dim_types(i) == proto::TorusTopology::RING);
    }
    return topo;
}

GroupingInfo finalize_mesh_grouping_with_device_topology(
    const GroupingInfo& grouping,
    const MgdDeviceTopology& device_topo,
    const std::map<uint32_t, uint32_t>* mgd_to_pgd_nodes = nullptr) {
    const bool has_ring =
        std::any_of(device_topo.ring_dims.begin(), device_topo.ring_dims.end(), [](bool is_ring) { return is_ring; });
    if (!has_ring) {
        return grouping;
    }

    int64_t num_nodes = 1;
    for (int32_t dim : device_topo.dims) {
        num_nodes *= dim;
    }

    std::vector<uint32_t> node_ids;
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
    // larger asic_count; reset it to the node count so the grouping stays self-consistent (is_flattened()
    // stays true and downstream PSD placement does not try to re-flatten an already-flattened mesh).
    result.asic_count = static_cast<uint32_t>(num_nodes);
    return result;
}

struct MeshTopologyMatch {
    std::string name;
    size_t idx = 0;
    MappingResult<uint32_t, uint32_t> mapping;
};

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

    // Build the graph with the MGD's declared per-dimension topology (RING vs LINE). Using the real wrap
    // edges is what restricts the match to the correct PGD topology variant: a RING/RING MGD is a torus, so
    // only the TORUSXY variant contains all its wrap edges and matches, while MESH/TORUSX/TORUSY (missing
    // some wraps) correctly fail to match. (A LINE-only graph here would embed in every variant.)
    std::vector<bool> ring_dims;
    ring_dims.reserve(device_topology.dim_types_size());
    for (int i = 0; i < device_topology.dim_types_size(); ++i) {
        ring_dims.push_back(device_topology.dim_types(i) == proto::TorusTopology::RING);
    }
    return build_row_major_mesh_graph(asic_ids, device_dims, "", 1, ring_dims);
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

    // LINE-only graph for topology matching (RING edges are added when groupings are committed).
    return build_row_major_mesh_graph(asic_ids, device_dims, "", 1);
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

std::set<uint32_t> find_pgd_nodes_at_asic_position(
    const GroupingInfo& pgd_grouping, const tt::tt_metal::ASICPosition& position) {
    std::set<uint32_t> pgd_nodes;
    for (const uint32_t node_id : pgd_grouping.adjacency_graph.get_nodes()) {
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
    const GroupingInfo& grouping, const std::map<uint32_t, uint32_t>& mgd_node_to_grouping_node) {
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

std::optional<std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>>> filter_pinnings_for_mesh_ids(
    const std::optional<std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>>>& pinnings,
    const std::set<uint32_t>& applicable_mesh_ids) {
    if (!pinnings.has_value()) {
        return std::nullopt;
    }
    std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>> filtered;
    filtered.reserve(pinnings->size());
    for (const auto& pinning : *pinnings) {
        if (applicable_mesh_ids.contains(pinning.second.mesh_id.get())) {
            filtered.push_back(pinning);
        }
    }
    if (filtered.empty()) {
        return std::nullopt;
    }
    return filtered;
}

// Returns the number of required constraints added (one per MGD node that resolved to >=1 PGD node). May be
// 0 even when `pinnings` is non-empty (e.g. no pinned ASIC position maps to a PGD node in this grouping); the
// caller skips that PGD variant when pinnings were required.
std::size_t add_mgd_to_pgd_asic_position_pinning_constraints(
    MappingConstraints<uint32_t, uint32_t>& constraints,
    const GroupingInfo& pgd_grouping,
    const std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>>& pinnings) {
    std::map<uint32_t, std::set<uint32_t>> mgd_node_to_pgd_nodes;
    for (const auto& [position, fabric_node] : pinnings) {
        const auto pgd_nodes = find_pgd_nodes_at_asic_position(pgd_grouping, position);
        if (pgd_nodes.empty()) {
            continue;
        }
        mgd_node_to_pgd_nodes[fabric_node.chip_id].insert(pgd_nodes.begin(), pgd_nodes.end());
    }
    std::size_t constraints_added = 0;
    for (const auto& [mgd_node, pgd_nodes] : mgd_node_to_pgd_nodes) {
        if (!pgd_nodes.empty()) {
            constraints.add_required_constraint(mgd_node, pgd_nodes);
            ++constraints_added;
        }
    }
    return constraints_added;
}

}  // namespace

namespace tt::tt_fabric {

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::optional<std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>>>& pinnings) const {
    return get_valid_groupings_for_mgd(mesh_graph_descriptor, &physical_system_descriptor, pinnings);
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor,
    const std::optional<std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>>>& pinnings) const {
    ValidGroupingsMap result;

    // ===== PHASE 0: Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts) =====
    // This step calculates required ASIC counts bottom-up and builds adjacency graphs
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos =
        PhysicalGroupingDescriptor::build_mgd_to_grouping_info_map(mesh_graph_descriptor);

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

        const std::set<uint32_t> applicable_mesh_ids =
            get_mesh_ids_for_mgd_instance_name(mesh_graph_descriptor, instance_name);
        const auto applicable_pinnings = filter_pinnings_for_mesh_ids(pinnings, applicable_mesh_ids);

        // Required nodes from MGD adjacency graph (this represents the topology pattern to match)
        size_t required_nodes = mgd_grouping_info.adjacency_graph.get_nodes().size();

        // Cheap necessary-condition prefilter for the (expensive) topology solve. solve_topology_mapping
        // looks for an injective edge-preserving map of the MGD graph (target) into a PGD variant (global),
        // so every MGD edge must land on a distinct PGD edge -> |E(PGD)| >= |E(MGD)| is required. A RING/RING
        // MGD is a full torus (degree 4 everywhere, ~2*N edges) while the MESH/TORUSX/TORUSY variants of the
        // same grid drop wrap edges, so they have strictly fewer edges and can never contain it. Counting
        // edges is O(V); the SAT solve it skips is many orders of magnitude slower (seconds per 128-node
        // candidate), so this eliminates the provably-impossible variants up front instead of solving them.
        auto count_undirected_edges = [](const AdjacencyGraph<uint32_t>& g) -> size_t {
            size_t directed = 0;
            for (uint32_t node : g.get_nodes()) {
                directed += g.get_neighbors(node).size();
            }
            return directed / 2;  // each undirected edge is stored from both endpoints
        };
        const size_t required_edges = count_undirected_edges(mgd_grouping_info.adjacency_graph);

        const auto device_topo = get_mgd_instance_device_topology(mesh_graph_descriptor, instance_name);

        auto normalized_dims = [](std::vector<int32_t> dims) {
            std::sort(dims.begin(), dims.end());
            return dims;
        };
        const std::vector<int32_t> required_grid_dims =
            device_topo.has_value() ? normalized_dims(device_topo->dims) : std::vector<int32_t>{};

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

        // Process difference levels from closest to farthest; commit only when embedding on PSD succeeds
        std::vector<MeshTopologyMatch> best_matches_topology;
        std::vector<MeshTopologyMatch> best_matches_psd_placed;
        size_t last_topology_match_count = 0;

        bool committed_pgd_matches = false;
        for (const auto& [node_diff, name_idx_pairs] : candidates_by_diff) {
            best_matches_topology.clear();
            best_matches_psd_placed.clear();

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

                const bool mgd_is_1xN_strip =
                    device_topo.has_value() && device_topo->dims.size() >= 2 &&
                    std::any_of(device_topo->dims.begin(), device_topo->dims.end(), [](int32_t d) { return d == 1; });

                // Same ASIC count but different grid factorization (e.g. MGD 1×32 vs PGD 4×8): still allow the
                // topology solve unless the MGD declares a full 2D grid (both dims > 1).
                if (node_diff == 0 && !required_grid_dims.empty() &&
                    grouping_info.flattened_node_grid_dims.size() >= 2 &&
                    normalized_dims(grouping_info.flattened_node_grid_dims) != required_grid_dims &&
                    !mgd_is_1xN_strip) {
                    log_debug(
                        tt::LogFabric,
                        "Skipping {} for {}: flattened node grid dims [{},{}] do not match MGD device topology",
                        name,
                        mgd_grouping_info.name,
                        grouping_info.flattened_node_grid_dims[0],
                        grouping_info.flattened_node_grid_dims[1]);
                    continue;
                }

                MappingConstraints<uint32_t, uint32_t> constraints;
                if (applicable_pinnings.has_value()) {
                    const std::size_t pinning_constraints_added = add_mgd_to_pgd_asic_position_pinning_constraints(
                        constraints, grouping_info, *applicable_pinnings);
                    // Ok if no pinning constraints were added, means this grouping might not be best fit
                    if (pinning_constraints_added == 0) {
                        continue;
                    }
                } else {
                    // No pinning for this MGD instance: keep the (0,0) anchor so the solve stays constrained
                    // instead of running unconstrained.
                    constraints.add_required_constraint(0, 0);
                }
                auto mapping_result = solve_topology_mapping<uint32_t, uint32_t>(
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
            // (tray_id, asic_location) slot labels is intentional so find_all_in_psd places on the same graph.
            auto make_committed_grouping = [&](const MeshTopologyMatch& match) -> GroupingInfo {
                GroupingInfo committed = mesh_flat_groupings.at(match.name)[match.idx];
                // The topology solve used the MGD mesh adjacency as target and this PGD variant as global, so
                // target_to_global is MGD-node -> PGD grouping-node. Compose logical chip_id -> PGD slot pinning
                // now so downstream consumes it directly without re-deriving the intermediate node pairing.
                committed.mesh_node_to_asic_position =
                    compose_mesh_node_to_asic_position_from_pgd_match(committed, match.mapping.target_to_global);
                return committed;
            };

            // Prefer the simplest topology that fits: order variants MESH -> TORUSX -> TORUSY -> TORUSXY so the
            // smallest topology that matches is used. The downstream set-packing solver de-duplicates variants
            // that cover the same physical ASIC set (find_all_in_psd / solve_for_many_groupings_to_psd), so
            // committing variants MESH-first means each physical region keeps its MESH form rather than a torus
            // form, while distinct physical regions (e.g. two tray-pairs for a 2x8) are each committed.
            auto variant_priority = [&](const MeshTopologyMatch& m) -> int {
                const std::string& type = mesh_flat_groupings.at(m.name)[m.idx].type;
                if (type == "MESH") {
                    return 0;
                }
                if (type == "TORUSX") {
                    return 1;
                }
                if (type == "TORUSY") {
                    return 2;
                }
                if (type == "TORUSXY") {
                    return 3;
                }
                return 4;
            };
            std::stable_sort(
                best_matches_topology.begin(),
                best_matches_topology.end(),
                [&](const MeshTopologyMatch& a, const MeshTopologyMatch& b) {
                    return variant_priority(a) < variant_priority(b);
                });

            if (physical_system_descriptor != nullptr) {
                for (const auto& match : best_matches_topology) {
                    const GroupingInfo committed_candidate = make_committed_grouping(match);
                    std::vector<std::string> psd_errors;
                    const auto mapped_asics =
                        find_any_in_psd(committed_candidate, *physical_system_descriptor, psd_errors);
                    if (!mapped_asics.empty()) {
                        best_matches_psd_placed.push_back(match);
                    } else if (!psd_errors.empty()) {
                        log_debug(
                            tt::LogFabric,
                            "PGD '{}' matched MGD '{}' topologically but could not be placed on PSD under current "
                            "constraints: {}",
                            committed_candidate.name,
                            mgd_grouping_info.name,
                            psd_errors.front());
                    } else {
                        log_debug(
                            tt::LogFabric,
                            "PGD '{}' matched MGD '{}' topologically but could not be placed on PSD (no ASIC embedding "
                            "found)",
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
                committed_pgd_matches = true;
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

        if (!committed_pgd_matches) {
            // No PGD grouping both matched MGD and placed on PSD — use the MGD grouping info itself
            log_info(
                tt::LogFabric,
                "Physical groupings: Mesh graph descriptor '{}': {} topology match(es), fallback to Mesh graph "
                "descriptor: {} ({})",
                mgd_grouping_info.name,
                last_topology_match_count,
                mgd_grouping_info.name,
                mgd_grouping_info.type);
            if (device_topo.has_value()) {
                result[instance_type][instance_name].push_back(
                    finalize_mesh_grouping_with_device_topology(mgd_grouping_info, *device_topo));
            } else {
                result[instance_type][instance_name].push_back(mgd_grouping_info);
            }
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

std::vector<GroupingInfo> PhysicalGroupingDescriptor::get_mgd_mesh_groupings_for_placement(
    const MeshGraphDescriptor& mesh_graph_descriptor) {
    const auto mgd_grouping_infos = build_mgd_to_grouping_info_map(mesh_graph_descriptor);
    const auto mesh_it = mgd_grouping_infos.find("MESH");
    if (mesh_it == mgd_grouping_infos.end()) {
        return {};
    }

    std::vector<GroupingInfo> meshes;
    meshes.reserve(mesh_it->second.size());
    for (const auto& [instance_name, mgd_grouping] : mesh_it->second) {
        const auto device_topo = get_mgd_instance_device_topology(mesh_graph_descriptor, instance_name);
        if (device_topo.has_value()) {
            meshes.push_back(finalize_mesh_grouping_with_device_topology(mgd_grouping, *device_topo));
        } else {
            meshes.push_back(mgd_grouping);
        }
    }
    return meshes;
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgds(
    const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const {
    ValidGroupingsMap out;
    // With multiple MGDs (split sub-contexts), different descriptors can reuse the same instance name (e.g. "M0").
    // Prefix each MGD's instance names with "mgd{i}_" so they stay distinct in the merged map; otherwise their
    // groupings (and the downstream physical mesh nodes) collapse together. Single-MGD keeps names unprefixed so the
    // common path is unchanged. The "mgd{i}_" key encodes the originating descriptor index for downstream lookup
    // (see build_physical_multi_mesh_adjacency_graph).
    const bool multi_mgd = mesh_graph_descriptors.size() > 1;
    for (size_t i = 0; i < mesh_graph_descriptors.size(); ++i) {
        auto one = get_valid_groupings_for_mgd(mesh_graph_descriptors[i], physical_system_descriptor);
        for (const auto& [type, by_name] : one) {
            for (const auto& [name, gvec] : by_name) {
                const std::string key = multi_mgd ? fmt::format("mgd{}_{}", i, name) : name;
                auto& dest = out[type][key];
                dest.insert(dest.end(), gvec.begin(), gvec.end());
            }
        }
    }
    return out;
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
    // Same-host is a PREFERENCE, not a hard requirement. We prefer keeping the whole mesh on one host when it
    // fits, but must allow cross-host placement when that is the only valid embedding of the requested topology
    // -- e.g. a 4x4 RING/RING torus that physically spans two galaxies through inter-host links. A required
    // same-rank constraint here wrongly forbids such legitimate cross-host meshes (it pins all nodes to one
    // host purely because the node count fits), so a torus that only closes across hosts can never be placed.
    // Using a preferred constraint keeps single-host meshes on one host while letting cross-host meshes embed.
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

// Build the MappingConstraints used by PGD→PSD embedding (trait + host alignment).
// Returns std::nullopt if a required trait constraint cannot be satisfied (e.g. slot count mismatch);
// `error_out` is set when that happens.
std::optional<MappingConstraints<uint32_t, AsicID>> build_pgd_to_psd_constraints(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<uint32_t, AsicID> initial_constraints,
    std::string* error_out = nullptr) {
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
            return std::nullopt;
        }
    }
    if (!relax_pgd_slot_traits && !target_location_traits.empty() && !global_location_traits.empty()) {
        if (!constraints.add_required_trait_constraint<ASICLocation>(target_location_traits, global_location_traits)) {
            if (error_out) {
                *error_out = "Failed to add required trait constraint for asic_location";
            }
            return std::nullopt;
        }
    }

    // PSD-only host partition (ASIC -> hostname): same-rank when the full mesh fits on one host, else unconstrained.
    configure_pgd_psd_host_alignment_constraints(
        grouping_info, physical_graph, physical_system_descriptor, constraints);

    return constraints;
}

// Helper function to solve the topology mapping with pinning constraints from GroupingInfo
MappingResult<uint32_t, AsicID> solve_for_one_grouping_to_psd(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<uint32_t, AsicID> initial_constraints = {}) {
    std::string err;
    auto constraints_opt = build_pgd_to_psd_constraints(
        grouping_info, physical_graph, physical_system_descriptor, std::move(initial_constraints), &err);
    if (!constraints_opt) {
        MappingResult<uint32_t, AsicID> failure;
        failure.success = false;
        failure.error_message = std::move(err);
        return failure;
    }
    return solve_topology_mapping(
        grouping_info.adjacency_graph, physical_graph, *constraints_opt, ConnectionValidationMode::STRICT, true);
}

// Enumerate up to `max_solutions` distinct image-set placements of `grouping_info` on `physical_graph`.
// Wraps solve_topology_mapping_n with unique_shapes=true so the solver skips permutations that hit the same ASIC set.
// Returns the (possibly empty) list of successful MappingResults.
std::vector<MappingResult<uint32_t, AsicID>> enumerate_distinct_placements_for_grouping(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    size_t max_solutions) {
    auto constraints_opt =
        build_pgd_to_psd_constraints(grouping_info, physical_graph, physical_system_descriptor, {}, nullptr);
    if (!constraints_opt) {
        return {};
    }
    return solve_topology_mapping_n<uint32_t, AsicID>(
        grouping_info.adjacency_graph,
        physical_graph,
        *constraints_opt,
        max_solutions,
        ConnectionValidationMode::STRICT,
        /*quiet_mode=*/true,
        TopologyMappingSolverEngine::Auto,
        /*unique_shapes=*/true);
}

// A single candidate placement considered by the set-packing solver.
struct PackingCandidate {
    size_t grouping_idx;             // index into the input groupings vector
    std::vector<size_t> asic_slots;  // dense ASIC indices (0..universe_size-1) used by this placement
    MappingResult<uint32_t, AsicID> result;
    size_t pool_order = 0;  // insertion order into the candidate pool (matches solver enumeration order)
    size_t host_count = 1;  // distinct hosts spanned by this placement
};

struct PackingResult {
    std::vector<PackingCandidate> selected;
    uint64_t total_weight = 0;
    bool proven_optimal = false;
};

// Maximum Weight Set Packing via branch-and-bound.
// Universe is [0, universe_size); each candidate's weight is asic_slots.size().
// At each DFS node the upper bound is current_weight + min(free_slots, suffix_weight_sum) — loose but cheap.
// When the wall-clock budget elapses, the best feasible solution found so far is returned with proven_optimal=false.
PackingResult solve_set_packing(
    std::vector<PackingCandidate> candidates, size_t universe_size, std::chrono::milliseconds budget) {
    PackingResult best;
    if (candidates.empty() || universe_size == 0) {
        best.proven_optimal = true;
        return best;
    }

    // Prefer heavier placements, then single-host, then earlier solver enumeration (preferred constraints).
    std::sort(candidates.begin(), candidates.end(), [](const PackingCandidate& a, const PackingCandidate& b) {
        if (a.asic_slots.size() != b.asic_slots.size()) {
            return a.asic_slots.size() > b.asic_slots.size();
        }
        if (a.host_count != b.host_count) {
            return a.host_count < b.host_count;
        }
        return a.pool_order < b.pool_order;
    });

    const size_t n = candidates.size();
    std::vector<uint64_t> suffix_weight(n + 1, 0);
    for (size_t i = n; i-- > 0;) {
        suffix_weight[i] = suffix_weight[i + 1] + candidates[i].asic_slots.size();
    }

    std::vector<bool> used(universe_size, false);
    size_t free_slots = universe_size;
    std::vector<size_t> current_path;  // positional indices into `candidates`
    std::vector<size_t> best_path;     // best feasible found so far
    uint64_t current_weight = 0;
    const auto deadline = std::chrono::steady_clock::now() + budget;
    bool timed_out = false;

    std::function<void(size_t)> dfs = [&](size_t i) {
        if (timed_out) {
            return;
        }
        // Any extension adds at most min(free_slots, sum-of-remaining-weights).
        const uint64_t bound = current_weight + std::min<uint64_t>(free_slots, suffix_weight[i]);
        if (bound <= best.total_weight) {
            return;
        }
        if (i == n) {
            if (current_weight > best.total_weight) {
                best.total_weight = current_weight;
                best_path = current_path;
            }
            return;
        }
        // Cheap deadline check: sample steady_clock periodically.
        if ((i & 0x3FFu) == 0 && std::chrono::steady_clock::now() > deadline) {
            timed_out = true;
            return;
        }

        const auto& c = candidates[i];
        bool conflict = false;
        for (size_t a : c.asic_slots) {
            if (used[a]) {
                conflict = true;
                break;
            }
        }
        if (!conflict) {
            for (size_t a : c.asic_slots) {
                used[a] = true;
            }
            free_slots -= c.asic_slots.size();
            current_path.push_back(i);
            current_weight += c.asic_slots.size();

            dfs(i + 1);

            current_weight -= c.asic_slots.size();
            current_path.pop_back();
            free_slots += c.asic_slots.size();
            for (size_t a : c.asic_slots) {
                used[a] = false;
            }
            if (timed_out) {
                return;
            }
        }
        dfs(i + 1);
    };

    dfs(0);
    best.proven_optimal = !timed_out;

    best.selected.reserve(best_path.size());
    for (size_t pos : best_path) {
        best.selected.push_back(std::move(candidates[pos]));
    }
    return best;
}

bool is_flattened(const GroupingInfo& grouping) {
    return grouping.asic_count == grouping.adjacency_graph.get_nodes().size();
}

}  // namespace

namespace tt::tt_fabric {

// Maximum placements to find before stopping (safeguard against infinite loops).
constexpr size_t kMaxPlacementsPerRun = 10000;
// Maximum distinct image-set placements enumerated per grouping during heterogeneous packing.
// Set high enough that practical heterogeneous packings (trait-constrained meshes on ~1000 ASICs) are exhausted,
// while still capping worst-case enumeration cost. A warning is logged if the cap is hit.
constexpr size_t kMaxPlacementsPerGrouping = 1024;
// Wall-clock budget for the set-packing branch-and-bound pass. On expiry the best feasible packing found so far
// is returned (may be empty or sub-optimal if the solver has not yet found any solution).
constexpr std::chrono::milliseconds kSetPackingBudget{5000};

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
// Algorithm (enumerate-then-pack):
//   Phase A — for each grouping, enumerate up to kMaxPlacementsPerGrouping distinct image-set placements
//             via solve_topology_mapping_n(unique_shapes=true). Identical ASIC sets across groupings are de-duped.
//   Phase B — Maximum Weight Set Packing via branch-and-bound to pick the disjoint subset that maximizes total
//             ASIC coverage. Wall-clock-budgeted; returns best feasible solution found on expiry.
// Returns map from each GroupingInfo* (by address into the input vector) to its vector of selected MappingResults.
std::unordered_map<const GroupingInfo*, std::vector<MappingResult<uint32_t, AsicID>>>
solve_for_many_groupings_to_psd_heterogeneous(
    const std::vector<GroupingInfo>& groupings,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    // Dense ASIC id → index assignment so the set-packing universe is [0, U).
    std::unordered_map<AsicID, size_t> asic_to_slot;
    asic_to_slot.reserve(physical_graph.get_nodes().size());
    for (const AsicID& asic : physical_graph.get_nodes()) {
        asic_to_slot.emplace(asic, asic_to_slot.size());
    }
    const size_t universe_size = asic_to_slot.size();

    // Phase A: enumerate candidates per grouping, de-duplicating identical ASIC sets across groupings.
    std::vector<PackingCandidate> candidates;
    std::unordered_set<std::string> seen_sets;  // key = sorted slot indices serialized as bytes
    size_t pool_order = 0;
    for (size_t gi = 0; gi < groupings.size(); ++gi) {
        const auto& grouping = groupings[gi];
        if (grouping.adjacency_graph.get_nodes().empty()) {
            continue;
        }
        auto placements = enumerate_distinct_placements_for_grouping(
            grouping, physical_graph, physical_system_descriptor, kMaxPlacementsPerGrouping);
        log_debug(
            tt::LogFabric,
            "Heterogeneous solver: grouping '{}' ({} nodes) enumerated {} distinct image-set placements",
            grouping.name,
            grouping.adjacency_graph.get_nodes().size(),
            placements.size());
        if (placements.size() == kMaxPlacementsPerGrouping) {
            log_warning(
                tt::LogFabric,
                "Heterogeneous solver: per-grouping enumeration cap hit for grouping '{}' (k={}). "
                "Set-packing remains optimal over the enumerated pool.",
                grouping.name,
                kMaxPlacementsPerGrouping);
        }
        for (auto& placement : placements) {
            if (!placement.success) {
                continue;
            }
            std::vector<size_t> slots;
            slots.reserve(placement.target_to_global.size());
            for (const auto& [_, asic_id] : placement.target_to_global) {
                auto it = asic_to_slot.find(asic_id);
                if (it == asic_to_slot.end()) {
                    // ASIC not in physical_graph — should not happen, but skip defensively.
                    slots.clear();
                    break;
                }
                slots.push_back(it->second);
            }
            if (slots.empty()) {
                continue;
            }
            std::sort(slots.begin(), slots.end());
            slots.erase(std::unique(slots.begin(), slots.end()), slots.end());

            std::string key(reinterpret_cast<const char*>(slots.data()), slots.size() * sizeof(size_t));
            if (!seen_sets.insert(std::move(key)).second) {
                continue;
            }
            std::set<std::string> hosts;
            for (const auto& [_, asic_id] : placement.target_to_global) {
                hosts.insert(physical_system_descriptor.get_host_name_for_asic(asic_id));
            }
            PackingCandidate candidate{gi, std::move(slots), std::move(placement)};
            candidate.pool_order = pool_order++;
            candidate.host_count = hosts.size();
            candidates.push_back(std::move(candidate));
        }
    }

    // Pre-seed the result map so every grouping has an entry, even if no placement is selected.
    std::unordered_map<const GroupingInfo*, std::vector<MappingResult<uint32_t, AsicID>>> map_result;
    for (const auto& grouping : groupings) {
        map_result.emplace(&grouping, std::vector<MappingResult<uint32_t, AsicID>>{});
    }
    if (candidates.empty()) {
        return map_result;
    }

    // Phase B: pick the disjoint subset with maximum total weight.
    log_debug(
        tt::LogFabric,
        "Heterogeneous solver: pool has {} unique candidates over {} ASICs; running set-packing",
        candidates.size(),
        universe_size);
    PackingResult packed = solve_set_packing(std::move(candidates), universe_size, kSetPackingBudget);
    log_debug(
        tt::LogFabric,
        "Heterogeneous solver: set-packing chose {} placements, total weight {} (proven_optimal={})",
        packed.selected.size(),
        packed.total_weight,
        packed.proven_optimal);
    if (!packed.proven_optimal) {
        log_warning(
            tt::LogFabric,
            "Heterogeneous solver: set-packing wall-clock budget ({}ms) expired; returning best feasible "
            "({} placements, {} ASICs covered).",
            kSetPackingBudget.count(),
            packed.selected.size(),
            packed.total_weight);
    }
    if (packed.selected.size() > kMaxPlacementsPerRun) {
        log_warning(
            tt::LogFabric, "Heterogeneous solver: hit max placements limit ({}) - truncating", kMaxPlacementsPerRun);
        packed.selected.resize(kMaxPlacementsPerRun);
    }

    for (auto& sel : packed.selected) {
        map_result[&groupings[sel.grouping_idx]].push_back(std::move(sel.result));
    }
    return map_result;
}
}  // namespace tt::tt_fabric

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
    for (uint32_t node_id : grouping_info.adjacency_graph.get_nodes()) {
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

std::vector<PsdPlacement> PhysicalGroupingDescriptor::find_all_in_psd(
    const std::vector<GroupingInfo>& groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const {
    PhysicalAdjacencyMap physical_adj_map = build_flat_adjacency_map_from_psd(physical_system_descriptor);
    AdjacencyGraph<AsicID> physical_graph(physical_adj_map);
    return find_all_in_psd(groupings, physical_system_descriptor, physical_graph);
}

// NOTE this only works on flattenable meshes right now
std::vector<PsdPlacement> PhysicalGroupingDescriptor::find_all_in_psd(
    const std::vector<GroupingInfo>& groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const AdjacencyGraph<AsicID>& physical_graph,
    std::vector<std::string>* errors_out) const {
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

    std::vector<PsdPlacement> placements;
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
                    PsdPlacement placement;
                    // Downstream only needs the pinning map, so copy just that (not the whole GroupingInfo,
                    // which would deep-copy items + adjacency_graph per placement).
                    placement.mesh_node_to_asic_position = grouping.mesh_node_to_asic_position;
                    // result.target_to_global is this grouping's node id -> AsicID; collect just the ASICs
                    // for the placement footprint (order unused, so iterate it directly).
                    for (const auto& [grouping_node, asic_id] : result.target_to_global) {
                        placement.asics.insert(asic_id);
                    }
                    placements.push_back(std::move(placement));
                }
            }
        }
    }

    if (errors_out != nullptr && placements.empty()) {
        if (flat_meshes.empty()) {
            errors_out->push_back("No valid groupings found for PSD");
        } else {
            const GroupingInfo& mesh_to_use = flat_meshes.back();
            auto result = solve_for_one_grouping_to_psd(mesh_to_use, physical_graph, physical_system_descriptor);
            errors_out->push_back(build_pgd_mapping_failure_message(mesh_to_use.name, mesh_to_use, result));
        }
    }

    return placements;
}
