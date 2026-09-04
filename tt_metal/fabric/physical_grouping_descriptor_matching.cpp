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
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <vector>
#include <tt_stl/fmt.hpp>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

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

// Applies the pinning groups this PGD grouping can host and returns how many were added; a group whose
// ASIC positions resolve to no node here does not apply and is dropped. Returns 0 when nothing applies or
// when the groups that do apply are not jointly satisfiable, and the caller then skips the grouping.
std::size_t add_mgd_to_pgd_asic_position_pinning_constraints(
    MappingConstraints<uint32_t, uint32_t>& constraints,
    const GroupingInfo& pgd_grouping,
    const std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>& pinnings) {
    std::size_t constraints_added = 0;
    for (const auto& group : pinnings) {
        std::set<uint32_t> mgd_nodes;
        std::set<uint32_t> pgd_nodes;
        for (const auto& fabric_node : group.fabric_nodes) {
            mgd_nodes.insert(fabric_node.chip_id);
        }
        for (const auto& position : group.asic_positions) {
            const auto found_pgd_nodes = find_pgd_nodes_at_asic_position(pgd_grouping, position);
            pgd_nodes.insert(found_pgd_nodes.begin(), found_pgd_nodes.end());
        }
        if (!mgd_nodes.empty() && !pgd_nodes.empty()) {
            if (!constraints.add_required_constraint(mgd_nodes, pgd_nodes)) {
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

// Host boundaries come only from the PSD (get_host_name_for_asic). Global groups are one set per host (variable
// size). One PGD mesh target group: hard same-rank if some host has >= mesh ASICs; otherwise preferred ASICs on a
// greedy minimal set of largest hosts (by ASIC count) to bias toward fewer cross-host hops.
void configure_pgd_psd_host_alignment_constraints(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    MappingConstraints<LogicalChipId, AsicID>& constraints) {
    // Collect hostname map for all asics in physical graph
    std::map<std::string, std::set<AsicID>> host_to_asics;
    for (const AsicID& asic_id : physical_graph.get_nodes()) {
        host_to_asics[physical_system_descriptor.get_host_name_for_asic(asic_id)].insert(asic_id);
    }

    // Collect all targets from PGD grouping info. These are LogicalChipIds: mesh-local nodes of
    // grouping_info.adjacency_graph, the same ids MappingConstraints<LogicalChipId, AsicID> constrains.
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
        for (const LogicalChipId& target : all_targets) {
            constraints.add_preferred_constraint(target, preferred_globals);
        }
    }
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

    // PSD-only host partition (ASIC -> hostname): same-rank when the full mesh fits on one host, else unconstrained.
    configure_pgd_psd_host_alignment_constraints(
        grouping_info, physical_graph, physical_system_descriptor, constraints);

    return true;
}

// Enumerate up to `max_solutions` distinct image-set placements of `grouping_info` on `physical_graph`.
// Wraps solve_topology_mapping_n with unique_shapes=true so the solver skips permutations that hit the same ASIC set.
// `constraints` is the caller's own object: it is used as-is and the grouping's trait and host-alignment
// constraints are added to it in place. Callers with nothing to anchor pass a default-constructed one, so
// that the object's lifetime and reuse across solves is always the caller's decision rather than a hidden
// temporary here.
// Returns the (possibly empty) list of successful MappingResults.
std::vector<MappingResult<LogicalChipId, AsicID>> enumerate_distinct_placements_for_grouping(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    size_t max_solutions,
    MappingConstraints<LogicalChipId, AsicID>& constraints) {
    if (!add_pgd_to_psd_constraints(grouping_info, physical_graph, physical_system_descriptor, constraints, nullptr)) {
        return {};
    }
    return solve_topology_mapping_n<uint32_t, AsicID>(
        grouping_info.adjacency_graph,
        physical_graph,
        constraints,
        max_solutions,
        ConnectionValidationMode::STRICT,
        /*quiet_mode=*/true,
        TopologyMappingSolverEngine::Auto,
        /*unique_shapes=*/true);
}

}  // namespace

namespace tt::tt_fabric {

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings) const {
    return get_valid_groupings_for_mgd(mesh_graph_descriptor, &physical_system_descriptor, pinnings);
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings) const {
    ValidGroupingsMap result;

    std::optional<AdjacencyGraph<tt::tt_metal::AsicID>> psd_physical_graph;
    if (physical_system_descriptor != nullptr) {
        psd_physical_graph.emplace(
            tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd(*physical_system_descriptor));
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

        // Process difference levels from closest to farthest; commit only when embedding on PSD succeeds.
        // Each pin set gets its own match/commit pass, so a shared descriptor accumulates the groupings of
        // every column it is pinned to.
        std::vector<MeshTopologyMatch> best_matches_topology;
        std::vector<MeshTopologyMatch> best_matches_psd_placed;
        size_t last_topology_match_count = 0;

        bool committed_pgd_matches = false;
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

                    const bool mgd_is_1xN_strip =
                        device_topo.has_value() && device_topo->dims.size() >= 2 &&
                        std::any_of(
                            device_topo->dims.begin(), device_topo->dims.end(), [](int32_t d) { return d == 1; });

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
                    if (!active_pinnings.empty()) {
                        // Keep only groupings that host at least one pin, with the pins that do apply
                        // required to hold together.
                        if (add_mgd_to_pgd_asic_position_pinning_constraints(
                                constraints, grouping_info, active_pinnings) == 0) {
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
                // that cover the same physical ASIC set (find_all_in_psd), so
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

                // Check and only use the Groupings found that can actually be placed on the PSD.
                // The committed candidate is already one flattened variant, so this uses the enumerating
                // solve rather than find_any_in_psd, which requires a still-hierarchical grouping.
                if (physical_system_descriptor != nullptr) {
                    for (const auto& match : best_matches_topology) {
                        const GroupingInfo committed_candidate = make_committed_grouping(match);
                        MappingConstraints<LogicalChipId, tt::tt_metal::AsicID> solve_constraints;
                        const auto placements = enumerate_distinct_placements_for_grouping(
                            committed_candidate,
                            *psd_physical_graph,
                            *physical_system_descriptor,
                            /*max_solutions=*/1,
                            solve_constraints);
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
        }  // end per-pin-set pass (pin_set_variants)

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
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::vector<std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>>& per_mgd_pinnings) const {
    ValidGroupingsMap out;
    // With multiple MGDs (split sub-contexts), different descriptors can reuse the same instance name (e.g. "M0").
    // Prefix each MGD's instance names with "mgd{i}_" so they stay distinct in the merged map; otherwise their
    // groupings (and the downstream physical mesh nodes) collapse together. Single-MGD keeps names unprefixed so the
    // common path is unchanged. The "mgd{i}_" key encodes the originating descriptor index for downstream lookup
    // (see build_physical_multi_mesh_adjacency_graph).
    for (size_t i = 0; i < mesh_graph_descriptors.size(); ++i) {
        // Pins for MGD i are in this descriptor's own local mesh-id space; forward them so the PGD<->MGD match
        // honours the pinned ASIC positions (same as the single-MGD get_valid_groupings_for_mgd(mgd, psd, pins)).
        std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh> pins;
        if (i < per_mgd_pinnings.size()) {
            pins = per_mgd_pinnings[i];
        }
        auto one = get_valid_groupings_for_mgd(mesh_graph_descriptors[i], physical_system_descriptor, pins);
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

namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::ASICLocation;
using tt::tt_metal::TrayID;
using tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd;
using tt::tt_metal::experimental::tt_fabric::PhysicalAdjacencyMap;

// Message for "this grouping has no embedding on this PSD". It reports the variants tried and their size
// rather than a partial mapping, because the enumerating solve yields successes only: when nothing places
// there is no partial result to describe.
std::string build_pgd_mapping_failure_message(
    const std::string& grouping_name, size_t flat_variant_count, size_t node_count) {
    return fmt::format(
        "PGD grouping '{}' could not be mapped to PSD: no embedding found for any of its {} flattened variant(s) "
        "({} nodes)",
        grouping_name,
        flat_variant_count,
        node_count);
}

// TODO: delete with solve_set_packing / find_all_in_psd. DFS uses PlacementCandidate instead.
struct PackingCandidate {
    size_t grouping_idx;             // index into the input groupings vector
    std::vector<size_t> asic_slots;  // dense ASIC indices (0..universe_size-1) used by this placement
    MappingResult<LogicalChipId, AsicID> result;
    size_t pool_order = 0;  // insertion order into the candidate pool (matches solver enumeration order)
    size_t host_count = 1;  // distinct hosts spanned by this placement
};

// TODO: delete with solve_set_packing / find_all_in_psd.
struct PackingResult {
    std::vector<PackingCandidate> selected;
    uint64_t total_weight = 0;
    bool proven_optimal = false;
};

// Maximum Weight Set Packing via branch-and-bound.
// Universe is [0, universe_size); each candidate's weight is asic_slots.size().
// At each DFS node the upper bound is current_weight + min(free_slots, suffix_weight_sum) — loose but cheap.
// When the wall-clock budget elapses, the best feasible solution found so far is returned with proven_optimal=false.
//
// TODO: delete with solve_for_many_groupings_to_psd_heterogeneous, its only caller.
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

// TODO: these three caps exist only for find_all_in_psd's packer. Delete with it.
constexpr size_t kMaxPlacementsPerRun = 10000;
constexpr size_t kMaxPlacementsPerGrouping = 1024;
constexpr std::chrono::milliseconds kSetPackingBudget{5000};

// Heterogeneous version: pack multiple different grouping types onto the physical graph.
// Each grouping can have a different topology. ASICs are shared globally - no overlap between any mappings.
// Algorithm (enumerate-then-pack):
//   Phase A — for each grouping, enumerate up to kMaxPlacementsPerGrouping distinct image-set placements
//             via solve_topology_mapping_n(unique_shapes=true). Identical ASIC sets across groupings are de-duped.
//   Phase B — Maximum Weight Set Packing via branch-and-bound to pick the disjoint subset that maximizes total
//             ASIC coverage. Wall-clock-budgeted; returns best feasible solution found on expiry.
// Returns map from each GroupingInfo* (by address into the input vector) to its vector of selected MappingResults.
//
// TODO: delete once the adjacency-guided DFS is the only placement producer. Phase B commits to a
// maximum-coverage tiling before anything has looked at the MGD's mesh-level edges, which is the root cause the
// DFS exists to remove. Its only caller is find_all_in_psd, so this and solve_set_packing go together.
std::unordered_map<const GroupingInfo*, std::vector<MappingResult<LogicalChipId, AsicID>>>
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
        MappingConstraints<LogicalChipId, AsicID> solve_constraints;
        auto placements = enumerate_distinct_placements_for_grouping(
            grouping, physical_graph, physical_system_descriptor, kMaxPlacementsPerGrouping, solve_constraints);
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
    std::unordered_map<const GroupingInfo*, std::vector<MappingResult<LogicalChipId, AsicID>>> map_result;
    for (const auto& grouping : groupings) {
        map_result.emplace(&grouping, std::vector<MappingResult<LogicalChipId, AsicID>>{});
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

// NOTE this only works on flattenable meshes right now
// TODO: Expand find_any_in_psd to non-flattenable meshes by doing recursive mapping
std::vector<MappingResult<LogicalChipId, AsicID>> PhysicalGroupingDescriptor::find_any_in_psd(
    const GroupingInfo& grouping,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t max_solutions,
    const std::optional<MappingConstraints<LogicalChipId, AsicID>>& extra_constraints,
    std::vector<std::string>* errors_out) const {
    AdjacencyGraph<AsicID> physical_graph(build_flat_adjacency_map_from_psd(physical_system_descriptor));
    return find_any_in_psd(
        grouping, physical_system_descriptor, physical_graph, max_solutions, extra_constraints, errors_out);
}

std::vector<MappingResult<LogicalChipId, AsicID>> PhysicalGroupingDescriptor::find_any_in_psd(
    const GroupingInfo& grouping,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const AdjacencyGraph<AsicID>& physical_graph,
    std::size_t max_solutions,
    const std::optional<MappingConstraints<LogicalChipId, AsicID>>& extra_constraints,
    std::vector<std::string>* errors_out) const {
    TT_FATAL(
        !is_flattened(grouping),
        "find_any_in_psd requires a hierarchical grouping (items still present, ASIC graph not yet built); "
        "'{}' is already flattened ({} ASIC nodes). Pass the PGD grouping from get_groupings_by_name, not a "
        "committed ValidGroupingsMap entry.",
        grouping.name,
        grouping.adjacency_graph.get_nodes().size());

    std::vector<GroupingInfo> flat_meshes = build_flattened_adjacency_mesh(grouping, physical_system_descriptor);

    // PSD filtering can remove every possibility. That is the grouping legitimately not fitting this
    // system, not an internal error, so report nothing placed rather than raising.
    if (flat_meshes.empty()) {
        return {};
    }

    std::vector<MappingResult<LogicalChipId, AsicID>> results;
    std::size_t nodes_in_largest_variant = 0;
    for (const auto& flat_mesh : flat_meshes) {
        const std::size_t node_count = flat_mesh.adjacency_graph.get_nodes().size();
        if (node_count == 0) {
            continue;
        }
        nodes_in_largest_variant = std::max(nodes_in_largest_variant, node_count);
        if (max_solutions != 0 && results.size() >= max_solutions) {
            break;
        }
        // 0 stays 0 so an unbounded request stays unbounded for every variant.
        const std::size_t remaining = max_solutions == 0 ? 0 : max_solutions - results.size();
        // Each variant solves against its own copy of the caller's constraints, because the solve adds
        // that variant's trait constraints in place and they must not leak into the next variant.
        MappingConstraints<LogicalChipId, AsicID> solve_constraints =
            extra_constraints.value_or(MappingConstraints<LogicalChipId, AsicID>{});
        auto placements = enumerate_distinct_placements_for_grouping(
            flat_mesh, physical_graph, physical_system_descriptor, remaining, solve_constraints);
        for (auto& placement : placements) {
            if (!placement.success) {
                continue;
            }
            results.push_back(std::move(placement));
            if (max_solutions != 0 && results.size() >= max_solutions) {
                break;
            }
        }
    }

    if (nodes_in_largest_variant == 0) {
        TT_THROW("Internal error: grouping '{}' produced empty graph", grouping.name);
    }

    if (results.empty() && errors_out != nullptr) {
        errors_out->push_back(
            build_pgd_mapping_failure_message(grouping.name, flat_meshes.size(), nodes_in_largest_variant));
    }

    log_debug(
        tt::LogFabric,
        "find_any_in_psd: grouping '{}' ({} flat variant(s)) returned {} placement(s){}",
        grouping.name,
        flat_meshes.size(),
        results.size(),
        extra_constraints.has_value() ? " under caller constraints" : "");
    return results;
}

// ---------------------------------------------------------------------------
// Adjacency-guided placement search (incremental domain generation)
//
// Places one mesh at a time by DFS with backtracking, generating each mesh's candidate regions only
// when it is reached, so the candidates are already constrained by what its placed neighbours took.
// ---------------------------------------------------------------------------
namespace {

using GlobalMeshId = MeshId;

// A placed mesh is one PsdPlacement: the ASIC footprint chosen for it, plus the winning grouping
// variant's mesh_node_to_asic_position. The pinning map is a property of the variant rather than of the
// footprint, so it has to be carried from the placement that produced it.

// TODO: forward_check — domain wipeout / union bound (deferred).
// bool forward_check(...);

// One mesh's chosen placement. Stored in a vector rather than a map: mesh count is modest (tens to
// low hundreds), we iterate the whole assignment often, and a contiguous vector is smaller and more
// cache-friendly than a tree node per entry.
struct PlacedMesh {
    GlobalMeshId mesh_id;
    PsdPlacement placement;
};
using AssignedMeshes = std::vector<PlacedMesh>;

bool assignment_has_mesh(const AssignedMeshes& assignment, const GlobalMeshId& mesh_id) {
    for (const PlacedMesh& placed : assignment) {
        if (placed.mesh_id == mesh_id) {
            return true;
        }
    }
    return false;
}

// Every ASIC claimed by an already-placed mesh. Derived from the assignment on demand rather than
// tracked alongside it, so undoing a choice stays a single vector copy with nothing else to keep in sync.
std::unordered_set<AsicID> collect_occupied_asics(const AssignedMeshes& assignment) {
    std::unordered_set<AsicID> occupied;
    for (const PlacedMesh& placed : assignment) {
        occupied.insert(placed.placement.asics.begin(), placed.placement.asics.end());
    }
    return occupied;
}

// Drop every occupied ASIC, and every edge pointing at one, so a solve against the result cannot land
// on a chip another mesh already holds. Parallel edges between two free chips are preserved, since
// channel multiplicity is carried as duplicate neighbour entries.
//
// Takes the occupied set rather than the assignment: the caller needs that set for its own constraint
// work, so deriving it once and passing it avoids walking every placed footprint twice.
AdjacencyGraph<AsicID> filter_mapped_placements_in_physical_graph(
    const std::unordered_set<AsicID>& occupied, const AdjacencyGraph<AsicID>& physical_graph) {
    // AdjacencyMap is a std::map, and we walk the source map in ascending key order and keep a
    // subsequence of it, so every key we insert is greater than the last. emplace_hint at end() turns
    // each insert from a tree descent into a constant-time append; operator[] would re-descend per node.
    AdjacencyGraph<AsicID>::AdjacencyMap free_adjacency;
    for (const auto& [asic_id, neighbors] : physical_graph.get_adjacency_map()) {
        if (occupied.contains(asic_id)) {
            continue;
        }
        std::vector<AsicID> free_neighbors;
        free_neighbors.reserve(neighbors.size());
        for (const AsicID& neighbor : neighbors) {
            if (!occupied.contains(neighbor)) {
                free_neighbors.push_back(neighbor);
            }
        }
        free_adjacency.emplace_hint(free_adjacency.end(), asic_id, std::move(free_neighbors));
    }
    return AdjacencyGraph<AsicID>(std::move(free_adjacency));
}

// The seam domain: the free chips with an ethernet link into `region`, each mapped to how many links
// it has into it. A mesh placed on any of these chips touches the region.
//
// Must be computed from the UNFILTERED physical graph. The filtered one has already deleted the
// region's own chips, so the links out of it are gone with them and this would come back empty.
std::map<AsicID, std::size_t> free_chips_bordering_region(
    const std::unordered_set<AsicID>& region,
    const std::unordered_set<AsicID>& occupied,
    const AdjacencyGraph<AsicID>& physical_graph) {
    std::map<AsicID, std::size_t> boundary;
    for (const AsicID& region_chip : region) {
        // Parallel links are duplicate neighbour entries, so this counts links and not chips.
        for (const AsicID& neighbor : physical_graph.get_neighbors(region_chip)) {
            if (!occupied.contains(neighbor)) {
                ++boundary[neighbor];
            }
        }
    }
    return boundary;
}

// The number of ethernet links running between two disjoint footprints, counting parallel links
// separately. Also derived from the unfiltered graph, for the same reason as above.
std::size_t count_links_between(
    const std::unordered_set<AsicID>& region_a,
    const std::unordered_set<AsicID>& region_b,
    const AdjacencyGraph<AsicID>& physical_graph) {
    std::size_t links = 0;
    for (const AsicID& chip : region_a) {
        for (const AsicID& neighbor : physical_graph.get_neighbors(chip)) {
            if (region_b.contains(neighbor)) {
                ++links;
            }
        }
    }
    return links;
}

// The already-placed neighbours of `mesh_id`, each mapped to the number of mesh-level edges joining
// them. mesh_level_graph carries channel multiplicity as duplicate neighbour entries, so that count is
// how many ethernet links the seam between the two meshes has to carry.
std::map<GlobalMeshId, std::size_t> placed_neighbors_of(
    const GlobalMeshId& mesh_id,
    const AssignedMeshes& assignment,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph) {
    std::map<GlobalMeshId, std::size_t> placed;
    for (const GlobalMeshId& neighbor : mesh_level_graph.get_neighbors(mesh_id)) {
        if (assignment_has_mesh(assignment, neighbor)) {
            ++placed[neighbor];
        }
    }
    return placed;
}

// Which mesh to place next, or nullopt once every mesh is placed (the search's base case). A pure
// function of the current state: among the unplaced meshes prefer the one with the most already-placed
// neighbours, so the search keeps growing the frontier it is most constrained by.
//
// When no unplaced mesh has a placed neighbour this returns one anyway, which is how a disconnected
// mesh graph seeds its next component without any component detection — the components are coupled only
// through ASIC occupancy, and a single search over them lets a later one force an earlier one to move.
//
// TODO: richer frontier heuristic — pinning seeds, smallest domain, largest shape.
std::optional<GlobalMeshId> select_next_mesh(
    const AssignedMeshes& assignment, const AdjacencyGraph<GlobalMeshId>& mesh_level_graph) {
    std::optional<GlobalMeshId> next_mesh;
    std::size_t best_placed_neighbor_count = 0;
    // mesh_level_graph must have a node per mesh, including meshes with no intermesh connections, or an
    // unconnected mesh is never selected and never placed. build_logical_multi_mesh_adjacency_graph seeds
    // every mesh as a node, and both the remap and the merge preserve isolated ones, so this holds.
    //
    // get_nodes() is ordered by mesh id and the comparison below is strict, so ties keep the lowest id
    // and the choice is deterministic.
    for (const GlobalMeshId& mesh_id : mesh_level_graph.get_nodes()) {
        if (assignment_has_mesh(assignment, mesh_id)) {
            continue;
        }
        const std::size_t placed_neighbor_count = placed_neighbors_of(mesh_id, assignment, mesh_level_graph).size();
        if (!next_mesh.has_value() || placed_neighbor_count > best_placed_neighbor_count) {
            next_mesh = mesh_id;
            best_placed_neighbor_count = placed_neighbor_count;
        }
    }
    return next_mesh;
}

// The candidate placements for `mesh_id` given what is already placed: every placement of every grouping
// variant accepted for this mesh that is disjoint from the regions already taken and that meets every
// mesh-level edge to an already-placed neighbour with enough ethernet links.
//
// Disjointness is enforced by the solve rather than filtered afterwards, so an overlapping placement is
// never constructed. Adjacency is enforced in two halves, because the solver counts mapped nodes and the
// requirement is on links: a cardinality constraint bounds how many of this mesh's chips must sit on the
// neighbour's boundary, which is necessary but not sufficient, and the exact link count is then checked
// on each finished placement.
std::vector<PsdPlacement> next_step_pool(
    const GlobalMeshId& mesh_id,
    const AssignedMeshes& assignment,
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t max_placements_per_variant) {
    const auto groupings_it = global_mesh_groupings.find(mesh_id);
    if (groupings_it == global_mesh_groupings.end()) {
        return {};
    }

    // Disjointness: occupied chips are absent from this graph, so no solve against it can pick one.
    // Built once here and consumed by every variant's solve below; it dies with this call, so the
    // trimmed graph is never alive across a recursion and never multiplied by search depth.
    const std::unordered_set<AsicID> occupied = collect_occupied_asics(assignment);
    const AdjacencyGraph<AsicID> free_physical_graph =
        filter_mapped_placements_in_physical_graph(occupied, physical_graph);

    // One seam per already-placed neighbour, found in a single pass over the assignment. Derived once
    // here because it depends only on the assignment, not on which grouping variant we are about to try.
    struct Seam {
        const std::unordered_set<AsicID>* neighbor_asics;
        // Free chips bordering the neighbour's region. Keys are candidate chips for this mesh; values
        // are how many links each one has into the region, which the bound below needs.
        std::map<AsicID, std::size_t> boundary;
        std::size_t required_links = 0;
        std::size_t min_boundary_chips = 0;
    };
    // Channel count per mesh-level edge out of `mesh_id`, keyed by neighbour. Multiplicity is carried as
    // duplicate neighbour entries, so counting them gives the links each seam has to carry. Built up
    // front so the single walk over the assignment below is a lookup per placed mesh.
    std::map<GlobalMeshId, std::size_t> required_links_by_neighbor;
    for (const GlobalMeshId& neighbor : mesh_level_graph.get_neighbors(mesh_id)) {
        ++required_links_by_neighbor[neighbor];
    }

    std::vector<Seam> seams;
    seams.reserve(required_links_by_neighbor.size());
    for (const PlacedMesh& placed : assignment) {
        const auto required_it = required_links_by_neighbor.find(placed.mesh_id);
        if (required_it == required_links_by_neighbor.end()) {
            // Placed, but shares no mesh-level edge with this mesh, so it constrains nothing here.
            continue;
        }
        const std::size_t required_links = required_it->second;
        std::map<AsicID, std::size_t> boundary =
            free_chips_bordering_region(placed.placement.asics, occupied, physical_graph);
        std::size_t max_links_per_chip = 0;
        for (const auto& [chip, links] : boundary) {
            max_links_per_chip = std::max(max_links_per_chip, links);
        }
        if (max_links_per_chip == 0) {
            // Nothing free borders this neighbour, so no placement of this mesh can reach it.
            return {};
        }
        // One chip carries at most max_links_per_chip links, so meeting required_links takes at least
        // this many of our chips on the boundary. A necessary condition only — a placement satisfying it
        // can still fall short once the links are actually counted, which is what the check below is for.
        Seam seam;
        seam.neighbor_asics = &placed.placement.asics;
        seam.boundary = std::move(boundary);
        seam.required_links = required_links;
        seam.min_boundary_chips = (required_links + max_links_per_chip - 1) / max_links_per_chip;
        seams.push_back(std::move(seam));
    }

    std::vector<PsdPlacement> pool;
    for (const GroupingInfo& grouping : groupings_it->second) {
        const std::vector<uint32_t>& grouping_nodes = grouping.adjacency_graph.get_nodes();
        if (grouping_nodes.empty()) {
            continue;
        }

        // One constraint object per variant: enumerate_distinct_placements_for_grouping adds the
        // variant's own trait and host-alignment constraints to it in place, and those must not leak
        // into the next variant's solve.
        MappingConstraints<LogicalChipId, AsicID> constraints;

        // Seam constraints are per variant too, since they are written over this variant's nodes.
        bool variant_feasible = true;
        for (const Seam& seam : seams) {
            if (seam.min_boundary_chips > grouping_nodes.size()) {
                // This variant does not have enough chips to carry the seam however it is placed.
                variant_feasible = false;
                break;
            }
            MappingConstraints<LogicalChipId, AsicID>::CardinalityPairSet seam_pairs;
            for (const LogicalChipId node : grouping_nodes) {
                for (const auto& [chip, links] : seam.boundary) {
                    seam_pairs.emplace(node, chip);
                }
            }
            if (!constraints.add_cardinality_constraint(seam_pairs, seam.min_boundary_chips)) {
                variant_feasible = false;
                break;
            }
        }
        if (!variant_feasible) {
            continue;
        }

        for (const auto& mapping : enumerate_distinct_placements_for_grouping(
                 grouping, free_physical_graph, physical_system_descriptor, max_placements_per_variant, constraints)) {
            if (!mapping.success) {
                continue;
            }
            PsdPlacement candidate;
            // Carry the variant's pinning map: it is a property of the variant, not of the footprint,
            // so it cannot be recovered once the grouping is out of scope.
            candidate.mesh_node_to_asic_position = grouping.mesh_node_to_asic_position;
            for (const auto& [grouping_node, asic_id] : mapping.target_to_global) {
                candidate.asics.insert(asic_id);
            }
            // The cardinality constraint bounds how many of our chips touch the neighbour, not how many
            // links they carry between them, so the real count is settled here.
            bool seams_satisfied = true;
            for (const Seam& seam : seams) {
                if (count_links_between(candidate.asics, *seam.neighbor_asics, physical_graph) < seam.required_links) {
                    seams_satisfied = false;
                    break;
                }
            }
            if (!seams_satisfied) {
                continue;
            }
            pool.push_back(std::move(candidate));
        }
    }
    return pool;
}

// Completes `assignment` into a placement for every remaining mesh.
//
// Returns an empty vector when this branch cannot be completed. A non-empty return means every mesh in
// `mesh_level_graph` has an entry (size matches). The assignment is taken BY VALUE: each branch
// works on its own copy, so a branch that fails just discards it and the caller's copy is untouched.
//
// `nodes_expanded` is shared across the whole search and must be passed by reference: a by-value
// counter would restart the budget down every branch and never actually bind.
AssignedMeshes place_remaining_meshes(
    AssignedMeshes assignment,

    // Global variables
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t& nodes_expanded,
    std::size_t node_budget) {
    // TODO: seed from MGD pinnings when present, and order seed candidates by
    // symmetry class so a symmetric dead end is not rediscovered once per image.
    const std::optional<GlobalMeshId> next_mesh = select_next_mesh(assignment, mesh_level_graph);
    if (!next_mesh.has_value()) {
        // Base case: every mesh is placed. The only way this is empty is a zero-mesh input, which can
        // only happen in the top-level call, so the recursion never mistakes it for a dead end.
        return assignment;
    }

    // Enumerate every placement per variant, uncapped: a pool truncated by a cap can hide the only
    // solution, and the search would have no way to find it. The trimmed physical graph lives inside this call and is
    // gone before we recurse, so it is never multiplied by search depth.
    //
    // TODO: this runs a fresh CSP enumeration at every node, which is by far the most expensive thing the
    // search does. Two ways to stop paying it:
    //
    //  - Memoize the pool in a transposition table keyed on (grouping variants, occupied set). Key on the
    //    variants rather than the mesh id, so every instance of the same mesh definition shares one entry —
    //    a descriptor with many identical meshes then solves each shape once per distinct occupancy instead
    //    of once per mesh. The key has to cover everything the pool depends on, so it must grow to include
    //    the placed neighbours' footprints, since the seam constraints are derived from them.
    //  - Better: enumerate each grouping variant once against the FULL physical graph before the search
    //    starts, and at each node filter that master list by occupancy instead of re-solving. Held as
    //    bitsets over ASIC index, disjointness is a few word ANDs per candidate, so the per-node cost drops
    //    from a CSP solve to a linear scan. The seam constraint filters the same list the same way. The
    //    cost is building the master list up front and holding it, which is wasted if the search only ever
    //    needs a handful of candidates.
    std::vector<PsdPlacement> candidates = next_step_pool(
        *next_mesh,
        assignment,
        global_mesh_groupings,
        mesh_level_graph,
        physical_graph,
        physical_system_descriptor,
        /*max_placements_per_variant=*/0);  // TRY ALL, but then see if you need to cap it

    // TODO: value ordering — try the least-constraining candidate first (the one leaving
    // the most live candidates for this mesh's unplaced neighbours), then prefer fewer hosts spanned.
    // Candidates are currently tried in enumeration order.

    for (PsdPlacement& candidate : candidates) {
        // Budget is on search nodes expanded rather than wall clock, so a failure is reproducible from
        // the MGD and PSD alone. 0 means no limit.
        ++nodes_expanded;
        if (node_budget != 0 && nodes_expanded > node_budget) {
            return {};
        }

        // Take this branch's own copy of the state and commit the candidate to it. The copy IS the undo
        // mechanism: if the branch fails, it is discarded and `assignment` was never touched.
        AssignedMeshes branch = assignment;
        branch.push_back(PlacedMesh{*next_mesh, std::move(candidate)});

        // TODO: forward_check — before recursing, recompute the domains of `*next_mesh`'s
        // unplaced neighbours and fail early on a domain wipeout or a violated union bound, so a dead
        // branch is caught here instead of several levels deeper.

        AssignedMeshes completed = place_remaining_meshes(
            std::move(branch),
            global_mesh_groupings,
            mesh_level_graph,
            physical_graph,
            physical_system_descriptor,
            nodes_expanded,
            node_budget);
        if (!completed.empty()) {
            return completed;
        }
        // Dead end. Nothing to undo — `branch` is already gone.
    }

    // TODO: keep the deepest partial assignment reached and report it as the diagnostic
    // ("placed 47 of 69 meshes; L48 needs a region adjacent to L47 at chips [...]"), rather than
    // returning a bare failure.
    return {};
}

AssignedMeshes start_adjacency_guided_dfs(
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t node_budget) {
    // Owned here so every branch shares one counter.
    std::size_t nodes_expanded = 0;

    // Start from an empty assignment. No seeding is needed: select_next_mesh picks the first mesh when
    // nothing is placed, and picks a fresh one again whenever the frontier runs dry, which is how
    // disconnected components are covered.
    AssignedMeshes assignment = place_remaining_meshes(
        /*assignment=*/{},
        global_mesh_groupings,
        mesh_level_graph,
        physical_graph,
        physical_system_descriptor,
        nodes_expanded,
        node_budget);

    // A complete placement has one entry per mesh; anything else is a search failure. Counted against the
    // graph, since that is what the search enumerates from.
    if (assignment.size() != mesh_level_graph.get_nodes().size()) {
        return {};
    }
    return assignment;
}

}  // namespace

std::vector<PsdPlacement> PhysicalGroupingDescriptor::solve_adjacency_guided_placement(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const ValidGroupingsMap& valid_groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t node_budget) const {
    return solve_adjacency_guided_placement(
        {&mesh_graph_descriptor}, valid_groupings, physical_system_descriptor, node_budget);
}

std::vector<PsdPlacement> PhysicalGroupingDescriptor::solve_adjacency_guided_placement(
    const std::vector<const MeshGraphDescriptor*>& mesh_graph_descriptors,
    const ValidGroupingsMap& valid_groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t node_budget) const {
    using tt::tt_metal::experimental::tt_fabric::build_logical_multi_mesh_adjacency_graph;
    using tt::tt_metal::experimental::tt_fabric::LogicalMultiMeshGraph;
    using tt::tt_metal::experimental::tt_fabric::merge_logical_multi_mesh_adjacency_graphs;

    if (mesh_graph_descriptors.empty()) {
        return {};
    }
    // NOTE: For now, only MESH groupings are supported, we will need to include support for hierarchical groupings in
    // the future.
    const auto mesh_it = valid_groupings.find("MESH");
    if (mesh_it == valid_groupings.end() || mesh_it->second.empty()) {
        return {};
    }

    // Per-MGD logical multi-mesh adjacency graphs (chip-level + mesh-level graphs for one descriptor each).
    std::vector<LogicalMultiMeshGraph> parts;
    parts.reserve(mesh_graph_descriptors.size());
    for (const MeshGraphDescriptor* descriptor : mesh_graph_descriptors) {
        parts.push_back(build_logical_multi_mesh_adjacency_graph(*descriptor));
    }
    // Local-to-global mesh ID maps produced while merging parts (one map per MGD index).
    // Unified logical topology: global mesh-level adjacency plus merged per-mesh chip graphs.
    std::vector<std::map<MeshId, MeshId>> local_to_global_mesh_ids;
    const LogicalMultiMeshGraph merged = merge_logical_multi_mesh_adjacency_graphs(parts, &local_to_global_mesh_ids);
    if (merged.mesh_adjacency_graphs_.empty()) {
        return {};
    }

    // Flat ASIC adjacency graph of the physical system (search domain for placement).
    const AdjacencyGraph<AsicID> physical_graph(
        tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd(physical_system_descriptor));

    // Instance-name keyed grouping variants from valid_groupings["MESH"].
    // Global mesh ID -> grouping variants, remapped from mesh_groupings via local_to_global_mesh_ids.
    const std::unordered_map<InstanceName, std::vector<GroupingInfo>>& mesh_groupings = mesh_it->second;
    std::map<MeshId, std::vector<GroupingInfo>> global_mesh_groupings;
    const std::size_t mgd_count = mesh_graph_descriptors.size();
    for (std::size_t mgd_index = 0; mgd_index < local_to_global_mesh_ids.size(); ++mgd_index) {
        if (mgd_index >= mgd_count) {
            break;
        }
        const MeshGraphDescriptor& mgd = *mesh_graph_descriptors[mgd_index];
        const auto mesh_id_to_instance_name = mgd.mesh_id_to_instance_name();
        for (const auto& [local_mesh_id, global_mesh_id] : local_to_global_mesh_ids[mgd_index]) {
            // Every mesh must be placeable. Skipping one here would drop it from the search silently and
            // still report success, since it would be missing from the completeness check too.
            const auto name_it = mesh_id_to_instance_name.find(local_mesh_id);
            TT_FATAL(
                name_it != mesh_id_to_instance_name.end(),
                "Internal error: Adjacency-guided placement: mesh {} of descriptor {} has no instance name, "
                "so its grouping variants cannot be looked up",
                *local_mesh_id,
                mgd_index);
            const InstanceName grouping_key = merged_instance_key(mgd_index, mgd_count, name_it->second);
            const auto groupings_it = mesh_groupings.find(grouping_key);
            TT_FATAL(
                groupings_it != mesh_groupings.end() && !groupings_it->second.empty(),
                "Internal error: Adjacency-guided placement: mesh '{}' (global mesh {}) has no valid grouping "
                "variants, so no region of this system can host it",
                grouping_key,
                *global_mesh_id);
            global_mesh_groupings.emplace(global_mesh_id, groupings_it->second);
        }
    }

    // Adjacency-guided DFS: the placement chosen for each global mesh ID. The mesh-level graph is what
    // enumerates the meshes, which is sound because its builder seeds a node per mesh before adding any
    // connection edges, so a mesh with no intermesh links is still a node with an empty neighbour list.
    auto mesh_placements = start_adjacency_guided_dfs(
        global_mesh_groupings, merged.mesh_level_graph_, physical_graph, physical_system_descriptor, node_budget);

    // Drop the mesh keying the caller does not consume and return the placements as a flat list.
    // TODO: Consider keeping GlobalmeshId as hint for later lookups
    std::vector<PsdPlacement> placements;
    placements.reserve(mesh_placements.size());
    for (PlacedMesh& placed : mesh_placements) {
        if (placed.placement.asics.empty()) {
            continue;
        }
        placements.push_back(std::move(placed.placement));
    }
    return placements;
}

// TODO: delete both overloads once solve_adjacency_guided_placement is the only producer.
std::vector<PsdPlacement> PhysicalGroupingDescriptor::find_all_in_psd(
    const std::vector<GroupingInfo>& groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const {
    PhysicalAdjacencyMap physical_adj_map = build_flat_adjacency_map_from_psd(physical_system_descriptor);
    AdjacencyGraph<AsicID> physical_graph(physical_adj_map);
    return find_all_in_psd(groupings, physical_system_descriptor, physical_graph);
}

// NOTE this only works on flattenable meshes right now
// TODO: delete with the overload above.
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
            errors_out->push_back(build_pgd_mapping_failure_message(
                mesh_to_use.name, flat_meshes.size(), mesh_to_use.adjacency_graph.get_nodes().size()));
        }
    }

    return placements;
}

}  // namespace tt::tt_fabric
