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
#include <string_view>
#include <tuple>
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

#include "topology_solver_sat_solver.hpp"

#include <google/protobuf/text_format.h>

using namespace tt::tt_fabric;

namespace {

// Helper function to build adjacency graph from row-major mesh connection.
// LINE neighbors are always included. When `ring_dims[d]` is true, also wrap both ends of dimension d.
// Missing `ring_dims` entries are treated as LINE (no wrap). RING wrap is skipped when dim < 3.
AdjacencyGraph<GroupingChipId> build_row_major_mesh_graph(
    const std::vector<GroupingChipId>& instance_ids,
    const std::vector<int32_t>& dims,
    const std::string& grouping_name,
    uint32_t connections_per_edge,
    const std::vector<bool>& ring_dims = {}) {
    std::map<GroupingChipId, std::vector<GroupingChipId>> adj_map;

    if (instance_ids.empty() || dims.empty()) {
        return AdjacencyGraph<GroupingChipId>(adj_map);
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
        GroupingChipId node_id = instance_ids[node_idx];
        std::vector<int32_t> coords = get_coords(node_idx);

        for (int32_t dim_idx = 0; dim_idx < static_cast<int32_t>(dims.size()); ++dim_idx) {
            const int32_t dim_size = dims[dim_idx];
            const int32_t coord_val = coords[dim_idx];
            const bool is_ring =
                dim_idx < static_cast<int32_t>(ring_dims.size()) && ring_dims[static_cast<size_t>(dim_idx)];

            auto add_neighbor_coord = [&](int32_t neighbor_coord_val) {
                std::vector<int32_t> neighbor_coords = coords;
                neighbor_coords[dim_idx] = neighbor_coord_val;
                GroupingChipId neighbor_id = instance_ids[get_index(neighbor_coords)];
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

    return AdjacencyGraph<GroupingChipId>(adj_map);
}

struct MgdDeviceTopology {
    std::vector<int32_t> dims;
    std::vector<bool> ring_dims;
};

// Size-1 and size-2 axes are ordinary mesh links; wrapping them does not add a distinct torus edge.
// TORUSX wraps flattened_node_grid_dims[0], TORUSY wraps [1] (same convention as flatten variants).
// Returns MESH=0 / TORUSX=1 / TORUSY=2 / TORUSXY=3 using only wraps on dims > 2.
int effective_torus_variant_priority(const GroupingInfo& grouping) {
    const auto& dims = grouping.flattened_node_grid_dims;
    auto genuine_wrap = [&](size_t i) {
        return i < dims.size() && dims[i] > 0 && tt::tt_fabric::is_genuine_torus_dim(static_cast<uint32_t>(dims[i]));
    };
    const std::string& type = grouping.type;
    const bool wrap_x = (type == "TORUSX" || type == "TORUSXY") && genuine_wrap(0);
    const bool wrap_y = (type == "TORUSY" || type == "TORUSXY") && genuine_wrap(1);
    if (wrap_x && wrap_y) {
        return 3;
    }
    if (wrap_y) {
        return 2;
    }
    if (wrap_x) {
        return 1;
    }
    return 0;
}

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
        const bool declared_ring = device_topology->dim_types(i) == proto::TorusTopology::RING;
        const int32_t dim_size = i < device_topology->dims_size() ? device_topology->dims(i) : 0;
        // RING on a dim of 2 or less is a no-op (same edges as LINE). Drop it so matching does not
        // look for a TORUS variant in that direction.
        topo.ring_dims.push_back(
            declared_ring && dim_size > 0 && tt::tt_fabric::is_genuine_torus_dim(static_cast<uint32_t>(dim_size)));
    }
    return topo;
}

GroupingInfo finalize_mesh_grouping_with_device_topology(
    const GroupingInfo& grouping,
    const MgdDeviceTopology& device_topo,
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
    // larger asic_count; reset it to the node count so the grouping stays self-consistent (is_flattened()
    // stays true and downstream PSD placement does not try to re-flatten an already-flattened mesh).
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

    const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(mesh_instance.desc);
    TT_FATAL(mesh_desc != nullptr, "Mesh descriptor is null");

    // Get device topology dimensions (represents ASIC-level layout)
    const auto& device_topology = mesh_desc->device_topology();
    std::vector<int32_t> device_dims(device_topology.dims().begin(), device_topology.dims().end());

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
    std::vector<bool> ring_dims;
    ring_dims.reserve(device_topology.dim_types_size());
    for (int i = 0; i < device_topology.dim_types_size(); ++i) {
        ring_dims.push_back(device_topology.dim_types(i) == proto::TorusTopology::RING);
    }
    return build_row_major_mesh_graph(asic_ids, device_dims, "", 1, ring_dims);
}

// Helper function to build adjacency graph from MGD switch instance
// Similar to build_mgd_mesh_instance_adjacency - builds row-major mesh graph from device_topology
AdjacencyGraph<GroupingChipId> build_mgd_switch_instance_adjacency(
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
        AdjacencyGraph<GroupingChipId> adjacency_graph =
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

// Applies MGD pinning groups as required constraints. `resolve_globals_at_position` maps each
// pinned ASIC position to the globals that occupy it (PGD grouping nodes, or PSD AsicIDs).
template <typename GlobalNode, typename ResolveGlobals>
std::size_t add_mgd_asic_position_pinning_constraints(
    MappingConstraints<LogicalChipId, GlobalNode>& constraints,
    const std::vector<tt::tt_metal::experimental::tt_fabric::PinningConstraint>& pinnings,
    ResolveGlobals&& resolve_globals_at_position) {
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
    MappingConstraints<LogicalChipId, AsicID>& constraints,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::STRICT,
    PlacementSolveStats* stats = nullptr) {
    if (!add_pgd_to_psd_constraints(grouping_info, physical_graph, physical_system_descriptor, constraints, nullptr)) {
        return {};
    }
    const auto solve_start = std::chrono::steady_clock::now();
    auto mappings = solve_topology_mapping_n<LogicalChipId, AsicID>(
        grouping_info.adjacency_graph,
        physical_graph,
        constraints,
        max_solutions,
        validation_mode,
        /*quiet_mode=*/true,
        TopologyMappingSolverEngine::Auto,
        /*unique_shapes=*/true);
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
        "  adjacency DFS nodes expanded: {}\n"
        "  next_step_pool calls: {}  ({} us)\n"
        "  inner solver calls: {} (SAT {}, DFS {})  ({} us)\n"
        "  inner DFS visits: {}  backtracks: {}  memo hits: {}\n"
        "  candidates generated: {}  inner solutions: {}\n"
        "  slowest inner: {} {}x{} in {} us\n"
        "  master (SAT joint placement): attempted {}  success {}  candidates {}  growth rounds {}  attempts {}  "
        "lists complete {}  closest partial {} meshes\n"
        "  master SAT: {} vars  {} clauses  enumerate {} us  encode {} us  solve {} us\n"
        "  total: {} us ({:.3f} ms)",
        success,
        meshes_placed,
        meshes_total,
        adjacency_nodes_expanded,
        next_step_pool_calls,
        next_step_pool_elapsed.count(),
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
        master_closest_meshes_placed,
        master_sat_vars,
        master_sat_clauses,
        master_enumeration_elapsed.count(),
        master_encode_elapsed.count(),
        master_solve_elapsed.count(),
        total_elapsed.count(),
        static_cast<double>(total_elapsed.count()) / 1000.0);
}

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
        auto count_undirected_edges = [](const AdjacencyGraph<GroupingChipId>& g) -> size_t {
            size_t directed = 0;
            for (GroupingChipId node : g.get_nodes()) {
                directed += g.get_neighbors(node).size();
            }
            return directed / 2;  // each undirected edge is stored from both endpoints
        };
        const size_t required_edges = count_undirected_edges(mgd_grouping_info.adjacency_graph);

        const auto device_topo = get_mgd_instance_device_topology(mesh_graph_descriptor, instance_name);

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
                    } else {
                        // No pinning for this MGD instance: keep the (0,0) anchor so the solve stays constrained
                        // instead of running unconstrained.
                        constraints.add_required_constraint(0, 0);
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
                // (and find_all_in_psd) places on the same graph.
                auto make_committed_grouping = [&](const MeshTopologyMatch& match) -> GroupingInfo {
                    GroupingInfo committed = mesh_flat_groupings.at(match.name)[match.idx];
                    // The topology solve used the MGD mesh adjacency as target and this PGD variant as global, so
                    // target_to_global is MGD-node -> PGD grouping-node. Compose logical chip_id -> PGD slot pinning
                    // now so downstream consumes it directly without re-deriving the intermediate node pairing.
                    committed.mesh_node_to_asic_position =
                        compose_mesh_node_to_asic_position_from_pgd_match(committed, match.mapping.target_to_global);
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
                // solve rather than find_any_in_psd, which requires a still-hierarchical grouping.
                // The MGD fallback is the same check, but it is not a PGD match, so it runs once
                // after this loop rather than once per candidate.
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

        // Offer the MGD grouping as a last-resort variant whenever it embeds on the PSD, even if a
        // PGD grouping already committed. Priority is encoded in this vector's order: PGD first,
        // MGD last. Placement walks that list, so PGD is preferred and this is only the fallback.
        // Do not insert it earlier or reorder this vector after this point.
        auto& committed = result[instance_type][instance_name];
        GroupingInfo mgd_fallback = device_topo.has_value()
                                        ? finalize_mesh_grouping_with_device_topology(mgd_grouping_info, *device_topo)
                                        : mgd_grouping_info;
        bool mgd_places = true;
        if (physical_system_descriptor != nullptr) {
            mgd_places = false;
            // Same position -> ASIC index the mapper uses for MGD pinnings (build_asic_positions_map).
            std::map<tt::tt_metal::ASICPosition, std::set<tt::tt_metal::AsicID>> asics_by_position;
            for (const tt::tt_metal::AsicID& asic_id : psd_physical_graph->get_nodes()) {
                asics_by_position[{physical_system_descriptor->get_tray_id(asic_id),
                                   physical_system_descriptor->get_asic_location(asic_id)}]
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
                if (!enumerate_distinct_placements_for_grouping(
                         mgd_fallback,
                         *psd_physical_graph,
                         *physical_system_descriptor,
                         /*max_solutions=*/1,
                         solve_constraints)
                         .empty()) {
                    mgd_places = true;
                    break;
                }
            }
        }
        if (mgd_places) {
            if (committed.empty() || committed.back().name != mgd_fallback.name) {
                log_info(
                    tt::LogFabric,
                    "Physical groupings: Mesh graph descriptor '{}': {} topology match(es), {} Mesh graph "
                    "descriptor: {} ({})",
                    mgd_grouping_info.name,
                    last_topology_match_count,
                    committed_pgd_matches ? "also offering" : "fallback to",
                    mgd_grouping_info.name,
                    mgd_grouping_info.type);
                committed.push_back(std::move(mgd_fallback));
            }
        }
        TT_FATAL(
            !committed.empty(),
            "Physical groupings: Mesh graph descriptor '{}': no PGD grouping and no MGD grouping "
            "could be placed on the PSD ({} topology match(es))",
            mgd_grouping_info.name,
            last_topology_match_count);
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
// Test-only packer (find_all_in_psd). Production placement is solve_adjacency_guided_placement.
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
//
// PGD_DFS_DEBUG — grep "PGD_DFS_DEBUG" to find and remove all adjacency-guided DFS debug
// instrumentation once grouping investigation is complete. Removal checklist:
//   - PlacementCandidate (revert next_step_pool to std::vector<PsdPlacement>)
//   - PlacedMesh::grouping_name / grouping_type
//   - maybe_record_deepest_partial + deepest_partial/deepest_count plumbing
//   - mesh_label_for_id + per-step DFS expand/commit/backtrack/dead-end logs
//   - log_adjacency_guided_placement_assignment + mesh_id_to_label in solve_adjacency_guided_placement
// PGD_DFS_DEBUG — start
struct PlacementCandidate {  // PGD_DFS_DEBUG
    PsdPlacement placement;
    std::string grouping_name;  // PGD_DFS_DEBUG
    std::string grouping_type;  // PGD_DFS_DEBUG
};

struct PlacedMesh {
    GlobalMeshId mesh_id;
    PsdPlacement placement;
    std::string grouping_name;  // PGD_DFS_DEBUG
    std::string grouping_type;  // PGD_DFS_DEBUG
};
// PGD_DFS_DEBUG — end (PlacedMesh keeps mesh_id/placement; drop grouping_* fields only)
using AssignedMeshes = std::vector<PlacedMesh>;

const PlacedMesh* find_placed_mesh(const AssignedMeshes& assignment, const GlobalMeshId& mesh_id) {
    for (const PlacedMesh& placed : assignment) {
        if (placed.mesh_id == mesh_id) {
            return &placed;
        }
    }
    return nullptr;
}

bool assignment_has_mesh(const AssignedMeshes& assignment, const GlobalMeshId& mesh_id) {
    return find_placed_mesh(assignment, mesh_id) != nullptr;
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

// One already-placed neighbour this mesh must touch. `boundary` is the free chips bordering that
// neighbour's region (keys are candidate chips; values are link count into the region, used as the
// cardinality pair weight). `requested_links` is the descriptor's channel count for this seam: the
// cardinality min_count under STRICT, and the amount tried first under RELAXED before falling back to 1.
struct Seam {
    std::map<AsicID, std::size_t> boundary;
    std::size_t requested_links = 0;
};

// Seams from `mesh_id` to its already-placed neighbours. Derived from the assignment and the
// mesh-level graph, not from a grouping variant. nullopt if a placed neighbour has no free chips
// bordering it, so no placement of this mesh can reach that seam.
std::optional<std::vector<Seam>> collect_seams_to_placed_neighbors(
    const GlobalMeshId& mesh_id,
    const AssignedMeshes& assignment,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const std::unordered_set<AsicID>& occupied,
    const AdjacencyGraph<AsicID>& physical_graph) {
    const std::map<GlobalMeshId, std::size_t> required_links_by_neighbor =
        placed_neighbors_of(mesh_id, assignment, mesh_level_graph);
    std::vector<Seam> seams;
    seams.reserve(required_links_by_neighbor.size());
    for (const auto& [neighbor_id, requested_links] : required_links_by_neighbor) {
        const PlacedMesh* placed = find_placed_mesh(assignment, neighbor_id);
        if (placed == nullptr) {
            continue;
        }
        std::map<AsicID, std::size_t> boundary =
            free_chips_bordering_region(placed->placement.asics, occupied, physical_graph);
        if (boundary.empty()) {
            return std::nullopt;
        }
        seams.push_back(Seam{.boundary = std::move(boundary), .requested_links = requested_links});
    }
    return seams;
}

// Under RELAXED, ask the solver to sit on the fattest chips it can instead of scoring embeddings after
// they come back. With a seam that is the highest-weight boundary chips; with none (the seed mesh) it
// is the highest-degree free chips, so later seams still have width to use.
void add_relaxed_preferred_chip_constraints(
    MappingConstraints<LogicalChipId, AsicID>& constraints,
    const std::vector<uint32_t>& grouping_nodes,
    const std::vector<Seam>& seams,
    const AdjacencyGraph<AsicID>& free_physical_graph) {
    std::set<AsicID> preferred_chips;
    std::size_t best_weight = 0;
    if (!seams.empty()) {
        for (const Seam& seam : seams) {
            for (const auto& [chip, links] : seam.boundary) {
                best_weight = std::max(best_weight, links);
            }
        }
        for (const Seam& seam : seams) {
            for (const auto& [chip, links] : seam.boundary) {
                if (links == best_weight) {
                    preferred_chips.insert(chip);
                }
            }
        }
    } else {
        for (const AsicID& chip : free_physical_graph.get_nodes()) {
            best_weight = std::max(best_weight, free_physical_graph.get_neighbors(chip).size());
        }
        if (best_weight > 0) {
            for (const AsicID& chip : free_physical_graph.get_nodes()) {
                if (free_physical_graph.get_neighbors(chip).size() == best_weight) {
                    preferred_chips.insert(chip);
                }
            }
        }
    }
    if (preferred_chips.empty()) {
        return;
    }
    for (const LogicalChipId node : grouping_nodes) {
        constraints.add_preferred_constraint(node, preferred_chips);
    }
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
// variant accepted for this mesh that is disjoint from the regions already taken and that reaches every
// mesh-level edge to an already-placed neighbour.
//
// Disjointness is enforced by the solve rather than filtered afterwards, so an overlapping placement is
// never constructed. Seam width is a weighted cardinality constraint: each (our-chip, border-ASIC) pair
// is worth that ASIC's link count into the neighbour, and min_count is the requested channel count, so
// the embeddings that come back already carry enough connections. Under RELAXED, if that amount cannot
// be met the constraint falls back to a single link (touching) and the solver is asked to prefer the
// fattest remaining chips / highest-degree free chips rather than ranking candidates after the fact.
//
// What a seam must carry depends on the descriptor's inter-mesh channel policy, and follows what the
// mapper does with the same policy (see check_local_consistency and compute_candidate_cost in
// topology_solver.tpp):
//
//   STRICT  - the mesh-level edge multiplicity is a hard requirement. A seam carrying fewer links than
//             the descriptor asked for is not a placement at all.
//   RELAXED - the count is a preference. The solve still asks for the full amount first; if that is
//             impossible the constraint falls back to a single link so the meshes still touch, and
//             preferred mappings (fattest boundary chips, or highest-degree free chips with no seam) steer
//             toward a wider seating.
//             Making the full count a hard filter with no fallback would be stricter than the mapper,
//             which accepts a narrow seam here and warns.
std::vector<PlacementCandidate> next_step_pool(  // PGD_DFS_DEBUG: revert to std::vector<PsdPlacement>
    const GlobalMeshId& mesh_id,
    const AssignedMeshes& assignment,
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    bool relaxed_inter_mesh_policy,
    std::size_t max_placements_per_variant,
    PlacementSolveStats* stats) {
    const auto pool_start = std::chrono::steady_clock::now();
    struct PoolElapsed {
        PlacementSolveStats* stats;
        std::chrono::steady_clock::time_point start;
        ~PoolElapsed() {
            if (stats != nullptr) {
                stats->next_step_pool_elapsed +=
                    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
            }
        }
    } pool_elapsed{stats, pool_start};
    if (stats != nullptr) {
        ++stats->next_step_pool_calls;
    }
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

    // One seam per already-placed neighbour. Derived once here because it depends only on the
    // assignment, not on which grouping variant we are about to try.
    const std::optional<std::vector<Seam>> seams_or_blocked =
        collect_seams_to_placed_neighbors(mesh_id, assignment, mesh_level_graph, occupied, physical_graph);
    if (!seams_or_blocked.has_value()) {
        // Nothing free borders a placed neighbour, so no placement of this mesh can reach it.
        return {};
    }
    const std::vector<Seam>& seams = *seams_or_blocked;

    std::vector<PlacementCandidate> pool;  // PGD_DFS_DEBUG
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
            MappingConstraints<LogicalChipId, AsicID>::CardinalityPairWeights seam_pair_weights;
            for (const LogicalChipId node : grouping_nodes) {
                for (const auto& [chip, links] : seam.boundary) {
                    seam_pair_weights.emplace(std::pair<LogicalChipId, AsicID>{node, chip}, links);
                }
            }
            if (!constraints.add_cardinality_constraint(seam_pair_weights, seam.requested_links)) {
                // RELAXED: the count is a preference. If the full amount cannot be met, require only
                // that the regions touch; preferred mappings below steer toward the wider chips.
                if (!relaxed_inter_mesh_policy ||
                    !constraints.add_cardinality_constraint(seam_pair_weights, /*min_count=*/1)) {
                    variant_feasible = false;
                    break;
                }
            }
        }
        if (!variant_feasible) {
            continue;
        }

        if (relaxed_inter_mesh_policy) {
            add_relaxed_preferred_chip_constraints(constraints, grouping_nodes, seams, free_physical_graph);
        }

        const ConnectionValidationMode validation_mode =
            relaxed_inter_mesh_policy ? ConnectionValidationMode::RELAXED : ConnectionValidationMode::STRICT;
        for (const auto& mapping : enumerate_distinct_placements_for_grouping(
                 grouping,
                 free_physical_graph,
                 physical_system_descriptor,
                 max_placements_per_variant,
                 constraints,
                 validation_mode,
                 stats)) {
            if (!mapping.success) {
                continue;
            }
            PlacementCandidate candidate;  // PGD_DFS_DEBUG
            // Carry the variant's pinning map: it is a property of the variant, not of the footprint,
            // so it cannot be recovered once the grouping is out of scope.
            candidate.placement.mesh_node_to_asic_position = grouping.mesh_node_to_asic_position;
            for (const auto& [grouping_node, asic_id] : mapping.target_to_global) {
                candidate.placement.asics.insert(asic_id);
            }
            candidate.grouping_name = grouping.name;  // PGD_DFS_DEBUG
            candidate.grouping_type = grouping.type;  // PGD_DFS_DEBUG
            pool.push_back(std::move(candidate));
            if (stats != nullptr) {
                ++stats->candidates_generated;
            }
        }
    }

    return pool;
}

// How many placements one grouping variant may contribute to a search node's candidate pool. A variant
// with no per-node tray/location traits to prune on -- an MGD fallback has none -- otherwise enumerates
// its whole symmetric solution space, which on a galaxy is minutes spent on candidates the search never
// reaches.
//
// Ten has been enough for every descriptor tried so far. Raise it if one that should place comes back
// unplaced: a truncated pool can hide the only seating that works, so this cap is the first thing to
// suspect before anything else in the search.
constexpr std::size_t kMaxPlacementsPerVariant = 10;

// Completes `assignment` into a placement for every remaining mesh.
//
// Returns an empty vector when this branch cannot be completed. A non-empty return means every mesh in
// `mesh_level_graph` has an entry (size matches). The assignment is taken BY VALUE: each branch
// works on its own copy, so a branch that fails just discards it and the caller's copy is untouched.
//
// `nodes_expanded` is shared across the whole search and must be passed by reference: a by-value
// counter would restart the budget down every branch and never actually bind.
// PGD_DFS_DEBUG — delete after grouping investigation.
std::string mesh_label_for_id(
    const GlobalMeshId& mesh_id, const std::map<GlobalMeshId, std::string>& mesh_id_to_label) {
    const auto label_it = mesh_id_to_label.find(mesh_id);
    return label_it != mesh_id_to_label.end() ? label_it->second : fmt::format("mesh {}", *mesh_id);
}

// PGD_DFS_DEBUG — delete after grouping investigation (deepest-partial logging on DFS failure).
void maybe_record_deepest_partial(
    const AssignedMeshes& assignment, AssignedMeshes* deepest_partial, std::size_t* deepest_count) {
    if (deepest_partial == nullptr || deepest_count == nullptr) {
        return;
    }
    if (assignment.size() > *deepest_count) {
        *deepest_count = assignment.size();
        *deepest_partial = assignment;
    }
}

AssignedMeshes place_remaining_meshes(
    AssignedMeshes assignment,

    // Global variables
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    bool relaxed_inter_mesh_policy,
    std::size_t& nodes_expanded,
    std::size_t node_budget,
    PlacementSolveStats* stats,
    AssignedMeshes* deepest_partial,                                // PGD_DFS_DEBUG
    std::size_t* deepest_count,                                     // PGD_DFS_DEBUG
    const std::map<GlobalMeshId, std::string>& mesh_id_to_label) {  // PGD_DFS_DEBUG
    // TODO: seed from MGD pinnings when present, and order seed candidates by
    // symmetry class so a symmetric dead end is not rediscovered once per image.
    const std::optional<GlobalMeshId> next_mesh = select_next_mesh(assignment, mesh_level_graph);
    if (!next_mesh.has_value()) {
        // Base case: every mesh is placed. The only way this is empty is a zero-mesh input, which can
        // only happen in the top-level call, so the recursion never mistakes it for a dead end.
        return assignment;
    }

    const std::string next_mesh_label = mesh_label_for_id(*next_mesh, mesh_id_to_label);  // PGD_DFS_DEBUG

    // Enumerate a bounded number of placements per variant. A truncated pool can in principle hide the
    // only solution, but an unbounded one lets a single unconstrained variant cost more than the whole
    // rest of the search. The trimmed physical graph lives inside this call and is
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
    std::vector<PlacementCandidate> candidates = next_step_pool(  // PGD_DFS_DEBUG
        *next_mesh,
        assignment,
        global_mesh_groupings,
        mesh_level_graph,
        physical_graph,
        physical_system_descriptor,
        relaxed_inter_mesh_policy,
        kMaxPlacementsPerVariant,
        stats);

    // PGD_DFS_DEBUG
    log_info(
        tt::LogFabric,
        "Adjacency-guided DFS expand depth {}: seating {} (global mesh {}), {} candidate(s) [search node {}]",
        assignment.size() + 1,
        next_mesh_label,
        *next_mesh,
        candidates.size(),
        nodes_expanded + 1);

    // TODO: value ordering — try the least-constraining candidate first (the one leaving
    // the most live candidates for this mesh's unplaced neighbours), then prefer fewer hosts spanned.
    // Under STRICT candidates are still tried in enumeration order; under RELAXED next_step_pool
    // already ranks them, but only by seam width, which is one part of least-constraining.

    for (PlacementCandidate& candidate : candidates) {  // PGD_DFS_DEBUG
        // Budget is on search nodes expanded rather than wall clock, so a failure is reproducible from
        // the MGD and PSD alone. 0 means no limit.
        ++nodes_expanded;
        if (node_budget != 0 && nodes_expanded > node_budget) {
            maybe_record_deepest_partial(assignment, deepest_partial, deepest_count);  // PGD_DFS_DEBUG
            return {};
        }

        // Take this branch's own copy of the state and commit the candidate to it. The copy IS the undo
        // mechanism: if the branch fails, it is discarded and `assignment` was never touched.
        AssignedMeshes branch = assignment;
        const std::string grouping_name = candidate.grouping_name;  // PGD_DFS_DEBUG
        const std::string grouping_type = candidate.grouping_type;  // PGD_DFS_DEBUG
        branch.push_back(PlacedMesh{
            *next_mesh,
            std::move(candidate.placement),
            grouping_name,                                                     // PGD_DFS_DEBUG
            grouping_type});                                                   // PGD_DFS_DEBUG
        maybe_record_deepest_partial(branch, deepest_partial, deepest_count);  // PGD_DFS_DEBUG
        const std::size_t commit_depth = branch.size();
        // PGD_DFS_DEBUG
        log_info(
            tt::LogFabric,
            "Adjacency-guided DFS commit depth {}: {} (global mesh {}) -> {} ({}) [search node {}]",
            commit_depth,
            next_mesh_label,
            *next_mesh,
            grouping_name,
            grouping_type,
            nodes_expanded);

        // TODO: forward_check — before recursing, recompute the domains of `*next_mesh`'s
        // unplaced neighbours and fail early on a domain wipeout or a violated union bound, so a dead
        // branch is caught here instead of several levels deeper.

        AssignedMeshes completed = place_remaining_meshes(
            std::move(branch),
            global_mesh_groupings,
            mesh_level_graph,
            physical_graph,
            physical_system_descriptor,
            relaxed_inter_mesh_policy,
            nodes_expanded,
            node_budget,
            stats,
            deepest_partial,    // PGD_DFS_DEBUG
            deepest_count,      // PGD_DFS_DEBUG
            mesh_id_to_label);  // PGD_DFS_DEBUG
        if (!completed.empty()) {
            return completed;
        }
        // PGD_DFS_DEBUG
        log_info(
            tt::LogFabric,
            "Adjacency-guided DFS backtrack depth {}: {} -> {} ({}) exhausted [search node {}]",
            commit_depth,
            next_mesh_label,
            grouping_name,
            grouping_type,
            nodes_expanded);
        // Dead end. Nothing to undo — `branch` is already gone.
    }

    if (candidates.empty()) {
        // PGD_DFS_DEBUG
        log_info(
            tt::LogFabric,
            "Adjacency-guided DFS dead-end depth {}: no candidates for {} (global mesh {}) [search node {}]",
            assignment.size() + 1,
            next_mesh_label,
            *next_mesh,
            nodes_expanded);
    }

    maybe_record_deepest_partial(assignment, deepest_partial, deepest_count);  // PGD_DFS_DEBUG
    return {};
}

// PGD_DFS_DEBUG — delete after grouping investigation (logs which PGD variant each seated mesh used).
void log_adjacency_guided_placement_assignment(
    const AssignedMeshes& assignment,
    const std::map<GlobalMeshId, std::string>& mesh_id_to_label,
    std::string_view outcome) {
    if (assignment.empty()) {
        return;
    }
    // PGD_DFS_DEBUG
    log_info(
        tt::LogFabric,
        "Adjacency-guided placement {}: {} mesh(es) seated with PGD grouping(s):",
        outcome,
        assignment.size());
    for (const PlacedMesh& placed : assignment) {
        const auto label_it = mesh_id_to_label.find(placed.mesh_id);
        const std::string mesh_label =
            label_it != mesh_id_to_label.end() ? label_it->second : fmt::format("mesh {}", *placed.mesh_id);
        // PGD_DFS_DEBUG
        log_info(
            tt::LogFabric,
            "  {} (global mesh {}): {} ({})",
            mesh_label,
            *placed.mesh_id,
            placed.grouping_name,
            placed.grouping_type);
    }
}

AssignedMeshes start_adjacency_guided_dfs(
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    bool relaxed_inter_mesh_policy,
    std::size_t node_budget,
    PlacementSolveStats* stats,
    const std::map<GlobalMeshId, std::string>& mesh_id_to_label,  // PGD_DFS_DEBUG
    AssignedMeshes* deepest_partial_out) {                        // PGD_DFS_DEBUG
    // Owned here so every branch shares one counter.
    std::size_t nodes_expanded = 0;
    AssignedMeshes deepest_partial;  // PGD_DFS_DEBUG
    std::size_t deepest_count = 0;   // PGD_DFS_DEBUG

    // Start from an empty assignment. No seeding is needed: select_next_mesh picks the first mesh when
    // nothing is placed, and picks a fresh one again whenever the frontier runs dry, which is how
    // disconnected components are covered.
    AssignedMeshes assignment = place_remaining_meshes(
        /*assignment=*/{},
        global_mesh_groupings,
        mesh_level_graph,
        physical_graph,
        physical_system_descriptor,
        relaxed_inter_mesh_policy,
        nodes_expanded,
        node_budget,
        stats,
        &deepest_partial,   // PGD_DFS_DEBUG
        &deepest_count,     // PGD_DFS_DEBUG
        mesh_id_to_label);  // PGD_DFS_DEBUG

    if (stats != nullptr) {
        stats->adjacency_nodes_expanded = nodes_expanded;
    }

    // A complete placement has one entry per mesh; anything else is a search failure. Counted against the
    // graph, since that is what the search enumerates from.
    if (assignment.size() != mesh_level_graph.get_nodes().size()) {
        if (!deepest_partial.empty()) {                 // PGD_DFS_DEBUG
            log_adjacency_guided_placement_assignment(  // PGD_DFS_DEBUG
                deepest_partial,
                mesh_id_to_label,
                fmt::format(
                    "deepest partial ({} of {} meshes)", deepest_partial.size(), mesh_level_graph.get_nodes().size()));
            if (deepest_partial_out != nullptr) {  // PGD_DFS_DEBUG
                *deepest_partial_out = std::move(deepest_partial);
            }
        }
        return {};
    }
    log_adjacency_guided_placement_assignment(assignment, mesh_id_to_label, "complete");  // PGD_DFS_DEBUG
    return assignment;
}

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
// (MasterCandidateLists::complete()).
// =====================================================================================================

// Dense 0..N-1 numbering of the fabric's ASICs. Built once from the physical graph, whose node order is
// already deterministic (AdjacencyMap is a std::map), so the numbering agrees across ranks and runs.
class AsicIndex {
public:
    explicit AsicIndex(const AdjacencyGraph<AsicID>& physical_graph) {
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
    AsicID asic(std::size_t dense) const { return dense_to_asic_[dense]; }

private:
    std::unordered_map<AsicID, std::size_t> asic_to_dense_;
    std::vector<AsicID> dense_to_asic_;
};

// Fixed-width bitset over the dense ASIC index. The word count is a runtime value (fabric size varies by
// system) so this is a vector rather than std::bitset, but it is never resized after construction.
class AsicBitset {
public:
    AsicBitset() = default;
    explicit AsicBitset(std::size_t bit_count) : words_((bit_count + 63) / 64, 0) {}

    void set(std::size_t bit) { words_[bit >> 6] |= (uint64_t{1} << (bit & 63)); }
    bool test(std::size_t bit) const { return ((words_[bit >> 6] >> (bit & 63)) & 1) != 0; }

    bool intersects(const AsicBitset& other) const {
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

// One legal seating of one grouping variant. Footprint-only: the per-node mapping from the inner solve is
// deliberately dropped, matching what next_step_pool keeps today -- downstream reconstructs positions
// from the variant's pinning map, not from the embedding.
struct MasterCandidate {
    AsicBitset footprint;
    std::vector<AsicID> asics;  // same content as the footprint, for building PsdPlacement directly
    // Dense index of the far end of every fabric link leaving the footprint. Parallel links are duplicate
    // entries, so intersecting this with another footprint counts links rather than chips.
    std::vector<uint32_t> boundary_dense;
    // Borrowed, never owned: name, type and mesh_node_to_asic_position live on the grouping, which is
    // identical across every placement of the variant. global_mesh_groupings holds the vectors by value
    // and outlives the solve; it must not be mutated once these pointers are taken.
    const GroupingInfo* variant = nullptr;
    uint16_t hosts_spanned = 1;
    // True when the variant is the definition's top-ranked grouping (same name as the first variant
    // get_valid_groupings_for_mgd committed). The master solve tries to seat every mesh on a preferred
    // candidate before it admits the rest, which is the global form of the DFS trying variants in order.
    bool preferred = true;
};

// A variant with no per-node tray / ASIC-location trait is the MGD fallback (or an unpinned PGD
// grouping). Its enumeration is unbounded by anything but the fabric size, so it is held back until the
// pinned variants alone have been shown not to suffice.
bool variant_is_trait_free(const GroupingInfo& grouping) {
    for (LogicalChipId node_id : grouping.adjacency_graph.get_nodes()) {
        if (node_id >= grouping.items.size()) {
            continue;
        }
        const GroupingItemInfo& item = grouping.items[node_id];
        if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
            continue;
        }
        if (*item.tray_id > 0 || *item.asic_location <= 8) {
            return false;
        }
    }
    return true;
}

// One live enumeration per (definition, variant). TopologyMappingEnumerationSession holds a single
// search engine and appends blocking clauses between next() calls, so resuming costs one solve rather
// than a re-encode. Heap-allocated and never moved: the DFS engine keeps pointers into the session's own
// graph/constraint snapshots.
struct VariantSource {
    const GroupingInfo* variant = nullptr;
    TopologyMappingEnumerationSession<LogicalChipId, AsicID> session;
    MappingConstraints<LogicalChipId, AsicID> constraints;  // trait / host-alignment, encoded once
    std::vector<std::map<LogicalChipId, AsicID>> excluded;  // mappings already returned
    std::vector<MasterCandidate> found;                     // stable: only ever appended to
    std::size_t solves = 0;
    bool exhausted = false;
    bool session_started = false;
    bool preferred = true;
};

struct MasterEnumerationContext {
    const AdjacencyGraph<AsicID>& physical_graph;
    const tt::tt_metal::PhysicalSystemDescriptor& psd;
    const AsicIndex& asic_index;
    ConnectionValidationMode validation_mode;
    // Per-definition footprints already listed by a higher-priority variant. Variants of one definition
    // often share footprints (torus variants differ only in adjacency), and a duplicate seat would be
    // pure symmetry for the master solve. First variant wins, which is the "PGD first, MGD last" order
    // get_valid_groupings_for_mgd committed.
    std::set<std::vector<uint64_t>>& seen_footprints;
    PlacementSolveStats* stats;
};

// Pull up to `batch` more placements from one variant. Returns how many were added. Sets `exhausted`
// when the session reports no further distinct mapping -- the only signal that this variant's list is
// COMPLETE, and therefore the only condition under which a later UNSAT is trustworthy.
std::size_t grow_variant(VariantSource& source, MasterEnumerationContext& ctx, std::size_t batch) {
    if (source.exhausted) {
        return 0;
    }
    if (!source.session_started) {
        // Same constraint construction enumerate_distinct_placements_for_grouping performs.
        if (!add_pgd_to_psd_constraints(*source.variant, ctx.physical_graph, ctx.psd, source.constraints, nullptr)) {
            source.exhausted = true;
            return 0;
        }
        source.session_started = true;
    }

    std::size_t added = 0;
    while (added < batch) {
        MappingResult<LogicalChipId, AsicID> mapping = source.session.next(
            source.variant->adjacency_graph,
            ctx.physical_graph,
            source.constraints,
            source.excluded,
            ctx.validation_mode,
            /*quiet_mode=*/true,
            TopologyMappingSolverEngine::Auto,
            /*unique_shapes=*/true);
        ++source.solves;
        if (ctx.stats != nullptr) {
            ++ctx.stats->inner_solver_calls;
            if (mapping.stats.used_sat) {
                ++ctx.stats->inner_solver_sat_calls;
            } else {
                ++ctx.stats->inner_solver_dfs_calls;
            }
        }
        if (!mapping.success) {
            source.exhausted = true;
            break;
        }
        source.excluded.push_back(mapping.target_to_global);
        if (ctx.stats != nullptr) {
            ++ctx.stats->inner_solutions_found;
        }

        MasterCandidate candidate;
        candidate.footprint = AsicBitset(ctx.asic_index.size());
        candidate.variant = source.variant;
        candidate.preferred = source.preferred;
        candidate.asics.reserve(mapping.target_to_global.size());
        std::set<std::string> hosts;
        for (const auto& [node, asic] : mapping.target_to_global) {
            candidate.footprint.set(ctx.asic_index.dense(asic));
            candidate.asics.push_back(asic);
            hosts.insert(ctx.psd.get_host_name_for_asic(asic));
        }
        if (!ctx.seen_footprints.insert(candidate.footprint.words()).second) {
            continue;  // a higher-priority variant already offers this footprint
        }
        candidate.hosts_spanned = static_cast<uint16_t>(hosts.size());
        for (const AsicID& chip : candidate.asics) {
            for (const AsicID& neighbor : ctx.physical_graph.get_neighbors(chip)) {
                const std::size_t dense = ctx.asic_index.dense(neighbor);
                if (!candidate.footprint.test(dense)) {
                    candidate.boundary_dense.push_back(static_cast<uint32_t>(dense));
                }
            }
        }
        source.found.push_back(std::move(candidate));
        ++added;
    }
    return added;
}

// Fabric links crossing from footprint `a` into footprint `b`. Symmetric, since every link is listed from
// both ends in the physical graph.
std::size_t seam_link_count(const MasterCandidate& a, const MasterCandidate& b) {
    std::size_t links = 0;
    for (const uint32_t dense : a.boundary_dense) {
        if (b.footprint.test(dense)) {
            ++links;
        }
    }
    return links;
}

// Candidate lists keyed by mesh DEFINITION (the address of the grouping-variant vector held in
// global_mesh_groupings, shared by every instance of that definition). All 52 S4x1 meshes of a
// descriptor share one entry.
class MasterCandidateLists {
public:
    using DefinitionKey = const std::vector<GroupingInfo>*;

    // Pinned variants become live sources immediately; trait-free ones are deferred until
    // enable_mgd_fallback. That split is what makes "prefer an all-PGD placement" a GLOBAL property: the
    // DFS could only prefer PGD locally, which is why its traces show MGD fallbacks committed at depth
    // 17-19 while PGD options remained elsewhere.
    void add_definition(DefinitionKey key) {
        Definition& definition = definitions_[key];
        // The committed list is in preference order, and every flattened variant of one PGD grouping
        // shares its name, so "same name as the first" marks the top-ranked grouping's variants.
        std::optional<std::string> preferred_name;
        for (const GroupingInfo& variant : *key) {
            if (variant.adjacency_graph.get_nodes().empty()) {
                continue;
            }
            if (!preferred_name.has_value()) {
                preferred_name = variant.name;
            }
            auto source = std::make_unique<VariantSource>();
            source->variant = &variant;
            source->preferred = (variant.name == *preferred_name);
            if (variant_is_trait_free(variant)) {
                definition.deferred.push_back(std::move(source));
            } else {
                definition.active.push_back(std::move(source));
            }
        }
    }

    // Promote the deferred (trait-free) variants of every definition to live sources. Returns how many.
    std::size_t enable_mgd_fallback() {
        std::size_t promoted = 0;
        for (auto& [_, definition] : definitions_) {
            for (auto& source : definition.deferred) {
                definition.active.push_back(std::move(source));
                ++promoted;
            }
            definition.deferred.clear();
        }
        return promoted;
    }

    // Pull up to `batch_per_variant` more candidates from every live source of `key`. Returns how many
    // were added across variants. Any addition invalidates `candidates()` ordering assumptions held by a
    // previous encoding, which is why the master problem is re-encoded after growth.
    std::size_t grow(DefinitionKey key, std::size_t batch_per_variant, MasterEnumerationContext& ctx) {
        Definition& definition = definitions_.at(key);
        std::size_t added = 0;
        for (auto& source : definition.active) {
            if (definition.total_found() >= kMaxCandidatesPerDefinition) {
                break;
            }
            added += grow_variant(*source, ctx, batch_per_variant);
        }
        if (added != 0) {
            rebuild_flat(definition);
            ++generation_;
        }
        return added;
    }

    const std::vector<const MasterCandidate*>& candidates(DefinitionKey key) const { return definitions_.at(key).flat; }

    // True only when every live variant of every definition reported exhaustion and nothing is deferred.
    // UNSAT is meaningless otherwise.
    bool complete() const {
        for (const auto& [_, definition] : definitions_) {
            if (!definition.deferred.empty()) {
                return false;
            }
            for (const auto& source : definition.active) {
                if (!source->exhausted) {
                    return false;
                }
            }
        }
        return true;
    }

    // Whether any live source can still yield candidates (bounded by the per-definition cap).
    bool can_grow() const {
        for (const auto& [_, definition] : definitions_) {
            if (definition.total_found() >= kMaxCandidatesPerDefinition) {
                continue;
            }
            for (const auto& source : definition.active) {
                if (!source->exhausted) {
                    return true;
                }
            }
        }
        return false;
    }

    std::size_t total_candidates() const {
        std::size_t total = 0;
        for (const auto& [_, definition] : definitions_) {
            total += definition.flat.size();
        }
        return total;
    }

    // Bumped on every growth; seam caches key on it so a stale matrix is never reused.
    std::size_t generation() const { return generation_; }

    std::string describe(DefinitionKey key) const {
        const Definition& definition = definitions_.at(key);
        std::string text;
        for (const auto& source : definition.active) {
            text += fmt::format(
                "{}[{} found, {} solves{}] ",
                source->variant->name,
                source->found.size(),
                source->solves,
                source->exhausted ? ", exhausted" : "");
        }
        if (!definition.deferred.empty()) {
            text += fmt::format("({} trait-free variant(s) deferred)", definition.deferred.size());
        }
        return text;
    }

    // Guards a trait-free variant from enumerating its whole symmetric solution space: past this many
    // candidates for one definition the list stops growing and is reported incomplete.
    static constexpr std::size_t kMaxCandidatesPerDefinition = 8192;

private:
    struct Definition {
        std::vector<std::unique_ptr<VariantSource>> active;
        std::vector<std::unique_ptr<VariantSource>> deferred;
        std::set<std::vector<uint64_t>> seen_footprints;
        std::vector<const MasterCandidate*> flat;  // rebuilt after each grow(); variant order, then discovery

        std::size_t total_found() const {
            std::size_t total = 0;
            for (const auto& source : active) {
                total += source->found.size();
            }
            return total;
        }
    };

    static void rebuild_flat(Definition& definition) {
        definition.flat.clear();
        for (const auto& source : definition.active) {
            for (const MasterCandidate& candidate : source->found) {
                definition.flat.push_back(&candidate);
            }
        }
    }

    std::map<DefinitionKey, Definition> definitions_;
    std::size_t generation_ = 0;

public:
    // The per-definition footprint dedup set, handed to the enumeration context of that definition.
    std::set<std::vector<uint64_t>>& seen_footprints(DefinitionKey key) { return definitions_.at(key).seen_footprints; }
};

// Link-count matrices between the candidate lists of two definitions, computed once per unordered pair
// and shared by every mesh-level edge joining instances of those definitions. On the Gemma descriptor
// that is 6 matrices standing in for 71 edges.
class SeamLinkMatrices {
public:
    using DefinitionKey = MasterCandidateLists::DefinitionKey;

    // Entry (i, j) is the number of fabric links between candidate i of `d1` and candidate j of `d2`,
    // saturated at 255. The row/column roles follow the argument order, so callers may pass either order.
    const std::vector<uint8_t>& matrix(
        DefinitionKey d1, DefinitionKey d2, const MasterCandidateLists& lists, std::size_t& cols_out) {
        if (lists.generation() != generation_) {
            cache_.clear();
            generation_ = lists.generation();
        }
        const auto& c1 = lists.candidates(d1);
        const auto& c2 = lists.candidates(d2);
        cols_out = c2.size();
        auto it = cache_.find({d1, d2});
        if (it != cache_.end()) {
            return it->second;
        }
        std::vector<uint8_t> m(c1.size() * c2.size(), 0);
        for (std::size_t i = 0; i < c1.size(); ++i) {
            for (std::size_t j = 0; j < c2.size(); ++j) {
                const std::size_t links = seam_link_count(*c1[i], *c2[j]);
                m[i * c2.size() + j] = static_cast<uint8_t>(std::min<std::size_t>(links, 255));
            }
        }
        return cache_.emplace(std::make_pair(d1, d2), std::move(m)).first->second;
    }

private:
    std::map<std::pair<DefinitionKey, DefinitionKey>, std::vector<uint8_t>> cache_;
    std::size_t generation_ = static_cast<std::size_t>(-1);
};

// One variable per (mesh INSTANCE, candidate). Instances cannot share variables even when they share a
// candidate list, because they take different seats.
struct SeatVars {
    std::map<GlobalMeshId, std::vector<int>> by_mesh;  // parallel to lists.candidates(definition_of(mesh))
    std::vector<std::vector<int>> by_dense_asic;       // asic -> vars whose footprint covers it
    std::vector<int> non_preferred;                    // seats on a lower-ranked variant; assumed false first
};

struct MasterEncodingSize {
    std::size_t vars = 0;
    std::size_t clauses = 0;
};

// At-most-one over `lits`. Pairwise below a small cutoff; sequential (Sinz) above it, which adds n-1
// auxiliary variables and ~3n binary clauses instead of n^2/2. Pairwise is not viable at these sizes:
// several hundred candidates per mesh and per ASIC would put both constraint families in the millions.
constexpr std::size_t kPairwiseAtMostOneCutoff = 8;

void add_at_most_one(
    tt::tt_fabric::detail::TopologySatSolver& sat, const std::vector<int>& lits, MasterEncodingSize& size) {
    if (lits.size() <= 1) {
        return;
    }
    if (lits.size() <= kPairwiseAtMostOneCutoff) {
        for (std::size_t i = 0; i < lits.size(); ++i) {
            for (std::size_t j = i + 1; j < lits.size(); ++j) {
                sat.add(-lits[i]);
                sat.add(-lits[j]);
                sat.add(0);
                ++size.clauses;
            }
        }
        return;
    }
    std::vector<int> chain(lits.size() - 1);
    for (int& s : chain) {
        s = sat.declare_one_more_variable();
        ++size.vars;
    }
    sat.add(-lits[0]);
    sat.add(chain[0]);
    sat.add(0);
    ++size.clauses;
    for (std::size_t i = 1; i + 1 < lits.size(); ++i) {
        sat.add(-lits[i]);
        sat.add(chain[i]);
        sat.add(0);
        sat.add(-chain[i - 1]);
        sat.add(chain[i]);
        sat.add(0);
        sat.add(-lits[i]);
        sat.add(-chain[i - 1]);
        sat.add(0);
        size.clauses += 3;
    }
    sat.add(-lits.back());
    sat.add(-chain.back());
    sat.add(0);
    ++size.clauses;
}

// Mesh-level edges as (m1 < m2) -> channel count. mesh_level_graph carries channel multiplicity as
// duplicate neighbour entries, the same convention placed_neighbors_of relies on.
std::map<std::pair<GlobalMeshId, GlobalMeshId>, std::size_t> collect_mesh_edges(
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph) {
    std::map<std::pair<GlobalMeshId, GlobalMeshId>, std::size_t> edges;
    for (const GlobalMeshId& m1 : mesh_level_graph.get_nodes()) {
        for (const GlobalMeshId& m2 : mesh_level_graph.get_neighbors(m1)) {
            if (m1 < m2) {
                ++edges[{m1, m2}];
            }
        }
    }
    return edges;
}

// Encodes the whole master problem into a fresh solver. Returns false if any mesh has an empty candidate
// list, which is UNSAT by construction and worth reporting directly rather than as a verdict.
//
// Three constraint families:
//   (1) exactly one seat per mesh instance;
//   (2) no ASIC serves two meshes;
//   (3) seams: a seat for m1 forces some seat for m2 that it reaches with >= `need` links, and the
//       reverse. Forward alone is logically sufficient given (1), but the reverse doubles propagation
//       strength and the clauses are cheap. A seat with no compatible partner degenerates to a unit clause
//       and is deleted at encode time -- the "mesh 19 has no candidates" discovery, found once instead of
//       895 times at depth 19.
bool encode_master_problem(
    tt::tt_fabric::detail::TopologySatSolver& sat,
    SeatVars& vars,
    MasterEncodingSize& size,
    const std::map<GlobalMeshId, MasterCandidateLists::DefinitionKey>& definition_of,
    const MasterCandidateLists& lists,
    const std::map<std::pair<GlobalMeshId, GlobalMeshId>, std::size_t>& mesh_edges,
    SeamLinkMatrices& seams,
    const AsicIndex& asic_index,
    bool relaxed_tier) {
    vars.by_dense_asic.assign(asic_index.size(), {});

    // (1) Variables, at-least-one, at-most-one.
    for (const auto& [mesh_id, definition] : definition_of) {
        const auto& candidates = lists.candidates(definition);
        if (candidates.empty()) {
            return false;
        }
        std::vector<int>& lits = vars.by_mesh[mesh_id];
        lits.reserve(candidates.size());
        for (const MasterCandidate* cand : candidates) {
            const int var = sat.declare_one_more_variable();
            ++size.vars;
            lits.push_back(var);
            if (!cand->preferred) {
                vars.non_preferred.push_back(var);
            }
            for (const AsicID& asic : cand->asics) {
                vars.by_dense_asic[asic_index.dense(asic)].push_back(var);
            }
        }
        for (const int lit : lits) {
            sat.add(lit);
        }
        sat.add(0);
        ++size.clauses;
        add_at_most_one(sat, lits, size);
    }

    // (2) Disjointness.
    for (const std::vector<int>& users : vars.by_dense_asic) {
        add_at_most_one(sat, users, size);
    }

    // (3) Seams, both directions.
    for (const auto& [edge, channels] : mesh_edges) {
        const auto& [m1, m2] = edge;
        const std::size_t need = std::min<std::size_t>(relaxed_tier ? 1 : channels, 255);
        const auto d1 = definition_of.at(m1);
        const auto d2 = definition_of.at(m2);
        const std::vector<int>& l1 = vars.by_mesh.at(m1);
        const std::vector<int>& l2 = vars.by_mesh.at(m2);
        std::size_t cols = 0;
        const std::vector<uint8_t>& m = seams.matrix(d1, d2, lists, cols);
        TT_ASSERT(cols == l2.size() && m.size() == l1.size() * l2.size());

        for (std::size_t i = 0; i < l1.size(); ++i) {
            sat.add(-l1[i]);
            const uint8_t* row = m.data() + i * cols;
            for (std::size_t j = 0; j < cols; ++j) {
                if (row[j] >= need) {
                    sat.add(l2[j]);
                }
            }
            sat.add(0);
            ++size.clauses;
        }
        for (std::size_t j = 0; j < l2.size(); ++j) {
            sat.add(-l2[j]);
            for (std::size_t i = 0; i < l1.size(); ++i) {
                if (m[i * cols + j] >= need) {
                    sat.add(l1[i]);
                }
            }
            sat.add(0);
            ++size.clauses;
        }
    }
    return true;
}

AssignedMeshes decode_master_model(
    const tt::tt_fabric::detail::TopologySatSolver& sat,
    const SeatVars& vars,
    const std::map<GlobalMeshId, MasterCandidateLists::DefinitionKey>& definition_of,
    const MasterCandidateLists& lists) {
    AssignedMeshes assignment;
    assignment.reserve(definition_of.size());
    for (const auto& [mesh_id, definition] : definition_of) {
        const auto& candidates = lists.candidates(definition);
        const std::vector<int>& lits = vars.by_mesh.at(mesh_id);
        for (std::size_t i = 0; i < lits.size(); ++i) {
            if (sat.val(lits[i]) <= 0) {
                continue;
            }
            PsdPlacement placement;
            // The pinning map belongs to the variant and cannot be recovered from the footprint.
            placement.mesh_node_to_asic_position = candidates[i]->variant->mesh_node_to_asic_position;
            placement.asics.insert(candidates[i]->asics.begin(), candidates[i]->asics.end());
            assignment.push_back(PlacedMesh{
                mesh_id,
                std::move(placement),
                candidates[i]->variant->name,    // PGD_DFS_DEBUG
                candidates[i]->variant->type});  // PGD_DFS_DEBUG
            break;                               // exactly-one guarantees no second true literal
        }
    }
    return assignment;
}

// Diagnostic for an UNSAT master problem: how many meshes CAN be seated together, and which cannot join.
//
// Re-encodes the problem with one relaxation literal per mesh appended to its at-least-one clause, so a
// mesh whose relaxation literal is true is allowed to go unplaced. Meshes are then admitted greedily under
// assumptions (assume the relaxation literal false = "this mesh must be placed"): each mesh is kept if the
// instance stays SAT with it, dropped otherwise. The result is a maximal placeable subset -- not the
// maximum, but a real partial placement -- plus the list of meshes that could not be added to it. Every
// solve here is incremental on one solver, so this costs one solve per mesh. Budgeted, purely
// informational, and only run when the caller asked for stats or the log would otherwise be silent.
constexpr int kDiagnosisConflictBudgetPerMesh = 20000;

// Returns the decoded partial placement of the maximal subset (empty if even one mesh could not be seated).
AssignedMeshes diagnose_unsat_master(
    const std::map<GlobalMeshId, MasterCandidateLists::DefinitionKey>& definition_of,
    const MasterCandidateLists& lists,
    const std::map<std::pair<GlobalMeshId, GlobalMeshId>, std::size_t>& mesh_edges,
    SeamLinkMatrices& seams,
    const AsicIndex& asic_index,
    bool relaxed_tier,
    const std::map<GlobalMeshId, std::string>& mesh_id_to_label) {
    using tt::tt_fabric::detail::TopologySatSolver;
    const auto start = std::chrono::steady_clock::now();

    // Same encoding as the real attempt, except the at-least-one clause of every mesh gains its relaxation
    // literal. Re-implemented inline rather than through encode_master_problem so that the production
    // encoding stays exactly what it is.
    TopologySatSolver sat;
    MasterEncodingSize size;
    SeatVars vars;
    vars.by_dense_asic.assign(asic_index.size(), {});
    std::map<GlobalMeshId, int> relax_of;
    for (const auto& [mesh_id, definition] : definition_of) {
        const auto& candidates = lists.candidates(definition);
        std::vector<int>& lits = vars.by_mesh[mesh_id];
        for (const MasterCandidate* cand : candidates) {
            const int var = sat.declare_one_more_variable();
            lits.push_back(var);
            for (const AsicID& asic : cand->asics) {
                vars.by_dense_asic[asic_index.dense(asic)].push_back(var);
            }
        }
        const int relax = sat.declare_one_more_variable();
        relax_of.emplace(mesh_id, relax);
        for (const int lit : lits) {
            sat.add(lit);
        }
        sat.add(relax);
        sat.add(0);
        add_at_most_one(sat, lits, size);
    }
    for (const std::vector<int>& users : vars.by_dense_asic) {
        add_at_most_one(sat, users, size);
    }
    for (const auto& [edge, channels] : mesh_edges) {
        const auto& [m1, m2] = edge;
        const std::size_t need = std::min<std::size_t>(relaxed_tier ? 1 : channels, 255);
        const std::vector<int>& l1 = vars.by_mesh.at(m1);
        const std::vector<int>& l2 = vars.by_mesh.at(m2);
        std::size_t cols = 0;
        const std::vector<uint8_t>& m = seams.matrix(definition_of.at(m1), definition_of.at(m2), lists, cols);
        // A seam only binds when BOTH meshes are placed: a placed m1 seat forces a compatible m2 seat
        // unless m2 is relaxed away, and vice versa.
        for (std::size_t i = 0; i < l1.size(); ++i) {
            sat.add(-l1[i]);
            sat.add(relax_of.at(m2));
            for (std::size_t j = 0; j < cols; ++j) {
                if (m[i * cols + j] >= need) {
                    sat.add(l2[j]);
                }
            }
            sat.add(0);
        }
        for (std::size_t j = 0; j < l2.size(); ++j) {
            sat.add(-l2[j]);
            sat.add(relax_of.at(m1));
            for (std::size_t i = 0; i < l1.size(); ++i) {
                if (m[i * cols + j] >= need) {
                    sat.add(l1[i]);
                }
            }
            sat.add(0);
        }
    }

    // Greedy maximal placeable subset. Mesh order is mesh id order, which for a pipeline descriptor walks
    // the chain, so the report reads as "how far along the pipeline the placement gets".
    std::vector<GlobalMeshId> placed;
    std::vector<GlobalMeshId> rejected;
    std::vector<GlobalMeshId> undecided;  // budget exhausted before a verdict
    AssignedMeshes closest;               // decoded from the last SAT model, so it seats exactly `placed`
    for (const auto& [mesh_id, relax] : relax_of) {
        for (const GlobalMeshId& kept : placed) {
            sat.assume(-relax_of.at(kept));
        }
        sat.assume(-relax);
        const int verdict = sat.solve_limited(kDiagnosisConflictBudgetPerMesh);
        if (verdict == TopologySatSolver::kSat) {
            placed.push_back(mesh_id);
            // The model seats every assumed mesh and possibly more; keep only the assumed ones so the
            // reported subset is exactly what was proven co-placeable.
            AssignedMeshes model = decode_master_model(sat, vars, definition_of, lists);
            closest.clear();
            for (PlacedMesh& seated : model) {
                if (std::find(placed.begin(), placed.end(), seated.mesh_id) != placed.end()) {
                    closest.push_back(std::move(seated));
                }
            }
        } else if (verdict == TopologySatSolver::kUnsat) {
            rejected.push_back(mesh_id);
        } else {
            undecided.push_back(mesh_id);
        }
    }

    auto describe = [&](const std::vector<GlobalMeshId>& meshes) {
        std::string text;
        for (const GlobalMeshId& mesh_id : meshes) {
            if (!text.empty()) {
                text += ", ";
            }
            text += fmt::format("{}#{}", mesh_label_for_id(mesh_id, mesh_id_to_label), *mesh_id);
        }
        return text.empty() ? std::string("(none)") : text;
    };
    log_warning(
        tt::LogFabric,
        "SAT joint placement diagnosis ({} seams, {} ms): a maximal co-placeable subset seats {} of {} mesh(es); "
        "{} cannot be added to it{}. Rejected: {}. Undecided: {}",
        relaxed_tier ? "relaxed" : "strict",
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count(),
        placed.size(),
        relax_of.size(),
        rejected.size(),
        undecided.empty() ? "" : fmt::format(" and {} ran out of conflict budget", undecided.size()),
        describe(rejected),
        describe(undecided));
    return closest;
}

// Batch sizes for column generation. Pinned PGD variants have on the order of one placement per host, so
// the initial batch usually exhausts them outright; the growth batch matters only for trait-free variants.
constexpr std::size_t kInitialBatchPerVariant = 64;
constexpr std::size_t kGrowthBatchPerVariant = 128;
// Conflict budget for the strict-seam tier under a RELAXED policy. That tier is a preference (the mapper
// accepts a narrower seam and warns), so it is not worth an unbounded UNSAT proof; the relaxed tier that
// follows is solved without a cap. 0 = unbounded.
constexpr int kStrictTierConflictBudget = 50000;
// Conflict budget for the preferred-variant pass of each attempt (see encode: SeatVars::non_preferred).
constexpr int kPreferredPassConflictBudget = 50000;

// TODO(multi-solution): the master solve returns the FIRST model only. Make it enumerate placements the way
// the inner solver does (TopologySatSolver::configure_for_blocking_clause_enumeration + a blocking clause
// over the chosen seat literals after each model), so that:
//   - a caller can ask for the next placement when the downstream inter-mesh mapping rejects this one,
//     instead of failing the whole solve;
//   - alternatives can be ranked (seam width, hosts spanned, preferred variants) rather than accepting
//     whichever model CaDiCaL happens to find first;
//   - "SAT" can be turned into "how many placements exist", which is what the plan's completeness
//     reporting needs to be useful on a solvable descriptor.
// Blocking on seat literals alone would re-enumerate the same footprint set under a different variant;
// block on footprints (one literal per (mesh, footprint), or the dedup already in MasterCandidateLists).
AssignedMeshes start_sat_placement(
    const std::map<GlobalMeshId, MasterCandidateLists::DefinitionKey>& definition_of,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    bool relaxed_inter_mesh_policy,
    PlacementSolveStats* stats,
    const std::map<GlobalMeshId, std::string>& mesh_id_to_label,  // PGD_DFS_DEBUG
    AssignedMeshes* closest_out) {
    using tt::tt_fabric::detail::TopologySatSolver;

    if (stats != nullptr) {
        stats->master_solve_attempted = true;
    }
    const AsicIndex asic_index(physical_graph);
    const ConnectionValidationMode validation_mode =
        relaxed_inter_mesh_policy ? ConnectionValidationMode::RELAXED : ConnectionValidationMode::STRICT;

    MasterCandidateLists lists;
    std::vector<MasterCandidateLists::DefinitionKey> definitions;  // distinct, in first-seen order
    for (const auto& [mesh_id, key] : definition_of) {
        if (std::find(definitions.begin(), definitions.end(), key) == definitions.end()) {
            definitions.push_back(key);
            lists.add_definition(key);
        }
    }
    // Every mesh in the graph must have a definition, or the encoding would silently drop it.
    for (const GlobalMeshId& mesh_id : mesh_level_graph.get_nodes()) {
        if (!definition_of.contains(mesh_id)) {
            log_warning(tt::LogFabric, "SAT joint placement: mesh {} has no grouping variants; falling back", *mesh_id);
            return {};
        }
    }
    const auto mesh_edges = collect_mesh_edges(mesh_level_graph);

    auto grow_all = [&](std::size_t batch) {
        const auto start = std::chrono::steady_clock::now();
        std::size_t grown = 0;
        for (const auto key : definitions) {
            MasterEnumerationContext ctx{
                physical_graph,
                physical_system_descriptor,
                asic_index,
                validation_mode,
                lists.seen_footprints(key),
                stats};
            grown += lists.grow(key, batch, ctx);
        }
        if (stats != nullptr) {
            stats->master_enumeration_elapsed +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        }
        return grown;
    };
    auto log_lists = [&](std::string_view when) {
        for (const auto key : definitions) {
            // Label by the first instance of this definition.
            std::string label;
            for (const auto& [mesh_id, definition] : definition_of) {
                if (definition == key) {
                    label = mesh_label_for_id(mesh_id, mesh_id_to_label);
                    break;
                }
            }
            log_info(
                tt::LogFabric,
                "SAT joint placement {}: definition {}: {} candidate(s): {}",
                when,
                label,
                lists.candidates(key).size(),
                lists.describe(key));
        }
    };

    grow_all(kInitialBatchPerVariant);
    log_lists("initial enumeration");

    SeamLinkMatrices seams;
    std::size_t growth_rounds = 0;
    std::size_t attempts = 0;
    // Tier 0: pinned (PGD) variants only. Tier 1: trait-free (MGD fallback) variants added. Within a tier,
    // under a RELAXED policy the strict seam threshold is tried first, then the threshold drops to 1 --
    // mirroring next_step_pool's per-seam fallback, but as a global preference rather than a local one.
    for (int tier = 0; tier < 2; ++tier) {
        if (tier == 1) {
            if (lists.enable_mgd_fallback() == 0) {
                break;  // nothing was deferred
            }
            grow_all(kInitialBatchPerVariant);
            log_lists("after enabling trait-free variants");
        }
        for (;;) {
            for (const bool relaxed_tier : {false, true}) {
                if (relaxed_tier && !relaxed_inter_mesh_policy) {
                    continue;
                }
                // Re-encoded per attempt rather than extended in place: CNF clauses cannot gain literals
                // after the fact, so a grown candidate list would need extension literals on every
                // at-least-one and support clause. Encoding is cheap and attempts are few; the EXPENSIVE
                // state (the enumeration sessions) persists across attempts.
                ++attempts;
                const auto encode_start = std::chrono::steady_clock::now();
                TopologySatSolver sat;
                SeatVars vars;
                MasterEncodingSize size;
                const bool encoded = encode_master_problem(
                    sat, vars, size, definition_of, lists, mesh_edges, seams, asic_index, relaxed_tier);
                const auto encode_end = std::chrono::steady_clock::now();
                if (stats != nullptr) {
                    stats->master_encode_elapsed +=
                        std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start);
                    stats->master_sat_vars = size.vars;
                    stats->master_sat_clauses = size.clauses;
                    stats->master_sat_attempts = attempts;
                }
                if (!encoded) {
                    log_info(
                        tt::LogFabric,
                        "SAT joint placement: attempt {} (tier {}, {} seams): a mesh has no candidates; skipping solve",
                        attempts,
                        tier,
                        relaxed_tier ? "relaxed" : "strict");
                    break;  // growing is the only thing that can help; skip the relaxed re-encode
                }
                // Preferred-variant pass: assume every lower-ranked seat false, so a model (if one exists)
                // seats every mesh on its definition's top-ranked grouping. Assumptions are retracted after
                // the solve, so a failure here costs one budgeted solve and the unconstrained one below still
                // sees every candidate.
                int verdict = 0;
                if (!vars.non_preferred.empty()) {
                    for (const int lit : vars.non_preferred) {
                        sat.assume(-lit);
                    }
                    verdict = sat.solve_limited(kPreferredPassConflictBudget);
                    log_info(
                        tt::LogFabric,
                        "SAT joint placement: attempt {} preferred-variant pass ({} lower-ranked seats assumed off) -> "
                        "{}",
                        attempts,
                        vars.non_preferred.size(),
                        verdict == TopologySatSolver::kSat     ? "SAT"
                        : verdict == TopologySatSolver::kUnsat ? "UNSAT"
                                                               : "unknown (conflict budget)");
                }
                if (verdict != TopologySatSolver::kSat) {
                    const bool budgeted = !relaxed_tier && relaxed_inter_mesh_policy && kStrictTierConflictBudget > 0;
                    verdict = budgeted ? sat.solve_limited(kStrictTierConflictBudget) : sat.solve();
                }
                const auto solve_end = std::chrono::steady_clock::now();
                if (stats != nullptr) {
                    stats->master_solve_elapsed +=
                        std::chrono::duration_cast<std::chrono::microseconds>(solve_end - encode_end);
                }
                log_info(
                    tt::LogFabric,
                    "SAT joint placement: attempt {} (tier {}, {} seams): {} vars, {} clauses, {} candidates; encode "
                    "{} ms, solve {} ms -> {}",
                    attempts,
                    tier,
                    relaxed_tier ? "relaxed" : "strict",
                    size.vars,
                    size.clauses,
                    lists.total_candidates(),
                    std::chrono::duration_cast<std::chrono::milliseconds>(encode_end - encode_start).count(),
                    std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - encode_end).count(),
                    verdict == TopologySatSolver::kSat     ? "SAT"
                    : verdict == TopologySatSolver::kUnsat ? "UNSAT"
                                                           : "unknown (conflict budget)");
                if (verdict == TopologySatSolver::kSat) {
                    if (stats != nullptr) {
                        stats->master_solve_success = true;
                        stats->master_growth_rounds = growth_rounds;
                        stats->master_candidates_enumerated = lists.total_candidates();
                        stats->candidate_lists_complete = lists.complete();
                    }
                    AssignedMeshes assignment = decode_master_model(sat, vars, definition_of, lists);
                    log_adjacency_guided_placement_assignment(
                        assignment, mesh_id_to_label, "complete (SAT joint placement)");  // PGD_DFS_DEBUG
                    return assignment;
                }
            }
            if (!lists.can_grow()) {
                break;  // every live session exhausted (or capped) at this tier
            }
            ++growth_rounds;
            const std::size_t grown = grow_all(kGrowthBatchPerVariant);
            log_info(tt::LogFabric, "SAT joint placement: growth round {} added {} candidate(s)", growth_rounds, grown);
            if (grown == 0) {
                break;
            }
        }
    }

    const bool complete = lists.complete();
    if (stats != nullptr) {
        stats->master_growth_rounds = growth_rounds;
        stats->master_candidates_enumerated = lists.total_candidates();
        stats->candidate_lists_complete = complete;
    }
    log_warning(
        tt::LogFabric,
        "SAT joint placement: no placement found after {} attempt(s) and {} growth round(s) over {} candidate(s); "
        "candidate lists {} -- the UNSAT verdict is {}",
        attempts,
        growth_rounds,
        lists.total_candidates(),
        complete ? "COMPLETE" : "TRUNCATED",
        complete ? "trustworthy" : "NOT trustworthy");

    // The closest thing to an answer: a maximal set of meshes that CAN be seated together, under the most
    // permissive seam tier the policy allows, with every enumerated candidate in play.
    AssignedMeshes closest = diagnose_unsat_master(
        definition_of, lists, mesh_edges, seams, asic_index, relaxed_inter_mesh_policy, mesh_id_to_label);
    if (stats != nullptr) {
        stats->master_closest_meshes_placed = closest.size();
    }
    if (closest_out != nullptr) {
        *closest_out = std::move(closest);
    }
    return {};
}

// Which placement search to run. TT_METAL_PLACEMENT_SOLVER selects: "sat" (two-layer SAT joint placement
// only), "dfs" (adjacency-guided DFS only), or "auto" (default: SAT first; DFS only if SAT fails without
// a trustworthy UNSAT).
enum class PlacementSolverChoice { Auto, Sat, Dfs };

PlacementSolverChoice placement_solver_choice_from_env() {
    const char* env = std::getenv("TT_METAL_PLACEMENT_SOLVER");
    if (env == nullptr || env[0] == '\0') {
        return PlacementSolverChoice::Auto;
    }
    std::string value(env);
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "sat") {
        return PlacementSolverChoice::Sat;
    }
    if (value == "dfs") {
        return PlacementSolverChoice::Dfs;
    }
    if (value != "auto") {
        log_warning(tt::LogFabric, "TT_METAL_PLACEMENT_SOLVER='{}' not recognised (sat|dfs|auto); using auto", env);
    }
    return PlacementSolverChoice::Auto;
}

}  // namespace

std::vector<PsdPlacement> PhysicalGroupingDescriptor::solve_adjacency_guided_placement(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const ValidGroupingsMap& valid_groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t node_budget,
    PlacementSolveStats* stats_out) const {
    return solve_adjacency_guided_placement(
        {&mesh_graph_descriptor}, valid_groupings, physical_system_descriptor, node_budget, stats_out);
}

std::vector<PsdPlacement> PhysicalGroupingDescriptor::solve_adjacency_guided_placement(
    const std::vector<const MeshGraphDescriptor*>& mesh_graph_descriptors,
    const ValidGroupingsMap& valid_groupings,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::size_t node_budget,
    PlacementSolveStats* stats_out) const {
    using tt::tt_metal::experimental::tt_fabric::build_logical_multi_mesh_adjacency_graph;
    using tt::tt_metal::experimental::tt_fabric::LogicalMultiMeshGraph;
    using tt::tt_metal::experimental::tt_fabric::merge_logical_multi_mesh_adjacency_graphs;

    const auto total_start = std::chrono::steady_clock::now();
    // Stats are always collected and logged; the caller's object is filled when one is supplied.
    PlacementSolveStats local_stats;
    PlacementSolveStats* stats = stats_out != nullptr ? stats_out : &local_stats;
    *stats = PlacementSolveStats{};
    auto finish_stats = [&](std::size_t meshes_placed) {
        stats->meshes_placed = meshes_placed;
        stats->success = stats->meshes_total != 0 && meshes_placed == stats->meshes_total;
        stats->total_elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - total_start);
        log_info(tt::LogFabric, "{}", stats->to_string());
    };

    if (mesh_graph_descriptors.empty()) {
        finish_stats(0);
        return {};
    }
    // NOTE: For now, only MESH groupings are supported, we will need to include support for hierarchical groupings in
    // the future.
    const auto mesh_it = valid_groupings.find("MESH");
    if (mesh_it == valid_groupings.end() || mesh_it->second.empty()) {
        finish_stats(0);
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
        finish_stats(0);
        return {};
    }
    stats->meshes_total = merged.mesh_level_graph_.get_nodes().size();

    // Flat ASIC adjacency graph of the physical system (search domain for placement).
    const AdjacencyGraph<AsicID> physical_graph(
        tt::tt_metal::experimental::tt_fabric::build_flat_adjacency_map_from_psd(physical_system_descriptor));

    // Instance-name keyed grouping variants from valid_groupings["MESH"].
    // Global mesh ID -> grouping variants, remapped from mesh_groupings via local_to_global_mesh_ids.
    const std::unordered_map<InstanceName, std::vector<GroupingInfo>>& mesh_groupings = mesh_it->second;
    std::map<MeshId, std::vector<GroupingInfo>> global_mesh_groupings;
    // Mesh instance -> its DEFINITION's variant vector inside `valid_groupings`. Every instance of one
    // mesh definition resolves to the same vector, which is what lets the SAT path enumerate each
    // definition once and share the candidate list across instances. global_mesh_groupings holds
    // per-instance COPIES, so its addresses would not do.
    std::map<MeshId, const std::vector<GroupingInfo>*> mesh_definition_of;
    std::map<GlobalMeshId, std::string> mesh_id_to_label;  // PGD_DFS_DEBUG
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
            mesh_definition_of.emplace(global_mesh_id, &groupings_it->second);
            mesh_id_to_label.emplace(global_mesh_id, grouping_key);  // PGD_DFS_DEBUG
        }
    }

    // One inter-mesh channel policy for the whole solve, which is all the rest of the stack supports: the
    // mapper applies a single validation mode to every seam. Descriptors merged together are required to
    // agree (validate_shared_inter_mesh_policy, called where they are assembled), so the first one to state
    // a policy speaks for the set and the rest either match it or state nothing. A set that states nothing
    // stays STRICT, matching MeshGraph's default.
    bool relaxed_inter_mesh_policy = false;
    for (const MeshGraphDescriptor* descriptor : mesh_graph_descriptors) {
        const auto policy = descriptor->inter_mesh_policy();
        if (policy.has_value()) {
            relaxed_inter_mesh_policy = (*policy == InterMeshChannelPolicy::Relaxed);
            break;
        }
    }

    // Adjacency-guided DFS: the placement chosen for each global mesh ID. The mesh-level graph is what
    // enumerates the meshes, which is sound because its builder seeds a node per mesh before adding any
    // connection edges, so a mesh with no intermesh links is still a node with an empty neighbour list.
    //
    // Two searches share these inputs and return the same AssignedMeshes. The two-layer SAT joint
    // placement (Plan 4) runs first: it enumerates each grouping variant once against the whole fabric
    // and picks one seat per mesh in a single solve. The DFS is kept as the fallback for the case where
    // the SAT path fails without a trustworthy verdict (a truncated candidate list), or when selected
    // explicitly via TT_METAL_PLACEMENT_SOLVER=dfs.
    const PlacementSolverChoice solver_choice = placement_solver_choice_from_env();
    AssignedMeshes mesh_placements;
    AssignedMeshes closest_partial;  // SAT path's maximal co-placeable subset when it fails
    if (solver_choice != PlacementSolverChoice::Dfs) {
        mesh_placements = start_sat_placement(
            mesh_definition_of,
            merged.mesh_level_graph_,
            physical_graph,
            physical_system_descriptor,
            relaxed_inter_mesh_policy,
            stats,
            mesh_id_to_label,  // PGD_DFS_DEBUG
            &closest_partial);
    }
    if (mesh_placements.empty() && !closest_partial.empty()) {
        // Reported, not returned: the caller keys placements by vector position, so a partial vector
        // would silently seat the wrong meshes. This is the "closest result" for a human to read.
        log_adjacency_guided_placement_assignment(
            closest_partial,
            mesh_id_to_label,
            fmt::format(
                "closest partial (SAT joint placement, {} of {} meshes)",
                closest_partial.size(),
                merged.mesh_level_graph_.get_nodes().size()));
        for (const PlacedMesh& placed : closest_partial) {
            std::map<std::string, std::size_t> chips_per_host;
            for (const AsicID& asic : placed.placement.asics) {
                ++chips_per_host[physical_system_descriptor.get_host_name_for_asic(asic)];
            }
            std::string hosts;
            for (const auto& [host, chips] : chips_per_host) {
                hosts += fmt::format("{}{}({} chips)", hosts.empty() ? "" : ", ", host, chips);
            }
            log_info(
                tt::LogFabric,
                "  closest partial: {} (global mesh {}) -> {} on {}",
                mesh_label_for_id(placed.mesh_id, mesh_id_to_label),
                *placed.mesh_id,
                placed.grouping_name,
                hosts);
        }
    }
    const bool sat_verdict_trustworthy = stats->candidate_lists_complete;
    const bool run_dfs =
        solver_choice == PlacementSolverChoice::Dfs ||
        (solver_choice == PlacementSolverChoice::Auto && mesh_placements.empty() && !sat_verdict_trustworthy);
    if (run_dfs) {
        if (solver_choice == PlacementSolverChoice::Auto) {
            log_info(
                tt::LogFabric, "SAT joint placement did not place every mesh; falling back to adjacency-guided DFS");
        }
        mesh_placements = start_adjacency_guided_dfs(
            global_mesh_groupings,
            merged.mesh_level_graph_,
            physical_graph,
            physical_system_descriptor,
            relaxed_inter_mesh_policy,
            node_budget,
            stats,
            mesh_id_to_label,                  // PGD_DFS_DEBUG
            /*deepest_partial_out=*/nullptr);  // PGD_DFS_DEBUG
    }

    // Drop the mesh keying the caller does not consume and return the placements as a flat list.
    std::vector<PsdPlacement> placements;
    placements.reserve(mesh_placements.size());
    for (PlacedMesh& placed : mesh_placements) {
        if (placed.placement.asics.empty()) {
            continue;
        }
        placements.push_back(std::move(placed.placement));
    }
    finish_stats(placements.size());
    return placements;
}

// TODO: delete both overloads; they are test-only now that build_physical_multi_mesh_adjacency_graph
// uses solve_adjacency_guided_placement.
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
