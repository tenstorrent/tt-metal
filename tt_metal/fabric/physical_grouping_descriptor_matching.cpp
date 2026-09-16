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
#include <span>
#include <vector>
#include <tt_stl/fmt.hpp>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "protobuf/physical_grouping_descriptor.pb.h"
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

// Split of candidate-enumeration wall time within one placement solve (file scope so both the enumerate
// helper and start_sat_placement, which live in separate anonymous-namespace blocks, share it). Reset at
// the top of start_sat_placement and logged after enumeration: "host match" is the one-time per-variant
// constraint build (trait pinning + PSD host alignment / host-split acceptance); "candidate finding" is
// the topology-solver session loop that actually produces footprints. Single-threaded per rank's
// placement, so plain counters suffice.
static std::chrono::microseconds g_enum_host_match_elapsed{};
static std::chrono::microseconds g_enum_candidate_find_elapsed{};

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
    std::vector<int32_t> host_dims;
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
    const auto declared = mesh_graph_descriptor.get_declared_topology(instance_ids[0]);
    if (declared.dims.empty()) {
        return std::nullopt;
    }

    MgdDeviceTopology topo;
    topo.dims = declared.dims;
    topo.host_dims = declared.host_dims;
    topo.ring_dims.reserve(declared.ring_dims.size());
    for (std::size_t i = 0; i < declared.ring_dims.size(); ++i) {
        const int32_t dim_size = i < declared.dims.size() ? declared.dims[i] : 0;
        // RING on a dim of 2 or less is a no-op (same edges as LINE). Drop it so matching does not
        // look for a TORUS variant in that direction.
        topo.ring_dims.push_back(
            declared.ring_dims[i] && dim_size > 0 &&
            tt::tt_fabric::is_genuine_torus_dim(static_cast<uint32_t>(dim_size)));
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

// Compose PGD grouping-node -> host partition index from an MGD<->PGD topology match and the matched MGD
// host_topology. Empty only when the MGD declared no host topology at all, so an empty result means "nothing
// was declared" rather than "one host". Called at PGD<->MGD commit time in get_valid_groupings_for_mgd.
//
// TODO: the host split's direction is unconstrained. This row-major-splits the MGD's own chip indices and stamps
// the result onto whichever PGD nodes the isomorphism picked, so the axis the split lands on is whatever
// orientation that match happened to choose -- the MGD host axis is never tied to the PGD's host structure. One
// declared host group can therefore end up half on one physical host and half on another (measured on the gemma
// 4x4 with host_topology [2,1]: an 8-chip group whose best single 8-ASIC PSD host could seat only 4 of its 8
// members, with every member trait-bound). That fails configure_pgd_psd_host_alignment_constraints, and because
// footprint discovery and the rank-bound path can settle on different matches out of the same candidate set, the
// two can disagree about the same descriptor.
//
// Fix, and where it now stands: configure_mgd_pgd_host_alignment_constraints below ties the declared split to the
// host division the PGD variant sits on, so for a descriptor that declares its hosts the isomorphism can only
// return an orientation this stamping is sound for. What is still unconstrained is descriptors that say nothing
// for it to match against -- no HOSTS groupings, or hosts that leave their chips unspecified -- and there the
// split lands in whatever orientation the match happened to choose, exactly as described above.
std::map<LogicalChipId, uint32_t> compose_mesh_node_to_host_group_from_mgd_match(
    const std::optional<MgdDeviceTopology>& mgd_topo,
    const std::map<LogicalChipId, GroupingChipId>& mgd_node_to_grouping_node) {
    std::map<LogicalChipId, uint32_t> node_to_host_group;
    if (!mgd_topo.has_value() || mgd_topo->host_dims.empty() || mgd_topo->dims.empty()) {
        return node_to_host_group;
    }

    // host_topology [1,1] is not "no opinion": it declares one rank owning the whole mesh, which is as binding
    // as any other split and is left in so the caller enforces it. host_partition_index_for_row_major_chip puts
    // every chip in partition 0 for it, giving a single group that must land inside a single host.
    for (const auto& [mgd_node, grouping_node] : mgd_node_to_grouping_node) {
        node_to_host_group.emplace(
            grouping_node,
            host_partition_index_for_row_major_chip(
                static_cast<uint32_t>(mgd_node), mgd_topo->dims, mgd_topo->host_dims));
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
    const std::optional<MgdDeviceTopology>& mgd_topo,
    MappingConstraints<LogicalChipId, GroupingChipId>& constraints) {
    const bool nothing_declared = grouping_info.mesh_node_to_pgd_host_group.empty() || !mgd_topo.has_value() ||
                                  mgd_topo->dims.empty() || mgd_topo->host_dims.empty();
    if (nothing_declared) {
        return true;
    }

    std::map<uint32_t, std::set<LogicalChipId>> mgd_nodes_by_rank;
    for (LogicalChipId mgd_node : mgd_grouping_info.adjacency_graph.get_nodes()) {
        mgd_nodes_by_rank[host_partition_index_for_row_major_chip(
                              static_cast<uint32_t>(mgd_node), mgd_topo->dims, mgd_topo->host_dims)]
            .insert(mgd_node);
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
// A rejection here is usually not a too-small host: see the TODO on
// compose_mesh_node_to_host_group_from_mgd_match, which stamps the declared split onto the match in an arbitrary
// orientation, so a group can straddle two hosts that are each large enough to hold it whole.
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
        log_debug(
            tt::LogFabric,
            "DIAG host alignment '{}': {} target(s), PSD exposes {} host partition(s) -> no host constraint applied",
            grouping_info.name,
            all_targets.size(),
            global_groups.size());
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
        std::map<uint32_t, std::set<LogicalChipId>> targets_by_group;
        for (LogicalChipId node_id : all_targets) {
            const auto group_it = grouping_info.mesh_node_to_host_group.find(node_id);
            if (group_it == grouping_info.mesh_node_to_host_group.end()) {
                log_debug(
                    tt::LogFabric,
                    "PGD host split '{}' is missing host group for node {}",
                    grouping_info.name,
                    node_id);
                return false;
            }
            targets_by_group[group_it->second].insert(node_id);
        }
        std::vector<std::set<LogicalChipId>> target_groups;
        target_groups.reserve(targets_by_group.size());
        std::vector<std::size_t> group_sizes;
        group_sizes.reserve(targets_by_group.size());
        for (auto& [_, group_targets] : targets_by_group) {
            group_sizes.push_back(group_targets.size());
            target_groups.push_back(std::move(group_targets));
        }

        // Hard, and only in this direction: no PSD host boundary may cut through a declared mesh host rank. Each
        // target group must therefore be carvable inside one PSD host. Groups are free to share a host, so a
        // host_topology finer than the physical hosts stays legal; what is rejected is a single declared rank
        // whose chips would have to come from two different hosts.
        if (!constraints.set_same_rank_groups_constraint(target_groups, global_groups)) {
            std::map<std::size_t, std::size_t> global_size_histogram;
            for (const auto& asics : global_groups) {
                ++global_size_histogram[asics.size()];
            }
            std::vector<std::string> global_size_text;
            global_size_text.reserve(global_size_histogram.size());
            for (const auto& [asic_count, host_count] : global_size_histogram) {
                global_size_text.push_back(fmt::format("{}x{}chips", host_count, asic_count));
            }
            log_debug(
                tt::LogFabric,
                "PGD host split '{}' REJECTED: no PSD host can hold one of the {} declared host group(s) sized "
                "[{}]; PSD exposes {} host partition(s) [{}]",
                grouping_info.name,
                target_groups.size(),
                fmt::join(group_sizes, ","),
                global_groups.size(),
                fmt::join(global_size_text, ","));
            // DIAG: distinguish "no host is big enough" from "traits already pinned the group across hosts".
            const auto& forbidden_pairs = constraints.get_forbidden_pairs();
            const auto& valid_mappings = constraints.get_valid_mappings();
            for (std::size_t group_index = 0; group_index < target_groups.size(); ++group_index) {
                const std::set<LogicalChipId>& group = target_groups[group_index];
                std::size_t unconstrained_members = 0;
                for (const LogicalChipId& target : group) {
                    unconstrained_members += valid_mappings.contains(target) ? 0 : 1;
                }
                std::size_t best_covered = 0;
                std::size_t best_host_asics = 0;
                for (const auto& partition : global_groups) {
                    std::size_t covered = 0;
                    for (const LogicalChipId& target : group) {
                        for (const AsicID& asic_id : partition) {
                            if (forbidden_pairs.contains({target, asic_id})) {
                                continue;
                            }
                            if (!valid_mappings.contains(target) || constraints.is_valid_mapping(target, asic_id)) {
                                ++covered;
                                break;
                            }
                        }
                    }
                    if (covered > best_covered) {
                        best_covered = covered;
                        best_host_asics = partition.size();
                    }
                }
                log_debug(
                    tt::LogFabric,
                    "DIAG host split '{}' group {}: {} chip(s) ({} unconstrained); best single PSD host covers "
                    "{}/{} member(s) and has {} asic(s)",
                    grouping_info.name,
                    group_index,
                    group.size(),
                    unconstrained_members,
                    best_covered,
                    group.size(),
                    best_host_asics);
            }
            return false;
        }
        // Cap the hosts used at the number of declared groups: the split may collapse onto fewer hosts, never
        // spread onto more than it declared.
        constraints.set_max_same_rank_groups_used(target_groups.size());
        log_debug(
            tt::LogFabric,
            "PGD host split '{}' ACCEPTED: {} declared host group(s) sized [{}] each required onto a single one of "
            "{} PSD host partition(s)",
            grouping_info.name,
            target_groups.size(),
            fmt::join(group_sizes, ","),
            global_groups.size());
        return true;
    }

    if (!prefer_minimal_host_cover(all_targets)) {
        log_debug(
            tt::LogFabric,
            "PGD host alignment '{}': target count {} exceeds largest single PSD partition; preferring minimal host "
            "cover across {} partition(s)",
            grouping_info.name,
            all_targets.size(),
            global_groups.size());
    }
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
    TopologyMappingEnumerationSession<LogicalChipId, AsicID> session;
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
std::vector<MappingResult<LogicalChipId, AsicID>> enumerate_distinct_placements_for_grouping(
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
        // Encode the grouping's trait / host-alignment constraints once; the session snapshots them on its
        // first next() and must not see them change afterward. This is the "host match" phase.
        const auto host_match_start = std::chrono::steady_clock::now();
        const bool encoded =
            add_pgd_to_psd_constraints(grouping_info, physical_graph, physical_system_descriptor, constraints, nullptr);
        g_enum_host_match_elapsed +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - host_match_start);
        if (!encoded) {
            log_debug(
                tt::LogFabric,
                "DIAG enumerate '{}': CONSTRAINT-ENCODE-FAILED (trait or host alignment)",
                grouping_info.name);
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
        MappingResult<LogicalChipId, AsicID> mapping = state.session.next(
            grouping_info.adjacency_graph,
            physical_graph,
            constraints,
            state.excluded,
            validation_mode,
            /*quiet_mode=*/true,
            TopologyMappingSolverEngine::Auto,
            unique_shapes);
        ++state.solves;
        if (!mapping.success) {
            if (i == 0) {
                log_debug(
                    tt::LogFabric,
                    "DIAG enumerate '{}': NO-EMBEDDING ({} target node(s) on {} asic(s), validation_mode={}, "
                    "unique_shapes={}): {}",
                    grouping_info.name,
                    grouping_info.adjacency_graph.get_nodes().size(),
                    physical_graph.get_nodes().size(),
                    static_cast<int>(validation_mode),
                    unique_shapes,
                    mapping.error_message);
            }
            state.exhausted = true;
            break;
        }
        state.excluded.push_back(mapping.target_to_global);
        mappings.push_back(std::move(mapping));
    }
    g_enum_candidate_find_elapsed +=
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - solve_start);

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
        "  adjacency DFS nodes expanded: {}\n"
        "  next_step_pool calls: {}  ({} us)\n"
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

}  // namespace

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings,
    bool require_placement) const {
    return get_valid_groupings_for_mgd(mesh_graph_descriptor, &physical_system_descriptor, pinnings, require_placement);
}

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor,
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor,
    const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings,
    bool require_placement) const {
    ValidGroupingsMap result;

    // Times the whole matcher call regardless of entry path (placement builder, multi-MGD, or the
    // rank-bound pinning enrichment fast path), logging on exit. require_placement separates the
    // placement call (true) from the pinning-only enrichment call (false).
    struct GvfmTimer {
        std::chrono::steady_clock::time_point start;
        bool require_placement;
        ~GvfmTimer() {
            log_info(
                tt::LogFabric,
                "TIMING get_valid_groupings_for_mgd (require_placement={}): {} ms",
                require_placement,
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                    .count());
        }
    } gvfm_timer{std::chrono::steady_clock::now(), require_placement};

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
                    }

                    // 5. The declared host split against the host division this variant sits on, so the match
                    // comes back in an orientation whose ranks each fall inside one of the descriptor's hosts.
                    if (!configure_mgd_pgd_host_alignment_constraints(
                            mgd_grouping_info, grouping_info, device_topo, constraints)) {
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
                // (and find_all_in_psd) places on the same graph.
                auto make_committed_grouping = [&](const MeshTopologyMatch& match) -> GroupingInfo {
                    GroupingInfo committed = mesh_flat_groupings.at(match.name)[match.idx];
                    // The topology solve used the MGD mesh adjacency as target and this PGD variant as global, so
                    // target_to_global is MGD-node -> PGD grouping-node. Compose logical chip_id -> PGD slot pinning
                    // now so downstream consumes it directly without re-deriving the intermediate node pairing.
                    committed.mesh_node_to_asic_position =
                        compose_mesh_node_to_asic_position_from_pgd_match(committed, match.mapping.target_to_global);
                    committed.mesh_node_to_host_group =
                        compose_mesh_node_to_host_group_from_mgd_match(device_topo, match.mapping.target_to_global);
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

        // The fallback is the MGD's own topology, so its nodes are the MGD's own chips and the declared split
        // applies to it directly, with no match to compose through -- the identity pairing below is what
        // "directly" means. Stamping it here puts the fallback under the same host containment rule as a
        // committed PGD grouping: held against the machine's hosts, since a fallback has no descriptor hosts
        // to be held against. Without it the split reaches the placement solve empty and decays to the soft
        // same-host preference, which lets a rank be seated across two hosts of the machine.
        std::map<LogicalChipId, GroupingChipId> fallback_nodes_are_mgd_chips;
        for (LogicalChipId node_id : mgd_fallback.adjacency_graph.get_nodes()) {
            fallback_nodes_are_mgd_chips.emplace(node_id, node_id);
        }
        mgd_fallback.mesh_node_to_host_group =
            compose_mesh_node_to_host_group_from_mgd_match(device_topo, fallback_nodes_are_mgd_chips);

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
        // Callers that only want preferred pinnings on an already rank-bound graph pass require_placement=false:
        // for them an unplaceable mesh is a missing hint, not a broken system, and they degrade to no pinning.
        if (committed.empty()) {
            TT_FATAL(
                !require_placement,
                "Physical groupings: Mesh graph descriptor '{}': no PGD grouping and no MGD grouping "
                "could be placed on the PSD ({} topology match(es))",
                mgd_grouping_info.name,
                last_topology_match_count);
            log_warning(
                tt::LogFabric,
                "Physical groupings: Mesh graph descriptor '{}': no PGD grouping and no MGD grouping could be "
                "placed on the PSD ({} topology match(es)); continuing without pinning hints for this mesh",
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

// One legal seating of one grouping variant. Footprint-only: the per-node mapping from the inner solve is
// deliberately dropped, matching what next_step_pool keeps today -- downstream reconstructs positions
// from the variant's pinning map, not from the embedding.
//
// Seam geometry between two seatings is fabric_links_to (link count). Disjointness in the master SAT
// uses by_dense_asic on dense_asics(), not footprint bitsets. boundary_bitset_ / internal_footprint_bitset_
// gate and count links via the private Bitset helpers.
class Candidate {
public:
    const std::vector<uint32_t>& dense_asics() const { return dense_asics_; }

    const std::vector<AsicID>& asics() const { return asics_; }

    const GroupingInfo* variant() const { return variant_; }

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

class AdjacencyMatrix;
class AdjacencyMatrixCache;

// All grouping variants and enumerated candidates for one global mesh instance.
class CandidatePool {
public:
    CandidatePool(
        const std::vector<GroupingInfo>& groupings,
        const AdjacencyGraph<AsicID>& physical_graph,
        const tt::tt_metal::PhysicalSystemDescriptor& psd,
        ConnectionValidationMode validation_mode) :
        physical_graph_(physical_graph), psd_(psd), validation_mode_(validation_mode), asic_index_(physical_graph) {
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

    // Seat order matches master SAT variables (SeatVars.by_mesh) and AdjacencyMatrix row/column indices.
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

private:
    // Resumable enumeration state for one grouping variant. Map nodes are never moved -- the session keeps
    // pointers into its own graph/constraint snapshots. Seatings live in candidates_ on the pool.
    struct GroupingVariant {
        const GroupingInfo* grouping = nullptr;
        GroupingVariantEnumeration enumeration;
        std::size_t candidates_found = 0;
    };

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
            const auto mappings = enumerate_distinct_placements_for_grouping(
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
};

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

    static AdjacencyMatrix build(const std::vector<Candidate>& from_seats, const std::vector<Candidate>& to_seats) {
        AdjacencyMatrix table;
        table.rows_ = from_seats.size();
        table.cols_ = to_seats.size();
        table.data_.assign(table.rows_ * table.cols_, 0);
        for (std::size_t from_seat = 0; from_seat < table.rows_; ++from_seat) {
            for (std::size_t to_seat = 0; to_seat < table.cols_; ++to_seat) {
                const std::size_t links = from_seats[from_seat].fabric_links_to(to_seats[to_seat]);
                table.data_[from_seat * table.cols_ + to_seat] =
                    static_cast<uint8_t>(std::min(links, std::size_t{255}));
            }
        }
        return table;
    }

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<uint8_t> data_;
};

// Lazy cache of AdjacencyMatrix tables keyed by (from_mesh, to_mesh), invalidated when the candidate pool grows.
class AdjacencyMatrixCache {
public:
    // Rows = from_mesh candidates; columns = to_mesh candidates (same order as SeatVars.by_mesh).
    const AdjacencyMatrix& adjacency_matrix(
        GlobalMeshId from_mesh,
        GlobalMeshId to_mesh,
        const std::map<GlobalMeshId, CandidatePool>& pools,
        std::size_t pools_generation) {
        if (pools_generation != generation_) {
            cache_.clear();
            generation_ = pools_generation;
        }
        const auto key = std::make_pair(from_mesh, to_mesh);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        const AdjacencyMatrix built =
            AdjacencyMatrix::build(pools.at(from_mesh).candidates(), pools.at(to_mesh).candidates());
        return cache_.emplace(key, std::move(built)).first->second;
    }

private:
    std::map<std::pair<GlobalMeshId, GlobalMeshId>, AdjacencyMatrix> cache_;
    std::size_t generation_ = static_cast<std::size_t>(-1);
};

// One variable per (mesh INSTANCE, candidate). Instances cannot share variables even when they share a
// candidate list, because they take different seats.
struct SeatVars {
    std::map<GlobalMeshId, std::vector<int>> by_mesh;  // parallel to pool.candidates(mesh_id)
    std::vector<std::vector<int>> by_dense_asic;       // asic -> vars whose footprint covers it
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
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const std::map<GlobalMeshId, CandidatePool>& pools,
    std::size_t pools_generation,
    const std::map<std::pair<GlobalMeshId, GlobalMeshId>, std::size_t>& mesh_edges,
    AdjacencyMatrixCache& adjacency_cache,
    bool relaxed_tier) {
    const std::size_t asic_count = pools.empty() ? 0 : pools.begin()->second.asic_count();
    vars.by_dense_asic.assign(asic_count, {});

    // (1) Variables, at-least-one, at-most-one.
    for (const auto& [mesh_id, _] : global_mesh_groupings) {
        const auto& candidates = pools.at(mesh_id).candidates();
        if (candidates.empty()) {
            return false;
        }
        std::vector<int>& lits = vars.by_mesh[mesh_id];
        lits.reserve(candidates.size());
        for (const Candidate& cand : candidates) {
            const int var = sat.declare_one_more_variable();
            ++size.vars;
            lits.push_back(var);
            for (const uint32_t dense : cand.dense_asics()) {
                vars.by_dense_asic[dense].push_back(var);
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
        const std::vector<int>& l1 = vars.by_mesh.at(m1);
        const std::vector<int>& l2 = vars.by_mesh.at(m2);
        const AdjacencyMatrix& adjacency = adjacency_cache.adjacency_matrix(m1, m2, pools, pools_generation);
        TT_ASSERT(adjacency.rows() == l1.size() && adjacency.cols() == l2.size());

        for (std::size_t from_seat = 0; from_seat < l1.size(); ++from_seat) {
            sat.add(-l1[from_seat]);
            for (std::size_t to_seat = 0; to_seat < adjacency.cols(); ++to_seat) {
                if (adjacency.satisfies_channel(from_seat, to_seat, need)) {
                    sat.add(l2[to_seat]);
                }
            }
            sat.add(0);
            ++size.clauses;
        }
        for (std::size_t to_seat = 0; to_seat < l2.size(); ++to_seat) {
            sat.add(-l2[to_seat]);
            for (std::size_t from_seat = 0; from_seat < adjacency.rows(); ++from_seat) {
                if (adjacency.satisfies_channel(from_seat, to_seat, need)) {
                    sat.add(l1[from_seat]);
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
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const std::map<GlobalMeshId, CandidatePool>& pools) {
    AssignedMeshes assignment;
    assignment.reserve(global_mesh_groupings.size());
    for (const auto& [mesh_id, _] : global_mesh_groupings) {
        const auto& candidates = pools.at(mesh_id).candidates();
        const std::vector<int>& lits = vars.by_mesh.at(mesh_id);
        for (std::size_t i = 0; i < lits.size(); ++i) {
            if (sat.val(lits[i]) <= 0) {
                continue;
            }
            PsdPlacement placement;
            // The pinning map belongs to the variant and cannot be recovered from the footprint.
            placement.mesh_node_to_asic_position = candidates[i].variant()->mesh_node_to_asic_position;
            placement.asics.insert(candidates[i].asics().begin(), candidates[i].asics().end());
            assignment.push_back(PlacedMesh{
                mesh_id,
                std::move(placement),
                candidates[i].variant()->name,    // PGD_DFS_DEBUG
                candidates[i].variant()->type});  // PGD_DFS_DEBUG
            break;                               // exactly-one guarantees no second true literal
        }
    }
    return assignment;
}

// Per-variant grow batch for column generation (initial enumeration and each UNSAT growth round).
// CandidatePool::grow passes this directly to enumerate_distinct_placements_for_grouping on every
// non-exhausted variant; growth rounds add more if the first solve is UNSAT.
constexpr std::size_t kGrowBudgetPerVariant = 32;
// Conflict budget for the strict-seam tier under a RELAXED policy. That tier is a preference (the mapper
// accepts a narrower seam and warns), so it is not worth an unbounded UNSAT proof; the relaxed tier that
// follows is solved without a cap. 0 = unbounded.
constexpr int kStrictTierConflictBudget = 50000;

// TODO(multi-solution): the master solve returns the FIRST model only. Make it enumerate placements the way
// the inner solver does (TopologySatSolver::configure_for_blocking_clause_enumeration + a blocking clause
// over the chosen seat literals after each model), so that:
//   - a caller can ask for the next placement when the downstream inter-mesh mapping rejects this one,
//     instead of failing the whole solve;
//   - alternatives can be ranked (seam width, hosts spanned, grouping variant quality) rather than accepting
//     whichever model CaDiCaL happens to find first;
//   - "SAT" can be turned into "how many placements exist", which is what the plan's completeness
//     reporting needs to be useful on a solvable descriptor.
// Blocking on seat literals alone would re-enumerate the same footprint set under a different variant;
// block on footprints (one literal per (mesh, footprint), or the dedup already in CandidatePool).
AssignedMeshes start_sat_placement(
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings,
    const AdjacencyGraph<GlobalMeshId>& mesh_level_graph,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    bool relaxed_inter_mesh_policy,
    const std::map<GlobalMeshId, ConnectionValidationMode>& sat_intra_mesh_mode_by_mesh,
    PlacementSolveStats* stats,
    const std::map<GlobalMeshId, std::string>& mesh_id_to_label) {
    using tt::tt_fabric::detail::TopologySatSolver;

    if (stats != nullptr) {
        stats->master_solve_attempted = true;
    }
    // Reset the enumerate-phase split (host match vs candidate finding) for this solve.
    g_enum_host_match_elapsed = std::chrono::microseconds{};
    g_enum_candidate_find_elapsed = std::chrono::microseconds{};

    std::map<GlobalMeshId, CandidatePool> pools;
    for (const auto& [mesh_id, groupings] : global_mesh_groupings) {
        const auto mode_it = sat_intra_mesh_mode_by_mesh.find(mesh_id);
        TT_FATAL(
            mode_it != sat_intra_mesh_mode_by_mesh.end(),
            "Internal error: SAT joint placement: mesh {} has no intra-mesh validation mode",
            *mesh_id);
        pools.emplace(mesh_id, CandidatePool(groupings, physical_graph, physical_system_descriptor, mode_it->second));
    }
    std::size_t pools_generation = 0;
    // Every mesh in the graph must have grouping variants, or the encoding would silently drop it.
    for (const GlobalMeshId& mesh_id : mesh_level_graph.get_nodes()) {
        if (!global_mesh_groupings.contains(mesh_id)) {
            log_warning(tt::LogFabric, "SAT joint placement: mesh {} has no grouping variants; falling back", *mesh_id);
            return {};
        }
    }
    const auto mesh_edges = collect_mesh_edges(mesh_level_graph);

    auto grow_all = [&](std::size_t batch_per_variant) {
        const auto start = std::chrono::steady_clock::now();
        std::size_t grown = 0;
        for (const auto& [mesh_id, _] : global_mesh_groupings) {
            grown += pools.at(mesh_id).grow(batch_per_variant);
        }
        if (grown != 0) {
            ++pools_generation;
        }
        if (stats != nullptr) {
            stats->master_enumeration_elapsed +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        }
        return grown;
    };

    grow_all(kGrowBudgetPerVariant);
    for (const auto& [mesh_id, pool] : pools) {
        log_info(
            tt::LogFabric,
            "SAT joint placement initial enumeration: {} (global mesh {}): {} candidate(s)",
            mesh_label_for_id(mesh_id, mesh_id_to_label),
            *mesh_id,
            pool.candidates().size());
    }
    // Attribute the enumerate phase: the one-time per-variant constraint build (trait + PSD host
    // alignment / host-split acceptance) versus the topology-solver session loop that finds footprints.
    {
        const double host_match_ms = static_cast<double>(g_enum_host_match_elapsed.count()) / 1000.0;
        const double candidate_find_ms = static_cast<double>(g_enum_candidate_find_elapsed.count()) / 1000.0;
        const double split_total_ms = host_match_ms + candidate_find_ms;
        log_info(
            tt::LogFabric,
            "SAT joint placement enumerate phase split: host match {:.1f} ms ({:.1f}%), candidate finding {:.1f} ms "
            "({:.1f}%), measured total {:.1f} ms",
            host_match_ms,
            split_total_ms > 0 ? 100.0 * host_match_ms / split_total_ms : 0.0,
            candidate_find_ms,
            split_total_ms > 0 ? 100.0 * candidate_find_ms / split_total_ms : 0.0,
            split_total_ms);
    }
    AdjacencyMatrixCache adjacency_cache;
    std::size_t growth_rounds = 0;
    std::size_t attempts = 0;
    auto count_candidates = [&]() {
        std::size_t total = 0;
        for (const auto& [_, pool] : pools) {
            total += pool.candidates().size();
        }
        return total;
    };
    auto pools_enumeration_complete = [&]() {
        for (const auto& [_, pool] : pools) {
            if (!pool.variants_exhausted()) {
                return false;
            }
        }
        return true;
    };
    // Under a RELAXED policy the strict seam threshold is tried first, then the threshold drops to 1 --
    // mirroring next_step_pool's per-seam fallback, but as a global preference rather than a local one.
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
                sat,
                vars,
                size,
                global_mesh_groupings,
                pools,
                pools_generation,
                mesh_edges,
                adjacency_cache,
                relaxed_tier);
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
                    "SAT joint placement: attempt {} ({} seams): a mesh has no candidates; skipping solve",
                    attempts,
                    relaxed_tier ? "relaxed" : "strict");
                break;  // growing is the only thing that can help; skip the relaxed re-encode
            }
            const bool budgeted = !relaxed_tier && relaxed_inter_mesh_policy && kStrictTierConflictBudget > 0;
            const int verdict = budgeted ? sat.solve_limited(kStrictTierConflictBudget) : sat.solve();
            const auto solve_end = std::chrono::steady_clock::now();
            if (stats != nullptr) {
                stats->master_solve_elapsed +=
                    std::chrono::duration_cast<std::chrono::microseconds>(solve_end - encode_end);
            }
            log_info(
                tt::LogFabric,
                "SAT joint placement: attempt {} ({} seams): {} vars, {} clauses, {} candidates; encode "
                "{} ms, solve {} ms -> {}",
                attempts,
                relaxed_tier ? "relaxed" : "strict",
                size.vars,
                size.clauses,
                count_candidates(),
                std::chrono::duration_cast<std::chrono::milliseconds>(encode_end - encode_start).count(),
                std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - encode_end).count(),
                verdict == TopologySatSolver::kSat     ? "SAT"
                : verdict == TopologySatSolver::kUnsat ? "UNSAT"
                                                       : "unknown (conflict budget)");
            if (verdict == TopologySatSolver::kSat) {
                if (stats != nullptr) {
                    stats->master_solve_success = true;
                    stats->master_growth_rounds = growth_rounds;
                    stats->master_candidates_enumerated = count_candidates();
                    stats->candidate_lists_complete = pools_enumeration_complete();
                }
                return decode_master_model(sat, vars, global_mesh_groupings, pools);
            }
        }
        ++growth_rounds;
        std::size_t grown = grow_all(kGrowBudgetPerVariant);
        log_info(tt::LogFabric, "SAT joint placement: growth round {} added {} candidate(s)", growth_rounds, grown);
        if (grown == 0) {
            break;
        }
    }

    const bool complete = pools_enumeration_complete();
    if (stats != nullptr) {
        stats->master_growth_rounds = growth_rounds;
        stats->master_candidates_enumerated = count_candidates();
        stats->candidate_lists_complete = complete;
    }
    log_warning(
        tt::LogFabric,
        "SAT joint placement: no placement found after {} attempt(s) and {} growth round(s) over {} candidate(s); "
        "candidate lists {} -- the UNSAT verdict is {}",
        attempts,
        growth_rounds,
        count_candidates(),
        complete ? "COMPLETE" : "TRUNCATED",
        complete ? "trustworthy" : "NOT trustworthy");
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
            mesh_id_to_label.emplace(global_mesh_id, grouping_key);  // PGD_DFS_DEBUG
        }
    }

    // Channel validation policy for the whole solve. Descriptors merged together must agree on
    // inter-mesh policy (validate_shared_inter_mesh_policy); every MGD defaults to STRICT when it states none.
    const bool relaxed_inter_mesh_policy = mesh_graph_descriptors.front()->is_inter_mesh_policy_relaxed();
    std::map<GlobalMeshId, ConnectionValidationMode> sat_intra_mesh_mode_by_mesh;
    for (std::size_t mgd_index = 0; mgd_index < local_to_global_mesh_ids.size() && mgd_index < mgd_count; ++mgd_index) {
        const MeshGraphDescriptor& mgd = *mesh_graph_descriptors[mgd_index];
        for (const auto& [local_mesh_id, global_mesh_id] : local_to_global_mesh_ids[mgd_index]) {
            sat_intra_mesh_mode_by_mesh.emplace(
                global_mesh_id,
                mgd.is_intra_mesh_policy_relaxed(local_mesh_id) ? ConnectionValidationMode::RELAXED
                                                                : ConnectionValidationMode::STRICT);
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
    if (solver_choice != PlacementSolverChoice::Dfs) {
        mesh_placements = start_sat_placement(
            global_mesh_groupings,
            merged.mesh_level_graph_,
            physical_graph,
            physical_system_descriptor,
            relaxed_inter_mesh_policy,
            sat_intra_mesh_mode_by_mesh,
            stats,
            mesh_id_to_label);
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
