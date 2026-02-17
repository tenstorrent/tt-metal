// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <queue>
#include <memory>
#include <cctype>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>

#include "protobuf/physical_grouping_descriptor.pb.h"
#include "protobuf/mesh_graph_descriptor.pb.h"
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-logger/tt-logger.hpp>
#include <cctype>
#include <map>

#include <google/protobuf/text_format.h>

using namespace tt::tt_fabric;

namespace {

std::string read_file_to_string(const std::filesystem::path& file_path) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

// Helper function to get grouping name string from proto
// Get the name field from a grouping (mandatory field)
std::string get_grouping_name(const proto::Grouping& grouping) { return grouping.name(); }

// Get the type string from a grouping (preset_type or custom_type)
std::string get_grouping_type_string(const proto::Grouping& grouping) {
    if (grouping.has_preset_type()) {
        switch (grouping.preset_type()) {
            case proto::TRAY_1: return "TRAY_1";
            case proto::TRAY_2: return "TRAY_2";
            case proto::TRAY_3: return "TRAY_3";
            case proto::TRAY_4: return "TRAY_4";
            case proto::HOSTS: return "HOSTS";
            case proto::MESH: return "MESH";
            default: return "";
        }
    } else if (grouping.has_custom_type()) {
        return grouping.custom_type();
    }
    return "";
}

// Legacy function for backward compatibility - returns type string
std::string get_grouping_name_string(const proto::Grouping& grouping) { return get_grouping_type_string(grouping); }

bool grouping_exists(const proto::PhysicalGroupings& proto, const std::string& grouping_name) {
    for (const auto& grouping : proto.groupings()) {
        std::string name = get_grouping_name_string(grouping);
        if (name == grouping_name) {
            return true;
        }
    }
    return false;
}

// Helper function to build adjacency graph from all-to-all connection
AdjacencyGraph<uint32_t> build_all_to_all_graph(const std::vector<uint32_t>& instance_ids) {
    std::map<uint32_t, std::vector<uint32_t>> adj_map;

    // All-to-all: every node connects to every other node
    // Always uses 1 connection per edge (bidirectional)
    // Process each edge only once to avoid duplicates
    for (size_t i = 0; i < instance_ids.size(); ++i) {
        for (size_t j = i + 1; j < instance_ids.size(); ++j) {
            // Add bidirectional edge (each edge processed once)
            adj_map[instance_ids[i]].push_back(instance_ids[j]);
            adj_map[instance_ids[j]].push_back(instance_ids[i]);
        }
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

// Helper function to build adjacency graph from row-major mesh connection
// Always uses LINE connectivity (no wrap-around) and 1 connection per edge
AdjacencyGraph<uint32_t> build_row_major_mesh_graph(
    const std::vector<uint32_t>& instance_ids,
    const std::vector<int32_t>& dims,
    const std::string& grouping_name = "") {
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

    auto coords_to_idx = [&](const std::vector<int32_t>& coords) -> int32_t {
        int32_t idx = 0;
        int32_t multiplier = 1;
        for (int32_t i = static_cast<int32_t>(dims.size()) - 1; i >= 0; --i) {
            idx += coords[i] * multiplier;
            multiplier *= dims[i];
        }
        return idx;
    };

    // Build adjacency: for each dimension, connect neighbors
    // Use a set to track processed edges to avoid double-counting
    std::set<std::pair<uint32_t, uint32_t>> processed_edges;

    for (uint32_t idx = 0; idx < instance_ids.size(); ++idx) {
        std::vector<int32_t> coords = get_coords(idx);

        // For each dimension
        // Always uses LINE connectivity (no wrap-around)
        for (size_t dim_idx = 0; dim_idx < dims.size(); ++dim_idx) {
            int32_t dim_size = dims[dim_idx];
            int32_t coord_val = coords[dim_idx];

            // Neighbor in positive direction
            // LINE: only connect if not at the end
            int32_t neighbor_idx = -1;
            if (coord_val < dim_size - 1) {
                std::vector<int32_t> coords_plus = coords;
                coords_plus[dim_idx] = coord_val + 1;
                neighbor_idx = coords_to_idx(coords_plus);
            }

            if (neighbor_idx >= 0 && neighbor_idx < static_cast<int32_t>(instance_ids.size())) {
                // Process edge only once (undirected)
                auto edge_pair = std::minmax(instance_ids[idx], instance_ids[neighbor_idx]);
                if (processed_edges.insert(edge_pair).second) {
                    // Always uses 1 connection per edge
                    adj_map[instance_ids[idx]].push_back(instance_ids[neighbor_idx]);
                    adj_map[instance_ids[neighbor_idx]].push_back(instance_ids[idx]);
                }
            }
        }
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

// Helper function to build adjacency graph from custom connections
// Ensures no duplicate connections and all connections are bidirectional
AdjacencyGraph<uint32_t> build_custom_connections_graph(
    const std::vector<uint32_t>& instance_ids, const proto::CustomConnections& custom_connections) {
    std::map<uint32_t, std::vector<uint32_t>> adj_map;

    // Build a map from instance index to instance ID
    std::map<uint32_t, uint32_t> index_to_id;
    for (size_t i = 0; i < instance_ids.size(); ++i) {
        index_to_id[static_cast<uint32_t>(i)] = instance_ids[i];
    }

    // Use a set to track processed edges to avoid duplicates
    // Edge pairs are normalized (min, max) to treat (A,B) and (B,A) as the same edge
    std::set<std::pair<uint32_t, uint32_t>> processed_edges;

    // Add edges from custom connections
    for (const auto& conn : custom_connections.connections()) {
        uint32_t src_idx = conn.src_instance();
        uint32_t dst_idx = conn.dst_instance();

        if (index_to_id.find(src_idx) == index_to_id.end() || index_to_id.find(dst_idx) == index_to_id.end()) {
            TT_THROW(
                "Custom connection references invalid instance index: src={}, dst={} (valid range: 0-{})",
                src_idx,
                dst_idx,
                instance_ids.size() - 1);
        }

        uint32_t src_id = index_to_id[src_idx];
        uint32_t dst_id = index_to_id[dst_idx];

        // Skip self-loops
        if (src_id == dst_id) {
            continue;
        }

        // Normalize edge pair to avoid duplicates (treat (A,B) and (B,A) as the same)
        auto edge_pair = std::minmax(src_id, dst_id);

        // Only add edge if not already processed (prevents duplicates)
        if (processed_edges.insert(edge_pair).second) {
            // Always uses 1 connection per edge (bidirectional)
            adj_map[src_id].push_back(dst_id);
            adj_map[dst_id].push_back(src_id);
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

    std::string dims_str = "[";
    for (size_t i = 0; i < device_dims.size(); ++i) {
        if (i > 0) {
            dims_str += ",";
        }
        dims_str += std::to_string(device_dims[i]);
    }
    dims_str += "]";
    log_critical(tt::LogFabric, "Building MGD mesh adjacency graph: device_dims={}, num_asics={}", dims_str, num_asics);

    // Create abstract ASIC node IDs (0, 1, 2, ..., num_asics-1)
    std::vector<uint32_t> asic_ids;
    asic_ids.reserve(num_asics);
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_asics); ++i) {
        asic_ids.push_back(i);
    }

    log_critical(
        tt::LogFabric,
        "Calling build_row_major_mesh_graph with {} nodes, dims={} (always uses LINE connectivity, 1 connection per "
        "edge)",
        asic_ids.size(),
        dims_str);

    // Build row-major mesh graph representing ASIC-level topology
    // Always uses LINE connectivity (no wrap-around) and 1 connection per edge
    auto result = build_row_major_mesh_graph(asic_ids, device_dims, "");

    log_critical(tt::LogFabric, "Built MGD mesh adjacency graph: {} nodes", result.get_nodes().size());

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
            if (graph_instance.sub_instances.count(src) && graph_instance.sub_instances.count(dst)) {
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

// Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts)
// Calculates required ASIC counts bottom-up and builds adjacency graphs
// Returns map: (type, name) -> GroupingInfo
std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> build_mgd_to_grouping_info_map(
    const MeshGraphDescriptor& mesh_graph_descriptor) {
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

    // Step 1b: Calculate required ASICs for graph instances bottom-up (children before parents)
    // Process graphs in topological order by iterating until all are processed
    std::unordered_set<GlobalNodeId> processed_graphs;
    bool progress_made = true;

    while (progress_made) {
        progress_made = false;

        for (GlobalNodeId graph_id : mesh_graph_descriptor.all_graphs()) {
            if (processed_graphs.find(graph_id) != processed_graphs.end()) {
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

                // Check if this sub-instance's required_asics is already calculated
                auto sub_type_it = required_asics_map.find(sub_instance.type);
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
        if (type_it == required_asics_map.end() || type_it->second.find(graph_instance.name) == type_it->second.end()) {
            TT_THROW(
                "Failed to calculate required ASIC count for graph instance '{}' (type '{}'). "
                "This may indicate a circular dependency in the MGD.",
                graph_instance.name,
                graph_instance.type);
        }
    }

    // ===== Step 2: Build GroupingInfo objects with adjacency graphs and ASIC counts =====

    // Process mesh instances
    for (GlobalNodeId mesh_id : mesh_graph_descriptor.all_meshes()) {
        const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_id);
        const std::string& mesh_type = mesh_instance.type;
        const std::string& mesh_name = mesh_instance.name;

        // Skip if already processed (same name/type)
        if (mgd_grouping_infos.find(mesh_type) != mgd_grouping_infos.end() &&
            mgd_grouping_infos[mesh_type].find(mesh_name) != mgd_grouping_infos[mesh_type].end()) {
            continue;
        }

        // Build adjacency graph for this mesh instance
        AdjacencyGraph<uint32_t> adjacency_graph = build_mgd_mesh_instance_adjacency(mesh_graph_descriptor, mesh_id);

        // Get required ASIC count (calculated above)
        uint32_t asic_count = required_asics_map.at(mesh_type).at(mesh_name);

        // Create GroupingInfo
        GroupingInfo grouping_info;
        grouping_info.name = mesh_name;
        grouping_info.type = mesh_type;
        grouping_info.asic_count = asic_count;
        grouping_info.adjacency_graph = std::move(adjacency_graph);
        // items left empty - not needed for matching

        mgd_grouping_infos[mesh_type][mesh_name] = std::move(grouping_info);
    }

    // Process graph instances
    for (GlobalNodeId graph_id : mesh_graph_descriptor.all_graphs()) {
        const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_id);
        const std::string& graph_type = graph_instance.type;
        const std::string& graph_name = graph_instance.name;

        // Skip if already processed (same name/type)
        if (mgd_grouping_infos.find(graph_type) != mgd_grouping_infos.end() &&
            mgd_grouping_infos[graph_type].find(graph_name) != mgd_grouping_infos[graph_type].end()) {
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

}  // namespace

namespace tt::tt_fabric {

PhysicalGroupingDescriptor::PhysicalGroupingDescriptor(const std::string& text_proto) {
    proto::PhysicalGroupings temp_proto;
    google::protobuf::TextFormat::Parser parser;

    // Strict validation - don't allow unknown fields
    parser.AllowUnknownField(false);
    parser.AllowUnknownExtension(false);

    TT_FATAL(parser.ParseFromString(text_proto, &temp_proto), "Failed to parse PhysicalGroupingDescriptor textproto");

    // Uniquify duplicate names before validation
    uniquify_duplicate_names(temp_proto);

    // Validate the proto
    const auto errors = static_validate(temp_proto);
    TT_FATAL(
        errors.empty(), "Failed to validate PhysicalGroupingDescriptor textproto: \n{}", get_validation_report(errors));

    proto_ = std::make_shared<proto::PhysicalGroupings>(temp_proto);

    populate();
}

PhysicalGroupingDescriptor::PhysicalGroupingDescriptor(const std::filesystem::path& text_proto_file_path) :
    PhysicalGroupingDescriptor(read_file_to_string(text_proto_file_path)) {}

PhysicalGroupingDescriptor::~PhysicalGroupingDescriptor() = default;

std::string PhysicalGroupingDescriptor::read_file_to_string(const std::filesystem::path& file_path) {
    return ::read_file_to_string(file_path);
}

bool PhysicalGroupingDescriptor::has_grouping(const std::string& grouping_name) const {
    return grouping_exists(*proto_, grouping_name);
}

size_t PhysicalGroupingDescriptor::get_grouping_count() const { return proto_->groupings_size(); }

GroupingInfo PhysicalGroupingDescriptor::convert_grouping_to_info(const proto::Grouping& grouping) const {
    GroupingInfo info;

    // Get grouping name (mandatory field) and type
    info.name = get_grouping_name(grouping);
    info.type = get_grouping_type_string(grouping);

    // Collect instance IDs and build items list
    std::vector<uint32_t> instance_ids;
    std::vector<uint32_t> asic_locations;  // For tray groupings, use ASIC locations as node IDs
    bool has_asic_locations = false;

    for (const auto& instance : grouping.instances()) {
        uint32_t instance_id = instance.id();
        instance_ids.push_back(instance_id);

        // Build GroupingItemInfo for backward compatibility with existing code
        GroupingItemInfo item_info;
        if (instance.has_asic_location()) {
            item_info.type = GroupingItemInfo::ItemType::ASIC_LOCATION;
            item_info.asic_location = static_cast<uint32_t>(instance.asic_location());
            info.items.push_back(item_info);
            asic_locations.push_back(item_info.asic_location);
            has_asic_locations = true;
        } else if (instance.has_grouping_ref()) {
            item_info.type = GroupingItemInfo::ItemType::GROUPING_REF;
            const auto& ref = instance.grouping_ref();
            if (ref.has_preset_type()) {
                // Convert preset_type enum to string
                switch (ref.preset_type()) {
                    case proto::TRAY_1: item_info.grouping_name = "TRAY_1"; break;
                    case proto::TRAY_2: item_info.grouping_name = "TRAY_2"; break;
                    case proto::TRAY_3: item_info.grouping_name = "TRAY_3"; break;
                    case proto::TRAY_4: item_info.grouping_name = "TRAY_4"; break;
                    case proto::HOSTS: item_info.grouping_name = "HOSTS"; break;
                    case proto::MESH: item_info.grouping_name = "MESH"; break;
                    default: break;
                }
            } else if (ref.has_custom_type()) {
                item_info.grouping_name = ref.custom_type();
            }
            info.items.push_back(item_info);
        }
    }

    // Build adjacency graph from connection specification
    // For tray groupings (with ASIC_LOCATION items), use ASIC locations as node IDs (1-8)
    // For other groupings (with GROUPING_REF items), use instance IDs (0, 1, 2, ...)
    // This ensures tray groupings match the PSD discovery which uses ASIC locations as node IDs
    std::vector<uint32_t> node_ids = has_asic_locations ? asic_locations : instance_ids;

    if (grouping.has_all_to_all()) {
        info.adjacency_graph = build_all_to_all_graph(node_ids);
    } else if (grouping.has_row_major_mesh()) {
        const auto& row_major_mesh = grouping.row_major_mesh();
        std::vector<int32_t> dims(row_major_mesh.dims().begin(), row_major_mesh.dims().end());
        info.adjacency_graph = build_row_major_mesh_graph(node_ids, dims, info.name);

        // Populate corner orientation for row-major mesh based on mesh shape and row-major order
        // Corner positions are determined by the mesh dimensions, not ASIC locations
        // For 1D meshes, endpoints can have multiple orientations (e.g., 1x4: first item has NW+SW, last has NE+SE)
        // For 1x1 mesh, the single item has all 4 orientations
        if (dims.size() >= 1 && info.items.size() > 0) {
            size_t total_items = info.items.size();

            if (dims.size() == 1) {
                // 1D mesh: single dimension
                int32_t length = dims[0];
                if (total_items == static_cast<size_t>(length)) {
                    // First item: NW + SW (top-left + bottom-left)
                    if (total_items > 0) {
                        info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                        info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                    }
                    // Last item: NE + SE (top-right + bottom-right)
                    if (total_items > 1) {
                        size_t last_idx = total_items - 1;
                        info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                        info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                    }
                }
            } else if (dims.size() >= 2) {
                int32_t rows = dims[0];
                int32_t cols = dims[1];

                if (rows == 1 && cols == 1) {
                    // 1x1 mesh: single item has all 4 orientations
                    if (total_items == 1) {
                        info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                        info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                        info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                        info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                    }
                } else if (rows == 1) {
                    // 1D row mesh (1xN): first item has NW+SW, last item has NE+SE
                    if (total_items == static_cast<size_t>(cols)) {
                        if (total_items > 0) {
                            info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                            info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                        }
                        if (total_items > 1) {
                            size_t last_idx = total_items - 1;
                            info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                            info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                        }
                    }
                } else if (cols == 1) {
                    // 1D column mesh (Nx1): first item has NW+NE, last item has SW+SE
                    if (total_items == static_cast<size_t>(rows)) {
                        if (total_items > 0) {
                            info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                            info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                        }
                        if (total_items > 1) {
                            size_t last_idx = total_items - 1;
                            info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                            info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                        }
                    }
                } else {
                    // 2D mesh: standard 4 corners
                    if (total_items == static_cast<size_t>(rows * cols)) {
                        size_t nw_idx = 0;
                        size_t ne_idx = static_cast<size_t>(cols - 1);
                        size_t sw_idx = static_cast<size_t>((rows - 1) * cols);
                        size_t se_idx = static_cast<size_t>((rows - 1) * cols + (cols - 1));

                        // Set corner orientations based on row-major mesh position
                        if (nw_idx < total_items) {
                            info.items[nw_idx].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                        }
                        if (ne_idx < total_items && ne_idx != nw_idx) {
                            info.items[ne_idx].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                        }
                        if (sw_idx < total_items && sw_idx != nw_idx && sw_idx != ne_idx) {
                            info.items[sw_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                        }
                        if (se_idx < total_items && se_idx != nw_idx && se_idx != ne_idx && se_idx != sw_idx) {
                            info.items[se_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                        }
                    }
                }
            }
        }
    } else if (grouping.has_custom()) {
        const auto& custom = grouping.custom();
        info.adjacency_graph = build_custom_connections_graph(node_ids, custom);
    } else {
        // No connection specified - empty adjacency graph (instances are not connected)
        info.adjacency_graph = AdjacencyGraph<uint32_t>();
    }

    return info;
}

uint32_t PhysicalGroupingDescriptor::get_grouping_asic_count(const std::string& grouping_name) const {
    auto it = resolved_groupings_cache_.find(grouping_name);
    if (it != resolved_groupings_cache_.end() && !it->second.empty()) {
        // Return the ASIC count from the first grouping with this name
        // (all groupings with same name should have same structure/count)
        return it->second[0].asic_count;
    }
    return 0;
}

std::vector<GroupingInfo> PhysicalGroupingDescriptor::get_groupings_by_name(const std::string& grouping_name) const {
    auto it = resolved_groupings_cache_.find(grouping_name);
    if (it != resolved_groupings_cache_.end()) {
        return it->second;
    }
    // Fallback: return empty vector if not found in cache
    return {};
}

std::vector<GroupingInfo> PhysicalGroupingDescriptor::get_all_groupings() const {
    std::vector<GroupingInfo> result;
    for (const auto& [name, groupings] : resolved_groupings_cache_) {
        for (const auto& grouping : groupings) {
            result.push_back(grouping);
        }
    }
    return result;
}

std::vector<std::string> PhysicalGroupingDescriptor::get_all_grouping_names() const {
    std::vector<std::string> types;
    for (const auto& grouping : proto_->groupings()) {
        types.push_back(get_grouping_type_string(grouping));
    }
    return types;
}

std::string PhysicalGroupingDescriptor::get_validation_report(const std::vector<std::string>& errors) {
    if (errors.empty()) {
        return "No validation errors found.\n";
    }

    std::ostringstream report;
    report << "=== PhysicalGroupingDescriptor Validation Report ===\n\n";
    report << "Errors:\n";
    for (const auto& error : errors) {
        report << "  - " << error << "\n";
    }
    report << "\n";

    return report.str();
}

// Uniquify duplicate names in the proto by adding unique IDs
void PhysicalGroupingDescriptor::uniquify_duplicate_names(proto::PhysicalGroupings& proto) {
    std::unordered_map<std::string, uint32_t> name_counters;
    std::unordered_set<std::string> used_names;

    for (int i = 0; i < proto.groupings_size(); ++i) {
        auto* grouping = proto.mutable_groupings(i);
        std::string current_name = get_grouping_name(*grouping);

        if (current_name.empty()) {
            continue;  // Skip if name is empty (will be caught by other validation)
        }

        // If this name is already used, uniquify it
        if (used_names.find(current_name) != used_names.end()) {
            // Generate unique name with ID suffix
            uint32_t& counter = name_counters[current_name];
            std::string unique_name;
            do {
                counter++;
                unique_name = fmt::format("{}_{}", current_name, counter);
            } while (used_names.find(unique_name) != used_names.end());

            // Update the proto with the unique name
            grouping->set_name(unique_name);
            used_names.insert(unique_name);
        } else {
            // First occurrence, keep as is
            used_names.insert(current_name);
            name_counters[current_name] = 0;  // Initialize counter
        }
    }
}

// Validate that all grouping names are unique (should be true after uniquification)
void PhysicalGroupingDescriptor::validate_unique_names(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    std::unordered_set<std::string> names;

    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = get_grouping_name(grouping);

        if (name.empty()) {
            continue;  // Empty names are caught by other validation
        }

        if (names.find(name) != names.end()) {
            errors.push_back(fmt::format(
                "Grouping name '{}' appears multiple times (internal error: uniquification failed).", name));
        }
        names.insert(name);
    }
}

std::vector<std::string> PhysicalGroupingDescriptor::static_validate(const proto::PhysicalGroupings& proto) {
    std::vector<std::string> all_errors;

    // Run validation groups with early exit checkpoints
    {
        validate_required_groupings(proto, all_errors);
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    {
        validate_grouping_references(proto, all_errors);
        validate_counts(proto, all_errors);
        validate_grouping_structure(proto, all_errors);
        validate_unique_names(proto, all_errors);
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    return all_errors;
}

uint32_t PhysicalGroupingDescriptor::calculate_base_grouping_asic_count(const GroupingInfo& grouping) {
    uint32_t count = 0;
    for (const auto& item : grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
            count += 1;
        }
    }
    return count;
}

uint32_t PhysicalGroupingDescriptor::calculate_dependent_grouping_asic_count(
    const GroupingInfo& grouping, const std::unordered_map<std::string, std::vector<GroupingInfo>>& groupings_by_name) {
    uint32_t total_asics = 0;

    // Set of preset names that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_names = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    for (const auto& item : grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
            total_asics += 1;
            continue;
        }

        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
            // Skip preset names that don't exist - they can be auto-populated later
            if (preset_names.find(item.grouping_name) != preset_names.end()) {
                auto ref_it = groupings_by_name.find(item.grouping_name);
                if (ref_it == groupings_by_name.end() || ref_it->second.empty()) {
                    // Preset name doesn't exist yet - skip it (will be auto-populated)
                    continue;
                }
            }

            auto ref_it = groupings_by_name.find(item.grouping_name);
            if (ref_it == groupings_by_name.end() || ref_it->second.empty()) {
                TT_THROW("Grouping '{}' references non-existent grouping '{}'", grouping.name, item.grouping_name);
            }

            uint32_t ref_count = ref_it->second[0].asic_count;
            if (ref_count == 0) {
                TT_THROW(
                    "Grouping '{}' references '{}' which has zero ASIC count (not yet resolved)",
                    grouping.name,
                    item.grouping_name);
            }
            total_asics += ref_count;
        }
    }

    return total_asics;
}

void PhysicalGroupingDescriptor::populate() {
    // Step 1: Convert all proto groupings and build dependency graph
    std::unordered_map<std::string, std::vector<GroupingInfo>> groupings_by_name;
    std::unordered_map<std::string, std::set<std::string>> dependencies;

    // Set of preset names that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_names = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    for (const auto& grouping : proto_->groupings()) {
        GroupingInfo info = convert_grouping_to_info(grouping);

        // Track dependencies (skip preset names that don't exist)
        std::set<std::string> deps;
        for (const auto& item : info.items) {
            if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                // Only track dependencies on groupings that exist or are not preset names
                // Preset names can be auto-populated, so don't treat them as blocking dependencies
                if (groupings_by_name.find(item.grouping_name) != groupings_by_name.end() ||
                    preset_names.find(item.grouping_name) == preset_names.end()) {
                    deps.insert(item.grouping_name);
                }
            }
        }
        dependencies[info.type] = deps;
        groupings_by_name[info.type].push_back(std::move(info));
    }

    // Step 2: Process base groupings (no dependencies)
    for (auto& [name, groupings] : groupings_by_name) {
        if (!dependencies[name].empty()) {
            continue;  // Skip dependent groupings for now
        }

        for (auto& grouping : groupings) {
            // Check if this grouping only references preset names
            bool only_preset_refs = true;
            bool has_any_refs = false;
            for (const auto& item : grouping.items) {
                if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                    has_any_refs = true;
                    if (preset_names.find(item.grouping_name) == preset_names.end()) {
                        only_preset_refs = false;
                        break;
                    }
                } else if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                    only_preset_refs = false;
                    break;
                }
            }

            if (only_preset_refs && has_any_refs) {
                // Grouping only references preset names - set count to 0 (will be populated later)
                grouping.asic_count = 0;
            } else {
                grouping.asic_count = calculate_base_grouping_asic_count(grouping);
                if (grouping.asic_count == 0) {
                    TT_THROW("Grouping '{}' has no ASIC_LOCATION items and cannot be resolved", name);
                }
            }
        }
    }

    // Step 3: Build topological sort data structures
    std::unordered_map<std::string, std::set<std::string>> incoming_edges;
    std::unordered_map<std::string, int> in_degree;

    for (const auto& [name, deps] : dependencies) {
        in_degree[name] = static_cast<int>(deps.size());
        for (const auto& dep : deps) {
            incoming_edges[dep].insert(name);
        }
    }

    // Step 4: Process dependent groupings in topological order
    std::queue<std::string> to_process;
    for (const auto& [name, deps] : dependencies) {
        if (deps.empty()) {
            to_process.push(name);
        }
    }

    std::vector<std::string> processed;
    while (!to_process.empty()) {
        std::string current = to_process.front();
        to_process.pop();
        processed.push_back(current);

        // Process all groupings that depend on current
        for (const auto& dependent : incoming_edges[current]) {
            in_degree[dependent]--;
            if (in_degree[dependent] > 0) {
                continue;  // Not ready yet
            }

            // All dependencies resolved, calculate ASIC count
            for (auto& grouping : groupings_by_name[dependent]) {
                grouping.asic_count = calculate_dependent_grouping_asic_count(grouping, groupings_by_name);
                if (grouping.asic_count == 0) {
                    TT_THROW(
                        "Grouping '{}' does not resolve to any ASIC locations (circular or missing references)",
                        dependent);
                }
            }

            to_process.push(dependent);
        }
    }

    // Step 5: Validate all groupings were processed
    if (processed.size() != dependencies.size()) {
        std::vector<std::string> unresolved;
        for (const auto& [name, _] : dependencies) {
            if (std::find(processed.begin(), processed.end(), name) == processed.end()) {
                unresolved.push_back(name);
            }
        }
        // Build error message manually
        std::string unresolved_str;
        for (size_t i = 0; i < unresolved.size(); ++i) {
            if (i > 0) {
                unresolved_str += ", ";
            }
            unresolved_str += unresolved[i];
        }
        TT_THROW("Circular dependencies detected. Unresolved: {}", unresolved_str);
    }

    // Step 6: Store resolved groupings
    resolved_groupings_cache_ = std::move(groupings_by_name);

    // Step 7: Final validation - all groupings should have ASIC counts > 0
    // Exception: groupings that only reference preset names (which can be auto-populated) may have 0 count
    for (const auto& [name, groupings] : resolved_groupings_cache_) {
        for (const auto& grouping : groupings) {
            if (grouping.asic_count == 0) {
                // Check if this grouping only references preset names
                bool only_preset_refs = true;
                bool has_any_refs = false;
                for (const auto& item : grouping.items) {
                    if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                        has_any_refs = true;
                        if (preset_names.find(item.grouping_name) == preset_names.end()) {
                            only_preset_refs = false;
                            break;
                        }
                    } else if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                        only_preset_refs = false;
                        break;
                    }
                }

                // Allow zero count only if grouping only references preset names
                if (!only_preset_refs || !has_any_refs) {
                    TT_THROW("Grouping '{}' has zero ASIC count after resolution", name);
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_required_groupings(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    // Note: MESH and preset types (TRAY_1, TRAY_2, TRAY_3, TRAY_4, HOSTS) are not required
    // They can be auto-populated from PhysicalSystemDescriptor if missing

    // Validate grouping names are non-empty and types are set
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = get_grouping_name(grouping);
        std::string type = get_grouping_type_string(grouping);

        if (name.empty()) {
            errors.push_back(
                fmt::format("Grouping at index {} has an empty name; grouping names must be non-empty", i));
        }

        if (type.empty()) {
            errors.push_back(fmt::format(
                "Grouping '{}' at index {} has no type set; exactly one of preset_type or custom_type must be set",
                name,
                i));
        }
    }
}

void PhysicalGroupingDescriptor::validate_grouping_references(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    // Build set of all grouping types
    std::unordered_set<std::string> grouping_types;
    for (const auto& grouping : proto.groupings()) {
        grouping_types.insert(get_grouping_type_string(grouping));
    }

    // Set of preset types that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_types = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    // Validate all grouping references
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = get_grouping_name(grouping);

        for (int j = 0; j < grouping.instances_size(); ++j) {
            const auto& instance = grouping.instances(j);

            if (instance.has_grouping_ref()) {
                const auto& ref = instance.grouping_ref();
                std::string ref_type;
                bool is_preset_type = false;

                if (ref.has_preset_type()) {
                    // Convert preset_type enum to string
                    switch (ref.preset_type()) {
                        case proto::TRAY_1: ref_type = "TRAY_1"; break;
                        case proto::TRAY_2: ref_type = "TRAY_2"; break;
                        case proto::TRAY_3: ref_type = "TRAY_3"; break;
                        case proto::TRAY_4: ref_type = "TRAY_4"; break;
                        case proto::HOSTS: ref_type = "HOSTS"; break;
                        case proto::MESH: ref_type = "MESH"; break;
                        default: ref_type = ""; break;
                    }
                    is_preset_type = true;
                } else if (ref.has_custom_type()) {
                    ref_type = ref.custom_type();
                    is_preset_type = false;
                }

                if (ref_type.empty()) {
                    errors.push_back(fmt::format("Grouping '{}' has a grouping_ref with empty grouping_type", name));
                    continue;
                }

                // Skip validation for preset types - they can be auto-populated
                if (is_preset_type || preset_types.contains(ref_type)) {
                    continue;
                }

                // For custom types, validate they exist
                if (!grouping_types.contains(ref_type)) {
                    errors.push_back(
                        fmt::format("Grouping '{}' references non-existent grouping type '{}'", name, ref_type));
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_counts(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = get_grouping_name(grouping);
        std::string type = get_grouping_type_string(grouping);
        uint32_t instance_count = static_cast<uint32_t>(grouping.instances_size());

        // Validate instance count
        if (type == "meshes" || type == "MESH") {
            // Meshes (both custom type and preset type) can have count >= 1
            if (instance_count < 1) {
                errors.push_back(fmt::format(
                    "Grouping '{}' (type '{}') has {} instances; meshes must have at least 1 instance",
                    name,
                    type,
                    instance_count));
            }
        } else {
            // For non-meshes groupings: must have at least 2 instances (unless single instance is explicitly allowed)
            if (instance_count < 2) {
                errors.push_back(fmt::format(
                    "Grouping '{}' (type '{}') has {} instances; groupings other than meshes must have at least 2 "
                    "instances",
                    name,
                    type,
                    instance_count));
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_grouping_structure(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = get_grouping_name_string(grouping);

        // Check that grouping has instances
        if (grouping.instances_size() == 0) {
            errors.push_back(fmt::format(
                "Grouping '{}' (type '{}') must have at least one instance", name, get_grouping_type_string(grouping)));
            continue;
        }

        // Validate each instance
        for (int j = 0; j < grouping.instances_size(); ++j) {
            const auto& instance = grouping.instances(j);

            // Check that exactly one of asic_location or grouping_ref is set (enforced by oneof, but validate anyway)
            bool has_asic_location = instance.has_asic_location();
            bool has_grouping_ref = instance.has_grouping_ref();

            if (!has_asic_location && !has_grouping_ref) {
                errors.push_back(
                    fmt::format("Grouping '{}' instance {} must have either asic_location or grouping_ref", name, j));
            }

            // Validate ASIC location enum value
            if (has_asic_location) {
                proto::AsicLocation loc = instance.asic_location();
                if (loc == proto::ASIC_LOCATION_UNSPECIFIED) {
                    errors.push_back(fmt::format(
                        "Grouping '{}' instance {} uses ASIC_LOCATION_UNSPECIFIED; must use ASIC_LOCATION_1 through "
                        "ASIC_LOCATION_8",
                        name,
                        j));
                }
                if (static_cast<int>(loc) < 1 || static_cast<int>(loc) > 8) {
                    errors.push_back(fmt::format(
                        "Grouping '{}' instance {} uses invalid ASIC location value {}",
                        name,
                        j,
                        static_cast<int>(loc)));
                }
            }
        }
    }
}

std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>>
PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(const MeshGraphDescriptor& mesh_graph_descriptor) const {
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> result;

    // ===== PHASE 0: Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts) =====
    // This step calculates required ASIC counts bottom-up and builds adjacency graphs
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos =
        build_mgd_to_grouping_info_map(mesh_graph_descriptor);

    // Build flattened adjacency graph for all mesh group infos
    // Get all mesh group infos from current grouping files
    std::unordered_map<std::string, AdjacencyGraph<uint32_t>> mesh_flat_adjacency_graphs;
    std::unordered_map<std::string, GroupingInfo> mesh_grouping_lookup;  // Map from name to GroupingInfo
    auto mesh_it = resolved_groupings_cache_.find("MESH");
    if (mesh_it != resolved_groupings_cache_.end()) {
        for (const auto& mesh_group_info : mesh_it->second) {
            mesh_flat_adjacency_graphs[mesh_group_info.name] = build_flattened_adjacency_graph(mesh_group_info);
            mesh_grouping_lookup[mesh_group_info.name] = mesh_group_info;
        }
    } else {
        TT_THROW("Internal error: MESH grouping not found in resolved_groupings_cache_");
    }

    // Try to fit the best mesh grouping for each MGD mesh instance using topology solver
    // Fit with the most number matched
    for (const auto& mesh_group_info_pair : mgd_grouping_infos["MESH"]) {
        const std::string& instance_name = mesh_group_info_pair.first;
        const GroupingInfo& mgd_grouping_info = mesh_group_info_pair.second;

        // Required nodes from MGD adjacency graph (this represents the topology pattern to match)
        size_t required_nodes = mgd_grouping_info.adjacency_graph.get_nodes().size();

        log_critical(
            tt::LogFabric,
            "Matching MGD mesh instance: {} (logical grouping: {}, ASIC count: {}, required adjacency nodes: {})",
            instance_name,
            mgd_grouping_info.name,
            mgd_grouping_info.asic_count,
            required_nodes);

        log_critical(tt::LogFabric, "  DEBUG: Required nodes from MGD adjacency graph: {}", required_nodes);

        // Print the logical (MGD) mesh adjacency graph
        log_critical(tt::LogFabric, "=== Logical (MGD) Mesh Adjacency Graph: {} ===", instance_name);
        mgd_grouping_info.adjacency_graph.print_adjacency_map(fmt::format("Logical_MGD_{}", instance_name));

        // Track the best match (most nodes matched)
        size_t best_match_count = 0;
        std::string best_match_name;

        for (const auto& mesh_flat_adjacency_graph : mesh_flat_adjacency_graphs) {
            const std::string& physical_grouping_name = mesh_flat_adjacency_graph.first;
            // mesh_flat_adjacency_graph.second is the flattened adjacency graph (ASIC-level)
            const auto& physical_adjacency_graph = mesh_flat_adjacency_graph.second;
            // Get flattened node count (ASIC-level nodes after flattening)
            size_t physical_flattened_nodes = physical_adjacency_graph.get_nodes().size();

            log_critical(
                tt::LogFabric,
                "  DEBUG: Physical grouping '{}' has {} flattened nodes (required: {})",
                physical_grouping_name,
                physical_flattened_nodes,
                required_nodes);

            // Filter: skip physical groupings that don't have enough flattened nodes
            if (physical_flattened_nodes < required_nodes) {
                log_critical(
                    tt::LogFabric,
                    "  Skipping physical grouping: {} (flattened nodes: {} < required: {})",
                    physical_grouping_name,
                    physical_flattened_nodes,
                    required_nodes);
                continue;
            }

            log_critical(
                tt::LogFabric,
                "  Trying physical grouping: {} (flattened nodes: {})",
                physical_grouping_name,
                physical_flattened_nodes);

            // Print the physical grouping mesh adjacency graph
            log_critical(tt::LogFabric, "=== Physical Grouping Mesh Adjacency Graph: {} ===", physical_grouping_name);
            physical_adjacency_graph.print_adjacency_map(fmt::format("Physical_{}", physical_grouping_name));

            auto mapping_result = solve_topology_mapping(
                mesh_flat_adjacency_graph.second,   // global: physical grouping that contains the pattern
                mgd_grouping_info.adjacency_graph,  // target: MGD mesh pattern to find
                {},
                ConnectionValidationMode::STRICT);

            size_t matched_nodes = mapping_result.target_to_global.size();
            size_t target_nodes = mgd_grouping_info.adjacency_graph.get_nodes().size();
            size_t global_nodes = physical_flattened_nodes;  // Use flattened node count

            log_critical(
                tt::LogFabric,
                "    Mapping result: success={}, matched_nodes={}/{}, target_nodes={}, global_nodes={}",
                mapping_result.success,
                matched_nodes,
                target_nodes,
                target_nodes,
                global_nodes);

            if (!mapping_result.error_message.empty()) {
                log_critical(tt::LogFabric, "    Error message: {}", mapping_result.error_message);
            }

            if (!mapping_result.warnings.empty()) {
                std::string warnings_str;
                for (size_t i = 0; i < mapping_result.warnings.size(); ++i) {
                    if (i > 0) {
                        warnings_str += "; ";
                    }
                    warnings_str += mapping_result.warnings[i];
                }
                log_critical(tt::LogFabric, "    Warnings: {}", warnings_str);
            }

            // If successful and matches more nodes than previous best, update best match
            if (mapping_result.success) {
                if (matched_nodes > best_match_count) {
                    log_critical(tt::LogFabric, "    -> New best match! (previous best: {} nodes)", best_match_count);
                    best_match_count = matched_nodes;
                    best_match_name = physical_grouping_name;
                } else {
                    log_critical(
                        tt::LogFabric,
                        "    -> Match found but not better than current best ({} nodes)",
                        best_match_count);
                }
            } else {
                log_critical(tt::LogFabric, "    -> Mapping failed");
            }
        }

        // Store the best match if found
        if (!best_match_name.empty()) {
            auto lookup_it = mesh_grouping_lookup.find(best_match_name);
            if (lookup_it != mesh_grouping_lookup.end()) {
                log_critical(
                    tt::LogFabric,
                    "  Final match for {}: {} (matched {} nodes)",
                    instance_name,
                    best_match_name,
                    best_match_count);
                result["MESH"][instance_name] = lookup_it->second;
            }
        } else {
            log_critical(tt::LogFabric, "  No match found for MGD mesh instance: {}", instance_name);
        }
    }

    return result;
}

AdjacencyGraph<uint32_t> PhysicalGroupingDescriptor::build_flattened_adjacency_graph(
    const GroupingInfo& grouping) const {
    (void)grouping;
    TT_THROW("Not implemented");
    return {};
}

void PhysicalGroupingDescriptor::validate_and_populate_preformed_groups_from_physical_system(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    // Get all ASIC descriptors
    const auto& asic_descriptors = physical_system_descriptor.get_asic_descriptors();

    if (asic_descriptors.empty()) {
        return;  // No ASICs to group
    }

    // Collect all existing names from resolved_groupings_cache_ to ensure uniqueness
    std::unordered_set<std::string> existing_names;
    for (const auto& [type, groupings] : resolved_groupings_cache_) {
        for (const auto& grouping : groupings) {
            existing_names.insert(grouping.name);
        }
    }

    // Helper function to generate a unique name
    auto generate_unique_name = [&](const std::string& base_type, uint32_t start_id = 0) -> std::string {
        uint32_t id = start_id;
        std::string candidate_name;
        do {
            candidate_name = fmt::format("{}_{}", base_type, id);
            id++;
        } while (existing_names.find(candidate_name) != existing_names.end());

        // Add the new name to the set to prevent future duplicates
        existing_names.insert(candidate_name);
        return candidate_name;
    };

    // ===== PHASE 1 & 2: Build structure from PSD (needed for both validation and creation) =====
    // Group ASICs by (hostname, tray_id) to form trays
    // Map: (hostname, tray_id) -> vector of (asic_id, asic_location)
    std::map<std::pair<std::string, uint32_t>, std::vector<std::pair<tt::tt_metal::AsicID, tt::tt_metal::ASICLocation>>>
        trays;

    // Map: (hostname, tray_id) -> set of ASIC IDs in this tray (for building adjacency graph)
    std::map<std::pair<std::string, uint32_t>, std::unordered_set<tt::tt_metal::AsicID>> tray_asic_id_sets;

    for (const auto& [asic_id, asic_desc] : asic_descriptors) {
        uint32_t tray_id_value = *asic_desc.tray_id;
        auto tray_key = std::make_pair(asic_desc.host_name, tray_id_value);
        trays[tray_key].push_back({asic_id, asic_desc.asic_location});
        tray_asic_id_sets[tray_key].insert(asic_id);
    }

    // Group trays by hostname
    std::map<std::string, std::vector<std::pair<std::string, uint32_t>>> host_trays;
    for (const auto& [host_tray_pair, asics] : trays) {
        const auto& [hostname, tray_id] = host_tray_pair;
        host_trays[hostname].push_back({hostname, tray_id});
    }

    // Sort trays within each host by tray_id
    for (auto& [hostname, tray_list] : host_trays) {
        std::sort(tray_list.begin(), tray_list.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    }

    // Map: (hostname, tray_id) -> tray name (TRAY_1, TRAY_2, etc.)
    std::map<std::pair<std::string, uint32_t>, std::string> tray_name_map;

    // Assign tray names (TRAY_1, TRAY_2, TRAY_3, TRAY_4) for each host
    for (auto& [hostname, tray_list] : host_trays) {
        for (size_t i = 0; i < tray_list.size() && i < 4; ++i) {
            std::string tray_name = "TRAY_" + std::to_string(i + 1);
            tray_name_map[tray_list[i]] = tray_name;
        }
    }

    // Helper function to build adjacency graph for a tray based on ASIC locations
    // Node IDs in the graph are ASIC location values (1-8)
    auto build_tray_adjacency_graph =
        [&](const std::vector<std::pair<tt::tt_metal::AsicID, tt::tt_metal::ASICLocation>>& tray_asics,
            const std::unordered_set<tt::tt_metal::AsicID>& tray_asic_ids) -> AdjacencyGraph<uint32_t> {
        std::map<uint32_t, std::vector<uint32_t>> adj_map;

        // Build a map from ASIC location to ASIC ID for quick lookup
        std::unordered_map<tt::tt_metal::AsicID, uint32_t> asic_id_to_location;
        for (const auto& [asic_id, asic_location] : tray_asics) {
            asic_id_to_location[asic_id] = *asic_location;
        }

        // Build adjacency graph: for each ASIC in the tray, find its neighbors
        // and add edges between their ASIC locations
        std::set<std::pair<uint32_t, uint32_t>> processed_edges;
        for (const auto& [asic_id, asic_location] : tray_asics) {
            uint32_t src_location = *asic_location;
            std::vector<tt::tt_metal::AsicID> neighbors = physical_system_descriptor.get_asic_neighbors(asic_id);

            for (const auto& neighbor_id : neighbors) {
                // Only consider neighbors that are in the same tray
                if (tray_asic_ids.find(neighbor_id) == tray_asic_ids.end()) {
                    continue;
                }

                // Get the neighbor's ASIC location
                auto neighbor_loc_it = asic_id_to_location.find(neighbor_id);
                if (neighbor_loc_it == asic_id_to_location.end()) {
                    continue;
                }

                uint32_t dst_location = neighbor_loc_it->second;

                // Skip self-loops
                if (src_location == dst_location) {
                    continue;
                }

                // Normalize edge pair to avoid duplicates (treat (A,B) and (B,A) as the same)
                auto edge_pair = std::minmax(src_location, dst_location);

                // Only add edge if not already processed (prevents duplicates)
                if (processed_edges.insert(edge_pair).second) {
                    // Add bidirectional edge (undirected graph)
                    adj_map[src_location].push_back(dst_location);
                    adj_map[dst_location].push_back(src_location);
                }
            }
        }

        return AdjacencyGraph<uint32_t>(adj_map);
    };

    // Helper function to check if two tray groupings are equivalent
    // (same ASIC locations and same adjacency graph)
    auto trays_are_equivalent = [](const GroupingInfo& a, const GroupingInfo& b) -> bool {
        if (a.type != b.type) {
            return false;
        }
        if (a.asic_count != b.asic_count) {
            return false;
        }
        if (a.items.size() != b.items.size()) {
            return false;
        }

        // Check that ASIC locations match
        for (size_t i = 0; i < a.items.size(); ++i) {
            if (a.items[i].type != GroupingItemInfo::ItemType::ASIC_LOCATION ||
                b.items[i].type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
                return false;
            }
            if (a.items[i].asic_location != b.items[i].asic_location) {
                return false;
            }
        }

        // Check that adjacency graphs are equivalent
        // Compare node sets and edges
        const auto& a_nodes = a.adjacency_graph.get_nodes();
        const auto& b_nodes = b.adjacency_graph.get_nodes();
        if (a_nodes != b_nodes) {
            return false;
        }

        // Check edges for each node
        for (uint32_t node : a_nodes) {
            const auto& a_neighbors = a.adjacency_graph.get_neighbors(node);
            const auto& b_neighbors = b.adjacency_graph.get_neighbors(node);
            std::vector<uint32_t> a_sorted(a_neighbors.begin(), a_neighbors.end());
            std::vector<uint32_t> b_sorted(b_neighbors.begin(), b_neighbors.end());
            std::sort(a_sorted.begin(), a_sorted.end());
            std::sort(b_sorted.begin(), b_sorted.end());
            if (a_sorted != b_sorted) {
                return false;
            }
        }

        return true;
    };

    // Helper function to build tray grouping from physical system
    auto build_tray_info_from_psd =
        [&](const std::string& tray_type,
            const std::vector<std::pair<tt::tt_metal::AsicID, tt::tt_metal::ASICLocation>>& tray_asics,
            const std::unordered_set<tt::tt_metal::AsicID>& tray_asic_ids) -> GroupingInfo {
        GroupingInfo tray_info;
        tray_info.type = tray_type;
        tray_info.asic_count = static_cast<uint32_t>(tray_asics.size());

        // Sort ASICs by location to ensure consistent ordering
        std::vector<std::pair<tt::tt_metal::AsicID, tt::tt_metal::ASICLocation>> sorted_asics = tray_asics;
        std::sort(sorted_asics.begin(), sorted_asics.end(), [](const auto& a, const auto& b) {
            return *a.second < *b.second;
        });

        // Create items with ASIC locations
        for (const auto& [asic_id, asic_location] : sorted_asics) {
            GroupingItemInfo item;
            item.type = GroupingItemInfo::ItemType::ASIC_LOCATION;
            item.asic_location = *asic_location;
            tray_info.items.push_back(item);
        }

        // Build adjacency graph based on per-tray ASIC locations
        tray_info.adjacency_graph = build_tray_adjacency_graph(tray_asics, tray_asic_ids);

        // Populate corner orientation for mesh groupings based on row-major mesh structure
        // Infer mesh dimensions from adjacency graph structure (analyze connectivity patterns)
        // For a row-major mesh, we can infer dimensions by analyzing the graph structure
        size_t num_asics = tray_info.items.size();
        int32_t rows = 0, cols = 0;

        // Try to infer mesh dimensions from adjacency graph structure
        // For a row-major mesh, corner nodes have degree 2, edge nodes have degree 3, interior nodes have degree 4
        const auto& nodes = tray_info.adjacency_graph.get_nodes();
        if (!nodes.empty() && num_asics > 0) {
            // Count nodes by degree to infer mesh structure
            std::map<size_t, size_t> degree_count;
            for (uint32_t node : nodes) {
                size_t degree = tray_info.adjacency_graph.get_neighbors(node).size();
                degree_count[degree]++;
            }

            // Standard configurations based on number of ASICs and degree patterns
            if (num_asics == 8) {
                // Standard tray: 2x4 mesh (4 corners with degree 2, 4 edges with degree 3)
                rows = 2;
                cols = 4;
            } else if (num_asics == 4) {
                // 2x2 mesh (4 corners, all with degree 2)
                rows = 2;
                cols = 2;
            } else if (num_asics == 16) {
                // 4x4 mesh
                rows = 4;
                cols = 4;
            } else if (num_asics == 32) {
                // 4x8 mesh
                rows = 4;
                cols = 8;
            }
            // Add more configurations as needed
        }

        // If we have valid dimensions, populate corner orientations based on row-major mesh position
        // For 1D meshes, endpoints can have multiple orientations
        // For 1x1 mesh, the single item has all 4 orientations
        if (rows > 0 && cols > 0 && num_asics == static_cast<size_t>(rows * cols)) {
            if (rows == 1 && cols == 1) {
                // 1x1 mesh: single item has all 4 orientations
                if (num_asics == 1) {
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                }
            } else if (rows == 1) {
                // 1D row mesh (1xN): first item has NW+SW, last item has NE+SE
                if (num_asics > 0) {
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                }
                if (num_asics > 1) {
                    size_t last_idx = num_asics - 1;
                    tray_info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                    tray_info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                }
            } else if (cols == 1) {
                // 1D column mesh (Nx1): first item has NW+NE, last item has SW+SE
                if (num_asics > 0) {
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                    tray_info.items[0].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                }
                if (num_asics > 1) {
                    size_t last_idx = num_asics - 1;
                    tray_info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                    tray_info.items[last_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                }
            } else {
                // 2D mesh: standard 4 corners
                size_t nw_idx = 0;
                size_t ne_idx = static_cast<size_t>(cols - 1);
                size_t sw_idx = static_cast<size_t>((rows - 1) * cols);
                size_t se_idx = static_cast<size_t>((rows - 1) * cols + (cols - 1));

                if (nw_idx < num_asics) {
                    tray_info.items[nw_idx].corners.push_back(GroupingItemInfo::CornerOrientation::NW);
                }
                if (ne_idx < num_asics && ne_idx != nw_idx) {
                    tray_info.items[ne_idx].corners.push_back(GroupingItemInfo::CornerOrientation::NE);
                }
                if (sw_idx < num_asics && sw_idx != nw_idx && sw_idx != ne_idx) {
                    tray_info.items[sw_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SW);
                }
                if (se_idx < num_asics && se_idx != nw_idx && se_idx != ne_idx && se_idx != sw_idx) {
                    tray_info.items[se_idx].corners.push_back(GroupingItemInfo::CornerOrientation::SE);
                }
            }
        }

        return tray_info;
    };

    // Helper function to check if two host groupings are equivalent
    // (same tray references in same order)
    auto hosts_are_equivalent = [](const GroupingInfo& a, const GroupingInfo& b) -> bool {
        if (a.type != b.type) {
            return false;
        }
        if (a.items.size() != b.items.size()) {
            return false;
        }

        // Check that tray references match in order
        for (size_t i = 0; i < a.items.size(); ++i) {
            if (a.items[i].type != GroupingItemInfo::ItemType::GROUPING_REF ||
                b.items[i].type != GroupingItemInfo::ItemType::GROUPING_REF) {
                return false;
            }
            if (a.items[i].grouping_name != b.items[i].grouping_name) {
                return false;
            }
        }

        return true;
    };

    // Helper function to build host grouping from physical system
    auto build_host_info_from_psd =
        [&](const std::string& /* hostname */,
            const std::vector<std::pair<std::string, uint32_t>>& tray_list) -> GroupingInfo {
        GroupingInfo host_info;
        host_info.type = "HOSTS";
        host_info.asic_count = 0;

        // Create items referencing trays
        for (const auto& tray_pair : tray_list) {
            auto tray_name_it = tray_name_map.find(tray_pair);
            if (tray_name_it == tray_name_map.end()) {
                continue;
            }

            const std::string& tray_name = tray_name_it->second;

            GroupingItemInfo item;
            item.type = GroupingItemInfo::ItemType::GROUPING_REF;
            item.grouping_name = tray_name;
            host_info.items.push_back(item);

            // Add ASIC count from tray
            auto tray_groupings_it = resolved_groupings_cache_.find(tray_name);
            if (tray_groupings_it != resolved_groupings_cache_.end() && !tray_groupings_it->second.empty()) {
                host_info.asic_count += tray_groupings_it->second[0].asic_count;
            }
        }

        // Build adjacency graph for host based on tray topology
        // For 4 trays, assume 2x2 grid topology (standard configuration)
        // Node IDs are instance indices (0, 1, 2, 3) corresponding to tray positions
        if (host_info.items.size() == 4) {
            // Build 2x2 row-major mesh: trays arranged as:
            // TRAY_1 (0)  TRAY_2 (1)
            // TRAY_3 (2)  TRAY_4 (3)
            std::vector<uint32_t> instance_ids = {0, 1, 2, 3};
            std::vector<int32_t> dims = {2, 2};
            host_info.adjacency_graph = build_row_major_mesh_graph(instance_ids, dims, "");
        } else {
            // For other configurations, use empty graph
            host_info.adjacency_graph = AdjacencyGraph<uint32_t>();
        }

        return host_info;
    };

    // ===== PHASE 1: If groupings file definitions already exist, validate them against PSD =====
    // Check if TRAY_* and HOSTS groupings already exist from file definitions
    bool has_tray_groupings_from_file = false;
    bool has_host_groupings_from_file = false;

    for (const std::string& tray_type : {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4"}) {
        auto it = resolved_groupings_cache_.find(tray_type);
        if (it != resolved_groupings_cache_.end() && !it->second.empty()) {
            has_tray_groupings_from_file = true;
            break;
        }
    }

    {
        auto it = resolved_groupings_cache_.find("HOSTS");
        if (it != resolved_groupings_cache_.end() && !it->second.empty()) {
            has_host_groupings_from_file = true;
        }
    }

    // If groupings exist from file, validate they match PSD structure
    if (has_tray_groupings_from_file) {
        // Validate each tray type that exists in PSD
        for (const auto& [host_tray_pair, tray_asics] : trays) {
            auto tray_name_it = tray_name_map.find(host_tray_pair);
            if (tray_name_it == tray_name_map.end()) {
                continue;  // Skip if tray name not assigned (more than 4 trays per host)
            }

            const std::string& tray_type = tray_name_it->second;
            const auto& tray_asic_ids = tray_asic_id_sets[host_tray_pair];

            // Build expected tray info from PSD
            GroupingInfo expected_tray_info = build_tray_info_from_psd(tray_type, tray_asics, tray_asic_ids);

            // Check if this tray type exists in cache
            auto existing_trays_it = resolved_groupings_cache_.find(tray_type);
            if (existing_trays_it != resolved_groupings_cache_.end() && !existing_trays_it->second.empty()) {
                // Validate that at least one existing grouping matches expected
                bool found_match = false;
                for (const auto& existing_tray : existing_trays_it->second) {
                    if (trays_are_equivalent(expected_tray_info, existing_tray)) {
                        found_match = true;
                        break;
                    }
                }

                if (!found_match) {
                    // Build debug strings showing ASIC locations in order
                    std::string discovered_locations;
                    for (size_t i = 0; i < expected_tray_info.items.size(); ++i) {
                        if (i > 0) {
                            discovered_locations += ", ";
                        }
                        discovered_locations +=
                            fmt::format("ASIC_LOCATION_{}", expected_tray_info.items[i].asic_location);
                    }

                    std::string file_locations;
                    std::string discovered_graph_edges, file_graph_edges;
                    std::string discovered_node_mapping, file_node_mapping;
                    if (!existing_trays_it->second.empty()) {
                        const auto& existing_tray = existing_trays_it->second[0];
                        for (size_t i = 0; i < existing_tray.items.size(); ++i) {
                            if (i > 0) {
                                file_locations += ", ";
                            }
                            file_locations += fmt::format("ASIC_LOCATION_{}", existing_tray.items[i].asic_location);
                        }

                        // Build node ID to ASIC location mappings
                        // Discovered: node IDs are ASIC locations
                        const auto& discovered_nodes = expected_tray_info.adjacency_graph.get_nodes();
                        for (uint32_t node : discovered_nodes) {
                            if (!discovered_node_mapping.empty()) {
                                discovered_node_mapping += ", ";
                            }
                            discovered_node_mapping += fmt::format("node_{}=ASIC_LOCATION_{}", node, node);
                        }

                        // File: node IDs are ASIC locations (for tray groupings), find corresponding item
                        const auto& file_nodes = existing_tray.adjacency_graph.get_nodes();
                        for (uint32_t node : file_nodes) {
                            if (!file_node_mapping.empty()) {
                                file_node_mapping += ", ";
                            }
                            // For tray groupings, node IDs are ASIC locations, so find the item with this ASIC location
                            bool found = false;
                            for (size_t i = 0; i < existing_tray.items.size(); ++i) {
                                if (existing_tray.items[i].type == GroupingItemInfo::ItemType::ASIC_LOCATION &&
                                    existing_tray.items[i].asic_location == node) {
                                    file_node_mapping += fmt::format("node_{}=ASIC_LOCATION_{}(idx_{})", node, node, i);
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                file_node_mapping += fmt::format("node_{}=unknown", node);
                            }
                        }

                        // Build adjacency graph edge descriptions
                        for (uint32_t node : discovered_nodes) {
                            const auto& neighbors = expected_tray_info.adjacency_graph.get_neighbors(node);
                            for (uint32_t neighbor : neighbors) {
                                if (node < neighbor) {  // Avoid duplicate edges
                                    if (!discovered_graph_edges.empty()) {
                                        discovered_graph_edges += ", ";
                                    }
                                    discovered_graph_edges +=
                                        fmt::format("ASIC_LOCATION_{}->ASIC_LOCATION_{}", node, neighbor);
                                }
                            }
                        }

                        for (uint32_t node : file_nodes) {
                            const auto& neighbors = existing_tray.adjacency_graph.get_neighbors(node);
                            for (uint32_t neighbor : neighbors) {
                                if (node < neighbor) {  // Avoid duplicate edges
                                    if (!file_graph_edges.empty()) {
                                        file_graph_edges += ", ";
                                    }
                                    // For tray groupings, node IDs are ASIC locations
                                    file_graph_edges +=
                                        fmt::format("ASIC_LOCATION_{}->ASIC_LOCATION_{}", node, neighbor);
                                }
                            }
                        }
                    }

                    TT_THROW(
                        "TRAY grouping type '{}' already exists but does not match physical system. "
                        "Expected {} ASICs with locations matching physical system, but found different configuration. "
                        "Discovered from PSD (row-major sorted): [{}]. "
                        "File definition: [{}]. "
                        "Discovered node mapping (node_id=ASIC_LOCATION): [{}]. "
                        "File node mapping (instance_id=ASIC_LOCATION): [{}]. "
                        "Discovered adjacency graph edges: [{}]. "
                        "File adjacency graph edges: [{}].",
                        tray_type,
                        expected_tray_info.asic_count,
                        discovered_locations,
                        file_locations,
                        discovered_node_mapping.empty() ? "none" : discovered_node_mapping,
                        file_node_mapping.empty() ? "none" : file_node_mapping,
                        discovered_graph_edges.empty() ? "none" : discovered_graph_edges,
                        file_graph_edges.empty() ? "none" : file_graph_edges);
                }
            }
        }
    }

    if (has_host_groupings_from_file) {
        // Validate host groupings match PSD structure
        for (const auto& [hostname, tray_list] : host_trays) {
            if (tray_list.size() > 4) {
                continue;  // Skip hosts with more than 4 trays (not standard)
            }

            // Build expected host info from PSD
            GroupingInfo expected_host_info = build_host_info_from_psd(hostname, tray_list);

            // Check if HOSTS grouping exists in cache
            auto existing_hosts_it = resolved_groupings_cache_.find("HOSTS");
            if (existing_hosts_it != resolved_groupings_cache_.end() && !existing_hosts_it->second.empty()) {
                // For host groupings, check if the number of trays matches first
                // If the number of trays doesn't match, skip Phase 1 validation and let Phase 3
                // handle it with topology solver validation (which will catch topology mismatches like 2x3 vs 2x2)
                bool tray_count_matches = false;
                for (const auto& existing_host : existing_hosts_it->second) {
                    if (existing_host.items.size() == expected_host_info.items.size()) {
                        tray_count_matches = true;
                        break;
                    }
                }

                // Only do strict Phase 1 validation if tray counts match
                // Otherwise, let Phase 3 topology solver handle the validation
                if (tray_count_matches) {
                    // Validate that at least one existing grouping matches expected
                    bool found_match = false;
                    for (const auto& existing_host : existing_hosts_it->second) {
                        if (hosts_are_equivalent(expected_host_info, existing_host)) {
                            found_match = true;
                            break;
                        }
                    }

                    if (!found_match) {
                        // Build tray names string for error message
                        std::string tray_names;
                        for (size_t i = 0; i < expected_host_info.items.size(); ++i) {
                            if (i > 0) {
                                tray_names += ", ";
                            }
                            tray_names += expected_host_info.items[i].grouping_name;
                        }
                        TT_THROW(
                            "HOSTS grouping type already exists but does not match physical system. "
                            "Expected {} trays ({}) but found different configuration.",
                            expected_host_info.items.size(),
                            tray_names);
                    }
                }
                // If tray counts don't match, skip Phase 1 validation - Phase 3 will catch it with topology solver
            }
        }
    }

    // ===== PHASE 2: Build grouping info out of physical system descriptor =====
    // Process each tray: find matching existing grouping or create new one
    if (!has_tray_groupings_from_file) {
        for (const auto& [host_tray_pair, tray_asics] : trays) {
            auto tray_name_it = tray_name_map.find(host_tray_pair);
            if (tray_name_it == tray_name_map.end()) {
                continue;  // Skip if tray name not assigned (more than 4 trays per host)
            }

            const std::string& tray_type = tray_name_it->second;  // e.g., "TRAY_1"
            const auto& tray_asic_ids = tray_asic_id_sets[host_tray_pair];

            // Build tray info from physical system descriptor
            GroupingInfo tray_info_from_psd = build_tray_info_from_psd(tray_type, tray_asics, tray_asic_ids);

            // Check if an equivalent tray grouping already exists
            auto existing_trays_it = resolved_groupings_cache_.find(tray_type);
            bool found_match = false;

            if (existing_trays_it != resolved_groupings_cache_.end()) {
                for (const auto& existing_tray : existing_trays_it->second) {
                    if (trays_are_equivalent(tray_info_from_psd, existing_tray)) {
                        found_match = true;
                        break;
                    }
                }
            }

            // If no match found, create a new tray grouping
            if (!found_match) {
                tray_info_from_psd.name = generate_unique_name(tray_type);
                resolved_groupings_cache_[tray_type].push_back(tray_info_from_psd);
            }
        }
    }

    // Process each host: find matching existing grouping or create new one
    if (!has_host_groupings_from_file) {
        for (const auto& [hostname, tray_list] : host_trays) {
            if (tray_list.size() > 4) {
                continue;  // Skip hosts with more than 4 trays (not standard)
            }

            // Build host info from physical system descriptor
            GroupingInfo host_info_from_psd = build_host_info_from_psd(hostname, tray_list);

            // Check if an equivalent host grouping already exists
            auto existing_hosts_it = resolved_groupings_cache_.find("HOSTS");
            bool found_match = false;

            if (existing_hosts_it != resolved_groupings_cache_.end()) {
                for (const auto& existing_host : existing_hosts_it->second) {
                    if (hosts_are_equivalent(host_info_from_psd, existing_host)) {
                        found_match = true;
                        break;
                    }
                }
            }

            // If no match found, create a new host grouping
            if (!found_match) {
                host_info_from_psd.name = generate_unique_name("HOSTS");
                resolved_groupings_cache_["HOSTS"].push_back(host_info_from_psd);
            }
        }
    }

    // ===== PHASE 3: Use topology solver to validate file groupings map onto discovered PSD groupings =====
    // This phase validates that groupings from file can be mapped onto discovered groupings from PSD
    // using topology solver with constraints ensuring trays match (ASIC locations)
    if (has_tray_groupings_from_file) {
        // For each tray type, validate using topology solver
        for (const auto& [host_tray_pair, tray_asics] : trays) {
            auto tray_name_it = tray_name_map.find(host_tray_pair);
            if (tray_name_it == tray_name_map.end()) {
                continue;  // Skip if tray name not assigned (more than 4 trays per host)
            }

            const std::string& tray_type = tray_name_it->second;
            const auto& tray_asic_ids = tray_asic_id_sets[host_tray_pair];

            // Get file grouping (target) and PSD grouping (global)
            auto file_tray_it = resolved_groupings_cache_.find(tray_type);
            if (file_tray_it == resolved_groupings_cache_.end() || file_tray_it->second.empty()) {
                continue;  // No file grouping to validate
            }

            // Build PSD grouping for this tray
            GroupingInfo psd_tray_info = build_tray_info_from_psd(tray_type, tray_asics, tray_asic_ids);

            // Try to map each file grouping onto the PSD grouping
            for (const auto& file_tray_info : file_tray_it->second) {
                // Build trait maps for ASIC locations (constraints)
                // Target nodes = ASIC locations from file grouping
                // Global nodes = ASIC locations from PSD grouping
                std::map<uint32_t, uint32_t> target_traits;  // target_node -> asic_location
                std::map<uint32_t, uint32_t> global_traits;  // global_node -> asic_location

                // Build target traits from file grouping items
                for (const auto& item : file_tray_info.items) {
                    if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                        // Use ASIC location as both node ID and trait value
                        uint32_t node_id = item.asic_location;
                        target_traits[node_id] = item.asic_location;
                    }
                }

                // Build global traits from PSD grouping items
                for (const auto& item : psd_tray_info.items) {
                    if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                        // Use ASIC location as both node ID and trait value
                        uint32_t node_id = item.asic_location;
                        global_traits[node_id] = item.asic_location;
                    }
                }

                // Build constraints: ASIC locations must match
                MappingConstraints<uint32_t, uint32_t> constraints;
                constraints.add_required_trait_constraint(target_traits, global_traits);

                // Solve mapping: map file grouping (target) onto PSD grouping (global)
                auto mapping_result = solve_topology_mapping(
                    file_tray_info.adjacency_graph,  // target: file grouping adjacency graph
                    psd_tray_info.adjacency_graph,   // global: PSD grouping adjacency graph
                    constraints,
                    ConnectionValidationMode::STRICT);

                // Validate that all target nodes are mapped AND all global nodes are used
                size_t target_nodes = file_tray_info.adjacency_graph.get_nodes().size();
                size_t global_nodes = psd_tray_info.adjacency_graph.get_nodes().size();
                size_t mapped_target_nodes = mapping_result.target_to_global.size();
                size_t used_global_nodes = mapping_result.global_to_target.size();

                if (!mapping_result.success || mapped_target_nodes != target_nodes ||
                    used_global_nodes != global_nodes) {
                    TT_THROW(
                        "TRAY grouping type '{}' from file cannot be mapped onto discovered PSD grouping using "
                        "topology "
                        "solver. "
                        "Target nodes: {} (mapped: {}), Global nodes: {} (used: {}). "
                        "Error: {}",
                        tray_type,
                        target_nodes,
                        mapped_target_nodes,
                        global_nodes,
                        used_global_nodes,
                        mapping_result.error_message.empty() ? "Mapping failed" : mapping_result.error_message);
                }
            }
        }
    }

    if (has_host_groupings_from_file) {
        // For host groupings, validate using topology solver
        // Host groupings reference trays, so we need to validate tray references match
        auto file_hosts_it = resolved_groupings_cache_.find("HOSTS");
        if (file_hosts_it != resolved_groupings_cache_.end() && !file_hosts_it->second.empty()) {
            // For each host in PSD, build expected host grouping
            for (const auto& [hostname, tray_list] : host_trays) {
                if (tray_list.size() > 4) {
                    continue;  // Skip hosts with more than 4 trays (not standard)
                }

                // Build PSD host grouping
                GroupingInfo psd_host_info = build_host_info_from_psd(hostname, tray_list);

                // Try to map each file host grouping onto the PSD host grouping
                for (const auto& file_host_info : file_hosts_it->second) {
                    // For host groupings, we validate by checking if tray references match
                    // Since hosts don't have adjacency graphs (empty), we use a simpler validation:
                    // Check that the tray reference sequence matches

                    // Build trait maps for tray references (using tray index as trait)
                    std::map<uint32_t, std::string> target_traits;  // target_node -> tray_name
                    std::map<uint32_t, std::string> global_traits;  // global_node -> tray_name

                    // Build target traits from file host grouping items
                    for (size_t i = 0; i < file_host_info.items.size(); ++i) {
                        const auto& item = file_host_info.items[i];
                        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                            target_traits[static_cast<uint32_t>(i)] = item.grouping_name;
                        }
                    }

                    // Build global traits from PSD host grouping items
                    for (size_t i = 0; i < psd_host_info.items.size(); ++i) {
                        const auto& item = psd_host_info.items[i];
                        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                            global_traits[static_cast<uint32_t>(i)] = item.grouping_name;
                        }
                    }

                    // Check if tray references match exactly
                    bool tray_refs_match = false;
                    if (target_traits.size() == global_traits.size()) {
                        tray_refs_match = true;
                        for (const auto& [idx, tray_name] : target_traits) {
                            auto global_it = global_traits.find(idx);
                            if (global_it == global_traits.end() || global_it->second != tray_name) {
                                tray_refs_match = false;
                                break;
                            }
                        }
                    }

                    // Get graph sizes
                    size_t target_nodes = file_host_info.adjacency_graph.get_nodes().size();
                    size_t global_nodes = psd_host_info.adjacency_graph.get_nodes().size();

                    // For hosts with empty adjacency graphs, validate by tray reference matching
                    if (target_nodes == 0 && global_nodes == 0) {
                        if (!tray_refs_match) {
                            std::string file_trays, psd_trays;
                            for (const auto& [_, name] : target_traits) {
                                if (!file_trays.empty()) {
                                    file_trays += ", ";
                                }
                                file_trays += name;
                            }
                            for (const auto& [_, name] : global_traits) {
                                if (!psd_trays.empty()) {
                                    psd_trays += ", ";
                                }
                                psd_trays += name;
                            }
                            TT_THROW(
                                "HOSTS grouping from file cannot be mapped onto discovered PSD grouping. "
                                "File trays: [{}], PSD trays: [{}]",
                                file_trays,
                                psd_trays);
                        }
                        // Tray references match, validation passes
                        continue;
                    }

                    // If one graph is empty but the other is not, this is an error
                    // Both should have the same topology specification (both empty or both non-empty)
                    if ((target_nodes == 0 && global_nodes > 0) || (target_nodes > 0 && global_nodes == 0)) {
                        std::string file_trays, psd_trays;
                        for (const auto& [_, name] : target_traits) {
                            if (!file_trays.empty()) {
                                file_trays += ", ";
                            }
                            file_trays += name;
                        }
                        for (const auto& [_, name] : global_traits) {
                            if (!psd_trays.empty()) {
                                psd_trays += ", ";
                            }
                            psd_trays += name;
                        }
                        TT_THROW(
                            "HOSTS grouping topology mismatch: file grouping has {} nodes, PSD grouping has {} nodes. "
                            "Both should have the same topology specification. "
                            "File trays: [{}], PSD trays: [{}]",
                            target_nodes,
                            global_nodes,
                            file_trays,
                            psd_trays);
                    }

                    // For non-empty graphs, use topology solver to validate topology matches
                    // First, check if graphs have the same number of nodes (required for isomorphism)
                    if (target_nodes != global_nodes) {
                        std::string file_trays, psd_trays;
                        for (const auto& [_, name] : target_traits) {
                            if (!file_trays.empty()) {
                                file_trays += ", ";
                            }
                            file_trays += name;
                        }
                        for (const auto& [_, name] : global_traits) {
                            if (!psd_trays.empty()) {
                                psd_trays += ", ";
                            }
                            psd_trays += name;
                        }
                        TT_THROW(
                            "HOSTS grouping topology mismatch: file grouping has {} nodes, PSD grouping has {} nodes. "
                            "Both should have the same number of nodes for topology matching. "
                            "File trays: [{}], PSD trays: [{}]. "
                            "This indicates the topology in the file (e.g., 2x3=6 nodes) does not match the expected "
                            "topology "
                            "from PSD discovery (e.g., 2x2=4 nodes).",
                            target_nodes,
                            global_nodes,
                            file_trays,
                            psd_trays);
                    }

                    // Count edges in both graphs to ensure they match (required for isomorphism)
                    size_t target_edges = 0;
                    for (uint32_t node : file_host_info.adjacency_graph.get_nodes()) {
                        target_edges += file_host_info.adjacency_graph.get_neighbors(node).size();
                    }
                    target_edges /= 2;  // Divide by 2 since edges are bidirectional

                    size_t global_edges = 0;
                    for (uint32_t node : psd_host_info.adjacency_graph.get_nodes()) {
                        global_edges += psd_host_info.adjacency_graph.get_neighbors(node).size();
                    }
                    global_edges /= 2;  // Divide by 2 since edges are bidirectional

                    if (target_edges != global_edges) {
                        std::string file_trays, psd_trays;
                        for (const auto& [_, name] : target_traits) {
                            if (!file_trays.empty()) {
                                file_trays += ", ";
                            }
                            file_trays += name;
                        }
                        for (const auto& [_, name] : global_traits) {
                            if (!psd_trays.empty()) {
                                psd_trays += ", ";
                            }
                            psd_trays += name;
                        }
                        TT_THROW(
                            "HOSTS grouping topology mismatch: file grouping has {} edges, PSD grouping has {} edges. "
                            "Both should have the same number of edges for topology matching. "
                            "File trays: [{}], PSD trays: [{}]. "
                            "This indicates the topology in the file (e.g., 2x3 has different edge count) does not "
                            "match the expected topology "
                            "from PSD discovery (e.g., 2x2).",
                            target_edges,
                            global_edges,
                            file_trays,
                            psd_trays);
                    }

                    // Build constraints: tray references must match (using instance index as trait)
                    // Note: Instance IDs in adjacency graphs are 0, 1, 2, 3 (tray indices)
                    MappingConstraints<uint32_t, uint32_t> constraints;

                    // Add trait constraints: instance index -> tray name
                    // This ensures that tray at position 0 maps to the same tray name, etc.
                    constraints.add_required_trait_constraint(target_traits, global_traits);

                    // Solve mapping: map file host grouping (target) onto PSD host grouping (global)
                    auto mapping_result = solve_topology_mapping(
                        file_host_info.adjacency_graph,  // target: file host grouping adjacency graph
                        psd_host_info.adjacency_graph,   // global: PSD host grouping adjacency graph
                        constraints,
                        ConnectionValidationMode::STRICT);

                    // Validate that all target nodes are mapped AND all global nodes are used
                    size_t mapped_target_nodes = mapping_result.target_to_global.size();
                    size_t used_global_nodes = mapping_result.global_to_target.size();

                    if (!mapping_result.success || mapped_target_nodes != target_nodes ||
                        used_global_nodes != global_nodes) {
                        std::string file_trays, psd_trays;
                        for (const auto& [_, name] : target_traits) {
                            if (!file_trays.empty()) {
                                file_trays += ", ";
                            }
                            file_trays += name;
                        }
                        for (const auto& [_, name] : global_traits) {
                            if (!psd_trays.empty()) {
                                psd_trays += ", ";
                            }
                            psd_trays += name;
                        }
                        TT_THROW(
                            "HOSTS grouping from file cannot be mapped onto discovered PSD grouping using topology "
                            "solver. "
                            "Target nodes: {} (mapped: {}), Global nodes: {} (used: {}). "
                            "File trays: [{}], PSD trays: [{}]. "
                            "This indicates the topology in the file (e.g., 1x4) does not match the expected topology "
                            "from PSD discovery (e.g., 2x2). "
                            "Error: {}",
                            target_nodes,
                            mapped_target_nodes,
                            global_nodes,
                            used_global_nodes,
                            file_trays,
                            psd_trays,
                            mapping_result.error_message.empty() ? "Mapping failed" : mapping_result.error_message);
                    }
                }
            }
        }
    }
}

}  // namespace tt::tt_fabric
