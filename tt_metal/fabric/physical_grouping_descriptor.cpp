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
AdjacencyGraph<uint32_t> build_all_to_all_graph(
    const std::vector<uint32_t>& instance_ids, uint32_t /* num_connections */) {
    std::map<uint32_t, std::vector<uint32_t>> adj_map;

    // All-to-all: every node connects to every other node
    for (size_t i = 0; i < instance_ids.size(); ++i) {
        for (size_t j = 0; j < instance_ids.size(); ++j) {
            if (i != j) {
                adj_map[instance_ids[i]].push_back(instance_ids[j]);
            }
        }
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

// Helper function to build adjacency graph from row-major mesh connection
AdjacencyGraph<uint32_t> build_row_major_mesh_graph(
    const std::vector<uint32_t>& instance_ids,
    const std::vector<int32_t>& dims,
    const std::vector<proto::RowMajorMeshConnection::DimType>& dim_types,
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
    for (uint32_t idx = 0; idx < instance_ids.size(); ++idx) {
        std::vector<int32_t> coords = get_coords(idx);

        // For each dimension
        for (size_t dim_idx = 0; dim_idx < dims.size(); ++dim_idx) {
            bool is_ring = (dim_idx < dim_types.size() && dim_types[dim_idx] == proto::RowMajorMeshConnection::RING);
            int32_t dim_size = dims[dim_idx];
            int32_t coord_val = coords[dim_idx];

            // Neighbor in positive direction
            if (is_ring) {
                // RING: always connect, wrapping around
                std::vector<int32_t> coords_plus = coords;
                coords_plus[dim_idx] = (coord_val + 1) % dim_size;
                int32_t neighbor_idx = coords_to_idx(coords_plus);
                if (neighbor_idx >= 0 && neighbor_idx < static_cast<int32_t>(instance_ids.size())) {
                    adj_map[instance_ids[idx]].push_back(instance_ids[neighbor_idx]);
                }
            } else {
                // LINE: only connect if not at the end
                if (coord_val < dim_size - 1) {
                    std::vector<int32_t> coords_plus = coords;
                    coords_plus[dim_idx] = coord_val + 1;
                    int32_t neighbor_idx = coords_to_idx(coords_plus);
                    if (neighbor_idx >= 0 && neighbor_idx < static_cast<int32_t>(instance_ids.size())) {
                        adj_map[instance_ids[idx]].push_back(instance_ids[neighbor_idx]);
                    }
                }
            }

            // Neighbor in negative direction
            if (is_ring) {
                // RING: always connect, wrapping around
                std::vector<int32_t> coords_minus = coords;
                coords_minus[dim_idx] = (coord_val - 1 + dim_size) % dim_size;
                int32_t neighbor_idx = coords_to_idx(coords_minus);
                if (neighbor_idx >= 0 && neighbor_idx < static_cast<int32_t>(instance_ids.size())) {
                    adj_map[instance_ids[idx]].push_back(instance_ids[neighbor_idx]);
                }
            } else {
                // LINE: only connect if not at the start
                if (coord_val > 0) {
                    std::vector<int32_t> coords_minus = coords;
                    coords_minus[dim_idx] = coord_val - 1;
                    int32_t neighbor_idx = coords_to_idx(coords_minus);
                    if (neighbor_idx >= 0 && neighbor_idx < static_cast<int32_t>(instance_ids.size())) {
                        adj_map[instance_ids[idx]].push_back(instance_ids[neighbor_idx]);
                    }
                }
            }
        }
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

// Helper function to build adjacency graph from custom connections
AdjacencyGraph<uint32_t> build_custom_connections_graph(
    const std::vector<uint32_t>& instance_ids, const proto::CustomConnections& custom_connections) {
    std::map<uint32_t, std::vector<uint32_t>> adj_map;

    // Build a map from instance index to instance ID
    std::map<uint32_t, uint32_t> index_to_id;
    for (size_t i = 0; i < instance_ids.size(); ++i) {
        index_to_id[static_cast<uint32_t>(i)] = instance_ids[i];
    }

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

        // Add bidirectional edge (assuming undirected, but can be made directional if needed)
        adj_map[src_id].push_back(dst_id);
        adj_map[dst_id].push_back(src_id);
    }

    return AdjacencyGraph<uint32_t>(adj_map);
}

// Helper function to build adjacency graph from MGD mesh instance's host topology
// Builds a row-major mesh graph based on the mesh's host_topology dims
// This represents the topology at the host level, which matches the grouping's instance-level topology
AdjacencyGraph<uint32_t> build_mgd_mesh_instance_adjacency(
    const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId mesh_instance_id) {
    const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_instance_id);
    TT_FATAL(mesh_instance.kind == NodeKind::Mesh, "build_mgd_mesh_instance_adjacency called on non-mesh instance");

    const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(mesh_instance.desc);
    TT_FATAL(mesh_desc != nullptr, "Mesh descriptor is null");

    // Get host topology dimensions (represents host layout)
    const auto& host_topology = mesh_desc->host_topology();
    std::vector<int32_t> host_dims(host_topology.dims().begin(), host_topology.dims().end());

    if (host_dims.empty()) {
        // No host topology - return empty graph (single host case)
        return AdjacencyGraph<uint32_t>();
    }

    // Calculate number of hosts
    int32_t num_hosts = 1;
    for (int32_t dim : host_dims) {
        num_hosts *= dim;
    }

    // Create abstract host node IDs (0, 1, 2, ...)
    std::vector<uint32_t> host_ids;
    host_ids.reserve(num_hosts);
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_hosts); ++i) {
        host_ids.push_back(i);
    }

    // Use device topology dim_types for host-level connections (hosts inherit device topology connectivity)
    const auto& device_topology = mesh_desc->device_topology();
    std::vector<proto::RowMajorMeshConnection::DimType> dim_types;
    dim_types.reserve(device_topology.dim_types_size());
    for (int dt : device_topology.dim_types()) {
        // Convert TorusTopology::Type to RowMajorMeshConnection::DimType
        // TorusTopology::Type::RING = 2, TorusTopology::Type::LINE = 1
        if (dt == 2) {  // RING
            dim_types.push_back(proto::RowMajorMeshConnection::RING);
        } else {  // LINE or default
            dim_types.push_back(proto::RowMajorMeshConnection::LINE);
        }
    }

    // Build row-major mesh graph representing host-level topology
    return build_row_major_mesh_graph(host_ids, host_dims, dim_types, "");
}

// Helper function to build adjacency graph from MGD graph instance
// The graph instance's sub_instances become nodes, and connections between them become edges
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
                // Add bidirectional edge (undirected graph)
                adj_map[src].push_back(dst);
                adj_map[dst].push_back(src);
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

// Helper to find best matching grouping using adjacency graph matching
// Returns the grouping with closest count that successfully maps the logical graph
GroupingInfo find_best_grouping_by_adjacency(
    const AdjacencyGraph<uint32_t>& logical_graph, const std::vector<GroupingInfo>& candidate_groupings) {
    if (candidate_groupings.empty()) {
        TT_THROW("No candidate groupings provided for adjacency matching");
    }

    // Sort candidates by count (ascending) to prefer closest match
    std::vector<GroupingInfo> sorted_candidates = candidate_groupings;
    std::sort(sorted_candidates.begin(), sorted_candidates.end(), [](const GroupingInfo& a, const GroupingInfo& b) {
        return a.adjacency_graph.get_nodes().size() < b.adjacency_graph.get_nodes().size();
    });

    // Get logical graph node count
    size_t logical_node_count = logical_graph.get_nodes().size();

    // Try to find exact match first, then closest match
    GroupingInfo best_match = sorted_candidates[0];
    size_t best_count_diff = SIZE_MAX;
    bool found_valid_mapping = false;

    for (const auto& candidate : sorted_candidates) {
        const auto& physical_graph = candidate.adjacency_graph;
        size_t physical_node_count = physical_graph.get_nodes().size();

        // Skip if physical graph is smaller than logical graph (can't map)
        if (physical_node_count < logical_node_count) {
            continue;
        }

        // Build mapping constraints (no constraints - allow any mapping)
        MappingConstraints<uint32_t, uint32_t> constraints;

        // Try to solve the mapping
        auto mapping_result =
            solve_topology_mapping(logical_graph, physical_graph, constraints, ConnectionValidationMode::RELAXED);

        if (mapping_result.success) {
            // Found a valid mapping
            size_t count_diff = physical_node_count - logical_node_count;
            if (!found_valid_mapping || count_diff < best_count_diff) {
                best_match = candidate;
                best_count_diff = count_diff;
                found_valid_mapping = true;
            }
        }
    }

    if (!found_valid_mapping) {
        // Build error message listing what we tried
        std::stringstream ss;
        ss << "No valid adjacency graph mapping found. Logical graph has " << logical_node_count << " nodes. ";
        ss << "Tried " << sorted_candidates.size() << " candidate groupings with sizes: ";
        for (size_t i = 0; i < sorted_candidates.size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << sorted_candidates[i].adjacency_graph.get_nodes().size();
        }
        TT_THROW("This system is not compatible with the following MGD: {}", ss.str());
    }

    return best_match;
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

    for (const auto& instance : grouping.instances()) {
        uint32_t instance_id = instance.id();
        instance_ids.push_back(instance_id);

        // Build GroupingItemInfo for backward compatibility with existing code
        GroupingItemInfo item_info;
        if (instance.has_asic_location()) {
            item_info.type = GroupingItemInfo::ItemType::ASIC_LOCATION;
            item_info.asic_location = static_cast<uint32_t>(instance.asic_location());
            info.items.push_back(item_info);
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
    // The adjacency graph represents the topology/connections between instances in this grouping.
    // Node IDs in the graph are instance IDs (from instance.id() fields).
    // This graph can be used by topology solvers to match logical graphs to physical groupings.
    if (grouping.has_all_to_all()) {
        const auto& all_to_all = grouping.all_to_all();
        info.adjacency_graph = build_all_to_all_graph(instance_ids, all_to_all.num_connections());
    } else if (grouping.has_row_major_mesh()) {
        const auto& row_major_mesh = grouping.row_major_mesh();
        std::vector<int32_t> dims(row_major_mesh.dims().begin(), row_major_mesh.dims().end());
        std::vector<proto::RowMajorMeshConnection::DimType> dim_types;
        dim_types.reserve(row_major_mesh.dim_types().size());
        for (int dt : row_major_mesh.dim_types()) {
            dim_types.push_back(static_cast<proto::RowMajorMeshConnection::DimType>(dt));
        }
        info.adjacency_graph = build_row_major_mesh_graph(instance_ids, dims, dim_types, info.name);
    } else if (grouping.has_custom()) {
        const auto& custom = grouping.custom();
        info.adjacency_graph = build_custom_connections_graph(instance_ids, custom);
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

GroupingInfo PhysicalGroupingDescriptor::find_best_meshes_grouping(
    uint32_t required_chips, const MeshGraphDescriptor& mesh_graph_descriptor, GlobalNodeId mesh_instance_id) const {
    // Get all "MESH" preset groupings only
    auto meshes_groupings = get_groupings_by_name("MESH");

    if (meshes_groupings.empty()) {
        TT_THROW("No 'MESH' groupings found in PhysicalGroupingDescriptor");
    }

    // Build adjacency graph from MGD mesh instance's device topology
    AdjacencyGraph<uint32_t> mesh_topology_graph =
        build_mgd_mesh_instance_adjacency(mesh_graph_descriptor, mesh_instance_id);
    size_t mesh_node_count = mesh_topology_graph.get_nodes().size();

    // Filter: Only consider groupings where asic_count >= required_chips
    std::vector<GroupingInfo> size_valid_candidates;
    for (const auto& grouping : meshes_groupings) {
        if (grouping.asic_count >= required_chips) {
            size_valid_candidates.push_back(grouping);
        }
    }

    if (size_valid_candidates.empty()) {
        // Return the largest grouping even if undersized (for error reporting)
        GroupingInfo largest = meshes_groupings[0];
        for (const auto& grouping : meshes_groupings) {
            if (grouping.asic_count > largest.asic_count) {
                largest = grouping;
            }
        }
        return largest;
    }

    // Priority 1: Exact ASIC count match (prefer these first)
    std::vector<GroupingInfo> exact_match_candidates;
    for (const auto& candidate : size_valid_candidates) {
        if (candidate.asic_count == required_chips) {
            exact_match_candidates.push_back(candidate);
        }
    }

    // If we have exact ASIC count matches, use topology solving to pick the best one
    if (!exact_match_candidates.empty()) {
        std::vector<GroupingInfo> topology_valid_exact_matches;
        MappingConstraints<uint32_t, uint32_t> constraints;  // No constraints - allow any mapping

        for (const auto& candidate : exact_match_candidates) {
            const auto& grouping_graph = candidate.adjacency_graph;
            size_t grouping_node_count = grouping_graph.get_nodes().size();

            // Skip if grouping graph is smaller than mesh graph (can't map)
            if (grouping_node_count < mesh_node_count) {
                continue;
            }

            // For single-instance groupings (empty adjacency graph), topology check not applicable
            if (grouping_node_count == 0) {
                topology_valid_exact_matches.push_back(candidate);
                continue;
            }

            // Try to solve the topology mapping
            auto mapping_result = solve_topology_mapping(
                mesh_topology_graph, grouping_graph, constraints, ConnectionValidationMode::RELAXED);

            if (mapping_result.success) {
                topology_valid_exact_matches.push_back(candidate);
            }
        }

        // If topology solving found valid matches, return the first one (they're all exact ASIC matches)
        if (!topology_valid_exact_matches.empty()) {
            return topology_valid_exact_matches[0];
        }

        // If no topology-valid exact matches, return first exact ASIC match (topology solving failed but ASIC count
        // matches)
        return exact_match_candidates[0];
    }

    // Priority 2: No exact ASIC count match - use topology solving to find best oversized match
    std::vector<GroupingInfo> topology_valid_candidates;
    MappingConstraints<uint32_t, uint32_t> constraints;  // No constraints - allow any mapping

    for (const auto& candidate : size_valid_candidates) {
        const auto& grouping_graph = candidate.adjacency_graph;
        size_t grouping_node_count = grouping_graph.get_nodes().size();

        // Skip if grouping graph is smaller than mesh graph (can't map)
        if (grouping_node_count < mesh_node_count) {
            continue;
        }

        // For single-instance groupings (empty adjacency graph), topology check not applicable
        if (grouping_node_count == 0) {
            topology_valid_candidates.push_back(candidate);
            continue;
        }

        // Try to solve the topology mapping
        auto mapping_result =
            solve_topology_mapping(mesh_topology_graph, grouping_graph, constraints, ConnectionValidationMode::RELAXED);

        if (mapping_result.success) {
            topology_valid_candidates.push_back(candidate);
        }
    }

    // If no topology-valid candidates, fall back to size-valid candidates (for error reporting)
    if (topology_valid_candidates.empty()) {
        // Build error message
        std::stringstream ss;
        ss << "No MESH groupings found that can map the mesh topology (" << mesh_node_count << " nodes). ";
        ss << "Tried " << size_valid_candidates.size() << " size-valid candidates: ";
        for (size_t i = 0; i < size_valid_candidates.size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << size_valid_candidates[i].name
               << " (nodes: " << size_valid_candidates[i].adjacency_graph.get_nodes().size() << ")";
        }
        TT_THROW("This system is not compatible with the following MGD: {}", ss.str());
    }

    // Priority 3: Closest oversized match (minimal waste) among topology-valid candidates
    GroupingInfo best_match = topology_valid_candidates[0];
    uint32_t min_waste = best_match.asic_count - required_chips;

    for (const auto& grouping : topology_valid_candidates) {
        uint32_t waste = grouping.asic_count - required_chips;
        if (waste < min_waste) {
            min_waste = waste;
            best_match = grouping;
        }
    }

    return best_match;
}

std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>>
PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(const MeshGraphDescriptor& mesh_graph_descriptor) const {
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> result;

    // ===== PHASE 0: Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts) =====
    // This step calculates required ASIC counts bottom-up and builds adjacency graphs
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos =
        build_mgd_to_grouping_info_map(mesh_graph_descriptor);

    // ===== PHASE 2: Match mesh types to "meshes" groupings =====
    // Get all unique mesh names
    auto all_names = mesh_graph_descriptor.all_names();

    for (const auto& name : all_names) {
        if (mesh_graph_descriptor.type_by_name(name) != "MESH") {
            continue;  // Skip non-mesh instances
        }

        // Get all instances with this name
        const auto& instance_ids = mesh_graph_descriptor.instances_by_name(name);

        // Get MGD grouping info (includes adjacency graph and ASIC count)
        const auto& mgd_grouping_info = mgd_grouping_infos.at("MESH").at(name);
        uint32_t required_chips = mgd_grouping_info.asic_count;

        // Find best matching "meshes" grouping using topology solving
        // Must contain at least required_chips AND pass topology mapping
        GroupingInfo best_meshes_grouping =
            find_best_meshes_grouping(required_chips, mesh_graph_descriptor, instance_ids[0]);

        // Store in result for all instances of this mesh type
        // Note: ASIC count validation is already done in find_best_meshes_grouping
        for (GlobalNodeId mesh_id : instance_ids) {
            const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_id);
            result[mesh_instance.type][mesh_instance.name] = best_meshes_grouping;
        }
    }

    // ===== PHASE 3: Match graph instances using adjacency graph matching =====
    // Group graph instances by type and name for matching
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<GlobalNodeId>>>
        graph_instances_by_type_name;

    for (GlobalNodeId graph_id : mesh_graph_descriptor.all_graphs()) {
        const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_id);
        graph_instances_by_type_name[graph_instance.type][graph_instance.name].push_back(graph_id);
    }

    // For each unique (type, name) combination, find matching grouping
    for (const auto& [graph_type, name_map] : graph_instances_by_type_name) {
        for (const auto& [graph_name, graph_ids] : name_map) {
            // Get MGD grouping info (includes adjacency graph and ASIC count)
            const auto& mgd_grouping_info = mgd_grouping_infos.at(graph_type).at(graph_name);
            const AdjacencyGraph<uint32_t>& logical_graph = mgd_grouping_info.adjacency_graph;
            uint32_t required_asics = mgd_grouping_info.asic_count;

            // Get all candidate groupings (skip base groupings like "meshes", "trays")
            std::vector<GroupingInfo> candidate_groupings;
            for (const auto& [grouping_name, groupings] : resolved_groupings_cache_) {
                if (grouping_name == "meshes" || grouping_name == "trays") {
                    continue;  // Skip base groupings
                }

                // Only consider groupings that have grouping references (higher-level groupings)
                for (const auto& grouping : groupings) {
                    bool has_grouping_ref = false;
                    for (const auto& item : grouping.items) {
                        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                            has_grouping_ref = true;
                            break;
                        }
                    }
                    if (has_grouping_ref) {
                        candidate_groupings.push_back(grouping);
                    }
                }
            }

            if (candidate_groupings.empty()) {
                TT_THROW(
                    "This system is not compatible with the following MGD: "
                    "No candidate groupings found for graph type '{}' name '{}'",
                    graph_type,
                    graph_name);
            }

            // Find best matching grouping using adjacency graph matching
            GroupingInfo matched_grouping = find_best_grouping_by_adjacency(logical_graph, candidate_groupings);

            // Validation: Verify matched grouping has sufficient ASIC count
            // For groupings with refs, the asic_count may be calculated from definition-time refs,
            // but required_asics is calculated from runtime-matched sub-instances, so we use the max
            uint32_t effective_asic_count = matched_grouping.asic_count;
            bool has_grouping_refs = false;
            for (const auto& item : matched_grouping.items) {
                if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                    has_grouping_refs = true;
                    break;
                }
            }

            if (has_grouping_refs && required_asics > 0) {
                // For groupings with refs, trust topology matching and use required_asics as effective count
                // since it's calculated from actual matched sub-instances
                effective_asic_count = std::max(effective_asic_count, required_asics);
            }

            if (effective_asic_count < required_asics) {
                TT_THROW(
                    "This system is not compatible with the following MGD: "
                    "Graph instance '{}' (type '{}') requires {} ASICs total from its components, "
                    "but the matched grouping '{}' has only {} ASICs",
                    graph_name,
                    graph_type,
                    required_asics,
                    matched_grouping.name,
                    effective_asic_count);
            }

            // Assign matched grouping to all graph instances with this type/name
            for (GlobalNodeId graph_id : graph_ids) {
                const auto& graph_instance = mesh_graph_descriptor.get_instance(graph_id);
                result[graph_instance.type][graph_instance.name] = matched_grouping;
            }
        }
    }

    // ===== VALIDATION: Verify sufficient ASIC counts for mesh instances =====
    // Note: Mesh matching already ensures sufficient ASIC count, but validate for completeness
    for (GlobalNodeId mesh_id : mesh_graph_descriptor.all_meshes()) {
        const auto& mesh_instance = mesh_graph_descriptor.get_instance(mesh_id);
        auto type_it = result.find(mesh_instance.type);
        if (type_it != result.end()) {
            auto name_it = type_it->second.find(mesh_instance.name);
            if (name_it != type_it->second.end()) {
                const auto& matched_grouping = name_it->second;
                uint32_t required_chips = mesh_graph_descriptor.get_chip_count(mesh_id);

                if (matched_grouping.asic_count < required_chips) {
                    TT_THROW(
                        "This system is not compatible with the following MGD: "
                        "Mesh instance '{}' requires {} chips, but the matched grouping '{}' has only {} ASICs",
                        mesh_instance.name,
                        required_chips,
                        matched_grouping.name,
                        matched_grouping.asic_count);
                }
            }
        }
    }

    return result;
}

void PhysicalGroupingDescriptor::validate_and_populate_preformed_groups_from_physical_system(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    // Get all ASIC descriptors
    const auto& asic_descriptors = physical_system_descriptor.get_asic_descriptors();

    if (asic_descriptors.empty()) {
        return;  // No ASICs to group
    }

    // Group ASICs by (hostname, tray_id) to form trays
    // Map: (hostname, tray_id) -> vector of (asic_id, asic_location)
    std::map<std::pair<std::string, uint32_t>, std::vector<std::pair<tt::tt_metal::AsicID, tt::tt_metal::ASICLocation>>>
        trays;

    for (const auto& [asic_id, asic_desc] : asic_descriptors) {
        uint32_t tray_id_value = *asic_desc.tray_id;
        trays[{asic_desc.host_name, tray_id_value}].push_back({asic_id, asic_desc.asic_location});
    }

    // Create TRAY groupings for each tray
    // Map: (hostname, tray_id) -> tray name (TRAY_1, TRAY_2, etc.)
    std::map<std::pair<std::string, uint32_t>, std::string> tray_name_map;

    // Group trays by hostname
    std::map<std::string, std::vector<std::pair<std::string, uint32_t>>> host_trays;

    for (const auto& [host_tray_pair, asics] : trays) {
        const auto& [hostname, tray_id] = host_tray_pair;
        host_trays[hostname].push_back({hostname, tray_id});
    }

    // Sort trays within each host by tray_id
    for (auto& [hostname, tray_list] : host_trays) {
        std::sort(tray_list.begin(), tray_list.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

        // Assign tray names (TRAY_1, TRAY_2, TRAY_3, TRAY_4)
        for (size_t i = 0; i < tray_list.size() && i < 4; ++i) {
            std::string tray_name = "TRAY_" + std::to_string(i + 1);
            tray_name_map[tray_list[i]] = tray_name;
        }
    }

    // Helper function to build expected tray grouping from physical system
    auto build_expected_tray_info =
        [&](const std::string& tray_type,
            const std::vector<std::pair<tt::tt_metal::AsicID, tt::tt_metal::ASICLocation>>& asics) -> GroupingInfo {
        GroupingInfo tray_info;
        tray_info.type = tray_type;  // Type identifier (TRAY_1, TRAY_2, etc.) - name will be set with unique ID
        tray_info.asic_count = static_cast<uint32_t>(asics.size());

        // Sort ASICs by location to ensure consistent ordering
        std::vector<std::pair<tt::tt_metal::AsicID, tt::tt_metal::ASICLocation>> sorted_asics = asics;
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

        // Empty adjacency graph (trays don't have connection topology defined)
        tray_info.adjacency_graph = AdjacencyGraph<uint32_t>();

        return tray_info;
    };

    // Helper function to validate tray grouping
    auto validate_tray_grouping = [](const GroupingInfo& expected, const GroupingInfo& actual) -> bool {
        if (expected.type != actual.type) {
            return false;
        }
        if (expected.asic_count != actual.asic_count) {
            return false;
        }
        if (expected.items.size() != actual.items.size()) {
            return false;
        }

        // Check that ASIC locations match
        for (size_t i = 0; i < expected.items.size(); ++i) {
            if (expected.items[i].type != GroupingItemInfo::ItemType::ASIC_LOCATION ||
                actual.items[i].type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
                return false;
            }
            if (expected.items[i].asic_location != actual.items[i].asic_location) {
                return false;
            }
        }

        return true;
    };

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

    // Process each tray: validate if exists, create if not
    for (const auto& [host_tray_pair, asics] : trays) {
        auto tray_name_it = tray_name_map.find(host_tray_pair);
        if (tray_name_it == tray_name_map.end()) {
            continue;  // Skip if tray name not assigned (more than 4 trays per host)
        }

        const std::string& tray_type = tray_name_it->second;  // e.g., "TRAY_1"

        // Build expected tray info from physical system (with type, not unique name yet)
        GroupingInfo expected_tray_info = build_expected_tray_info(tray_type, asics);

        // Check if tray grouping already exists (by type)
        auto existing_trays = resolved_groupings_cache_.find(tray_type);
        if (existing_trays != resolved_groupings_cache_.end() && !existing_trays->second.empty()) {
            // Validate existing grouping matches expected
            bool is_valid = false;
            for (const auto& existing_tray : existing_trays->second) {
                if (validate_tray_grouping(expected_tray_info, existing_tray)) {
                    is_valid = true;
                    break;
                }
            }

            if (!is_valid) {
                TT_THROW(
                    "TRAY grouping type '{}' already exists but does not match physical system. "
                    "Expected {} ASICs with locations matching physical system, but found different configuration.",
                    tray_type,
                    expected_tray_info.asic_count);
            }
            // Grouping exists and is valid, skip creation
            continue;
        }

        // Grouping doesn't exist, create it with unique name
        expected_tray_info.name = generate_unique_name(tray_type);
        expected_tray_info.type = tray_type;
        resolved_groupings_cache_[tray_type].push_back(expected_tray_info);
    }

    // Helper function to build expected host grouping from physical system
    auto build_expected_host_info =
        [&](const std::string& /* hostname */,
            const std::vector<std::pair<std::string, uint32_t>>& tray_list) -> GroupingInfo {
        GroupingInfo host_info;
        host_info.type = "HOSTS";  // Type identifier (name will be set with unique ID)
        host_info.asic_count = 0;

        // Create items referencing trays
        for (size_t i = 0; i < tray_list.size(); ++i) {
            auto tray_name_it = tray_name_map.find(tray_list[i]);
            if (tray_name_it == tray_name_map.end()) {
                continue;
            }

            const std::string& tray_name = tray_name_it->second;

            GroupingItemInfo item;
            item.type = GroupingItemInfo::ItemType::GROUPING_REF;
            item.grouping_name = tray_name;
            host_info.items.push_back(item);

            // Add ASIC count from tray
            auto tray_groupings = resolved_groupings_cache_[tray_name];
            if (!tray_groupings.empty()) {
                host_info.asic_count += tray_groupings[0].asic_count;
            }
        }

        // Empty adjacency graph (hosts don't have connection topology defined by default)
        host_info.adjacency_graph = AdjacencyGraph<uint32_t>();

        return host_info;
    };

    // Helper function to validate host grouping
    auto validate_host_grouping = [](const GroupingInfo& expected, const GroupingInfo& actual) -> bool {
        if (expected.type != actual.type) {
            return false;
        }
        if (expected.items.size() != actual.items.size()) {
            return false;
        }

        // Check that tray references match
        for (size_t i = 0; i < expected.items.size(); ++i) {
            if (expected.items[i].type != GroupingItemInfo::ItemType::GROUPING_REF ||
                actual.items[i].type != GroupingItemInfo::ItemType::GROUPING_REF) {
                return false;
            }
            if (expected.items[i].grouping_name != actual.items[i].grouping_name) {
                return false;
            }
        }

        return true;
    };

    // Process each host: validate if exists, create if not
    for (const auto& [hostname, tray_list] : host_trays) {
        if (tray_list.size() > 4) {
            continue;  // Skip hosts with more than 4 trays (not standard)
        }

        // Build expected host info from physical system
        GroupingInfo expected_host_info = build_expected_host_info(hostname, tray_list);

        // Check if HOSTS grouping already exists (by type)
        auto existing_hosts = resolved_groupings_cache_.find("HOSTS");
        if (existing_hosts != resolved_groupings_cache_.end() && !existing_hosts->second.empty()) {
            // Validate existing grouping matches expected
            bool is_valid = false;
            for (const auto& existing_host : existing_hosts->second) {
                if (validate_host_grouping(expected_host_info, existing_host)) {
                    is_valid = true;
                    break;
                }
            }

            if (!is_valid) {
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
            // Grouping exists and is valid, skip creation
            continue;
        }

        // Grouping doesn't exist, create it with unique name (checking against all existing names)
        expected_host_info.name = generate_unique_name("HOSTS");
        expected_host_info.type = "HOSTS";
        resolved_groupings_cache_["HOSTS"].push_back(expected_host_info);
    }
}

}  // namespace tt::tt_fabric
