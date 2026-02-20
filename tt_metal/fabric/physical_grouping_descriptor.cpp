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
// Helper function to assign corner orientations to items in a GroupingInfo based on mesh dimensions
// This is used for both PGD groupings (with multiple items) and MGD mesh instances (with a single item)
void assign_corner_orientations_to_grouping(GroupingInfo& info, const std::vector<int32_t>& dims) {
    using CO = GroupingItemInfo::CornerOrientation;

    if (dims.empty() || info.items.empty()) {
        return;
    }

    size_t total_items = info.items.size();

    if (dims.size() == 1) {
        // 1D mesh: single dimension
        int32_t length = dims[0];
        if (total_items == static_cast<size_t>(length)) {
            // First item: NW + SW (top-left + bottom-left)
            if (total_items > 0) {
                info.items[0].corners.push_back(CO::NW);
                info.items[0].corners.push_back(CO::SW);
            }
            // Last item: NE + SE (top-right + bottom-right)
            if (total_items > 1) {
                size_t last_idx = total_items - 1;
                info.items[last_idx].corners.push_back(CO::NE);
                info.items[last_idx].corners.push_back(CO::SE);
            }
        }
    } else if (dims.size() >= 2) {
        int32_t rows = dims[0];
        int32_t cols = dims[1];

        if (rows == 1 && cols == 1) {
            // 1x1 mesh: single item has all 4 orientations
            if (total_items == 1) {
                info.items[0].corners.push_back(CO::NW);
                info.items[0].corners.push_back(CO::NE);
                info.items[0].corners.push_back(CO::SW);
                info.items[0].corners.push_back(CO::SE);
            }
        } else if (rows == 1) {
            // 1D row mesh (1xN): first item has NW+SW, last item has NE+SE
            if (total_items == static_cast<size_t>(cols)) {
                if (total_items > 0) {
                    info.items[0].corners.push_back(CO::NW);
                    info.items[0].corners.push_back(CO::SW);
                }
                if (total_items > 1) {
                    size_t last_idx = total_items - 1;
                    info.items[last_idx].corners.push_back(CO::NE);
                    info.items[last_idx].corners.push_back(CO::SE);
                }
            }
        } else if (cols == 1) {
            // 1D column mesh (Nx1): first item has NW+NE, last item has SW+SE
            if (total_items == static_cast<size_t>(rows)) {
                if (total_items > 0) {
                    info.items[0].corners.push_back(CO::NW);
                    info.items[0].corners.push_back(CO::NE);
                }
                if (total_items > 1) {
                    size_t last_idx = total_items - 1;
                    info.items[last_idx].corners.push_back(CO::SW);
                    info.items[last_idx].corners.push_back(CO::SE);
                }
            }
        } else {
            // 2D mesh: standard 4 corners
            if (total_items == static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
                // Multiple items: assign corners to specific items based on position
                size_t nw_idx = 0;
                size_t ne_idx = static_cast<size_t>(cols) - 1;
                size_t sw_idx = (static_cast<size_t>(rows) - 1) * static_cast<size_t>(cols);
                size_t se_idx =
                    ((static_cast<size_t>(rows) - 1) * static_cast<size_t>(cols)) + (static_cast<size_t>(cols) - 1);

                // Set corner orientations based on row-major mesh position
                if (nw_idx < total_items) {
                    info.items[nw_idx].corners.push_back(CO::NW);
                }
                if (ne_idx < total_items && ne_idx != nw_idx) {
                    info.items[ne_idx].corners.push_back(CO::NE);
                }
                if (sw_idx < total_items && sw_idx != nw_idx && sw_idx != ne_idx) {
                    info.items[sw_idx].corners.push_back(CO::SW);
                }
                if (se_idx < total_items && se_idx != nw_idx && se_idx != ne_idx && se_idx != sw_idx) {
                    info.items[se_idx].corners.push_back(CO::SE);
                }
            } else if (total_items == 1) {
                // Single item representing entire mesh (e.g., MGD mesh instance)
                // Assign all 4 corners to indicate it's a complete 2D mesh
                info.items[0].corners.push_back(CO::NW);
                info.items[0].corners.push_back(CO::NE);
                info.items[0].corners.push_back(CO::SW);
                info.items[0].corners.push_back(CO::SE);
            }
        }
    }
}

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
// Always uses LINE connectivity (no wrap-around) with configurable connections per edge
AdjacencyGraph<uint32_t> build_row_major_mesh_graph(
    const std::vector<uint32_t>& instance_ids,
    const std::vector<int32_t>& dims,
    const std::string& grouping_name = "",
    uint32_t connections_per_edge = 1) {
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
                    // Add multiple connections per edge if specified
                    for (uint32_t conn = 0; conn < connections_per_edge; ++conn) {
                        adj_map[instance_ids[idx]].push_back(instance_ids[neighbor_idx]);
                        adj_map[instance_ids[neighbor_idx]].push_back(instance_ids[idx]);
                    }
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
    std::vector<std::string> all_errors = static_validate(temp_proto);

    TT_FATAL(
        all_errors.empty(),
        "Failed to validate PhysicalGroupingDescriptor textproto: \n{}",
        get_validation_report(all_errors));

    proto_ = std::make_shared<proto::PhysicalGroupings>(temp_proto);

    populate();

    // Collect grouping validation errors and add to the same error vector
    instance_validate(all_errors);

    TT_FATAL(
        all_errors.empty(),
        "Failed to validate PhysicalGroupingDescriptor textproto: \n{}",
        get_validation_report(all_errors));
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
        assign_corner_orientations_to_grouping(info, dims);
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

std::vector<GroupingInfo> PhysicalGroupingDescriptor::get_groupings_by_type(const std::string& grouping_type) const {
    auto it = resolved_groupings_cache_.find(grouping_type);
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
        if (used_names.contains(current_name)) {
            // Generate unique name with ID suffix
            uint32_t& counter = name_counters[current_name];
            std::string unique_name;
            do {
                counter++;
                unique_name = fmt::format("{}_{}", current_name, counter);
            } while (used_names.contains(unique_name));

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

        if (names.contains(name)) {
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
            if (preset_names.contains(item.grouping_name)) {
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
                if (groupings_by_name.contains(item.grouping_name) || !preset_names.contains(item.grouping_name)) {
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
                    if (!preset_names.contains(item.grouping_name)) {
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

    // Step 5: Store resolved groupings
    // Note: Cycle detection is now handled by validate_no_cycles() in grouping_validate()
    resolved_groupings_cache_ = std::move(groupings_by_name);
}

void PhysicalGroupingDescriptor::grouping_validate() const {
    std::vector<std::string> errors;
    instance_validate(errors);

    // Throw if any errors found
    if (!errors.empty()) {
        std::string error_msg = "Grouping validation failed:\n";
        for (size_t i = 0; i < errors.size(); ++i) {
            error_msg += fmt::format("  {}. {}\n", i + 1, errors[i]);
        }
        TT_THROW("{}", error_msg);
    }
}

void PhysicalGroupingDescriptor::instance_validate(std::vector<std::string>& errors) const {
    validate_leaf_groupings(errors);
    validate_asic_location_usage(errors);
    validate_no_cycles(errors);
    validate_instance_counts(errors);
}

void PhysicalGroupingDescriptor::validate_leaf_groupings(std::vector<std::string>& errors) const {
    // Build dependency graph and identify leaf vs non-leaf groupings
    std::unordered_map<std::string, bool> has_asic_locations;  // grouping -> true if uses ASIC locations
    std::unordered_map<std::string, bool> has_grouping_refs;   // grouping -> true if uses grouping refs
    std::unordered_set<std::string> all_grouping_types;

    // First pass: identify all groupings and their characteristics
    for (const auto& [type, groupings] : resolved_groupings_cache_) {
        all_grouping_types.insert(type);
        bool has_asic = false;
        bool has_refs = false;

        for (const auto& grouping : groupings) {
            for (const auto& item : grouping.items) {
                if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                    has_asic = true;
                } else if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                    has_refs = true;
                }
            }
        }

        has_asic_locations[type] = has_asic;
        has_grouping_refs[type] = has_refs;
    }

    // Validation: At least one leaf grouping uses ASIC locations
    // A leaf grouping is one that has ASIC locations and no grouping references
    bool has_leaf_with_asic = false;
    for (const auto& type : all_grouping_types) {
        if (has_asic_locations[type] && !has_grouping_refs[type]) {
            has_leaf_with_asic = true;
            break;
        }
    }
    if (!has_leaf_with_asic) {
        errors.push_back("At least one leaf grouping must use ASIC locations");
    }
}

void PhysicalGroupingDescriptor::validate_asic_location_usage(std::vector<std::string>& errors) const {
    // Build dependency graph and identify groupings
    std::unordered_map<std::string, bool> has_asic_locations;  // grouping -> true if uses ASIC locations
    std::unordered_map<std::string, bool> has_grouping_refs;   // grouping -> true if uses grouping refs
    std::unordered_set<std::string> all_grouping_types;

    // First pass: identify all groupings and their characteristics
    for (const auto& [type, groupings] : resolved_groupings_cache_) {
        all_grouping_types.insert(type);
        bool has_asic = false;
        bool has_refs = false;

        for (const auto& grouping : groupings) {
            for (const auto& item : grouping.items) {
                if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                    has_asic = true;
                } else if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                    has_refs = true;
                }
            }
        }

        has_asic_locations[type] = has_asic;
        has_grouping_refs[type] = has_refs;
    }

    // Validation: Only leaf groupings should use ASIC locations, others should not
    // A grouping that has grouping references should not have ASIC locations
    for (const auto& type : all_grouping_types) {
        if (has_grouping_refs[type] && has_asic_locations[type]) {
            errors.push_back(fmt::format(
                "Grouping '{}' uses ASIC locations but also has grouping references. Only leaf groupings should use "
                "ASIC locations",
                type));
        }
    }
}

void PhysicalGroupingDescriptor::validate_no_cycles(std::vector<std::string>& errors) const {
    // Build dependency graph
    std::unordered_map<std::string, std::set<std::string>> dependencies;  // grouping -> set of dependencies
    std::unordered_set<std::string> all_grouping_types;

    for (const auto& [type, groupings] : resolved_groupings_cache_) {
        all_grouping_types.insert(type);
        for (const auto& grouping : groupings) {
            for (const auto& item : grouping.items) {
                if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                    dependencies[type].insert(item.grouping_name);
                }
            }
        }
    }

    // Use DFS to detect cycles
    std::unordered_map<std::string, int> color;  // 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
    for (const auto& type : all_grouping_types) {
        color[type] = 0;
    }

    std::function<bool(const std::string&)> has_cycle = [&](const std::string& node) -> bool {
        if (color[node] == 1) {
            // Gray node - cycle detected
            return true;
        }
        if (color[node] == 2) {
            // Black node - already processed
            return false;
        }

        color[node] = 1;  // Mark as gray (visiting)

        // Check all dependencies
        auto deps_it = dependencies.find(node);
        if (deps_it != dependencies.end()) {
            for (const auto& dep : deps_it->second) {
                if (all_grouping_types.contains(dep)) {
                    if (has_cycle(dep)) {
                        return true;
                    }
                }
            }
        }

        color[node] = 2;  // Mark as black (visited)
        return false;
    };

    for (const auto& type : all_grouping_types) {
        if (color[type] == 0) {
            if (has_cycle(type)) {
                errors.push_back(
                    fmt::format("Circular dependencies detected in grouping hierarchy involving '{}'", type));
                break;  // Only report one cycle
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_instance_counts(std::vector<std::string>& errors) const {
    // Set of preset names that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_names = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    // Validation: all groupings should have ASIC counts > 0
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
                    errors.push_back(fmt::format("Grouping '{}' has zero ASIC count after resolution", name));
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_required_groupings(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
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

    // TRAY, HOST, and MESH definitions are no longer required - validation removed
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

        // Validate instance count - all groupings must have at least 1 instance
        if (instance_count < 1) {
            errors.push_back(fmt::format(
                "Grouping '{}' (type '{}') has {} instances; all groupings must have at least 1 instance",
                name,
                type,
                instance_count));
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

namespace {

// -----------------------------------------------------------------------------
// Phase 0-2: MGD to GroupingInfo conversion helpers
// -----------------------------------------------------------------------------

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
    auto result = build_row_major_mesh_graph(asic_ids, device_dims, "");

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
        assign_corner_orientations_to_grouping(grouping_info, device_dims);

        // Store keyed by mesh definition name (not instance key)
        mgd_grouping_infos[mesh_type][mesh_name] = std::move(grouping_info);
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
    const std::unordered_map<std::string, std::vector<GroupingInfo>>& resolved_groupings_cache_,
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
    for (const auto& [pgd_type, pgd_groupings] : resolved_groupings_cache_) {
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

ValidGroupingsMap PhysicalGroupingDescriptor::get_valid_groupings_for_mgd(
    const MeshGraphDescriptor& mesh_graph_descriptor) const {
    ValidGroupingsMap result;

    // ===== PHASE 0: Convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts) =====
    // This step calculates required ASIC counts bottom-up and builds adjacency graphs
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> mgd_grouping_infos =
        build_mgd_to_grouping_info_map(mesh_graph_descriptor);

    // ===== PHASE 1: Build flattened adjacency graphs for all mesh group infos =====
    // Each possibility from build_flattened_adjacency_mesh gets a uniquified key (name_0, name_1, ...)
    std::unordered_map<std::string, GroupingInfo> mesh_flat_groupings;  // Lookup map for flattened GroupingInfo by key
    auto mesh_it = resolved_groupings_cache_.find("MESH");
    if (mesh_it != resolved_groupings_cache_.end()) {
        for (const auto& mesh_group_info : mesh_it->second) {
            auto meshes = build_flattened_adjacency_mesh(mesh_group_info);
            for (size_t i = 0; i < meshes.size(); ++i) {
                std::string uniquified_key = mesh_group_info.name + "_" + std::to_string(i);
                mesh_flat_groupings[uniquified_key] = std::move(meshes[i]);
            }
        }
    } else {
        TT_THROW("Internal error: MESH grouping not found in resolved_groupings_cache_");
    }

    // ===== PHASE 2: Match MESH mgd groupings to MESH groupings =====
    // For each MGD mesh instance, find all valid PGD mesh groupings that can contain it
    for (const auto& [mgd_instance_key, mgd_mesh_grouping] : mgd_grouping_infos["MESH"]) {
        const std::string& instance_name = mgd_instance_key;  // Use unique instance key (includes mesh_id)
        const GroupingInfo& mgd_grouping_info = mgd_mesh_grouping;
        const std::string& instance_type = mgd_grouping_info.type;  // Should be "MESH"

        // Required nodes from MGD adjacency graph (this represents the topology pattern to match)
        size_t required_nodes = mgd_grouping_info.adjacency_graph.get_nodes().size();

        // Group valid candidates by node difference (map is ordered by key ascending)
        std::map<size_t, std::vector<std::string>> candidates_by_diff;
        for (const auto& [name, grouping_info] : mesh_flat_groupings) {
            size_t n = grouping_info.adjacency_graph.get_nodes().size();
            if (n >= required_nodes) {
                candidates_by_diff[n - required_nodes].push_back(name);
            }
        }

        // Process difference levels from closest to farthest; stop at first level with any match
        std::vector<std::string> best_matches;
        for (const auto& [node_diff, names] : candidates_by_diff) {
            (void)node_diff;
            for (const std::string& name : names) {
                const auto& grouping_info = mesh_flat_groupings.at(name);
                // NOTE: If we ever want to support mixed type topologies, we need to add constraints to match the types
                auto mapping_result = solve_topology_mapping<uint32_t, uint32_t>(
                    mgd_grouping_info.adjacency_graph,
                    grouping_info.adjacency_graph,
                    {},
                    ConnectionValidationMode::STRICT,
                    true);
                if (mapping_result.success) {
                    best_matches.push_back(name);
                }
            }
            if (!best_matches.empty()) {
                break;  // Found matches at this (best) level
            }
        }

        // Store all best matches (add all entries that are possible)
        if (best_matches.empty()) {
            // No match found - use the MGD grouping info itself
            result[instance_type][instance_name].push_back(mgd_grouping_info);
        } else {
            for (const std::string& match_name : best_matches) {
                // Look up the GroupingInfo from lookup map
                auto lookup_it = mesh_flat_groupings.find(match_name);
                if (lookup_it != mesh_flat_groupings.end()) {
                    result[instance_type][instance_name].push_back(lookup_it->second);
                }
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

// =============================================================================
// build_flattened_adjacency_mesh – flattened ASIC-level mesh from hierarchy
// =============================================================================

namespace {

// Infer row_major_mesh dims [rows, cols] from items' corner orientations.
// Returns empty if corners don't indicate a row_major layout (fallback to [1, n]).
std::vector<int32_t> infer_dims_from_corners(const GroupingInfo& g) {
    using CO = GroupingItemInfo::CornerOrientation;
    const size_t n = g.items.size();
    if (n == 0) {
        return {};
    }

    auto has = [&](size_t i, CO c) {
        const auto& corners = g.items[i].corners;
        return std::find(corners.begin(), corners.end(), c) != corners.end();
    };

    if (n == 1) {
        if (has(0, CO::NW) && has(0, CO::SE)) {
            return {1, 1};
        }
        return {};
    }

    bool first_has_nw_sw = has(0, CO::NW) && has(0, CO::SW);
    bool first_has_nw_ne = has(0, CO::NW) && has(0, CO::NE);
    bool last_has_ne_se = has(n - 1, CO::NE) && has(n - 1, CO::SE);
    bool last_has_sw_se = has(n - 1, CO::SW) && has(n - 1, CO::SE);

    if (first_has_nw_sw && last_has_ne_se) {
        return {1, static_cast<int32_t>(n)};
    }
    if (first_has_nw_ne && last_has_sw_se) {
        return {static_cast<int32_t>(n), 1};
    }

    size_t nw_idx = n, ne_idx = n, sw_idx = n, se_idx = n;
    for (size_t i = 0; i < n; ++i) {
        if (has(i, CO::NW) && nw_idx == n) {
            nw_idx = i;
        }
        if (has(i, CO::NE) && ne_idx == n) {
            ne_idx = i;
        }
        if (has(i, CO::SW) && sw_idx == n) {
            sw_idx = i;
        }
        if (has(i, CO::SE) && se_idx == n) {
            se_idx = i;
        }
    }
    if (nw_idx >= n || ne_idx >= n || sw_idx >= n || se_idx >= n) {
        return {};
    }
    int32_t cols = static_cast<int32_t>(ne_idx - nw_idx + 1);
    if (cols <= 0 || n % static_cast<size_t>(cols) != 0) {
        return {};
    }
    int32_t rows = static_cast<int32_t>(n / static_cast<size_t>(cols));
    return {rows, cols};
}

enum class CardinalDirection { North, South, East, West };
enum class AdjacencyDirection { A_LEFT_OF_B, A_ABOVE_B, A_RIGHT_OF_B, A_BELOW_B };

// Metadata for flattened mesh nodes
struct NodeMetadata {
    uint32_t tray_id = 0;
    std::vector<std::string> grouping_path;
};

struct FlattenedMesh {
    AdjacencyGraph<uint32_t> graph;
    std::vector<int32_t> dims;  // Always [rows, cols]
    std::vector<uint32_t> nodes_row_major;
    std::unordered_map<uint32_t, NodeMetadata> node_metadata;  // Maps node ID to metadata
    std::vector<FlattenedMesh> sub_meshes;  // Empty for leaf meshes

    // Extract nodes along a cardinal edge (North/South/East/West)
    // For leaf meshes: extracts from nodes_row_major based on 2D grid position
    // For compound meshes: recursively extracts from sub-meshes on the corresponding edge
    std::vector<uint32_t> get_edge(CardinalDirection dir) const {
        constexpr int32_t MIN_DIMS_FOR_2D = 2;
        constexpr int32_t FIRST_ROW = 0;
        constexpr int32_t FIRST_COL = 0;

        if (sub_meshes.empty()) {
            // Leaf mesh: extract nodes from nodes_row_major based on edge direction
            const bool is_valid = !nodes_row_major.empty() && dims.size() >= MIN_DIMS_FOR_2D;
            if (!is_valid) {
                return {};
            }

            const int32_t rows = dims[0];
            const int32_t cols = dims[1];
            const int32_t num_nodes = static_cast<int32_t>(nodes_row_major.size());
            std::vector<uint32_t> edge_nodes;

            // Helper to safely add node at index if within bounds
            auto add_node_at_index = [&](int32_t idx) {
                if (idx < num_nodes) {
                    edge_nodes.push_back(nodes_row_major[idx]);
                }
            };

            switch (dir) {
                case CardinalDirection::North:
                    // Top row: indices 0 to cols-1
                    for (int32_t col = FIRST_COL; col < cols && col < num_nodes; ++col) {
                        add_node_at_index(col);
                    }
                    break;
                case CardinalDirection::South:
                    // Bottom row: indices (rows-1)*cols to rows*cols-1
                    {
                        const int32_t bottom_row_start = (rows - 1) * cols;
                        const int32_t bottom_row_end = rows * cols;
                        for (int32_t col = bottom_row_start; col < bottom_row_end && col < num_nodes; ++col) {
                            add_node_at_index(col);
                        }
                    }
                    break;
                case CardinalDirection::West:
                    // Left column: indices 0, cols, 2*cols, ..., (rows-1)*cols
                    for (int32_t row = FIRST_ROW; row < rows; ++row) {
                        add_node_at_index(row * cols);
                    }
                    break;
                case CardinalDirection::East:
                    // Right column: indices cols-1, 2*cols-1, ..., rows*cols-1
                    {
                        const int32_t right_col_offset = cols - 1;
                        for (int32_t row = FIRST_ROW; row < rows; ++row) {
                            add_node_at_index((row * cols) + right_col_offset);
                        }
                    }
                    break;
            }
            return edge_nodes;
        }

        // Compound mesh: collect edges from sub-meshes that lie on the requested edge
        if (dims.size() < MIN_DIMS_FOR_2D) {
            return {};
        }

        const int32_t item_rows = dims[0];
        const int32_t item_cols = dims[1];
        const int32_t total_items = item_rows * item_cols;
        const int32_t num_sub_meshes = static_cast<int32_t>(sub_meshes.size());
        std::vector<uint32_t> edge_nodes;

        for (int32_t item_idx = 0; item_idx < total_items && item_idx < num_sub_meshes; ++item_idx) {
            const int32_t row = item_idx / item_cols;
            const int32_t col = item_idx % item_cols;

            // Check if this sub-mesh is on the requested edge
            const bool is_on_north_edge = (dir == CardinalDirection::North) && (row == FIRST_ROW);
            const bool is_on_south_edge = (dir == CardinalDirection::South) && (row == item_rows - 1);
            const bool is_on_west_edge = (dir == CardinalDirection::West) && (col == FIRST_COL);
            const bool is_on_east_edge = (dir == CardinalDirection::East) && (col == item_cols - 1);
            const bool is_on_requested_edge =
                is_on_north_edge || is_on_south_edge || is_on_west_edge || is_on_east_edge;

            if (is_on_requested_edge) {
                // Recursively get edge from this sub-mesh and append
                const auto sub_edge = sub_meshes[item_idx].get_edge(dir);
                edge_nodes.insert(edge_nodes.end(), sub_edge.begin(), sub_edge.end());
            }
        }
        return edge_nodes;
    }
};

// Join two adjacent meshes by connecting their corresponding boundary edges
// Maps adjacency direction to the cardinal edges that should be connected
void join_two_adjacent_meshes(
    std::map<uint32_t, std::set<uint32_t>>& adj_set,
    const FlattenedMesh& mesh_a,
    const FlattenedMesh& mesh_b,
    AdjacencyDirection adjacency_dir) {
    // Map adjacency direction to the cardinal edges to connect:
    // - A_LEFT_OF_B: connect A's East edge to B's West edge
    // - A_ABOVE_B: connect A's South edge to B's North edge
    // - A_RIGHT_OF_B: connect A's West edge to B's East edge
    // - A_BELOW_B: connect A's North edge to B's South edge
    static const std::pair<CardinalDirection, CardinalDirection> edge_mapping[] = {
        {CardinalDirection::East, CardinalDirection::West},    // A_LEFT_OF_B
        {CardinalDirection::South, CardinalDirection::North},  // A_ABOVE_B
        {CardinalDirection::West, CardinalDirection::East},    // A_RIGHT_OF_B
        {CardinalDirection::North, CardinalDirection::South}   // A_BELOW_B
    };

    const auto [edge_a_dir, edge_b_dir] = edge_mapping[static_cast<int>(adjacency_dir)];
    const auto edge_a_nodes = mesh_a.get_edge(edge_a_dir);
    const auto edge_b_nodes = mesh_b.get_edge(edge_b_dir);

    // Helper to convert CardinalDirection to string
    auto dir_to_string = [](CardinalDirection dir) -> const char* {
        switch (dir) {
            case CardinalDirection::North: return "North";
            case CardinalDirection::South: return "South";
            case CardinalDirection::East: return "East";
            case CardinalDirection::West: return "West";
            default: return "Unknown";
        }
    };

    // Error if mesh has no directions (empty edges)
    TT_FATAL(
        !edge_a_nodes.empty(),
        "Mesh A has no {} edge (no directions available). Mesh must have valid 2D dimensions to determine edges.",
        dir_to_string(edge_a_dir));
    TT_FATAL(
        !edge_b_nodes.empty(),
        "Mesh B has no {} edge (no directions available). Mesh must have valid 2D dimensions to determine edges.",
        dir_to_string(edge_b_dir));

    // Error if mesh boundaries don't match (bigger or smaller than expected)
    TT_FATAL(
        edge_a_nodes.size() == edge_b_nodes.size(),
        "Mesh boundary size mismatch: mesh A has {} nodes on {} edge, mesh B has {} nodes on {} edge. Meshes must have "
        "matching boundary sizes.",
        edge_a_nodes.size(),
        dir_to_string(edge_a_dir),
        edge_b_nodes.size(),
        dir_to_string(edge_b_dir));

    // Connect corresponding nodes on the two edges (1-to-1 mapping)
    for (size_t i = 0; i < edge_a_nodes.size(); ++i) {
        adj_set[edge_a_nodes[i]].insert(edge_b_nodes[i]);
        adj_set[edge_b_nodes[i]].insert(edge_a_nodes[i]);
    }
}

// Normalize dimensions to always be [rows, cols] format
// - Empty dims: default to single row [1, total_items]
// - Single dim: treat as column count [1, dims[0]]
// - Two dims: already normalized [rows, cols]
std::vector<int32_t> normalize_dims(const std::vector<int32_t>& dims, size_t total_items) {
    constexpr int32_t SINGLE_ROW = 1;
    constexpr size_t SINGLE_DIM = 1;

    if (dims.empty()) {
        return {SINGLE_ROW, static_cast<int32_t>(total_items)};
    }
    if (dims.size() == SINGLE_DIM) {
        return {SINGLE_ROW, dims[0]};
    }
    // Already in [rows, cols] format (size == 2)
    return dims;
}

// Join multiple meshes arranged in a 2D grid into a single adjacency graph
// Algorithm:
//   1. Copy all internal edges from each mesh
//   2. Connect adjacent meshes along their shared boundaries (horizontal and vertical)
//   3. Convert from set-based to vector-based adjacency representation
AdjacencyGraph<uint32_t> join_mesh_level(const std::vector<FlattenedMesh>& meshes, const std::vector<int32_t>& dims) {
    constexpr size_t SINGLE_MESH = 1;
    constexpr int32_t ROW_INDEX = 0;
    constexpr int32_t COL_INDEX = 1;

    if (meshes.empty()) {
        return AdjacencyGraph<uint32_t>();
    }
    if (meshes.size() == SINGLE_MESH) {
        // Validate single mesh has valid dimensions
        const auto& mesh = meshes[0];
        TT_FATAL(
            mesh.dims.size() >= 2 && mesh.dims[ROW_INDEX] > 0 && mesh.dims[COL_INDEX] > 0,
            "Single mesh has invalid dimensions [rows={}, cols={}]. Mesh must have valid 2D dimensions.",
            !mesh.dims.empty() ? mesh.dims[ROW_INDEX] : 0,
            mesh.dims.size() > 1 ? mesh.dims[COL_INDEX] : 0);
        return mesh.graph;
    }

    const std::vector<int32_t> normalized_dims = normalize_dims(dims, meshes.size());
    const int32_t rows = normalized_dims[ROW_INDEX];
    const int32_t cols = normalized_dims[COL_INDEX];

    // Validate that we have the expected number of meshes for the layout
    const int32_t expected_mesh_count = rows * cols;
    TT_FATAL(
        static_cast<int32_t>(meshes.size()) == expected_mesh_count,
        "Mesh count mismatch: expected {} meshes for {}x{} layout, but got {}. Mesh layout must match the number of "
        "meshes.",
        expected_mesh_count,
        rows,
        cols,
        meshes.size());

    std::map<uint32_t, std::set<uint32_t>> adj_set;

    // Step 1: Copy all existing internal edges from each mesh
    // Also validate each mesh has valid dimensions
    for (size_t i = 0; i < meshes.size(); ++i) {
        const auto& mesh = meshes[i];
        TT_FATAL(
            mesh.dims.size() >= 2 && mesh.dims[ROW_INDEX] > 0 && mesh.dims[COL_INDEX] > 0,
            "Mesh {} has invalid dimensions [rows={}, cols={}]. Each mesh must have valid 2D dimensions.",
            i,
            !mesh.dims.empty() ? mesh.dims[ROW_INDEX] : 0,
            mesh.dims.size() > 1 ? mesh.dims[COL_INDEX] : 0);

        for (const auto& node : mesh.nodes_row_major) {
            for (const auto& neighbor : mesh.graph.get_neighbors(node)) {
                adj_set[node].insert(neighbor);
            }
        }
    }

    // Step 2: Connect adjacent meshes along their boundaries
    for (int32_t row = 0; row < rows; ++row) {
        for (int32_t col = 0; col < cols; ++col) {
            const size_t mesh_idx = (static_cast<size_t>(row) * static_cast<size_t>(cols)) + static_cast<size_t>(col);
            const bool has_right_neighbor = (col + 1 < cols);
            const bool has_bottom_neighbor = (row + 1 < rows);

            // Connect to right neighbor (horizontal connection)
            if (has_right_neighbor) {
                const size_t right_mesh_idx = mesh_idx + 1;
                join_two_adjacent_meshes(
                    adj_set, meshes[mesh_idx], meshes[right_mesh_idx], AdjacencyDirection::A_LEFT_OF_B);
            }

            // Connect to bottom neighbor (vertical connection)
            if (has_bottom_neighbor) {
                const size_t bottom_mesh_idx =
                    (static_cast<size_t>(row + 1) * static_cast<size_t>(cols)) + static_cast<size_t>(col);
                join_two_adjacent_meshes(
                    adj_set, meshes[mesh_idx], meshes[bottom_mesh_idx], AdjacencyDirection::A_ABOVE_B);
            }
        }
    }

    // Step 3: Convert from set-based to vector-based adjacency (required by AdjacencyGraph)
    std::map<uint32_t, std::vector<uint32_t>> adj_map;
    for (const auto& [node, neighbors] : adj_set) {
        adj_map[node] = std::vector<uint32_t>(neighbors.begin(), neighbors.end());
    }
    return AdjacencyGraph<uint32_t>(adj_map);
}

// Helper to extract tray ID from grouping name (e.g., "tray_1" -> 1, "TRAY_2" -> 2)
uint32_t extract_tray_id(const std::string& grouping_name) {
    const std::string lower_name = [&]() {
        std::string result = grouping_name;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }();

    if (lower_name.starts_with("tray_")) {
        try {
            return static_cast<uint32_t>(std::stoul(lower_name.substr(5)));
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

// Recursively build flattened meshes from a grouping item.
// Returns a vector of meshes - one per possibility (based on possible groupings that can be formed).
// Algorithm:
//   - Leaf (ASIC_LOCATION): create single-node mesh (one possibility)
//   - Single-item grouping: recurse for each possible sub_grouping, don't join - one entry per possibility
//   - Multi-item grouping: build sub-meshes for each item (each can have multiple possibilities),
//     then for each combination in the Cartesian product, join and add one entry
std::vector<FlattenedMesh> build_flattened_meshes_for_item(
    const GroupingItemInfo& item,
    uint32_t& next_global_id,
    const std::unordered_map<std::string, std::vector<GroupingInfo>>& cache,
    const PhysicalGroupingDescriptor* desc,
    const std::vector<std::string>& grouping_path = {}) {
    constexpr int32_t SINGLE_NODE_ROWS = 1;
    constexpr int32_t SINGLE_NODE_COLS = 1;
    constexpr size_t SINGLE_ITEM = 1;

    if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
        // Leaf case: single ASIC node - one possibility
        FlattenedMesh mesh;
        mesh.dims = {SINGLE_NODE_ROWS, SINGLE_NODE_COLS};
        const uint32_t node_id = next_global_id++;

        // Extract tray ID from grouping path
        NodeMetadata metadata;
        for (const auto& path_elem : grouping_path) {
            uint32_t tray_id = extract_tray_id(path_elem);
            if (tray_id > 0) {
                metadata.tray_id = tray_id;
                break;
            }
        }

        // Build grouping path: copy existing path and add ASIC location
        metadata.grouping_path = grouping_path;
        metadata.grouping_path.push_back("ASIC_LOCATION_" + std::to_string(item.asic_location));

        mesh.nodes_row_major = {node_id};
        mesh.node_metadata[node_id] = metadata;
        mesh.graph = AdjacencyGraph<uint32_t>({{node_id, {}}});
        return {std::move(mesh)};
    }

    // Compound case: resolve grouping reference - iterate over ALL possible groupings
    const auto cache_it = cache.find(item.grouping_name);
    TT_FATAL(cache_it != cache.end() && !cache_it->second.empty(), "Unknown grouping: {}", item.grouping_name);
    const std::vector<GroupingInfo>& possible_groupings = cache_it->second;

    std::vector<FlattenedMesh> all_results;
    for (const GroupingInfo& sub_grouping : possible_groupings) {
        // Build new grouping path using grouping name (not type)
        std::vector<std::string> new_path = grouping_path;
        new_path.push_back(sub_grouping.name);

        // Single-item grouping: recurse directly, add each possibility as its own entry (no join)
        if (sub_grouping.items.size() == SINGLE_ITEM) {
            std::vector<FlattenedMesh> sub_results =
                build_flattened_meshes_for_item(sub_grouping.items[0], next_global_id, cache, desc, new_path);
            for (FlattenedMesh& m : sub_results) {
                all_results.push_back(std::move(m));
            }
            continue;
        }

        // Multi-item grouping: build sub-meshes for each item (each returns vector of possibilities)
        std::vector<std::vector<FlattenedMesh>> sub_meshes_per_item;
        sub_meshes_per_item.reserve(sub_grouping.items.size());
        for (const auto& sub_item : sub_grouping.items) {
            sub_meshes_per_item.push_back(
                build_flattened_meshes_for_item(sub_item, next_global_id, cache, desc, new_path));
        }

        // Infer layout from corner orientations
        const std::vector<int32_t> inferred_dims = infer_dims_from_corners(sub_grouping);
        TT_FATAL(
            !inferred_dims.empty(),
            "Cannot infer mesh dimensions from grouping '{}' with {} items - grouping must use row_major_mesh "
            "connection type to determine layout. "
            "Groupings with all_to_all or custom connections cannot be flattened into a mesh.",
            sub_grouping.name,
            sub_grouping.items.size());

        const std::vector<int32_t> layout = normalize_dims(inferred_dims, sub_grouping.items.size());

        // Cartesian product: for each combination of one mesh from each item, join and add one entry
        std::vector<size_t> indices(sub_grouping.items.size(), 0);
        bool done = false;
        while (!done) {
            std::vector<FlattenedMesh> chosen;
            chosen.reserve(sub_grouping.items.size());
            for (size_t i = 0; i < sub_grouping.items.size(); ++i) {
                chosen.push_back(sub_meshes_per_item[i][indices[i]]);
            }

            FlattenedMesh mesh;
            mesh.graph = join_mesh_level(chosen, layout);
            mesh.dims = layout;
            mesh.sub_meshes = std::move(chosen);

            for (const auto& sub_mesh : mesh.sub_meshes) {
                mesh.nodes_row_major.insert(
                    mesh.nodes_row_major.end(), sub_mesh.nodes_row_major.begin(), sub_mesh.nodes_row_major.end());
            }
            all_results.push_back(std::move(mesh));

            // Advance to next combination
            size_t d = sub_grouping.items.size();
            while (d > 0) {
                d--;
                indices[d]++;
                if (indices[d] < sub_meshes_per_item[d].size()) {
                    break;
                }
                indices[d] = 0;
                if (d == 0) {
                    done = true;
                }
            }
        }
    }
    return all_results;
}

}  // unnamed namespace

// Top-level entry point: builds meshes for each item, returns a vector - one per possibility
std::vector<GroupingInfo> PhysicalGroupingDescriptor::build_flattened_adjacency_mesh(
    const GroupingInfo& grouping) const {
    if (grouping.items.empty()) {
        GroupingInfo result = grouping;
        result.adjacency_graph = AdjacencyGraph<uint32_t>();
        return {result};
    }

    // Build flattened meshes for each item (each returns vector of possibilities)
    uint32_t next_node_id = 0;
    std::vector<std::string> initial_path = {grouping.name};

    if (grouping.items.size() == 1) {
        // Single item: return all possibilities directly (no join)
        std::vector<FlattenedMesh> meshes = build_flattened_meshes_for_item(
            grouping.items[0], next_node_id, resolved_groupings_cache_, this, initial_path);
        std::vector<GroupingInfo> result;
        result.reserve(meshes.size());
        for (FlattenedMesh& m : meshes) {
            GroupingInfo info = grouping;
            info.adjacency_graph = std::move(m.graph);
            result.push_back(std::move(info));
        }
        return result;
    }

    // Multi-item: build per-item possibilities, then Cartesian product
    std::vector<std::vector<FlattenedMesh>> meshes_per_item;
    meshes_per_item.reserve(grouping.items.size());
    for (const auto& item : grouping.items) {
        meshes_per_item.push_back(
            build_flattened_meshes_for_item(item, next_node_id, resolved_groupings_cache_, this, initial_path));
    }

    const std::vector<int32_t> inferred_dims = infer_dims_from_corners(grouping);
    TT_FATAL(
        !inferred_dims.empty(),
        "Cannot infer mesh dimensions from grouping '{}' with {} items - grouping must use row_major_mesh "
        "connection type to determine layout. "
        "Groupings with all_to_all or custom connections cannot be flattened into a mesh.",
        grouping.name,
        grouping.items.size());

    const std::vector<int32_t> layout = normalize_dims(inferred_dims, grouping.items.size());

    std::vector<GroupingInfo> result;
    std::vector<size_t> indices(grouping.items.size(), 0);
    bool done = false;
    while (!done) {
        std::vector<FlattenedMesh> chosen;
        chosen.reserve(grouping.items.size());
        for (size_t i = 0; i < grouping.items.size(); ++i) {
            chosen.push_back(meshes_per_item[i][indices[i]]);
        }

        AdjacencyGraph<uint32_t> joined_graph = join_mesh_level(chosen, layout);

        GroupingInfo info = grouping;
        info.adjacency_graph = std::move(joined_graph);
        result.push_back(std::move(info));

        size_t d = grouping.items.size();
        while (d > 0) {
            d--;
            indices[d]++;
            if (indices[d] < meshes_per_item[d].size()) {
                break;
            }
            indices[d] = 0;
            if (d == 0) {
                done = true;
            }
        }
    }
    return result;
}

// ================================
// Validate current preformed mesh groups with the physical system descriptor
// ================================

namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::ASICLocation;
using tt::tt_metal::TrayID;

AdjacencyGraph<AsicID> build_physical_adjacency_graph_for_cluster(const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    std::map<AsicID, std::vector<AsicID>> adj_map;
    for (const std::string& hostname : psd.get_all_hostnames()) {
        const auto& asic_graph = psd.get_system_graph().asic_connectivity_graph.at(hostname);
        for (const auto& [asic_id, edges] : asic_graph) {
            std::vector<AsicID> neighbors;
            for (const auto& [dst_asic, _] : edges) {
                neighbors.push_back(dst_asic);
            }
            adj_map[asic_id] = std::move(neighbors);
        }
    }
    return AdjacencyGraph<AsicID>(adj_map);
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

}  // namespace

namespace {

// Helper function to solve the topology mapping
MappingResult<uint32_t, AsicID> solve_for_one_grouping_to_psd(
    const GroupingInfo& grouping_info,
    const AdjacencyGraph<AsicID>& physical_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& /* physical_system_descriptor */) {
    MappingConstraints<uint32_t, AsicID> constraints;
    // No trait constraints - solve without tray/location matching requirements
    return solve_topology_mapping(
        grouping_info.adjacency_graph, physical_graph, constraints, ConnectionValidationMode::RELAXED, true);
}

}  // namespace

std::unordered_set<tt::tt_metal::AsicID> PhysicalGroupingDescriptor::find_any_in_psd(
    const GroupingInfo& grouping,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    std::vector<std::string>* errors_out) const {
    auto physical_graph = build_physical_adjacency_graph_for_cluster(physical_system_descriptor);
    auto flat_meshes = build_flattened_adjacency_mesh(grouping);

    std::unordered_set<tt::tt_metal::AsicID> asic_ids;
    const GroupingInfo* last_mesh_tried = nullptr;
    MappingResult<uint32_t, AsicID> last_result;

    // Use the first flat mesh that actually fits
    for (const auto& flat_mesh : flat_meshes) {
        if (flat_mesh.adjacency_graph.get_nodes().empty()) {
            continue;
        }

        last_mesh_tried = &flat_mesh;
        auto result = solve_for_one_grouping_to_psd(flat_mesh, physical_graph, physical_system_descriptor);
        last_result = result;

        if (result.success) {
            for (const auto& [target_node, asic_id] : result.target_to_global) {
                asic_ids.insert(asic_id);
            }
            return asic_ids;
        }
    }

    if (flat_meshes.empty() || flat_meshes.front().adjacency_graph.get_nodes().empty()) {
        TT_THROW("Internal error: grouping produced empty graph");
    }

    if (errors_out != nullptr && last_mesh_tried != nullptr) {
        errors_out->push_back(build_pgd_mapping_failure_message(grouping.name, *last_mesh_tried, last_result));
    }

    return asic_ids;
}

}  // namespace tt::tt_fabric
