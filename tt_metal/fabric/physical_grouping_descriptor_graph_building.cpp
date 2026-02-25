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
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-logger/tt-logger.hpp>
#include <map>

#include <google/protobuf/text_format.h>

namespace tt::tt_fabric {

// Ensure tt::assert resolves correctly (not tt::tt_fabric::tt::assert)
using namespace ::tt::assert;

// Helper function to iterate through Cartesian product of multiple vectors.
// Calls callback(indices) for each combination, where indices[i] is the index into the i-th vector.
template <typename Callback>
void iterate_cartesian_product(const std::vector<size_t>& sizes, Callback callback) {
    if (sizes.empty()) {
        return;
    }

    std::vector<size_t> indices(sizes.size(), 0);
    bool done = false;

    while (!done) {
        callback(indices);

        // Advance to next combination (like an odometer)
        size_t d = sizes.size();
        while (d > 0) {
            d--;
            indices[d]++;
            if (indices[d] < sizes[d]) {
                break;
            }
            indices[d] = 0;
            if (d == 0) {
                done = true;
            }
        }
    }
}

namespace {

using tt::tt_fabric::AdjacencyGraph;
using tt::tt_fabric::iterate_cartesian_product;

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

// Helper function to assign corner orientations to grouping items based on mesh dimensions
// Implemented as a private static class method
void PhysicalGroupingDescriptor::assign_corner_orientations_to_grouping(
    GroupingInfo& info, const std::vector<int32_t>& dims) {
    using CO = tt::tt_fabric::GroupingItemInfo::CornerOrientation;

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

GroupingInfo PhysicalGroupingDescriptor::convert_grouping_to_info(const proto::Grouping& grouping) const {
    GroupingInfo info;

    // Get grouping name (mandatory field) and type
    info.name = PhysicalGroupingDescriptor::get_grouping_name(grouping);
    info.type = PhysicalGroupingDescriptor::get_grouping_type_string(grouping);

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
            item_info.type = tt::tt_fabric::GroupingItemInfo::ItemType::ASIC_LOCATION;
            item_info.asic_location = ::tt::tt_metal::ASICLocation{static_cast<uint32_t>(instance.asic_location())};
            info.items.push_back(item_info);
            asic_locations.push_back(*item_info.asic_location);
            has_asic_locations = true;
        } else if (instance.has_grouping_ref()) {
            item_info.type = tt::tt_fabric::GroupingItemInfo::ItemType::GROUPING_REF;
            const auto& ref = instance.grouping_ref();
            if (ref.has_preset_type()) {
                // Convert preset_type enum to string and populate tray_id for TRAY_* types
                switch (ref.preset_type()) {
                    case proto::TRAY_1:
                        item_info.grouping_name = "TRAY_1";
                        item_info.tray_id = ::tt::tt_metal::TrayID{1};
                        break;
                    case proto::TRAY_2:
                        item_info.grouping_name = "TRAY_2";
                        item_info.tray_id = ::tt::tt_metal::TrayID{2};
                        break;
                    case proto::TRAY_3:
                        item_info.grouping_name = "TRAY_3";
                        item_info.tray_id = ::tt::tt_metal::TrayID{3};
                        break;
                    case proto::TRAY_4:
                        item_info.grouping_name = "TRAY_4";
                        item_info.tray_id = ::tt::tt_metal::TrayID{4};
                        break;
                    case proto::HOSTS:
                        item_info.grouping_name = "HOSTS";
                        // tray_id remains 0 (default)
                        break;
                    case proto::MESH:
                        item_info.grouping_name = "MESH";
                        // tray_id remains 0 (default)
                        break;
                    default:
                        // tray_id remains 0 (default)
                        break;
                }
            } else if (ref.has_custom_type()) {
                item_info.grouping_name = ref.custom_type();
                // tray_id remains 0 (default) for custom types
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
        PhysicalGroupingDescriptor::assign_corner_orientations_to_grouping(info, dims);
    } else if (grouping.has_custom()) {
        const auto& custom = grouping.custom();
        info.adjacency_graph = build_custom_connections_graph(node_ids, custom);
    } else {
        // No connection specified - empty adjacency graph (instances are not connected)
        info.adjacency_graph = tt::tt_fabric::AdjacencyGraph<uint32_t>();
    }

    return info;
}

}  // namespace tt::tt_fabric

namespace {

// Infer row_major_mesh dims [rows, cols] from items' corner orientations.
// Returns empty if corners don't indicate a row_major layout (fallback to [1, n]).
std::vector<int32_t> infer_dims_from_corners(const tt::tt_fabric::GroupingInfo& g) {
    using CO = tt::tt_fabric::GroupingItemInfo::CornerOrientation;
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
    ::tt::tt_metal::TrayID tray_id{0};
    std::vector<std::string> grouping_path;
};

struct FlattenedMesh {
    tt::tt_fabric::AdjacencyGraph<uint32_t> graph;
    std::vector<int32_t> dims;  // Always [rows, cols]
    std::vector<uint32_t> nodes_row_major;
    std::unordered_map<uint32_t, NodeMetadata> node_metadata;  // Maps node ID to metadata
    std::vector<FlattenedMesh> sub_meshes;                     // Empty for leaf meshes

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
tt::tt_fabric::AdjacencyGraph<uint32_t> join_mesh_level(
    const std::vector<FlattenedMesh>& meshes, const std::vector<int32_t>& dims) {
    constexpr size_t SINGLE_MESH = 1;
    constexpr int32_t ROW_INDEX = 0;
    constexpr int32_t COL_INDEX = 1;

    if (meshes.empty()) {
        return tt::tt_fabric::AdjacencyGraph<uint32_t>();
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
    return tt::tt_fabric::AdjacencyGraph<uint32_t>(adj_map);
}

// Helper to extract tray ID from grouping name (e.g., "tray_1" -> TrayID{1}, "TRAY_2" -> TrayID{2})
::tt::tt_metal::TrayID extract_tray_id(const std::string& grouping_name) {
    const std::string lower_name = [&]() {
        std::string result = grouping_name;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }();

    if (lower_name.starts_with("tray_")) {
        try {
            return ::tt::tt_metal::TrayID{static_cast<uint32_t>(std::stoul(lower_name.substr(5)))};
        } catch (...) {
            return ::tt::tt_metal::TrayID{0};
        }
    }
    return ::tt::tt_metal::TrayID{0};
}

// Helper to extract ASIC location from grouping path string (e.g., "ASIC_LOCATION_1" -> ASICLocation{1})
::tt::tt_metal::ASICLocation extract_asic_location_from_path(const std::vector<std::string>& grouping_path) {
    for (const auto& path_elem : grouping_path) {
        if (path_elem.starts_with("ASIC_LOCATION_")) {
            try {
                return ::tt::tt_metal::ASICLocation{static_cast<uint32_t>(std::stoul(path_elem.substr(14)))};
            } catch (...) {
                return ::tt::tt_metal::ASICLocation{0};
            }
        }
    }
    return ::tt::tt_metal::ASICLocation{0};
}

// Rebuild GroupingInfo.items from FlattenedMesh. Graph nodes are 0..n-1; items[i] is the item for node i.
void rebuild_items_from_flattened_mesh(tt::tt_fabric::GroupingInfo& info, const FlattenedMesh& mesh) {
    info.items.clear();
    const auto& node_ids = mesh.graph.get_nodes();
    info.items.reserve(node_ids.size());

    for (uint32_t node_id : node_ids) {
        tt::tt_fabric::GroupingItemInfo item;
        item.type = tt::tt_fabric::GroupingItemInfo::ItemType::ASIC_LOCATION;
        item.asic_location = ::tt::tt_metal::ASICLocation{0};
        item.tray_id = ::tt::tt_metal::TrayID{0};

        auto metadata_it = mesh.node_metadata.find(node_id);
        if (metadata_it != mesh.node_metadata.end()) {
            const NodeMetadata& metadata = metadata_it->second;
            item.tray_id = metadata.tray_id;
            item.asic_location = extract_asic_location_from_path(metadata.grouping_path);
            item.grouping_path = metadata.grouping_path;
        }
        info.items.push_back(std::move(item));
    }
}

// Recursively build flattened meshes from a grouping item.
// Returns a vector of meshes - one per possibility (based on possible groupings that can be formed).
// Algorithm:
//   - Leaf (ASIC_LOCATION): create single-node mesh (one possibility)
//   - Single-item grouping: recurse for each possible sub_grouping, don't join - one entry per possibility
//   - Multi-item grouping: build sub-meshes for each item (each can have multiple possibilities),
//     then for each combination in the Cartesian product, join and add one entry
// Note: PSD validation is done at the top level in build_flattened_adjacency_mesh, not here.
std::vector<FlattenedMesh> build_flattened_meshes_for_item(
    const tt::tt_fabric::GroupingItemInfo& item,
    uint32_t& next_global_id,
    const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<tt::tt_fabric::GroupingInfo>>>&
        cache,
    const tt::tt_fabric::PhysicalGroupingDescriptor* desc,
    const std::vector<std::string>& grouping_path = {}) {
    constexpr int32_t SINGLE_NODE_ROWS = 1;
    constexpr int32_t SINGLE_NODE_COLS = 1;
    constexpr size_t SINGLE_ITEM = 1;

    if (item.type == tt::tt_fabric::GroupingItemInfo::ItemType::ASIC_LOCATION) {
        // Leaf case: single ASIC node - one possibility
        FlattenedMesh mesh;
        mesh.dims = {SINGLE_NODE_ROWS, SINGLE_NODE_COLS};
        const uint32_t node_id = next_global_id++;

        // Extract tray ID from grouping path
        NodeMetadata metadata;
        for (const auto& path_elem : grouping_path) {
            ::tt::tt_metal::TrayID tray_id = extract_tray_id(path_elem);
            if (*tray_id > 0) {
                metadata.tray_id = tray_id;
                break;
            }
        }

        // Build grouping path: copy existing path and add ASIC location
        metadata.grouping_path = grouping_path;
        metadata.grouping_path.push_back("ASIC_LOCATION_" + std::to_string(*item.asic_location));

        mesh.nodes_row_major = {node_id};
        mesh.node_metadata[node_id] = metadata;
        mesh.graph = tt::tt_fabric::AdjacencyGraph<uint32_t>({{node_id, {}}});

        // PSD validation is done at the top level only, not for individual leaf nodes
        return {std::move(mesh)};
    }

    // Compound case: resolve grouping reference - iterate over ALL possible groupings
    // Look up by name first; preset types (TRAY_1, HOSTS, MESH) may be keyed by type, so fall back to type lookup
    std::vector<tt::tt_fabric::GroupingInfo> possible_groupings;
    auto name_it = cache.find(item.grouping_name);
    if (name_it != cache.end()) {
        for (const auto& [type, groupings] : name_it->second) {
            possible_groupings.insert(possible_groupings.end(), groupings.begin(), groupings.end());
        }
    }
    if (possible_groupings.empty()) {
        // Fallback: search by type (handles preset refs like TRAY_1, HOSTS where cache key is name "tray_1",
        // "hosts_required")
        for (const auto& [name, type_map] : cache) {
            auto type_it = type_map.find(item.grouping_name);
            if (type_it != type_map.end()) {
                possible_groupings.insert(possible_groupings.end(), type_it->second.begin(), type_it->second.end());
            }
        }
    }
    TT_FATAL(!possible_groupings.empty(), "Unknown grouping: {}", item.grouping_name);

    std::vector<FlattenedMesh> all_results;
    for (const tt::tt_fabric::GroupingInfo& sub_grouping : possible_groupings) {
        // Build new grouping path using grouping name (not type)
        std::vector<std::string> new_path = grouping_path;
        new_path.push_back(sub_grouping.name);

        // Ensure tray_id is extractable in leaf nodes: add type (e.g. "TRAY_1") if it encodes tray
        // and the name didn't already provide it. This handles cases where name might not match "tray_N".
        ::tt::tt_metal::TrayID name_tray_id = extract_tray_id(sub_grouping.name);
        ::tt::tt_metal::TrayID type_tray_id = extract_tray_id(sub_grouping.type);
        if (*name_tray_id == 0 && *type_tray_id > 0) {
            new_path.push_back(sub_grouping.type);
        }
        // Fallback: ref item has tray_id (e.g. TRAY_1) but neither name nor type encodes it
        if (*name_tray_id == 0 && *type_tray_id == 0 && *item.tray_id > 0) {
            new_path.push_back("tray_" + std::to_string(*item.tray_id));
        }

        // Single-item grouping: recurse directly, add each possibility as its own entry (no join)
        if (sub_grouping.items.size() == SINGLE_ITEM) {
            std::vector<FlattenedMesh> sub_results =
                build_flattened_meshes_for_item(sub_grouping.items[0], next_global_id, cache, desc, new_path);
            // PSD validation is done at the top level only, not at intermediate levels
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
        std::vector<size_t> sizes;
        sizes.reserve(sub_meshes_per_item.size());
        for (const auto& meshes : sub_meshes_per_item) {
            sizes.push_back(meshes.size());
        }

        size_t combination_index = 0;
        tt::tt_fabric::iterate_cartesian_product(sizes, [&](const std::vector<size_t>& indices) {
            std::vector<FlattenedMesh> chosen;
            chosen.reserve(sub_grouping.items.size());
            for (size_t i = 0; i < sub_grouping.items.size(); ++i) {
                chosen.push_back(sub_meshes_per_item[i][indices[i]]);
            }

            FlattenedMesh mesh;
            mesh.graph = join_mesh_level(chosen, layout);
            mesh.dims = layout;
            mesh.sub_meshes = std::move(chosen);

            // Collect nodes and metadata from all sub-meshes
            for (const auto& sub_mesh : mesh.sub_meshes) {
                mesh.nodes_row_major.insert(
                    mesh.nodes_row_major.end(), sub_mesh.nodes_row_major.begin(), sub_mesh.nodes_row_major.end());
                // Copy node metadata from sub-mesh
                for (const auto& [node_id, metadata] : sub_mesh.node_metadata) {
                    mesh.node_metadata[node_id] = metadata;
                }
            }

            // Note: We don't filter here because build_flattened_meshes_for_item already filtered sub-items
            // and the top-level join will be validated in build_flattened_adjacency_mesh
            all_results.push_back(std::move(mesh));
            combination_index++;
        });
    }
    return all_results;
}

}  // namespace

namespace tt::tt_fabric {

std::vector<GroupingInfo> PhysicalGroupingDescriptor::build_flattened_adjacency_mesh(
    const GroupingInfo& grouping) const {
    return build_flattened_adjacency_mesh(
        grouping, static_cast<const tt::tt_metal::PhysicalSystemDescriptor*>(nullptr));
}

std::vector<GroupingInfo> PhysicalGroupingDescriptor::build_flattened_adjacency_mesh(
    const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const {
    return build_flattened_adjacency_mesh(grouping, &physical_system_descriptor);
}

// Top-level entry point: builds meshes for each item, returns a vector - one per possibility
std::vector<GroupingInfo> PhysicalGroupingDescriptor::build_flattened_adjacency_mesh(
    const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) const {
    (void)physical_system_descriptor;  // Reserved for future use - can be used for validation/filtering

    if (grouping.items.empty()) {
        GroupingInfo result = grouping;
        result.adjacency_graph = tt::tt_fabric::AdjacencyGraph<uint32_t>();
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

        for (size_t i = 0; i < meshes.size(); ++i) {
            GroupingInfo info = grouping;
            if (meshes.size() > 1) {
                info.name = grouping.name + "_" + std::to_string(i);
            }

            // Rebuild items from flattened mesh node metadata BEFORE moving the graph
            // (rebuild_items_from_flattened_mesh needs mesh.graph.get_nodes() and mesh.node_metadata)
            rebuild_items_from_flattened_mesh(info, meshes[i]);
            info.adjacency_graph = std::move(meshes[i].graph);

            // If PSD is provided, validate that the graph can be mapped to it
            if (physical_system_descriptor != nullptr) {
                // solve_for_one_grouping_to_psd uses items[node_id] for trait constraints
                auto mapping_result = find_any_in_psd(info, *physical_system_descriptor);
                if (mapping_result.empty()) {
                    continue;  // Skip this combination if it can't be mapped
                }
            }

            result.push_back(std::move(info));
        }
        return result;
    }

    // Multi-item: build per-item possibilities, then Cartesian product
    std::vector<std::vector<FlattenedMesh>> meshes_per_item;
    meshes_per_item.reserve(grouping.items.size());
    for (const auto& item : grouping.items) {
        auto item_meshes =
            build_flattened_meshes_for_item(item, next_node_id, resolved_groupings_cache_, this, initial_path);
        // If item has no meshes (shouldn't happen for valid groupings), return empty result
        if (item_meshes.empty()) {
            return {};
        }
        meshes_per_item.push_back(std::move(item_meshes));
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
    std::vector<size_t> sizes;
    sizes.reserve(meshes_per_item.size());
    for (const auto& meshes : meshes_per_item) {
        sizes.push_back(meshes.size());
    }
    size_t total_combinations = 1;
    for (size_t s : sizes) {
        total_combinations *= s;
    }

    size_t combination_index = 0;
    tt::tt_fabric::iterate_cartesian_product(sizes, [&](const std::vector<size_t>& indices) {
        std::vector<FlattenedMesh> chosen;
        chosen.reserve(grouping.items.size());
        for (size_t i = 0; i < grouping.items.size(); ++i) {
            chosen.push_back(meshes_per_item[i][indices[i]]);
        }

        tt::tt_fabric::AdjacencyGraph<uint32_t> joined_graph = join_mesh_level(chosen, layout);

        // Collect node metadata from all chosen meshes
        FlattenedMesh joined_mesh;
        joined_mesh.graph = std::move(joined_graph);
        joined_mesh.dims = layout;
        joined_mesh.sub_meshes = chosen;

        // Collect nodes and metadata from all sub-meshes
        for (const auto& sub_mesh : chosen) {
            joined_mesh.nodes_row_major.insert(
                joined_mesh.nodes_row_major.end(), sub_mesh.nodes_row_major.begin(), sub_mesh.nodes_row_major.end());
            // Copy node metadata from sub-mesh
            for (const auto& [node_id, metadata] : sub_mesh.node_metadata) {
                joined_mesh.node_metadata[node_id] = metadata;
            }
        }

        GroupingInfo info = grouping;
        if (total_combinations > 1) {
            info.name = grouping.name + "_" + std::to_string(combination_index);
        }

        // Rebuild items from flattened mesh node metadata BEFORE moving the graph
        // (rebuild_items_from_flattened_mesh needs joined_mesh.graph.get_nodes() and joined_mesh.node_metadata)
        rebuild_items_from_flattened_mesh(info, joined_mesh);
        info.adjacency_graph = std::move(joined_mesh.graph);

        // If PSD is provided, validate that the top-level joined graph can be mapped to it
        if (physical_system_descriptor != nullptr) {
            // solve_for_one_grouping_to_psd uses items[node_id] for trait constraints
            auto mapping_result = find_any_in_psd(info, *physical_system_descriptor);
            if (mapping_result.empty()) {
                return;  // Skip this combination if it can't be mapped
            }
        }

        result.push_back(std::move(info));
        combination_index++;
    });
    return result;
}

}  // namespace tt::tt_fabric
