// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "control_plane.hpp"
#include "fabric_host_utils.hpp"

#include <tt-metalium/fabric.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/assert.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include "erisc_datamover_builder.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include "fabric/hw/inc/fabric_routing_mode.h"
#include "fabric_context.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace tt::tt_fabric {

namespace {

// Computes BFS distance map from a start chip to all reachable chips using the provided adjacency map.
std::unordered_map<chip_id_t, std::uint32_t> compute_distances(
    chip_id_t start_chip, const std::unordered_map<chip_id_t, std::vector<chip_id_t>>& adjacency_map) {
    std::unordered_map<chip_id_t, std::uint32_t> dist;
    std::queue<chip_id_t> q;
    dist[start_chip] = 0;
    q.push(start_chip);

    while (!q.empty()) {
        auto cur = q.front();
        q.pop();

        auto it = adjacency_map.find(cur);
        if (it != adjacency_map.end()) {
            for (auto nbr : it->second) {
                if (dist.find(nbr) == dist.end()) {
                    dist[nbr] = dist.at(cur) + 1;
                    q.push(nbr);
                }
            }
        }
    }
    return dist;
}

void create_1d_mesh_view_with_dfs(
    const std::unordered_map<chip_id_t, std::vector<chip_id_t>>& adjacency_map,
    chip_id_t start_chip,
    uint32_t num_chips,
    std::vector<chip_id_t>& path) {
    std::unordered_set<chip_id_t> visited;
    visited.insert(start_chip);

    path.reserve(num_chips);
    path.push_back(start_chip);

    // Internal recursive helper function
    std::function<bool(chip_id_t)> dfs = [&](chip_id_t current_chip) -> bool {
        if (path.size() == num_chips) {
            return true;
        }

        auto it = adjacency_map.find(current_chip);
        if (it == adjacency_map.end()) {
            return false;  // No neighbors
        }

        for (chip_id_t nbr : it->second) {
            if (visited.find(nbr) == visited.end()) {
                path.push_back(nbr);
                visited.insert(nbr);

                if (dfs(nbr)) {
                    return true;
                }

                // Backtrack
                path.pop_back();
                visited.erase(nbr);
            }
        }
        return false;  // No valid path found from this node
    };

    dfs(start_chip);
}

}  // namespace

bool is_tt_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return is_1d_fabric_config(fabric_config) || is_2d_fabric_config(fabric_config);
}

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config) {
    switch (fabric_config) {
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: return FabricType::TORUS_XY;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X: return FabricType::TORUS_X;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y: return FabricType::TORUS_Y;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY: return FabricType::TORUS_XY;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_X: return FabricType::TORUS_X;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_Y: return FabricType::TORUS_Y;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY: return FabricType::TORUS_XY;
        default: return FabricType::MESH;
    }
}

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id, RoutingDirection direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const bool is_2d_fabric = control_plane.get_fabric_context().is_2D_routing_enabled();

    const std::vector<chan_id_t>& fabric_channels =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);

    // the subset of routers that support forwarding b/w those chips
    std::vector<chan_id_t> forwarding_channels;
    if (is_2d_fabric) {
        forwarding_channels =
            control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, direction);
    } else {
        // TODO: not going to work for Big Mesh
        const auto src_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
        const auto dst_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);
        // for 1D check if each port has an active connection to the dst_chip_id
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const auto& soc_desc = cluster.get_soc_desc(src_chip_id);

        for (const auto& channel : fabric_channels) {
            const auto eth_core = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
            auto [connected_chip_id, connected_eth_core] =
                cluster.get_connected_ethernet_core(std::make_tuple(src_chip_id, CoreCoord{eth_core.x, eth_core.y}));
            if (connected_chip_id == dst_chip_id) {
                forwarding_channels.push_back(channel);
            }
        }
    }

    std::vector<uint32_t> link_indices;
    for (uint32_t i = 0; i < fabric_channels.size(); i++) {
        if (std::find(forwarding_channels.begin(), forwarding_channels.end(), fabric_channels[i]) !=
            forwarding_channels.end()) {
            link_indices.push_back(i);
        }
    }

    return link_indices;
}

void set_routing_mode(uint16_t routing_mode) {
    // override for forced routing mode
    if (routing_mode == ROUTING_MODE_UNDEFINED) {
        return;
    }

    // Validate dimension flags are orthogonal (only one can be set)
    TT_FATAL(
        __builtin_popcount(routing_mode & (ROUTING_MODE_1D | ROUTING_MODE_2D | ROUTING_MODE_3D)) == 1,
        "Only one dimension mode (1D, 2D, 3D) can be active at once");

    // Validate topology flags are orthogonal
    TT_FATAL(
        __builtin_popcount(
            routing_mode & (ROUTING_MODE_RING | ROUTING_MODE_LINE | ROUTING_MODE_MESH | ROUTING_MODE_TORUS)) == 1,
        "Only one topology mode (RING, LINE, MESH, TORUS) can be active at once");

    // Validate 1D can't be used with MESH or TORUS
    TT_FATAL(
        !(routing_mode & ROUTING_MODE_1D) || !(routing_mode & (ROUTING_MODE_MESH | ROUTING_MODE_TORUS)),
        "1D routing mode cannot be combined with MESH or TORUS topology");

    // Validate 2D can't be used with LINE or RING
    TT_FATAL(
        !(routing_mode & ROUTING_MODE_2D) || !(routing_mode & (ROUTING_MODE_LINE | ROUTING_MODE_RING)),
        "2D routing mode cannot be combined with LINE or RING topology");

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    control_plane.set_routing_mode(routing_mode);
}

void set_routing_mode(Topology topology, tt::tt_fabric::FabricConfig fabric_config, uint32_t dimension /*, take more*/) {
    // TODO: take more parameters to set detail routing mode
    TT_FATAL(
        dimension == 1 || dimension == 2 || dimension == 3,
        "Invalid dimension {}. Supported dimensions are 1, 2, or 3",
        dimension);

    uint16_t mode = (dimension == 3 ? ROUTING_MODE_3D : 0);
    if (topology == Topology::Ring) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_RING);
    } else if (topology == Topology::Linear) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_LINE);
    } else if (topology == Topology::Mesh) {
        mode |= (ROUTING_MODE_2D | ROUTING_MODE_MESH);
    } else if (topology == Topology::Torus) {
        mode |= (ROUTING_MODE_2D | ROUTING_MODE_TORUS);
    }

    if (tt::tt_fabric::FabricContext::is_dynamic_routing_config(fabric_config)) {
        mode |= ROUTING_MODE_DYNAMIC;
    } else {
        mode |= ROUTING_MODE_LOW_LATENCY;
    }
    set_routing_mode(mode);
}


IntraMeshAdjacencyMap build_mesh_adjacency_map(
    const std::set<chip_id_t>& user_chip_ids,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    std::function<std::vector<chip_id_t>(chip_id_t)> get_adjacent_chips_func,
    std::optional<chip_id_t> start_chip_id /* = std::nullopt */) {
    constexpr size_t CORNER_1D_ADJACENT_CHIPS = 1;
    constexpr size_t CORNER_ADJACENT_CHIPS = 2;
    constexpr size_t EDGE_ADJACENT_CHIPS = 3;
    IntraMeshAdjacencyMap topology_info;

    // Store mesh dimensions in topology info
    topology_info.ns_size = mesh_shape[0];
    topology_info.ew_size = mesh_shape[1];

    // Determine the starting chip ID for BFS (use first chip from mesh container unless specified)
    chip_id_t chip_0 = start_chip_id.has_value() ? *start_chip_id : *user_chip_ids.begin();

    // BFS to populate mesh of chips based on adjacency from chip 0
    std::queue<chip_id_t> chip_queue;
    std::unordered_set<chip_id_t> visited_chips;
    chip_queue.push(chip_0);
    visited_chips.insert(chip_0);

    bool is_1d_mesh = (topology_info.ns_size == 1) || (topology_info.ew_size == 1);

    while (!chip_queue.empty()) {
        chip_id_t current_chip = chip_queue.front();
        chip_queue.pop();

        // Get adjacent chips using the provided adjacency function
        std::vector<chip_id_t> adjacent_chips = get_adjacent_chips_func(current_chip);
        // Count neighbours: CORNER_ADJACENT_CHIPS → corner, EDGE_ADJACENT_CHIPS → edge, INTERIOR_ADJACENT_CHIPS →
        // interior (we treat only links with ≥ num_ports_per_side lanes as neighbours)
        bool is_corner = false;

        if (is_1d_mesh) {
            // For 1D meshes, corners have exactly 1 adjacent chip (endpoints)
            is_corner = (adjacent_chips.size() == CORNER_1D_ADJACENT_CHIPS);
        } else {
            // For 2D meshes, corners have exactly 2 adjacent chips
            is_corner = (adjacent_chips.size() == CORNER_ADJACENT_CHIPS);
        }

        if (is_corner) {
            // NOTE: First one added is the corner closest to chip 0
            //       this will be the pinned nw corner
            topology_info.corners.push_back(current_chip);
        } else if (adjacent_chips.size() == EDGE_ADJACENT_CHIPS) {
            topology_info.edges.push_back(current_chip);
        }

        for (const auto& adjacent_chip : adjacent_chips) {
            // Add chip to adjacent chips map
            topology_info.adjacency_map[current_chip].push_back(adjacent_chip);

            if (visited_chips.find(adjacent_chip) != visited_chips.end()) {
                continue;
            }

            // Add chip to queue and visited next set
            chip_queue.push(adjacent_chip);
            visited_chips.insert(adjacent_chip);
        }
    }

    return topology_info;
}

std::pair<std::unordered_map<chip_id_t, std::vector<chip_id_t>>, chip_id_t> sort_adjacency_map_by_eth_coords(
    const IntraMeshAdjacencyMap& topology_info) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto eth_coords = cluster.get_user_chip_ethernet_coordinates();
    if (eth_coords.size() != topology_info.adjacency_map.size()) {
        return {
            topology_info.adjacency_map,
            std::min_element(topology_info.adjacency_map.begin(), topology_info.adjacency_map.end())->first};
    }
    auto min_eth_chip = std::min_element(eth_coords.begin(), eth_coords.end(), [](const auto& a, const auto& b) {
        if (a.second.y != b.second.y) {
            return a.second.y < b.second.y;
        }
        return a.second.x < b.second.x;
    });

    auto adjacency_map = topology_info.adjacency_map;
    for (auto& [chip_id, neighbors] : adjacency_map) {
        std::sort(neighbors.begin(), neighbors.end(), [&eth_coords](chip_id_t a, chip_id_t b) {
            const auto& coord_a = eth_coords.at(a);
            const auto& coord_b = eth_coords.at(b);
            if (coord_a.y != coord_b.y) {
                return coord_a.y < coord_b.y;
            }
            return coord_a.x < coord_b.x;
        });
    }
    return {adjacency_map, min_eth_chip->first};
}

std::pair<std::unordered_map<chip_id_t, std::vector<chip_id_t>>, chip_id_t> sort_adjacency_map_by_ubb_id(
    const IntraMeshAdjacencyMap& topology_info) {
    auto adjacency_map = topology_info.adjacency_map;
    chip_id_t first_chip = 0;
    for (auto& [chip_id, neighbors] : adjacency_map) {
        auto ubb_id = tt::tt_fabric::get_ubb_id(chip_id);
        if (ubb_id.tray_id == 1 && ubb_id.asic_id == 1) {
            first_chip = chip_id;
            break;
        }
    }

    for (auto& [chip_id, neighbors] : adjacency_map) {
        std::sort(neighbors.begin(), neighbors.end(), [&](chip_id_t a, chip_id_t b) {
            auto ubb_id_a = tt::tt_fabric::get_ubb_id(a);
            auto ubb_id_b = tt::tt_fabric::get_ubb_id(b);
            return ubb_id_a.tray_id < ubb_id_b.tray_id ||
                   (ubb_id_a.tray_id == ubb_id_b.tray_id && ubb_id_a.asic_id < ubb_id_b.asic_id);
        });
    }
    return {adjacency_map, first_chip};
}

std::vector<chip_id_t> convert_1d_mesh_adjacency_to_row_major_vector(
    const IntraMeshAdjacencyMap& topology_info,
    std::optional<std::function<std::pair<AdjacencyMap, chip_id_t>(const IntraMeshAdjacencyMap&)>> graph_sorter) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    chip_id_t first_chip = 0;
    auto adj_map = topology_info.adjacency_map;
    if (cluster.get_board_type(0) == BoardType::N300) {
        // #26987: On N300 based systems we currently use Ethernet Coordinates to sort the adjacency map.
        // This ensures that the Fabric Node IDs are deterministically mapped to physical chips across hosts.
        // If this is not the case, we will need to regenerate MGDs depending on the host being run on, due to
        // the current infra tightly coupling logical and physical representations.
        // This will not be an issue once we have MGD 2.0 in logical space and algorithms to bind logical connections
        // in the MGD to physical ethernet channels.
        if (!graph_sorter.has_value()) {
            // Default behavior: sort adjacency map by Ethernet coordinates
            std::tie(adj_map, first_chip) = sort_adjacency_map_by_eth_coords(topology_info);
        } else {
            // User provided a sorting function. This is primarily done for testing.
            std::tie(adj_map, first_chip) = graph_sorter.value()(topology_info);
        }
    } else if (cluster.get_board_type(0) == BoardType::UBB) {
        if (!graph_sorter.has_value()) {
            // Default behavior: sort adjacency map by Ethernet coordinates
            std::tie(adj_map, first_chip) = sort_adjacency_map_by_ubb_id(topology_info);
        } else {
            // User provided a sorting function. This is primarily done for testing.
            std::tie(adj_map, first_chip) = graph_sorter.value()(topology_info);
        }
    } else if (topology_info.ns_size == 1 && topology_info.ew_size == 1) {
        // Single chip mesh
        first_chip = 0;
    } else {
        first_chip = std::min_element(topology_info.adjacency_map.begin(), topology_info.adjacency_map.end())->first;
    }

    // This vector contains a 1D view of devices in the mesh, constructed using DFS. This works since we are using a
    // mesh topology which guarantees connectivity.
    std::vector<chip_id_t> physical_chip_ids;
    create_1d_mesh_view_with_dfs(adj_map, first_chip, topology_info.ns_size * topology_info.ew_size, physical_chip_ids);
    return physical_chip_ids;
}

std::vector<chip_id_t> convert_2d_mesh_adjacency_to_row_major_vector(
    const IntraMeshAdjacencyMap& topology_info, std::optional<chip_id_t> nw_corner_chip_id) {
    // Check number of corners for 2D meshes
    TT_FATAL(
        topology_info.corners.size() == 4, "Expected 4 corners for 2D mesh, got {}.", topology_info.corners.size());

    // Determine the northwest corner
    chip_id_t nw_corner;
    if (nw_corner_chip_id.has_value()) {
        // Use the provided northwest corner chip ID if it's valid
        nw_corner = nw_corner_chip_id.value();
        // Verify that the provided chip is actually a corner
        TT_FATAL(
            std::find(topology_info.corners.begin(), topology_info.corners.end(), nw_corner) !=
                topology_info.corners.end(),
            "Provided chip ID {} is not a corner chip. Expected one of: {}",
            nw_corner,
            [&topology_info]() {
                std::string result;
                for (size_t i = 0; i < topology_info.corners.size(); ++i) {
                    if (i > 0) {
                        result += ", ";
                    }
                    result += std::to_string(topology_info.corners[i]);
                }
                return result;
            }());
    } else {
        // Default behavior: use the first corner found (closest to chip 0)
        nw_corner = topology_info.corners[0];
    }

    std::vector<chip_id_t> physical_chip_ids(topology_info.ns_size * topology_info.ew_size);

    // Place northwest corner at (0, 0)
    physical_chip_ids[0] = nw_corner;

    // -----------------------------------------------------------------------------
    // Corner discovery complete: we now have four corners, NW is fixed at index 0
    // -----------------------------------------------------------------------------

    // Step 1: BFS from the NW corner to get Manhattan distances dNW[chip].

    // Pre-compute signed mesh dimensions once to avoid repetitive casts.
    const int mesh_cols = static_cast<int>(topology_info.ew_size);
    const int mesh_rows = static_cast<int>(topology_info.ns_size);

    // 1) Distances from NW corner.
    auto dist_from_nw = compute_distances(nw_corner, topology_info.adjacency_map);

    // 2) Identify the NE corner (distance of mesh_ew_size-1 from NW) and run a second
    //    BFS from it to obtain dNE[chip].
    chip_id_t ne_corner = nw_corner;  // initialise
    bool ne_found = false;
    for (auto corner : topology_info.corners) {
        if (corner == nw_corner) {
            continue;
        }
        auto it = dist_from_nw.find(corner);
        if (it != dist_from_nw.end() && it->second == topology_info.ew_size - 1) {
            ne_corner = corner;
            ne_found = true;
            break;
        }
    }
    if (!ne_found) {
        // Fall back: pick any other corner; grid may be square so distance == mesh_ew_size-1 may not hold.
        for (auto corner : topology_info.corners) {
            if (corner != nw_corner) {
                ne_corner = corner;
                ne_found = true;
                break;
            }
        }
    }
    TT_FATAL(
        ne_found,
        "Ethernet mesh discovered does not match expected shape {}x{}.",
        topology_info.ew_size,
        topology_info.ns_size);

    // BFS from NE corner.
    auto dist_from_ne = compute_distances(ne_corner, topology_info.adjacency_map);

    // Step 3: compute (row, col) for every chip using the distance formulas
    std::fill(physical_chip_ids.begin(), physical_chip_ids.end(), static_cast<chip_id_t>(-1));

    for (const auto& [chip, d_nw] : dist_from_nw) {
        TT_FATAL(dist_from_ne.count(chip), "Mesh disconnected: chip {} missing in NE BFS.", chip);
        int d_ne = static_cast<int>(dist_from_ne.at(chip));
        int d_nw_int = static_cast<int>(d_nw);
        // Solve the 2-equation system:
        //   dNW = row + col
        //   dNE = row + (mesh_cols-1 - col)
        int col = (mesh_cols - 1 + d_nw_int - d_ne) / 2;
        int row = d_nw_int - col;

        TT_FATAL(row >= 0 && row < mesh_rows, "Row {} out of bounds.", row);
        TT_FATAL(col >= 0 && col < mesh_cols, "Col {} out of bounds.", col);

        size_t idx = static_cast<size_t>(row) * static_cast<size_t>(mesh_cols) + static_cast<size_t>(col);
        TT_FATAL(physical_chip_ids[idx] == static_cast<chip_id_t>(-1), "Duplicate mapping at index {}.", idx);
        physical_chip_ids[idx] = chip;
    }

    TT_FATAL(physical_chip_ids[0] == nw_corner, "NW corner not at index 0 after embedding.");

    for (std::uint32_t i = 0; i < physical_chip_ids.size(); ++i) {
        TT_FATAL(physical_chip_ids[i] != static_cast<chip_id_t>(-1), "Mesh embedding incomplete at index {}.", i);
    }

    return physical_chip_ids;
}

}  // namespace tt::tt_fabric
