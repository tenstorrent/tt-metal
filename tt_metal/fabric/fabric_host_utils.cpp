// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "control_plane.hpp"
#include "fabric_host_utils.hpp"

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/assert.hpp>
#include <magic_enum/magic_enum.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <set>
#include <vector>
#include <algorithm>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "fabric/hw/inc/fabric_routing_mode.h"
#include "fabric_context.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace tt::tt_fabric {

bool is_tt_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_RING ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
}

bool is_2d_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
}

uint32_t get_sender_channel_count(tt::tt_fabric::Topology topology) {
    if (topology == Topology::Mesh) {
        return FabricEriscDatamoverConfig::num_sender_channels_2d;
    } else {
        return FabricEriscDatamoverConfig::num_sender_channels_1d;
    }
}

uint32_t get_downstream_edm_count(tt::tt_fabric::Topology topology) {
    if (topology == Topology::Mesh) {
        return FabricEriscDatamoverConfig::num_downstream_edms_2d;
    } else {
        return FabricEriscDatamoverConfig::num_downstream_edms;
    }
}

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config, tt::ClusterType cluster_type) {
    if (cluster_type == tt::ClusterType::GALAXY && fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_RING) {
        return FabricType::TORUS_XY;
    }
    return FabricType::MESH;
}

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id, RoutingDirection direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const bool is_2d_fabric = control_plane.get_fabric_context().get_fabric_topology() == Topology::Mesh;

    const std::vector<chan_id_t>& fabric_channels =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);

    // the subset of routers that support forwarding b/w those chips
    std::vector<chan_id_t> forwarding_channels;
    if (is_2d_fabric) {
        forwarding_channels =
            control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, direction);
    } else {
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

    // Validate push/pull flags are orthogonal
    TT_FATAL(
        __builtin_popcount(routing_mode & (ROUTING_MODE_PUSH | ROUTING_MODE_PULL)) <= 1,
        "PUSH and PULL routing modes cannot be used together");

    // Validate push/pull flags are only for 2D
    TT_FATAL(
        !(routing_mode & (ROUTING_MODE_PUSH | ROUTING_MODE_PULL)) || (routing_mode & ROUTING_MODE_2D),
        "PUSH and PULL routing modes can only be used with 2D topology");

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
    }
    if (fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC) {
        mode |= ROUTING_MODE_DYNAMIC;
    } else {
        mode |= ROUTING_MODE_LOW_LATENCY;
    }
    set_routing_mode(mode);
}

void get_optimal_noc_for_edm(
    tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder1,
    tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder2,
    const uint32_t num_links,
    const tt_fabric::Topology topology) {
    constexpr uint32_t ring_noc_selection_link_threshold = 3;
    constexpr uint32_t line_noc_selection_link_threshold = 2;
    bool enable_noc_selection_opt = false;
    if (topology == tt_fabric::Topology::Ring) {
        enable_noc_selection_opt =
            (num_links > ring_noc_selection_link_threshold) && (edm_builder1.my_noc_y != edm_builder2.my_noc_y);
    } else {
        enable_noc_selection_opt =
            (num_links > line_noc_selection_link_threshold) && (edm_builder1.my_noc_y != edm_builder2.my_noc_y);
    }
    log_debug(
        tt::LogTest,
        "Fabric MeshId {} ChipId {} edm_builder1 {} {} is connecting to edm_builder2 {} {} num links {}",
        *(edm_builder1.local_fabric_node_id.mesh_id),
        edm_builder1.local_fabric_node_id.chip_id,
        edm_builder1.my_noc_x,
        edm_builder1.my_noc_y,
        edm_builder2.my_noc_x,
        edm_builder2.my_noc_y,
        num_links);

    if (enable_noc_selection_opt) {
        if (edm_builder1.my_noc_x < edm_builder2.my_noc_x) {
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_forwarding_noc_ids[i] = 0;
                edm_builder2.config.receiver_channel_forwarding_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_local_write_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_local_write_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_sender_channels; i++) {
                edm_builder1.config.sender_channel_ack_noc_ids[i] = 1;
                edm_builder2.config.sender_channel_ack_noc_ids[i] = 0;
            }
        } else if (edm_builder1.my_noc_x > edm_builder2.my_noc_x) {
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_forwarding_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_forwarding_noc_ids[i] = 0;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_local_write_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_local_write_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_sender_channels; i++) {
                edm_builder1.config.sender_channel_ack_noc_ids[i] = 0;
                edm_builder2.config.sender_channel_ack_noc_ids[i] = 1;
            }
        }
    }
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

std::vector<chip_id_t> convert_1d_mesh_adjacency_to_row_major_vector(const IntraMeshAdjacencyMap& topology_info) {
    // For 1D meshes, we expect exactly 2 corners (the endpoints)
    TT_FATAL(
        topology_info.corners.size() == 2, "Expected 2 corners for 1D mesh, got {}.", topology_info.corners.size());

    std::vector<chip_id_t> physical_chip_ids(topology_info.ns_size * topology_info.ew_size);
    std::fill(physical_chip_ids.begin(), physical_chip_ids.end(), static_cast<chip_id_t>(-1));

    // Place the first corner (closest to chip 0) at index 0
    chip_id_t first_corner = topology_info.corners[0];
    physical_chip_ids[0] = first_corner;

    // Place the second corner at the last index
    chip_id_t second_corner = topology_info.corners[1];
    physical_chip_ids[physical_chip_ids.size() - 1] = second_corner;

    // Fill in the middle chips using BFS distances
    auto dist_from_first = compute_distances(first_corner, topology_info.adjacency_map);

    for (const auto& [chip, distance] : dist_from_first) {
        if (chip != first_corner && chip != second_corner) {
            // For 1D mesh, distance directly corresponds to the index
            size_t idx = static_cast<size_t>(distance);
            TT_FATAL(idx < physical_chip_ids.size(), "Index {} out of bounds for 1D mesh.", idx);
            TT_FATAL(physical_chip_ids[idx] == static_cast<chip_id_t>(-1), "Duplicate mapping at index {}.", idx);
            physical_chip_ids[idx] = chip;
        }
    }

    // Verify all chips are mapped
    for (std::uint32_t i = 0; i < physical_chip_ids.size(); ++i) {
        TT_FATAL(physical_chip_ids[i] != static_cast<chip_id_t>(-1), "1D mesh embedding incomplete at index {}.", i);
    }

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
