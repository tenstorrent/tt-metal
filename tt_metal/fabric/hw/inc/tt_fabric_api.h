// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include <hostdevcommon/common_values.hpp>
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include "hostdevcommon/fabric_common.h"
#include "tt_metal/hw/inc/risc_common.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include <type_traits>

using namespace tt::tt_fabric;

namespace tt::tt_fabric {

inline eth_chan_directions get_next_hop_router_direction(uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    tt_l1_ptr tensix_routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr tensix_routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    if (dst_mesh_id == routing_table->mesh_id) {
        return static_cast<eth_chan_directions>(
            routing_table->intra_mesh_routing_table.get_original_direction(dst_dev_id));
    } else {
        return static_cast<eth_chan_directions>(
            routing_table->inter_mesh_routing_table.get_original_direction(dst_mesh_id));
    }
}

template <bool mcast = false>
void fabric_set_route(
    volatile tt_l1_ptr LowLatencyMeshPacketHeader* packet_header,
    eth_chan_directions direction,
    uint32_t branch_forward,
    uint32_t start_hop,
    uint32_t num_hops,
    bool terminate = false) {
    uint32_t local_packet = 0;
    uint32_t forward_packet = 0;
    uint32_t value = 0;
    switch (direction) {
        case eth_chan_directions::EAST:
            local_packet = LowLatencyMeshRoutingFields::FORWARD_WEST;
            forward_packet = LowLatencyMeshRoutingFields::FORWARD_EAST;
            if constexpr (mcast) {
                packet_header->routing_fields.branch_east_offset = start_hop;
            } else {
                packet_header->routing_fields.branch_east_offset = start_hop + 1;
            }
            break;
        case eth_chan_directions::WEST:
            local_packet = LowLatencyMeshRoutingFields::FORWARD_EAST;
            forward_packet = LowLatencyMeshRoutingFields::FORWARD_WEST;
            if constexpr (mcast) {
                packet_header->routing_fields.branch_west_offset = start_hop;
            } else {
                packet_header->routing_fields.branch_west_offset = start_hop + 1;
            }
            break;
        case eth_chan_directions::NORTH:
            local_packet = LowLatencyMeshRoutingFields::FORWARD_SOUTH;
            forward_packet = LowLatencyMeshRoutingFields::FORWARD_NORTH | branch_forward;
            break;
        case eth_chan_directions::SOUTH:
            local_packet = LowLatencyMeshRoutingFields::FORWARD_NORTH;
            forward_packet = LowLatencyMeshRoutingFields::FORWARD_SOUTH | branch_forward;
            break;
        default: ASSERT(false);
    }

    volatile tt_l1_ptr uint8_t* route_vector = packet_header->route_buffer;
    uint32_t local_val;
    uint32_t forward_val;
    uint32_t end_hop = start_hop + num_hops;
    for (uint32_t i = start_hop; i < end_hop; i++) {
        if constexpr (mcast) {
            // If forward north or forward south is set, then it may be 2d mcast and requires east/west forwarding, in
            // addition to spine forwards on north/south. forward_packet bit 0 and 1 determine if mcast has to branch
            // east/west from spine. If this is not a north/south mcast, then it cannot be a 2D mcast, and we dont need
            // to branch.
            uint32_t mcast_branch = forward_packet & LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_NS
                                        ? forward_packet & LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_EW
                                        : 0;
            forward_val = i == end_hop - 1 ? mcast_branch : forward_packet;
            local_val = local_packet;
        } else {
            forward_val = terminate ? (i == end_hop - 1 ? 0 : forward_packet) : forward_packet;
            local_val = terminate ? (i == end_hop - 1 ? local_packet : 0) : 0;
        }
        route_vector[i] = local_val | forward_val;
    }
    packet_header->routing_fields.hop_index = 0;
}

void fabric_set_unicast_route(
    volatile tt_l1_ptr MeshPacketHeader* packet_header,
    uint16_t my_dev_id,  // Ignore this: Dynamic Routing does not need src chip ID
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t ew_dim  // Ignore this: Dynamic Routing does not need mesh dimensions
) {
    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    // Minimize writes to L1 by doing 1 u64 write (decomposed to 2 u32 writes) instead of 4 u16 writes
    packet_header->mcast_params_64 = 0;
    packet_header->is_mcast_active = 0;
}

void fabric_set_mcast_route(
    volatile tt_l1_ptr MeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops) {
    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    // Minimize writes to L1 by doing 1 u64 write (decomposed to 2 u32 writes) instead of 4 u16 writes
    packet_header->mcast_params_64 = ((uint64_t)s_num_hops << 48) | ((uint64_t)n_num_hops << 32) |
                                     ((uint64_t)w_num_hops << 16) | ((uint64_t)e_num_hops);
    packet_header->is_mcast_active = 0;
}

void fabric_set_unicast_route(
    volatile tt_l1_ptr LowLatencyMeshPacketHeader* packet_header,
    uint16_t my_dev_id,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,  // Ignore this, since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
    uint16_t ew_dim) {
    uint32_t ns_hops = 0;
    uint32_t target_dev = dst_dev_id;
    uint32_t target_col = 0;

    while (target_dev >= ew_dim) {
        target_dev -= ew_dim;
        target_col++;
    }
    uint32_t my_col = 0;
    uint32_t my_dev = my_dev_id;
    while (my_dev >= ew_dim) {
        my_dev -= ew_dim;
        my_col++;
    }

    eth_chan_directions outgoing_direction;
    uint32_t ew_hops = 0;
    if (target_col == my_col) {
        if (my_dev < target_dev) {
            // My device is west of target device
            outgoing_direction = eth_chan_directions::EAST;
            ew_hops = target_dev - my_dev;
        } else {
            // My device is east of target device
            outgoing_direction = eth_chan_directions::WEST;
            ew_hops = my_dev - target_dev;
        }
        fabric_set_route(packet_header, outgoing_direction, 0, 0, ew_hops, true);
    } else {
        // First hop is north/south. Calculate the number of required hops before turning east/west
        uint32_t ns_hops = 0;
        if (target_col > my_col) {
            // Target device is south of my device
            ns_hops = target_col - my_col;
            outgoing_direction = eth_chan_directions::SOUTH;
        } else {
            // Target device is north of my device
            ns_hops = my_col - target_col;
            outgoing_direction = eth_chan_directions::NORTH;
        }

        // determine the east/west hops
        uint32_t turn_direction = my_dev < target_dev ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        uint32_t ew_hops = (my_dev < target_dev) ? target_dev - my_dev : my_dev - target_dev;
        fabric_set_route(
            packet_header, (eth_chan_directions)outgoing_direction, 0, 0, ns_hops - bool(ew_hops), ew_hops == 0);
        if (ew_hops) {
            // +1 because this branch is now implementing the turn
            fabric_set_route(
                packet_header, (eth_chan_directions)turn_direction, 0, ns_hops - bool(ew_hops), ew_hops + 1, true);
        }
    }
}

void fabric_set_mcast_route(
    volatile tt_l1_ptr LowLatencyMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,   // Ignore this, since Low Latency Mesh Fabric does not support arbitrary 2D Mcasts yet
    uint16_t dst_mesh_id,  // Ignore this, since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops) {
    uint32_t spine_hops = 0;
    uint32_t mcast_branch = 0;

    // For 2D Mcast, mcast spine runs N/S and branches are E/W
    // If api is called with east and/or west hops != 0, it may be a 2D mcast
    // If so, set the forwarding flags for east and/or west branchs.
    if (e_num_hops) {
        mcast_branch |= LowLatencyMeshRoutingFields::FORWARD_EAST;
    }
    if (w_num_hops) {
        mcast_branch |= LowLatencyMeshRoutingFields::FORWARD_WEST;
    }

    if (n_num_hops) {
        // Is a 2D mcast if mcast_branch != 0
        fabric_set_route<true>(packet_header, eth_chan_directions::NORTH, mcast_branch, 0, n_num_hops);
        spine_hops = n_num_hops;
    } else if (s_num_hops) {
        // Is a 2D mcast if mcast_branch != 0
        fabric_set_route<true>(packet_header, eth_chan_directions::SOUTH, mcast_branch, 0, s_num_hops);
        spine_hops = s_num_hops;
    }
    if (e_num_hops) {
        // Is a line mcast if spine_hops == 0
        fabric_set_route<true>(packet_header, eth_chan_directions::EAST, 0, spine_hops, e_num_hops);
        spine_hops += e_num_hops;
    }
    if (w_num_hops) {
        // Is a line mcast if spine_hops == 0
        fabric_set_route<true>(packet_header, eth_chan_directions::WEST, 0, spine_hops, w_num_hops);
    }
}

uint8_t get_router_direction(uint32_t eth_channel) {
    tt_l1_ptr tensix_fabric_connections_l1_info_t* connection_info =
        reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(MEM_TENSIX_FABRIC_CONNECTIONS_BASE);
    return connection_info->read_only[eth_channel].edm_direction;
}

}  // namespace tt::tt_fabric
