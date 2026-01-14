// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric;

// old runtime implementation moved from tt_fabric_api.h
void fabric_set_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t my_dev_id,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,  // Ignore this, since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
    uint16_t ew_dim) {
    uint32_t ns_hops = 0;
    uint32_t target_dev = dst_dev_id;
    uint32_t target_col = 0;

    tt_l1_ptr routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    uint16_t my_mesh_id = routing_table->my_mesh_id;
    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    packet_header->routing_fields.value = 0;
    packet_header->mcast_params_64 = 0;
    if (my_mesh_id != dst_mesh_id) {
        // TODO: https://github.com/tenstorrent/tt-metal/issues/27881
        // dst_dev_id = exit_node;
    }

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

void kernel_main() {
    uint32_t src_mesh_id = get_arg_val<uint32_t>(0);
    uint32_t src_fabric_dev_id = get_arg_val<uint32_t>(1);
    uint32_t result_addr = get_arg_val<uint32_t>(2);
    uint32_t num_devices = get_arg_val<uint32_t>(3);
    uint32_t ew_dim = get_arg_val<uint32_t>(4);

    volatile tt_l1_ptr uint32_t* result_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);
    uint8_t expected_buffer[PACKET_HEADER_MAX_SIZE];
    uint8_t actual_buffer[PACKET_HEADER_MAX_SIZE];

#ifdef FABRIC_2D
    constexpr uint32_t MAX_ROUTE_BUFFER_SIZE = FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE;
    auto expected_packet_header = reinterpret_cast<volatile tt_l1_ptr HybridMeshPacketHeader*>(expected_buffer);
    auto actual_packet_header = reinterpret_cast<volatile tt_l1_ptr HybridMeshPacketHeader*>(actual_buffer);
    volatile uint8_t* actual_route_buffer = actual_packet_header->route_buffer;
    volatile uint8_t* expected_route_buffer = expected_packet_header->route_buffer;
#else
    constexpr uint32_t MAX_ROUTE_BUFFER_SIZE =
        SINGLE_ROUTE_SIZE_1D;  // 1D: store only 4 bytes (single 32-bit routing field)
    auto expected_packet_header = reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(expected_buffer);
    auto actual_packet_header = reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(actual_buffer);
    volatile uint8_t* actual_route_buffer = (uint8_t*)&actual_packet_header->routing_fields.value;
    volatile uint8_t* expected_route_buffer = (uint8_t*)&expected_packet_header->routing_fields.value;
#endif
    for (uint32_t dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(5 + dst_idx * 2);
        uint32_t dst_fabric_dev_id = get_arg_val<uint32_t>(5 + dst_idx * 2 + 1);

        for (uint32_t i = 0; i < PACKET_HEADER_MAX_SIZE; i++) {
            reinterpret_cast<volatile uint8_t*>(actual_packet_header)[i] = 0;
            reinterpret_cast<volatile uint8_t*>(expected_packet_header)[i] = 0;
        }

        if (src_mesh_id == dst_mesh_id) {
#ifdef FABRIC_2D
            fabric_set_unicast_route(actual_packet_header, dst_fabric_dev_id, src_mesh_id);
            fabric_set_unicast_route(expected_packet_header, src_fabric_dev_id, dst_fabric_dev_id, dst_mesh_id, ew_dim);
#else
            uint8_t distance_in_hops = (dst_fabric_dev_id > src_fabric_dev_id)
                                           ? (dst_fabric_dev_id - src_fabric_dev_id)
                                           : (src_fabric_dev_id - dst_fabric_dev_id);
            fabric_set_unicast_route(actual_packet_header, dst_fabric_dev_id);
            if (distance_in_hops != 0) {
                // For 1D fabric, use HybridMeshPacketHeader with distance in hops
                expected_packet_header->to_chip_unicast(distance_in_hops);
            }
#endif
        } else {
            // TODO: Inter-mesh routing
            //       https://github.com/tenstorrent/tt-metal/issues/27881
        }

        // Store results
        uint32_t result_offset = dst_idx * (MAX_ROUTE_BUFFER_SIZE * 2);
        for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
            result_ptr[result_offset + i] = static_cast<uint32_t>(actual_route_buffer[i]);
            result_ptr[result_offset + MAX_ROUTE_BUFFER_SIZE + i] = static_cast<uint32_t>(expected_route_buffer[i]);
        }
    }
}
