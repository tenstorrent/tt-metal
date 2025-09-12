// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric;

void kernel_main() {
    uint32_t src_mesh_id = get_arg_val<uint32_t>(0);
    uint32_t src_fabric_dev_id = get_arg_val<uint32_t>(1);
    uint32_t result_addr = get_arg_val<uint32_t>(2);
    uint32_t num_devices = get_arg_val<uint32_t>(3);
    uint32_t ew_dim = get_arg_val<uint32_t>(4);

    volatile tt_l1_ptr uint32_t* result_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

#ifdef FABRIC_2D
    constexpr uint32_t MAX_ROUTE_BUFFER_SIZE = 32;  // 2D: store 32 bytes (32 packed command bytes)
#else
    constexpr uint32_t MAX_ROUTE_BUFFER_SIZE = 4;  // 1D: store only 4 bytes (single 32-bit routing field)
#endif
    volatile uint8_t expected_route_buffer[MAX_ROUTE_BUFFER_SIZE];
    volatile uint8_t route_buffer[MAX_ROUTE_BUFFER_SIZE];

    auto packet_header = PacketHeaderPool::allocate_header();
    for (uint32_t dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(5 + dst_idx * 2);
        uint32_t dst_fabric_dev_id = get_arg_val<uint32_t>(5 + dst_idx * 2 + 1);

        bool routing_success = false;
        for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
            route_buffer[i] = 0;
        }
        for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
            expected_route_buffer[i] = 0;
        }

        if (src_mesh_id == dst_mesh_id) {
#ifdef FABRIC_2D
            routing_success = get_routing_info<2, true>(dst_fabric_dev_id, route_buffer);
            for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
                packet_header->route_buffer[i] = 0;
            }
            fabric_set_unicast_route(packet_header, src_fabric_dev_id, dst_fabric_dev_id, dst_mesh_id, ew_dim);
            for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
                expected_route_buffer[i] = packet_header->route_buffer[i];
            }
#else
            // Calculate distance in hops for 1D fabric
            uint8_t distance_in_hops = (dst_fabric_dev_id > src_fabric_dev_id)
                                           ? (dst_fabric_dev_id - src_fabric_dev_id)
                                           : (src_fabric_dev_id - dst_fabric_dev_id);
            routing_success = get_routing_info<1, true>(dst_fabric_dev_id, route_buffer);
            if (distance_in_hops != 0) {
                // For 1D fabric, use LowLatencyPacketHeader with distance in hops
                packet_header->to_chip_unicast(distance_in_hops);
                uint32_t routing_value = packet_header->routing_fields.value;
                expected_route_buffer[0] = (routing_value >> 0) & 0xFF;
                expected_route_buffer[1] = (routing_value >> 8) & 0xFF;
                expected_route_buffer[2] = (routing_value >> 16) & 0xFF;
                expected_route_buffer[3] = (routing_value >> 24) & 0xFF;
            }
#endif
        } else {
            // TODO: Inter-mesh routing
        }

        // Store results
        uint32_t result_offset = dst_idx * (MAX_ROUTE_BUFFER_SIZE * 2);
        for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
            result_ptr[result_offset + i] = static_cast<uint32_t>(route_buffer[i]);
            result_ptr[result_offset + MAX_ROUTE_BUFFER_SIZE + i] = static_cast<uint32_t>(expected_route_buffer[i]);
        }
    }
}
