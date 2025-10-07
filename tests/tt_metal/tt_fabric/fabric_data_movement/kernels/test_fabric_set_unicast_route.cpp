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
    auto expected_packet_header = PacketHeaderPool::allocate_header();
    auto actual_packet_header = PacketHeaderPool::allocate_header();

#ifdef FABRIC_2D
    constexpr uint32_t MAX_ROUTE_BUFFER_SIZE = 32;  // 2D: store 32 bytes (32 packed command bytes)
    volatile uint8_t* actual_route_buffer = actual_packet_header->route_buffer;
    volatile uint8_t* expected_route_buffer = expected_packet_header->route_buffer;
#else
    constexpr uint32_t MAX_ROUTE_BUFFER_SIZE = 4;  // 1D: store only 4 bytes (single 32-bit routing field)
    volatile uint8_t* actual_route_buffer = (uint8_t*)&actual_packet_header->routing_fields.value;
    volatile uint8_t* expected_route_buffer = (uint8_t*)&expected_packet_header->routing_fields.value;
#endif
    for (uint32_t dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(5 + dst_idx * 2);
        uint32_t dst_fabric_dev_id = get_arg_val<uint32_t>(5 + dst_idx * 2 + 1);

        bool routing_success = false;
        for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
            actual_route_buffer[i] = 0;
            expected_route_buffer[i] = 0;
        }

        if (src_mesh_id == dst_mesh_id) {
#ifdef FABRIC_2D
            routing_success = fabric_set_unicast_route(dst_fabric_dev_id, actual_packet_header);
            fabric_set_unicast_route(expected_packet_header, src_fabric_dev_id, dst_fabric_dev_id, dst_mesh_id, ew_dim);
#else
            uint8_t distance_in_hops = (dst_fabric_dev_id > src_fabric_dev_id)
                                           ? (dst_fabric_dev_id - src_fabric_dev_id)
                                           : (src_fabric_dev_id - dst_fabric_dev_id);
            routing_success = fabric_set_unicast_route(dst_fabric_dev_id, actual_packet_header);
            if (distance_in_hops != 0) {
                // For 1D fabric, use LowLatencyPacketHeader with distance in hops
                expected_packet_header->to_chip_unicast(distance_in_hops);
            }
#endif
        } else {
            // TODO: Inter-mesh routing
        }

        // Store results
        uint32_t result_offset = dst_idx * (MAX_ROUTE_BUFFER_SIZE * 2);
        for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
            result_ptr[result_offset + i] = static_cast<uint32_t>(actual_route_buffer[i]);
            result_ptr[result_offset + MAX_ROUTE_BUFFER_SIZE + i] = static_cast<uint32_t>(expected_route_buffer[i]);
        }
    }
}
