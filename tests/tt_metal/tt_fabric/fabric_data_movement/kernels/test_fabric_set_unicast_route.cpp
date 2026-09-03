// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric;

void kernel_main() {
    uint32_t src_mesh_id = get_arg_val<uint32_t>(0);
#ifndef FABRIC_2D
    uint32_t src_fabric_dev_id = get_arg_val<uint32_t>(1);
#endif
    uint32_t result_addr = get_arg_val<uint32_t>(2);
    uint32_t num_devices = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* result_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

#ifdef FABRIC_2D
    constexpr uint32_t ROUTE_BUFFER_SIZE = FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE;
    constexpr uint32_t METADATA_WORDS = 4;  // dst_start_node_id, routing_fields, mcast_params low/high
    constexpr uint32_t RESULT_WORDS_PER_DESTINATION = ROUTE_BUFFER_SIZE + METADATA_WORDS;
    constexpr uint8_t UNTOUCHED = 0xA5;

    alignas(16) uint8_t actual_buffer[PACKET_HEADER_MAX_SIZE];
    auto actual_packet_header = reinterpret_cast<volatile tt_l1_ptr HybridMeshPacketHeader*>(actual_buffer);
    for (uint32_t dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(4 + dst_idx * 2);
        uint32_t dst_fabric_dev_id = get_arg_val<uint32_t>(4 + dst_idx * 2 + 1);

        for (uint32_t i = 0; i < PACKET_HEADER_MAX_SIZE; i++) {
            reinterpret_cast<volatile uint8_t*>(actual_packet_header)[i] = UNTOUCHED;
        }

        if (src_mesh_id == dst_mesh_id) {
            fabric_set_unicast_route(actual_packet_header, dst_fabric_dev_id, dst_mesh_id);
        }

        const uint32_t result_offset = dst_idx * RESULT_WORDS_PER_DESTINATION;
        for (uint32_t i = 0; i < ROUTE_BUFFER_SIZE; i++) {
            result_ptr[result_offset + i] = static_cast<uint32_t>(actual_packet_header->route_buffer[i]);
        }
        result_ptr[result_offset + ROUTE_BUFFER_SIZE] = actual_packet_header->dst_start_node_id;
        result_ptr[result_offset + ROUTE_BUFFER_SIZE + 1] = actual_packet_header->routing_fields.value;
        const uint64_t mcast_params = actual_packet_header->mcast_params_64;
        result_ptr[result_offset + ROUTE_BUFFER_SIZE + 2] = static_cast<uint32_t>(mcast_params);
        result_ptr[result_offset + ROUTE_BUFFER_SIZE + 3] = static_cast<uint32_t>(mcast_params >> 32);
    }
#else
    constexpr uint32_t ROUTE_BUFFER_SIZE = SINGLE_ROUTE_SIZE_1D;
    constexpr uint32_t RESULT_WORDS_PER_DESTINATION = ROUTE_BUFFER_SIZE * 2;

    alignas(16) uint8_t expected_buffer[PACKET_HEADER_MAX_SIZE];
    alignas(16) uint8_t actual_buffer[PACKET_HEADER_MAX_SIZE];
    auto expected_packet_header = reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(expected_buffer);
    auto actual_packet_header = reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(actual_buffer);
    volatile uint8_t* actual_route_buffer =
        reinterpret_cast<volatile uint8_t*>(&actual_packet_header->routing_fields.value);
    volatile uint8_t* expected_route_buffer =
        reinterpret_cast<volatile uint8_t*>(&expected_packet_header->routing_fields.value);

    for (uint32_t dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(4 + dst_idx * 2);
        uint32_t dst_fabric_dev_id = get_arg_val<uint32_t>(4 + dst_idx * 2 + 1);

        for (uint32_t i = 0; i < PACKET_HEADER_MAX_SIZE; i++) {
            reinterpret_cast<volatile uint8_t*>(actual_packet_header)[i] = 0;
            reinterpret_cast<volatile uint8_t*>(expected_packet_header)[i] = 0;
        }

        if (src_mesh_id == dst_mesh_id) {
            uint8_t distance_in_hops = (dst_fabric_dev_id > src_fabric_dev_id)
                                           ? (dst_fabric_dev_id - src_fabric_dev_id)
                                           : (src_fabric_dev_id - dst_fabric_dev_id);
            fabric_set_unicast_route(actual_packet_header, dst_fabric_dev_id);
            if (distance_in_hops != 0) {
                expected_packet_header->to_chip_unicast(distance_in_hops);
            }
        }

        const uint32_t result_offset = dst_idx * RESULT_WORDS_PER_DESTINATION;
        for (uint32_t i = 0; i < ROUTE_BUFFER_SIZE; i++) {
            result_ptr[result_offset + i] = static_cast<uint32_t>(actual_route_buffer[i]);
            result_ptr[result_offset + ROUTE_BUFFER_SIZE + i] = static_cast<uint32_t>(expected_route_buffer[i]);
        }
    }
#endif
}
