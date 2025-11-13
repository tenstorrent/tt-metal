// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

FORCE_INLINE uint32_t recompute_path(
    PACKET_HEADER_TYPE* packet_header,
    ROUTING_FIELDS_TYPE& cached_routing_fields,
    const tt::tt_fabric::routing_l1_info_t& routing_table) {
    if (packet_header->mcast_params_64 != 0 && packet_header->dst_start_mesh_id == routing_table.my_mesh_id) {
        // if (packet_header->dst_start_chip_id != routing_table.my_device_id) {
        //     fabric_set_unicast_route<true, static_cast<eth_chan_directions>(my_direction)>(
        //         packet_header, packet_header->dst_start_chip_id, packet_header->dst_start_mesh_id);
        //     packet_header->route_buffer[1] = LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_NE;
        //     packet_header->route_buffer[2] = LowLatencyMeshRoutingFields::FORWARD_WEST;
        //     packet_header->routing_fields.branch_east_offset = 2;
        // } else {
        //     fabric_set_mcast_route(packet_header);
        //     // packet_header->route_buffer[1] = LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_NE;
        // }
        fabric_set_mcast_route<static_cast<eth_chan_directions>(my_direction)>(packet_header);
        // packet_header->route_buffer[1] = LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_NE;
        // packet_header->routing_fields.branch_east_offset = 2;

        // fabric_set_mcast_route(packet_header);

        // auto dump = reinterpret_cast<tt_l1_ptr uint32_t*>(ROUTING_PATH_BASE_1D);
        // dump[0] = packet_header->route_buffer[0];
        // dump[1] = packet_header->route_buffer[1];
        // dump[2] = packet_header->route_buffer[2];
        // dump[3] = packet_header->route_buffer[3];
        // dump[4] = 0xDDDDDDDD;
        // dump[5] = cached_routing_fields.hop_index;
        // dump[12] =
        //     packet_header->routing_fields.branch_east_offset << 16 |
        //     packet_header->routing_fields.branch_west_offset;
        // dump[13] = routing_table.my_mesh_id << 16 | routing_table.my_device_id;
        // dump[14] = packet_header->dst_start_mesh_id << 16 | packet_header->dst_start_chip_id;
        // dump[15] = 0xdeadbeef;
        // while (true) {}
    } else {
        fabric_set_unicast_route<true, static_cast<eth_chan_directions>(my_direction)>(
            packet_header, packet_header->dst_start_chip_id, packet_header->dst_start_mesh_id);
    }
    cached_routing_fields.hop_index = 0;
    packet_header->routing_fields.hop_index = 0;
    return (uint32_t)packet_header->route_buffer[0];
}

FORCE_INLINE uint32_t get_cmd_with_mesh_boundary_adjustment(
    PACKET_HEADER_TYPE* packet_header,
    ROUTING_FIELDS_TYPE& cached_routing_fields,
    const tt::tt_fabric::routing_l1_info_t& routing_table) {
    uint32_t hop_cmd = packet_header->route_buffer[cached_routing_fields.hop_index];
    if constexpr (is_intermesh_router_on_edge || is_intramesh_router_on_edge) {
        if (hop_cmd == LowLatencyMeshRoutingFields::NOOP) {
            // Arrive at another mesh
            hop_cmd = recompute_path(packet_header, cached_routing_fields, routing_table);
        } else {
            if constexpr (is_intramesh_router_on_edge) {
                // Arrive at exit_node from its mesh. when src != exit node
                if (packet_header->dst_start_mesh_id != routing_table.my_mesh_id) {
                    if constexpr (my_direction == EAST) {
                        if (hop_cmd == LowLatencyMeshRoutingFields::FORWARD_EAST) {
                            hop_cmd = recompute_path(packet_header, cached_routing_fields, routing_table);
                        }
                    } else if constexpr (my_direction == WEST) {
                        if (hop_cmd == LowLatencyMeshRoutingFields::FORWARD_WEST) {
                            hop_cmd = recompute_path(packet_header, cached_routing_fields, routing_table);
                        }
                    } else if constexpr (my_direction == NORTH) {
                        if (hop_cmd == LowLatencyMeshRoutingFields::FORWARD_NORTH) {
                            hop_cmd = recompute_path(packet_header, cached_routing_fields, routing_table);
                        }
                    } else if constexpr (my_direction == SOUTH) {
                        if (hop_cmd == LowLatencyMeshRoutingFields::FORWARD_SOUTH) {
                            hop_cmd = recompute_path(packet_header, cached_routing_fields, routing_table);
                        }
                    } else {
                        ASSERT(false);
                    }
                }
            }
        }
    }
    return hop_cmd;
}
