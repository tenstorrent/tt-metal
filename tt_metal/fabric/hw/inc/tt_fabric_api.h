// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_fabric.h"
#include "tt_fabric_interface.h"
#include "eth_chan_noc_mapping.h"
#include "fabric_edm_packet_header.hpp"
#include <type_traits>

namespace tt::tt_fabric {

enum AsyncWriteMode : uint8_t {
    ADD_PR = 0x01,
    SEND_PR = 0x02,
    PUSH = SEND_PR,
    ADD_HEADER = 0x04,
    ADD_AND_SEND_PR = ADD_PR | SEND_PR,
    ALL = ADD_HEADER | ADD_PR | SEND_PR,
};

enum ClientDataMode : uint8_t {
    PACKETIZED_DATA = 0x0,
    RAW_DATA = 0x1,
};

enum RoutingType : uint8_t {
    ROUTING_TABLE,
    ROUTER_XY,
};

template <typename ClientInterfaceType>
inline uint32_t get_next_hop_router_noc_xy(
    ClientInterfaceType client_interface, uint32_t routing_plane, uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    ASSERT(routing_plane < client_interface->num_routing_planes);
    fabric_router_l1_config_t* routing_table = (fabric_router_l1_config_t*)client_interface->routing_tables_l1_offset;
    if (dst_mesh_id != routing_table[routing_plane].my_mesh_id) {
        uint32_t next_port = routing_table[routing_plane].inter_mesh_table.dest_entry[dst_mesh_id];
        ASSERT(next_port != INVALID_DIRECTION);
        return eth_chan_to_noc_xy[noc_index][next_port];
    } else {
        uint32_t next_port = routing_table[routing_plane].intra_mesh_table.dest_entry[dst_dev_id];
        ASSERT(next_port != INVALID_DIRECTION);
        return eth_chan_to_noc_xy[noc_index][next_port];
    }
}

inline eth_chan_directions get_next_hop_router_direction(uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    tt_l1_ptr tensix_routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr tensix_routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    if (dst_mesh_id == routing_table->mesh_id) {
        return routing_table->intra_mesh_routing_table[dst_dev_id];
    } else {
        return routing_table->inter_mesh_routing_table[dst_mesh_id];
    }
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA>
inline void fabric_setup_pull_request(
    volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface, uint32_t src_addr, uint32_t size) {
    uint32_t size_in_words = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
    // TODO: Could return this value to the user and take this as an arg to avoid repeated lookup
    // Added here to avoid user having to declare globals
    uint64_t xy_local_addr = get_noc_addr(0);
    client_interface->local_pull_request.pull_request.wr_ptr = size_in_words;
    client_interface->local_pull_request.pull_request.rd_ptr = 0;
    client_interface->local_pull_request.pull_request.size = size;
    client_interface->local_pull_request.pull_request.buffer_size = size_in_words;
    client_interface->local_pull_request.pull_request.buffer_start = xy_local_addr + src_addr;
    client_interface->local_pull_request.pull_request.words_written = size_in_words;
    client_interface->local_pull_request.pull_request.words_read = 0;
    client_interface->local_pull_request.pull_request.ack_addr =
        xy_local_addr + (uint32_t)&client_interface->local_pull_request.pull_request.words_read;
    if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
        client_interface->local_pull_request.pull_request.flags = FORWARD;
    } else {
        client_interface->local_pull_request.pull_request.flags = PACK_N_FORWARD;
    }
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA, RoutingType routing_type = RoutingType::ROUTER_XY>
inline void fabric_send_pull_request(
    volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface,
    uint32_t routing,  // routing refers to the router noc xy to use when using ROUTER_XY,
                       // and the routing plane to use when using ROUTING_TABLE
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    volatile tt_l1_ptr packet_header_t* header) {
    uint64_t router_addr;
    if constexpr (routing_type == RoutingType::ROUTING_TABLE) {
        router_addr = get_noc_addr_helper(
            get_next_hop_router_noc_xy(client_interface, routing, dst_mesh_id, dst_dev_id),
            FABRIC_ROUTER_REQ_QUEUE_START);
    } else {
        router_addr = get_noc_addr_helper(routing, FABRIC_ROUTER_REQ_QUEUE_START);
    }

    volatile local_pull_request_t* pull_request = (volatile local_pull_request_t*)&client_interface->local_pull_request;

    uint32_t increment;
    if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
        increment = 1;
    } else {
        // when sending raw data, we reserve two request slots in router.
        // first slot is pull request, second slot holds packet header
        // since the client data buffer does not contain the packet header.
        increment = 2;
    }

    tt_fabric_reserve_pull_request_slot(router_addr, pull_request, increment);
    uint32_t wrptr = pull_request->wrptr.ptr;
    if constexpr (data_mode == ClientDataMode::RAW_DATA) {
        uint32_t header_wrptr = (wrptr + 1) & CHAN_REQ_BUF_PTR_MASK;
        tt_fabric_check_pull_request_slot<true>(router_addr, pull_request, header_wrptr);
        uint32_t header_wr_index = header_wrptr & CHAN_REQ_BUF_SIZE_MASK;
        uint64_t noc_addr = router_addr + offsetof(chan_req_buf, chan_req) + header_wr_index * sizeof(pull_request_t);
        noc_async_write_one_packet((uint32_t)header, noc_addr, sizeof(pull_request_t), noc_index);
    } else {
        tt_fabric_check_pull_request_slot<true>(router_addr, pull_request, wrptr);
    }

    tt_fabric_send_pull_request(router_addr, pull_request);
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA, RoutingType routing_type = RoutingType::ROUTER_XY>
inline void fabric_send_pull_request(
    volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface,
    uint32_t routing,  // routing refers to the router noc xy to use when using ROUTER_XY,
                       // and the routing plane to use when using ROUTING_TABLE
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint32_t header_id = 0) {
    fabric_send_pull_request<data_mode, routing_type>(
        client_interface,
        routing,
        dst_mesh_id,
        dst_dev_id,
        reinterpret_cast<volatile tt_l1_ptr packet_header_t*>(
            &client_interface->header_buffer[header_id]) /*only used for raw mode*/);
}

FORCE_INLINE void fabric_wait_for_pull_request_words_flushed(
    volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface, uint32_t words) {
    while (client_interface->local_pull_request.pull_request.words_read < words) {
        invalidate_l1_cache();
#pragma GCC unroll 4
        for (int i = 0; i < 4; i++) {
            asm("nop");
        }
    }
}

inline void fabric_wait_for_pull_request_bytes_flushed(
    volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface, uint32_t size) {
    uint32_t size_in_words = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
    fabric_wait_for_pull_request_words_flushed(client_interface, size_in_words);
}

inline void fabric_wait_for_pull_request_flushed(volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface) {
    uint32_t words_written = client_interface->local_pull_request.pull_request.words_written;
    fabric_wait_for_pull_request_words_flushed(client_interface, words_written);
}

template <typename ClientInterfaceType, ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA>
static inline
#if defined(FVC_MODE_PULL) || !defined(LOW_LATENCY_ROUTING)
    packet_header_t*
#else
    low_latency_packet_header_t*
#endif
    extract_packet_header(
        tt_l1_ptr ClientInterfaceType client_interface,
        uint32_t src_addr,
        uint16_t dst_mesh_id,
        uint16_t dst_dev_id,
        uint32_t header_id = 0) {
    static_assert(
        (data_mode == ClientDataMode::RAW_DATA &&
         (std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
          std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>)) ||
            data_mode == ClientDataMode::PACKETIZED_DATA,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");
#if defined(FVC_MODE_PULL) || !defined(LOW_LATENCY_ROUTING)
    packet_header_t* packet_header;
    if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
        packet_header = (packet_header_t*)(src_addr);
    } else {
        packet_header = (packet_header_t*)&client_interface->header_buffer[header_id];
    }
#else
    low_latency_packet_header_t* packet_header;
    if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
        packet_header = (low_latency_packet_header_t*)(src_addr);
    } else {
        packet_header = (low_latency_packet_header_t*)&client_interface->header_buffer[header_id];
    }
#endif
    return packet_header;
}

template <typename HeaderType>
static inline void fabric_async_write_add_header_impl(
    HeaderType packet_header,
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size  // number of bytes to write to remote destination
) {
    static_assert(
        std::is_same_v<HeaderType, packet_header_t*> || std::is_same_v<HeaderType, low_latency_packet_header_t*>);
    if constexpr (std::is_same_v<HeaderType, packet_header_t*>) {
        packet_header->routing.flags = FORWARD;
        packet_header->routing.packet_size_bytes = size;
        packet_header->routing.dst_mesh_id = dst_mesh_id;
        packet_header->routing.dst_dev_id = dst_dev_id;
        packet_header->session.command = ASYNC_WR;
        packet_header->session.target_offset_l = (uint32_t)dst_addr;
        packet_header->session.target_offset_h = dst_addr >> 32;
        tt_fabric_add_header_checksum(packet_header);
    } else {
        packet_header->routing.packet_size_bytes = size;
        packet_header->routing.target_offset_l = (uint32_t)dst_addr;
        packet_header->routing.target_offset_h = dst_addr >> 32;
        packet_header->routing.command = ASYNC_WR;
    }
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA, typename ClientInterfaceType>
inline void fabric_async_write_add_header(
    tt_l1_ptr ClientInterfaceType client_interface,
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");
    fabric_async_write_add_header_impl(
        extract_packet_header<ClientInterfaceType, data_mode>(
            client_interface, src_addr, dst_mesh_id, dst_dev_id, header_id),
        src_addr,
        dst_mesh_id,
        dst_dev_id,
        dst_addr,
        size);
}

template <bool mcast = false>
void fabric_set_route(
    low_latency_packet_header_t* packet_header,
    eth_chan_directions direction,
    uint32_t start_hop,
    uint32_t num_hops,
    bool terminate = false) {
    uint32_t local_packet = 0;
    uint32_t forward_packet = 0;
    uint32_t value = 0;
    switch (direction) {
        case eth_chan_directions::EAST:
            local_packet = tt_low_latency_routing_vector::FORWARD_WEST;
            forward_packet = tt_low_latency_routing_vector::FORWARD_EAST;
            break;
        case eth_chan_directions::WEST:
            local_packet = tt_low_latency_routing_vector::FORWARD_EAST;
            forward_packet = tt_low_latency_routing_vector::FORWARD_WEST;
            break;
        case eth_chan_directions::NORTH:
            local_packet = tt_low_latency_routing_vector::FORWARD_SOUTH;
            forward_packet = tt_low_latency_routing_vector::FORWARD_NORTH;
            break;
        case eth_chan_directions::SOUTH:
            local_packet = tt_low_latency_routing_vector::FORWARD_NORTH;
            forward_packet = tt_low_latency_routing_vector::FORWARD_SOUTH;
            break;
        default: ASSERT(false);
    }

    uint8_t* route_vector = (uint8_t*)packet_header->routing.route_vector.value;
    uint32_t local_val;
    uint32_t forward_val;
    uint32_t end_hop = start_hop + num_hops;
    for (uint32_t i = start_hop; i < end_hop; i++) {
        if constexpr (mcast) {
            forward_val = i == end_hop - 1 ? 0 : forward_packet;
            local_val = local_packet;
        } else {
            forward_val = terminate ? (i == end_hop - 1 ? 0 : forward_packet) : forward_packet;
            local_val = terminate ? (i == end_hop - 1 ? local_packet : 0) : 0;
        }
        route_vector[i] = local_val | forward_val;
    }
    packet_header->routing.route_vector.hop_index = 0;
}

void fabric_set_unicast_route(
    volatile fabric_push_client_interface_t* client_interface,
    low_latency_packet_header_t* packet_header,
    uint32_t outgoing_direction,
    uint16_t dst_dev_id) {
    fabric_router_l1_config_t* routing_table = (fabric_router_l1_config_t*)client_interface->routing_tables_l1_offset;
    uint32_t my_dev_id = routing_table->my_device_id;
    uint32_t ew_dim = routing_table->east_dim;
    if (outgoing_direction == eth_chan_directions::EAST || outgoing_direction == eth_chan_directions::WEST) {
        uint32_t ew_hops = my_dev_id < dst_dev_id ? dst_dev_id - my_dev_id : my_dev_id - dst_dev_id;
        fabric_set_route(packet_header, (eth_chan_directions)outgoing_direction, 0, ew_hops, true);
    } else {
        // First hop is north/south. Calculate the number of required hops before turning east/west
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
        ns_hops = target_col > my_col ? target_col - my_col : my_col - target_col;
        // determine the east/west hops
        uint32_t turn_direction = my_dev < target_dev ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        uint32_t ew_hops = (my_dev < target_dev) ? target_dev - my_dev : my_dev - target_dev;
        if (ew_hops) {
            ns_hops--;
            ew_hops++;
        }
        fabric_set_route(packet_header, (eth_chan_directions)outgoing_direction, 0, ns_hops, ew_hops == 0);
        if (ew_hops) {
            fabric_set_route(packet_header, (eth_chan_directions)turn_direction, ns_hops, ew_hops, true);
        }
    }
}

void fabric_set_mcast_route(low_latency_packet_header_t* packet_header, eth_chan_directions direction, uint32_t hops) {
    fabric_set_route<true>(packet_header, (eth_chan_directions)direction, 0, hops);
}

template <bool mcast = false>
void fabric_set_route(
    LowLatencyMeshPacketHeader* packet_header,
    eth_chan_directions direction,
    uint32_t start_hop,
    uint32_t num_hops,
    bool terminate = false) {
    uint32_t local_packet = 0;
    uint32_t forward_packet = 0;
    uint32_t value = 0;
    switch (direction) {
        case eth_chan_directions::EAST:
            local_packet = tt_low_latency_routing_vector::FORWARD_WEST;
            forward_packet = tt_low_latency_routing_vector::FORWARD_EAST;
            break;
        case eth_chan_directions::WEST:
            local_packet = tt_low_latency_routing_vector::FORWARD_EAST;
            forward_packet = tt_low_latency_routing_vector::FORWARD_WEST;
            break;
        case eth_chan_directions::NORTH:
            local_packet = tt_low_latency_routing_vector::FORWARD_SOUTH;
            forward_packet = tt_low_latency_routing_vector::FORWARD_NORTH;
            break;
        case eth_chan_directions::SOUTH:
            local_packet = tt_low_latency_routing_vector::FORWARD_NORTH;
            forward_packet = tt_low_latency_routing_vector::FORWARD_SOUTH;
            break;
        default: ASSERT(false);
    }

    uint8_t* route_vector = packet_header->route_buffer;
    uint32_t local_val;
    uint32_t forward_val;
    uint32_t end_hop = start_hop + num_hops;
    for (uint32_t i = start_hop; i < end_hop; i++) {
        if constexpr (mcast) {
            forward_val = i == end_hop - 1 ? 0 : forward_packet;
            local_val = local_packet;
        } else {
            forward_val = terminate ? (i == end_hop - 1 ? 0 : forward_packet) : forward_packet;
            local_val = terminate ? (i == end_hop - 1 ? local_packet : 0) : 0;
        }
        route_vector[i] = local_val | forward_val;
    }
    packet_header->routing_fields.value = 0;
}

void fabric_set_unicast_route(
    MeshPacketHeader* packet_header,
    eth_chan_directions outgoing_direction,  // Ignore this: Dynamic Routing does not need outgoing_direction specified
    uint16_t my_dev_id,                      // Ignore this: Dynamic Routing does not need src chip ID
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t ew_dim  // Ignore this: Dynamic Routing does not need mesh dimensions
) {
    packet_header->dst_start_chip_id = dst_dev_id;
    packet_header->dst_start_mesh_id = dst_mesh_id;
    packet_header->mcast_params[0] = 0;
    packet_header->mcast_params[1] = 0;
    packet_header->mcast_params[2] = 0;
    packet_header->mcast_params[3] = 0;
    packet_header->is_mcast_active = 0;
}

void fabric_set_mcast_route(
    MeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops) {
    packet_header->dst_start_chip_id = dst_dev_id;
    packet_header->dst_start_mesh_id = dst_mesh_id;
    packet_header->mcast_params[0] = e_num_hops;
    packet_header->mcast_params[1] = w_num_hops;
    packet_header->mcast_params[2] = n_num_hops;
    packet_header->mcast_params[3] = s_num_hops;
    packet_header->is_mcast_active = 0;
}

void fabric_set_unicast_route(
    LowLatencyMeshPacketHeader* packet_header,
    eth_chan_directions outgoing_direction,
    uint16_t my_dev_id,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,  // Ignore this, since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
    uint16_t ew_dim) {
    if (outgoing_direction == eth_chan_directions::EAST || outgoing_direction == eth_chan_directions::WEST) {
        uint32_t ew_hops = my_dev_id < dst_dev_id ? dst_dev_id - my_dev_id : my_dev_id - dst_dev_id;
        fabric_set_route(packet_header, (eth_chan_directions)outgoing_direction, 0, ew_hops, true);
    } else {
        // First hop is north/south. Calculate the number of required hops before turning east/west
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
        ns_hops = target_col > my_col ? target_col - my_col : my_col - target_col;
        // determine the east/west hops
        uint32_t turn_direction = my_dev < target_dev ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        uint32_t ew_hops = (my_dev < target_dev) ? target_dev - my_dev : my_dev - target_dev;
        if (ew_hops) {
            ns_hops--;
            ew_hops++;
        }
        fabric_set_route(packet_header, (eth_chan_directions)outgoing_direction, 0, ns_hops, ew_hops == 0);
        if (ew_hops) {
            fabric_set_route(packet_header, (eth_chan_directions)turn_direction, ns_hops, ew_hops, true);
        }
    }
}

void fabric_set_mcast_route(
    LowLatencyMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,   // Ignore this, since Low Latency Mesh Fabric does not support arbitrary 2D Mcasts yet
    uint16_t dst_mesh_id,  // Ignore this, since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops) {
    if (e_num_hops) {
        fabric_set_route<true>(packet_header, eth_chan_directions::EAST, 0, e_num_hops);
    } else if (w_num_hops) {
        fabric_set_route<true>(packet_header, eth_chan_directions::WEST, 0, w_num_hops);
    } else if (n_num_hops) {
        fabric_set_route<true>(packet_header, eth_chan_directions::NORTH, 0, n_num_hops);
    } else if (s_num_hops) {
        fabric_set_route<true>(packet_header, eth_chan_directions::SOUTH, 0, s_num_hops);
    }
}

inline void fabric_client_connect(
    volatile fabric_push_client_interface_t* client_interface,
    int32_t routing_plane,
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id) {
    uint32_t direction = get_next_hop_router_direction(dst_mesh_id, dst_dev_id);
    uint32_t router_addr_h = get_next_hop_router_noc_xy(client_interface, routing_plane, dst_mesh_id, dst_dev_id);

    uint64_t client_q_addr = get_noc_addr_helper(router_addr_h, FABRIC_ROUTER_CLIENT_QUEUE_START);
    volatile fabric_push_client_queue_local_t* local_req_entry = &(client_interface->local_client_req_entry);

    // get client 'id'
    noc_fast_atomic_increment<DM_DEDICATED_NOC, true>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        client_q_addr + offsetof(fabric_push_client_queue_t, client_idx_counter),
        NOC_UNICAST_WRITE_VC,
        1,
        31,
        false,
        false,
        (uint32_t)&(local_req_entry->my_client_idx.ptr));
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));

    uint64_t curr_client_idx_addr = client_q_addr + offsetof(fabric_push_client_queue_t, curr_client_idx);
    // wait until the client ahead in the queue disconnects
    while (true) {
        noc_async_read_one_packet(curr_client_idx_addr, (uint32_t)&(local_req_entry->remote_curr_client_idx.ptr), 4);
        noc_async_read_barrier();
        if (local_req_entry->my_client_idx.ptr == local_req_entry->remote_curr_client_idx.ptr) {
            break;
        }
    }

    uint64_t router_wr_ptr_addr = client_q_addr + offsetof(fabric_push_client_queue_t, router_wr_ptr);
    noc_async_read_one_packet(router_wr_ptr_addr, (uint32_t)&(local_req_entry->remote_router_wr_ptr.ptr), 4);
    noc_async_read_barrier();

    uint64_t router_addr = get_noc_addr_helper(router_addr_h, FABRIC_ROUTER_REQ_QUEUE_START);
    router_addr += direction * sizeof(uint64_t);
    // stream register to receive router buffer space available updates.
    uint64_t xy_local_addr = get_noc_addr(0);
    noc_inline_dw_write<true>(
        router_addr,
        (STREAM_REG_ADDR(
            STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX)));
    noc_inline_dw_write(router_addr + sizeof(uint32_t), xy_local_addr >> NOC_ADDR_COORD_SHIFT);
    client_interface->router_addr_h = router_addr_h;
    client_interface->buffer_size = FABRIC_ROUTER_OUTBOUND_BUF_SLOTS;
    client_interface->wr_ptr = local_req_entry->remote_router_wr_ptr.ptr;
    client_interface->buffer_start = FABRIC_ROUTER_DATA_BUF_START + direction * FABRIC_ROUTER_OUTBOUND_BUF_SIZE;
    client_interface->router_push_addr = (STREAM_REG_ADDR(
        STREAM_ID_NOC_WORDS_RECEIVED + direction, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
    client_interface->router_space =
        (STREAM_REG_ADDR(STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
    client_interface->update_router_space =
        (STREAM_REG_ADDR(STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
    *(uint32_t*)(STREAM_REG_ADDR(STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) =
        client_interface->buffer_size;
}

inline void fabric_client_disconnect(volatile tt_l1_ptr fabric_push_client_interface_t* client_interface) {
    // wait for slots to drain
    while (*(uint32_t*)(client_interface->router_space) != FABRIC_ROUTER_OUTBOUND_BUF_SLOTS) {
        invalidate_l1_cache();
    }

    uint64_t client_q_addr = get_noc_addr_helper(client_interface->router_addr_h, FABRIC_ROUTER_CLIENT_QUEUE_START);

    // update wr ptr for the next client
    noc_inline_dw_write<true>(
        client_q_addr + offsetof(fabric_push_client_queue_t, router_wr_ptr), client_interface->wr_ptr);

    // update curr client index so that the next client in the queue can connect
    noc_fast_atomic_increment<DM_DEDICATED_NOC, true>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        client_q_addr + offsetof(fabric_push_client_queue_t, curr_client_idx),
        NOC_UNICAST_WRITE_VC,
        1,
        31,
        false,
        false);
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA>
inline void fabric_async_write_push_data(
    volatile tt_l1_ptr fabric_push_client_interface_t* client_interface,
    uint32_t src_addr,
    uint32_t size,
    volatile tt_l1_ptr packet_header_t* header) {
    uint64_t push_addr = get_noc_addr_helper(client_interface->router_addr_h, client_interface->router_push_addr);
    uint32_t router_buf_space = *(volatile uint32_t*)client_interface->router_space;
    while (router_buf_space == 0) {
        invalidate_l1_cache();
        router_buf_space = *(volatile uint32_t*)client_interface->router_space;
    }

    uint64_t buffer_wr_addr = get_noc_addr_helper(
        client_interface->router_addr_h,
        (client_interface->buffer_start + (client_interface->wr_ptr * FABRIC_ROUTER_BUF_SLOT_SIZE)));
    if constexpr (data_mode == ClientDataMode::RAW_DATA) {
        // In raw mode, pick up the header from header buffer in client interface.
        noc_async_write_one_packet((uint32_t)header, buffer_wr_addr, PACKET_HEADER_SIZE_BYTES, noc_index);
        buffer_wr_addr += PACKET_HEADER_SIZE_BYTES;
        size -= PACKET_HEADER_SIZE_BYTES;
    }
    noc_async_write_one_packet(src_addr, buffer_wr_addr, size, noc_index);
    noc_inline_dw_write<true>(push_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
    client_interface->wr_ptr++;
    *(volatile uint32_t*)client_interface->update_router_space = (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC;
    if (client_interface->wr_ptr >= client_interface->buffer_size) {
        client_interface->wr_ptr -= client_interface->buffer_size;
    }
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA>
inline void fabric_async_write_push_data(
    volatile tt_l1_ptr fabric_push_client_interface_t* client_interface,
    uint32_t src_addr,
    uint32_t size,
    uint32_t header_id = 0) {
    fabric_async_write_push_data<data_mode>(
        client_interface,
        src_addr,
        size,
        reinterpret_cast<volatile tt_l1_ptr packet_header_t*>(&client_interface->header_buffer[header_id]));
}

template <
    ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA,
    AsyncWriteMode mode = AsyncWriteMode::ALL,
    RoutingType routing_type = RoutingType::ROUTER_XY,
    typename ClientInterfaceType>
inline void fabric_async_write(
    tt_l1_ptr ClientInterfaceType client_interface,
    uint32_t routing,   // routing refers to router noc xy or routing plane
                        // and the routing plane to use when using ROUTING_TABLE
                        // or the network plane to use for this transaction for push mode
    uint32_t src_addr,  // source address in sender memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");

    if constexpr (mode & AsyncWriteMode::ADD_HEADER) {
        fabric_async_write_add_header<data_mode, ClientInterfaceType>(
            client_interface, src_addr, dst_mesh_id, dst_dev_id, dst_addr, size, header_id);
#if !defined(FVC_MODE_PULL) && defined(LOW_LATENCY_ROUTING)
        uint32_t outgoing_direction = (uint32_t)get_next_hop_router_direction(dst_mesh_id, dst_dev_id);
        if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
            fabric_set_unicast_route(
                client_interface, (low_latency_packet_header_t*)(src_addr), outgoing_direction, dst_dev_id);
        } else {
            fabric_set_unicast_route(
                client_interface,
                (low_latency_packet_header_t*)&client_interface->header_buffer[header_id],
                outgoing_direction,
                dst_dev_id);
        }
#endif
    }

    if constexpr (std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*>) {
        if constexpr (mode & AsyncWriteMode::ADD_PR) {
            if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
                fabric_setup_pull_request<data_mode>(client_interface, src_addr, size);
            } else {
                fabric_setup_pull_request<data_mode>(client_interface, src_addr, size - PACKET_HEADER_SIZE_BYTES);
            }
        }
        if constexpr (mode & AsyncWriteMode::SEND_PR) {
            fabric_send_pull_request<data_mode, routing_type>(
                client_interface, routing, dst_mesh_id, dst_dev_id, header_id);
        }
    } else {
        if constexpr (mode & AsyncWriteMode::PUSH) {
            fabric_async_write_push_data<data_mode>(client_interface, src_addr, size, header_id);
        }
    }
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA, typename ClientInterfaceType>
inline void fabric_async_write_multicast_add_header(
    tt_l1_ptr ClientInterfaceType client_interface,
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint16_t e_depth,
    uint16_t w_depth,
    uint16_t n_depth,
    uint16_t s_depth,
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");
    if constexpr (std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*>) {
        packet_header_t* packet_header;
        if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
            packet_header = (packet_header_t*)(src_addr);
        } else {
            packet_header = (packet_header_t*)&client_interface->header_buffer[header_id];
        }
        packet_header->routing.flags = FORWARD | MCAST_DATA;
        packet_header->routing.packet_size_bytes = size;
        packet_header->routing.dst_mesh_id = dst_mesh_id;
        packet_header->routing.dst_dev_id = dst_dev_id;
        packet_header->session.command = ASYNC_WR;
        packet_header->session.target_offset_l = (uint32_t)dst_addr;
        packet_header->session.target_offset_h = dst_addr >> 32;
        packet_header->packet_parameters.mcast_parameters.east = e_depth;
        packet_header->packet_parameters.mcast_parameters.west = w_depth;
        packet_header->packet_parameters.mcast_parameters.north = n_depth;
        packet_header->packet_parameters.mcast_parameters.south = s_depth;
        tt_fabric_add_header_checksum(packet_header);
    } else {
        low_latency_packet_header_t* packet_header;
        if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
            packet_header = (low_latency_packet_header_t*)(src_addr);
        } else {
            packet_header = (low_latency_packet_header_t*)&client_interface->header_buffer[header_id];
        }
        packet_header->routing.packet_size_bytes = size;
        packet_header->routing.command = ASYNC_WR;
        packet_header->routing.target_offset_l = (uint32_t)dst_addr;
        packet_header->routing.target_offset_h = dst_addr >> 32;
        if (e_depth) {
            fabric_set_mcast_route(packet_header, eth_chan_directions::EAST, e_depth);
        } else if (w_depth) {
            fabric_set_mcast_route(packet_header, eth_chan_directions::WEST, w_depth);
        } else if (n_depth) {
            fabric_set_mcast_route(packet_header, eth_chan_directions::NORTH, n_depth);
        } else if (s_depth) {
            fabric_set_mcast_route(packet_header, eth_chan_directions::SOUTH, s_depth);
        }
    }
}

// Write packetized data over fabric to dst_mesh, dst_dev.
// Packet is at src_addr in sender L1.
template <
    ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA,
    AsyncWriteMode mode = AsyncWriteMode::ALL,
    RoutingType routing_type = RoutingType::ROUTER_XY,
    typename ClientInterfaceType>
inline void fabric_async_write_multicast(
    tt_l1_ptr ClientInterfaceType client_interface,
    uint32_t routing,   // routing refers to the router noc xy to use when using ROUTER_XY,
                        // and the routing plane to use when using ROUTING_TABLE
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint16_t e_depth,
    uint16_t w_depth,
    uint16_t n_depth,
    uint16_t s_depth,
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");

    if constexpr (mode & AsyncWriteMode::ADD_HEADER) {
        fabric_async_write_multicast_add_header<data_mode, ClientInterfaceType>(
            client_interface,
            src_addr,
            dst_mesh_id,
            dst_dev_id,
            dst_addr,
            size,
            e_depth,
            w_depth,
            n_depth,
            s_depth,
            header_id);
    }

    if constexpr (std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*>) {
        if constexpr (mode & AsyncWriteMode::ADD_PR) {
            if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
                fabric_setup_pull_request<data_mode>(client_interface, src_addr, size);
            } else {
                fabric_setup_pull_request<data_mode>(client_interface, src_addr, size - PACKET_HEADER_SIZE_BYTES);
            }
        }

        if constexpr (mode & AsyncWriteMode::SEND_PR) {
            fabric_send_pull_request<data_mode, routing_type>(client_interface, routing, dst_mesh_id, dst_dev_id);
        }
    } else {
        if constexpr (mode & AsyncWriteMode::PUSH) {
            fabric_async_write_push_data<data_mode>(client_interface, src_addr, size, header_id);
        }
    }
}

template <typename HeaderType>
static inline void fabric_atomic_inc_add_header_impl(
    HeaderType packet_header,
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t atomic_inc,
    uint32_t wrap_boundary) {
    static_assert(
        std::is_same_v<HeaderType, packet_header_t*> || std::is_same_v<HeaderType, low_latency_packet_header_t*>);
    if constexpr (std::is_same_v<HeaderType, packet_header_t*>) {
        packet_header->routing.flags = INLINE_FORWARD;
        packet_header->routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
        packet_header->routing.dst_mesh_id = dst_mesh_id;
        packet_header->routing.dst_dev_id = dst_dev_id;
        packet_header->session.command = ATOMIC_INC;
        packet_header->session.target_offset_l = (uint32_t)dst_addr;
        packet_header->session.target_offset_h = dst_addr >> 32;
        packet_header->packet_parameters.atomic_parameters.wrap_boundary = wrap_boundary;
        packet_header->packet_parameters.atomic_parameters.increment = atomic_inc;
        tt_fabric_add_header_checksum(packet_header);
    } else {
        packet_header->routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
        packet_header->routing.atomic_offset_l = (uint32_t)dst_addr;
        packet_header->routing.atomic_offset_h = dst_addr >> 32;
        packet_header->routing.atomic_increment = atomic_inc;
        packet_header->routing.atomic_wrap = wrap_boundary;
        packet_header->routing.command = ATOMIC_INC;
    }
}

inline void fabric_atomic_inc_add_header(
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t atomic_inc,
    uint32_t wrap_boundary) {
    fabric_atomic_inc_add_header_impl(
        extract_packet_header(nullptr, src_addr, dst_mesh_id, dst_dev_id),
        src_addr,
        dst_mesh_id,
        dst_dev_id,
        dst_addr,
        atomic_inc,
        wrap_boundary);
}

// Write packetized data over fabric to dst_mesh, dst_dev.
// Packet is at src_addr in sender L1.
template <
    ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA,
    AsyncWriteMode mode = AsyncWriteMode::ALL,
    RoutingType routing_type = RoutingType::ROUTER_XY,
    typename ClientInterfaceType>
inline void fabric_atomic_inc(
    tt_l1_ptr ClientInterfaceType client_interface,
    uint32_t routing,   // routing refers to the router noc xy to use when using ROUTER_XY,
                        // and the routing plane to use when using ROUTING_TABLE
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t atomic_inc,
    uint32_t wrap_boundary,
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");

    if constexpr (mode & AsyncWriteMode::ADD_HEADER) {
        fabric_atomic_inc_add_header(src_addr, dst_mesh_id, dst_dev_id, dst_addr, atomic_inc, wrap_boundary);
#if !defined(FVC_MODE_PULL) && defined(LOW_LATENCY_ROUTING)
        uint32_t outgoing_direction = (uint32_t)get_next_hop_router_direction(dst_mesh_id, dst_dev_id);
        fabric_set_unicast_route(
            client_interface, (low_latency_packet_header_t*)(src_addr), outgoing_direction, dst_dev_id);
#endif
    }

    if constexpr (std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*>) {
        if constexpr (mode & AsyncWriteMode::ADD_PR) {
            fabric_setup_pull_request(client_interface, src_addr, PACKET_HEADER_SIZE_BYTES);
        }

        if constexpr (mode & AsyncWriteMode::SEND_PR) {
            fabric_send_pull_request<data_mode, routing_type>(client_interface, routing, dst_mesh_id, dst_dev_id);
        }
    } else {
        if constexpr (mode & AsyncWriteMode::PUSH) {
            fabric_async_write_push_data<data_mode>(client_interface, src_addr, PACKET_HEADER_SIZE_BYTES, header_id);
        }
    }
}

template <typename HeaderType>
static inline void fabric_async_write_atomic_inc_add_header_impl(
    HeaderType packet_header,
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_write_addr,
    uint64_t dst_atomic_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t atomic_inc,
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<HeaderType, packet_header_t*> || std::is_same_v<HeaderType, low_latency_packet_header_t*>);
    if constexpr (std::is_same_v<HeaderType, packet_header_t*>) {
        packet_header->routing.flags = FORWARD;
        packet_header->routing.packet_size_bytes = size;
        packet_header->routing.dst_mesh_id = dst_mesh_id;
        packet_header->routing.dst_dev_id = dst_dev_id;
        packet_header->session.command = ASYNC_WR | ATOMIC_INC;
        packet_header->session.target_offset_l = (uint32_t)dst_write_addr;
        packet_header->session.target_offset_h = dst_atomic_addr >> 32;
        packet_header->packet_parameters.async_wr_atomic_parameters.noc_xy = dst_atomic_addr >> 32;
        packet_header->packet_parameters.async_wr_atomic_parameters.l1_offset = (uint32_t)dst_atomic_addr;
        packet_header->packet_parameters.async_wr_atomic_parameters.increment = atomic_inc;
        tt_fabric_add_header_checksum(packet_header);
    } else {
        packet_header->routing.packet_size_bytes = size;
        packet_header->routing.target_offset_l = (uint32_t)dst_write_addr;
        packet_header->routing.target_offset_h = dst_write_addr >> 32;
        packet_header->routing.atomic_offset_l = (uint32_t)dst_atomic_addr;
        packet_header->routing.atomic_offset_h = dst_atomic_addr >> 32;
        packet_header->routing.atomic_increment = atomic_inc;
        packet_header->routing.command = ASYNC_WR | ATOMIC_INC;
    }
}

template <ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA, typename ClientInterfaceType>
inline void fabric_async_write_atomic_inc_add_header(
    tt_l1_ptr ClientInterfaceType client_interface,
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_write_addr,
    uint64_t dst_atomic_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t atomic_inc,
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");
    fabric_async_write_atomic_inc_add_header_impl(
        extract_packet_header<ClientInterfaceType, data_mode>(
            client_interface, src_addr, dst_mesh_id, dst_dev_id, header_id),
        src_addr,
        dst_mesh_id,
        dst_dev_id,
        dst_write_addr,
        dst_atomic_addr,
        size,
        atomic_inc);
}

// Write packetized data over fabric to dst_mesh, dst_dev.
// Packet is at src_addr in sender L1.
template <
    ClientDataMode data_mode = ClientDataMode::PACKETIZED_DATA,
    AsyncWriteMode mode = AsyncWriteMode::ALL,
    RoutingType routing_type = RoutingType::ROUTER_XY,
    typename ClientInterfaceType>
inline void fabric_async_write_atomic_inc(
    tt_l1_ptr ClientInterfaceType client_interface,
    uint32_t routing,   // routing refers to the router noc xy to use when using ROUTER_XY,
                        // and the routing plane to use when using ROUTING_TABLE
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_write_addr,
    uint64_t dst_atomic_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t atomic_inc,
    uint32_t header_id = 0) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");

    if constexpr (mode & AsyncWriteMode::ADD_HEADER) {
        fabric_async_write_atomic_inc_add_header<data_mode, ClientInterfaceType>(
            client_interface,
            src_addr,
            dst_mesh_id,
            dst_dev_id,
            dst_write_addr,
            dst_atomic_addr,
            size,
            atomic_inc,
            header_id);
#if !defined(FVC_MODE_PULL) && defined(LOW_LATENCY_ROUTING)
        uint32_t outgoing_direction = (uint32_t)get_next_hop_router_direction(dst_mesh_id, dst_dev_id);
        if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
            fabric_set_unicast_route(
                client_interface, (low_latency_packet_header_t*)(src_addr), outgoing_direction, dst_dev_id);
        } else {
            fabric_set_unicast_route(
                client_interface,
                (low_latency_packet_header_t*)&client_interface->header_buffer[header_id],
                outgoing_direction,
                dst_dev_id);
        }
#endif
    }

    if constexpr (std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*>) {
        if constexpr (mode & AsyncWriteMode::ADD_PR) {
            if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
                fabric_setup_pull_request<data_mode>(client_interface, src_addr, size);
            } else {
                fabric_setup_pull_request<data_mode>(client_interface, src_addr, size - PACKET_HEADER_SIZE_BYTES);
            }
        }

        if constexpr (mode & AsyncWriteMode::SEND_PR) {
            fabric_send_pull_request<data_mode, routing_type>(
                client_interface, routing, dst_mesh_id, dst_dev_id, header_id);
        }
    } else {
        if constexpr (mode & AsyncWriteMode::PUSH) {
            fabric_async_write_push_data<data_mode>(client_interface, src_addr, size, header_id);
        }
    }
}

template <RoutingType routing_type = RoutingType::ROUTER_XY, typename ClientInterfaceType>
inline void fabric_endpoint_init(tt_l1_ptr ClientInterfaceType client_interface, uint32_t outbound_eth_chan) {
    static_assert(
        std::is_same_v<ClientInterfaceType, volatile fabric_pull_client_interface_t*> ||
            std::is_same_v<ClientInterfaceType, volatile fabric_push_client_interface_t*>,
        "ClientInterfaceType must be either volatile fabric_pull_client_interface_t* or volatile "
        "fabric_push_client_interface_t*");
    // TODO: Should not assume routing tables are immediately after the client interface
    // This should be a separate address we take in
    uint32_t routing_tables_offset = (uint32_t)client_interface + sizeof(*client_interface);
    zero_l1_buf((uint32_t*)client_interface, sizeof(*client_interface));
    client_interface->routing_tables_l1_offset = routing_tables_offset;
    client_interface->num_routing_planes = 1;

    if constexpr (routing_type == RoutingType::ROUTING_TABLE) {
        // read routing table
        uint64_t dest_addr = get_noc_addr_helper(
            eth_chan_to_noc_xy[noc_index][outbound_eth_chan], eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);
        noc_async_read_one_packet(dest_addr, routing_tables_offset, sizeof(fabric_router_l1_config_t));
        noc_async_read_barrier();
    }
}

}  // namespace tt::tt_fabric
