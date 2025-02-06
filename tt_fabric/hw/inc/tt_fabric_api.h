// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include <hostdevcommon/common_values.hpp>
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_fabric_interface.h"

using namespace tt::tt_fabric;

extern volatile local_pull_request_t* local_pull_request;
extern volatile fabric_client_interface_t* client_interface;
extern fabric_client_push_interface_t client_push_interface;

#define ASYNC_WR_ALL 1
#define ASYNC_WR_ADD_PR 2
#define ASYNC_WR_SEND 3
#define ASYNC_WR_ADD_HEADER 4

inline uint32_t get_next_hop_router_noc_xy(uint32_t routing_plane, uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    ASSERT(routing_plane < client_interface->num_routing_planes);
    fabric_router_l1_config_t* routing_table = (fabric_router_l1_config_t*)client_interface->routing_tables_l1_offset;
    if (dst_mesh_id != routing_table[routing_plane].my_mesh_id) {
        uint32_t next_port = routing_table[routing_plane].inter_mesh_table.dest_entry[dst_mesh_id];
        return eth_chan_to_noc_xy[noc_index][next_port];
    } else {
        uint32_t next_port = routing_table[routing_plane].intra_mesh_table.dest_entry[dst_dev_id];
        return eth_chan_to_noc_xy[noc_index][next_port];
    }
}

inline uint32_t get_next_hop_router_direction(uint32_t routing_plane, uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    ASSERT(routing_plane < client_interface->num_routing_planes);
    fabric_router_l1_config_t* routing_table = (fabric_router_l1_config_t*)client_interface->routing_tables_l1_offset;
    uint32_t next_port = 0;
    uint32_t direction = 0;
    if (dst_mesh_id != routing_table[routing_plane].my_mesh_id) {
        next_port = routing_table[routing_plane].inter_mesh_table.dest_entry[dst_mesh_id];
    } else {
        next_port = routing_table[routing_plane].intra_mesh_table.dest_entry[dst_dev_id];
    }

    if (routing_table[routing_plane].port_direction.directions[eth_chan_directions::EAST] == next_port) {
        direction = eth_chan_directions::EAST;
    } else if (routing_table[routing_plane].port_direction.directions[eth_chan_directions::WEST] == next_port) {
        direction = eth_chan_directions::WEST;
    } else if (routing_table[routing_plane].port_direction.directions[eth_chan_directions::NORTH] == next_port) {
        direction = eth_chan_directions::NORTH;
    } else if (routing_table[routing_plane].port_direction.directions[eth_chan_directions::SOUTH] == next_port) {
        direction = eth_chan_directions::SOUTH;
    }
    return direction;
}

inline void fabric_setup_pull_request(uint32_t src_addr, uint32_t size) {
    uint32_t size_in_words = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
    client_interface->local_pull_request.pull_request.wr_ptr = size_in_words;
    client_interface->local_pull_request.pull_request.rd_ptr = 0;
    client_interface->local_pull_request.pull_request.size = size;
    client_interface->local_pull_request.pull_request.buffer_size = size_in_words;
    client_interface->local_pull_request.pull_request.buffer_start = xy_local_addr + src_addr;
    client_interface->local_pull_request.pull_request.words_written = size_in_words;
    client_interface->local_pull_request.pull_request.words_read = 0;
    client_interface->local_pull_request.pull_request.ack_addr =
        xy_local_addr + (uint32_t)&client_interface->local_pull_request.pull_request.words_read;
    client_interface->local_pull_request.pull_request.flags = FORWARD;
}

inline void fabric_send_pull_request(uint32_t routing_plane, uint16_t dst_mesh_id, uint16_t dst_dev_id) {
    uint64_t router_addr = ((uint64_t)get_next_hop_router_noc_xy(routing_plane, dst_mesh_id, dst_dev_id) << 32) |
                           FABRIC_ROUTER_REQ_QUEUE_START;
    tt_fabric_send_pull_request(router_addr, (volatile local_pull_request_t*)&client_interface->local_pull_request);
}

inline void fabric_async_write_add_header(
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size  // number of bytes to write to remote destination
) {
    packet_header_t* packet_header = (packet_header_t*)(src_addr);
    packet_header->routing.flags = FORWARD;
    packet_header->routing.packet_size_bytes = size;
    packet_header->routing.dst_mesh_id = dst_mesh_id;
    packet_header->routing.dst_dev_id = dst_dev_id;
    packet_header->session.command = ASYNC_WR;
    packet_header->session.target_offset_l = (uint32_t)dst_addr;
    packet_header->session.target_offset_h = dst_addr >> 32;
    tt_fabric_add_header_checksum(packet_header);
}

#ifdef FVC_MODE_PULL
// Write packetized data over fabric to dst_mesh, dst_dev.
// Packet is at src_addr in sender L1.
template <uint8_t mode = ASYNC_WR_ALL>
inline void fabric_async_write(
    uint32_t routing_plane,  // the network plane to use for this transaction
    uint32_t src_addr,       // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size  // number of bytes to write to remote destination
) {
    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_ADD_HEADER) {
        fabric_async_write_add_header(src_addr, dst_mesh_id, dst_dev_id, dst_addr, size);
    }

    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_ADD_PR) {
        fabric_setup_pull_request(src_addr, size);
    }

    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_SEND) {
        fabric_send_pull_request(routing_plane, dst_mesh_id, dst_dev_id);
    }
}
#else
inline void fabric_client_router_reserve(int32_t routing_plane, uint16_t dst_mesh_id, uint16_t dst_dev_id) {
    uint32_t direction = get_next_hop_router_direction(routing_plane, dst_mesh_id, dst_dev_id);
    uint32_t router_addr_h = get_next_hop_router_noc_xy(routing_plane, dst_mesh_id, dst_dev_id);
    uint64_t router_addr = ((uint64_t)router_addr_h << 32) | FABRIC_ROUTER_REQ_QUEUE_START;
    // direction = 3;
    router_addr += direction * 8;
    // stream register to receive router buffer space available updates.
    noc_inline_dw_write(router_addr, (STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX)));
    noc_inline_dw_write(router_addr + 4, xy_local_addr >> 32);
    client_push_interface.router_addr_h = router_addr_h;
    client_push_interface.buffer_size = FABRIC_ROUTER_OUTBOUND_BUF_SIZE / PACKET_WORD_SIZE_BYTES;
    client_push_interface.wr_ptr = 0;
    client_push_interface.buffer_start = FABRIC_ROUTER_DATA_BUF_START + direction * FABRIC_ROUTER_OUTBOUND_BUF_SIZE;
    client_push_interface.router_push_addr =
        (STREAM_REG_ADDR(6 + direction, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
    client_push_interface.router_space =
        reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
    client_push_interface.update_router_space =
        reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
    *(uint32_t*)(STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) = client_push_interface.buffer_size;
}

inline void fabric_async_write_push_data(uint32_t src_addr, uint32_t size) {
    uint32_t total_words_to_send = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
    uint64_t push_addr =
        ((uint64_t)client_push_interface.router_addr_h << 32) | (client_push_interface.router_push_addr);

    while (total_words_to_send > 0) {
        uint32_t router_buf_space = *client_push_interface.router_space;
        uint32_t words_before_wrap = client_push_interface.buffer_size - client_push_interface.wr_ptr;
        uint32_t words_to_send = min(total_words_to_send, words_before_wrap);
        words_to_send = min(router_buf_space, words_to_send);
        if (words_to_send) {
            uint64_t buffer_wr_addr =
                ((uint64_t)client_push_interface.router_addr_h << 32) |
                (client_push_interface.buffer_start + (client_push_interface.wr_ptr * PACKET_WORD_SIZE_BYTES));
            noc_async_write_one_packet(src_addr, buffer_wr_addr, words_to_send * PACKET_WORD_SIZE_BYTES, noc_index);
            noc_inline_dw_write(push_addr, words_to_send << REMOTE_DEST_BUF_WORDS_FREE_INC);
            client_push_interface.wr_ptr += words_to_send;
            *client_push_interface.update_router_space = (-words_to_send) << REMOTE_DEST_BUF_WORDS_FREE_INC;
            if (client_push_interface.wr_ptr >= client_push_interface.buffer_size) {
                client_push_interface.wr_ptr -= client_push_interface.buffer_size;
            }
            total_words_to_send -= words_to_send;
            src_addr += words_to_send * PACKET_WORD_SIZE_BYTES;
        }
    }
}

template <uint8_t mode = ASYNC_WR_ALL>
inline void fabric_async_write(
    uint32_t routing_plane,  // the network plane to use for this transaction
    uint32_t src_addr,       // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size  // number of bytes to write to remote destination
) {
    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_ADD_HEADER) {
        fabric_async_write_add_header(src_addr, dst_mesh_id, dst_dev_id, dst_addr, size);
    }

    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_SEND) {
        fabric_async_write_push_data(src_addr, size);
    }
}
#endif

inline void fabric_async_write_multicast_add_header(
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t e_depth,
    uint32_t w_depth,
    uint32_t n_depth,
    uint32_t s_depth) {
    packet_header_t* packet_header = (packet_header_t*)(src_addr);
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
}

/*
inline void fabric_async_write_multicast_set_state(
    packet_header_t * header_pointer,
    uint32_t routing_plane,  // the network plane to use for this transaction
    uint32_t src_addr,       // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t e_depth,
    uint32_t w_depth,
    uint32_t n_depth,
    uint32_t s_depth)

}

inline void fabric_async_write_multicast_with_state(
    uint32_t state_id
    uint32_t src_addr,       // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
)
*/

// Write packetized data over fabric to dst_mesh, dst_dev.
// Packet is at src_addr in sender L1.
template <uint8_t mode = ASYNC_WR_ALL>
inline void fabric_async_write_multicast(
    uint32_t routing_plane,  // the network plane to use for this transaction
    uint32_t src_addr,       // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t e_depth,
    uint32_t w_depth,
    uint32_t n_depth,
    uint32_t s_depth) {
    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_ADD_HEADER) {
        fabric_async_write_multicast_add_header(
            src_addr, dst_mesh_id, dst_dev_id, dst_addr, size, e_depth, w_depth, n_depth, s_depth);
    }

    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_ADD_PR) {
        fabric_setup_pull_request(src_addr, size);
    }

    if constexpr (mode == ASYNC_WR_ALL or mode == ASYNC_WR_SEND) {
        fabric_send_pull_request(routing_plane, dst_mesh_id, dst_dev_id);
    }
}

inline void send_message_to_gk() {
    uint64_t gk_noc_base = client_interface->gk_msg_buf_addr;
    uint64_t noc_addr = gk_noc_base + offsetof(ctrl_chan_msg_buf, wrptr);
    noc_fast_atomic_increment<DM_DYNAMIC_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        noc_addr,
        NOC_UNICAST_WRITE_VC,
        1,
        FVCC_BUF_LOG_SIZE,
        false,
        false,
        (uint32_t)&client_interface->wrptr.ptr);
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
    uint32_t wrptr = client_interface->wrptr.ptr;
    noc_addr = gk_noc_base + offsetof(ctrl_chan_msg_buf, rdptr);
    while (1) {
        noc_async_read_one_packet(noc_addr, (uint32_t)(&client_interface->rdptr.ptr), 4);
        noc_async_read_barrier();
        if (!fvcc_buf_ptrs_full(wrptr, client_interface->rdptr.ptr)) {
            break;
        }
    }
    uint32_t dest_wr_index = wrptr & FVCC_SIZE_MASK;
    noc_addr = gk_noc_base + offsetof(ctrl_chan_msg_buf, msg_buf) + dest_wr_index * sizeof(packet_header_t);
    noc_async_write_one_packet((uint32_t)(&client_interface->gk_message), noc_addr, sizeof(packet_header_t), noc_index);
    noc_async_write_barrier();
}

inline socket_handle_t* fabric_socket_open(
    uint32_t routing_plane,   // the network plane to use for this socket
    uint16_t epoch_id,        // Temporal epoch for which the socket is being opened
    uint16_t socket_id,       // Socket Id to open
    uint8_t socket_type,      // Unicast, Multicast, SSocket, DSocket
    uint8_t direction,        // Send or Receive
    uint16_t remote_mesh_id,  // Remote mesh/device that is the socket data sender/receiver.
    uint16_t remote_dev_id,
    uint8_t fvc  // fabric virtual channel.
) {
    uint32_t socket_count = client_interface->socket_count;
    socket_handle_t* socket_handle = (socket_handle_t*)&client_interface->socket_handles[socket_count];
    socket_count++;
    client_interface->socket_count = socket_count;
    socket_handle->socket_state = SocketState::OPENING;

    if (direction == SOCKET_DIRECTION_SEND) {
        client_interface->gk_message.packet_header.routing.dst_mesh_id = remote_mesh_id;
        client_interface->gk_message.packet_header.routing.dst_dev_id = remote_dev_id;
    } else {
        client_interface->gk_message.packet_header.routing.src_mesh_id = remote_mesh_id;
        client_interface->gk_message.packet_header.routing.src_dev_id = remote_dev_id;
    }
    client_interface->gk_message.packet_header.routing.flags = SYNC;
    client_interface->gk_message.packet_header.session.command = SOCKET_OPEN;
    client_interface->gk_message.packet_header.session.target_offset_h = client_interface->pull_req_buf_addr >> 32;
    client_interface->gk_message.packet_header.session.target_offset_l = (uint32_t)client_interface->pull_req_buf_addr;
    client_interface->gk_message.packet_header.session.ack_offset_h = xy_local_addr >> 32;
    client_interface->gk_message.packet_header.session.ack_offset_l = (uint32_t)socket_handle;
    client_interface->gk_message.packet_header.packet_parameters.socket_parameters.socket_id = socket_id;
    client_interface->gk_message.packet_header.packet_parameters.socket_parameters.epoch_id = epoch_id;
    client_interface->gk_message.packet_header.packet_parameters.socket_parameters.socket_type = socket_type;
    client_interface->gk_message.packet_header.packet_parameters.socket_parameters.socket_direction = direction;
    client_interface->gk_message.packet_header.packet_parameters.socket_parameters.routing_plane = routing_plane;
    tt_fabric_add_header_checksum((packet_header_t*)&client_interface->gk_message.packet_header);
    send_message_to_gk();
    return socket_handle;
}

inline void fabric_socket_close(socket_handle_t* socket_handle) {
    packet_header_t* packet_header = (packet_header_t*)&client_interface->gk_message.packet_header;
    uint32_t dst_mesh_id = socket_handle->rcvr_mesh_id;
    uint32_t dst_dev_id = socket_handle->rcvr_dev_id;
    packet_header->routing.flags = INLINE_FORWARD;
    packet_header->routing.dst_mesh_id = dst_mesh_id;
    packet_header->routing.dst_dev_id = dst_dev_id;
    packet_header->routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
    packet_header->session.command = SOCKET_CLOSE;
    packet_header->session.target_offset_l = (uint32_t)socket_handle->pull_notification_adddr;
    packet_header->session.target_offset_h = socket_handle->pull_notification_adddr >> 32;
    tt_fabric_add_header_checksum(packet_header);

    uint32_t* dst = (uint32_t*)&client_interface->local_pull_request.pull_request;
    uint32_t* src = (uint32_t*)packet_header;
    for (uint32_t i = 0; i < sizeof(pull_request_t) / 4; i++) {
        dst[i] = src[i];
    }
    uint64_t dest_addr =
        ((uint64_t)get_next_hop_router_noc_xy(socket_handle->routing_plane, dst_mesh_id, dst_dev_id) << 32) |
        FABRIC_ROUTER_REQ_QUEUE_START;
    tt_fabric_send_pull_request(dest_addr, (volatile local_pull_request_t*)&client_interface->local_pull_request);
}

inline void fabric_socket_connect(socket_handle_t* socket_handle) {
    // wait for socket state to change to Active.
    // Gatekeeper will update local socket handle when the receiver for send socket
    // is ready.
    while (((volatile socket_handle_t*)socket_handle)->socket_state != SocketState::ACTIVE);
}

inline void fabric_endpoint_init(uint32_t base_address, uint32_t gk_interface_addr_l, uint32_t gk_interface_addr_h) {
    tt_fabric_init();

    client_interface = (volatile fabric_client_interface_t*)base_address;
    uint32_t routing_tables_offset = base_address + sizeof(fabric_client_interface_t);

    zero_l1_buf((uint32_t*)client_interface, sizeof(fabric_client_interface_t));
    client_interface->gk_interface_addr = ((uint64_t)gk_interface_addr_h << 32) | gk_interface_addr_l;
    client_interface->gk_msg_buf_addr =
        (((uint64_t)gk_interface_addr_h << 32) | gk_interface_addr_l) + offsetof(gatekeeper_info_t, gk_msg_buf);
    client_interface->routing_tables_l1_offset = routing_tables_offset;

    // make sure fabric node gatekeeper is available.
    uint64_t noc_addr = client_interface->gk_interface_addr + offsetof(gatekeeper_info_t, ep_sync);
    client_interface->return_status[0] = 0;
    while (1) {
        noc_async_read_one_packet(noc_addr, (uint32_t)&client_interface->return_status[0], 4);
        noc_async_read_barrier();
        if (client_interface->return_status[0] != 0) {
            break;
        }
    }

    // read the gk info first at routing table addr and later override with routing tables
    noc_async_read_one_packet(
        client_interface->gk_interface_addr, client_interface->routing_tables_l1_offset, sizeof(gatekeeper_info_t));
    noc_async_read_barrier();

    client_interface->num_routing_planes = ((gatekeeper_info_t*)routing_tables_offset)->routing_planes;

    // read routing tables
    uint64_t gk_rt_noc_addr = client_interface->gk_interface_addr - sizeof(fabric_router_l1_config_t) * 4;
    uint32_t table_offset;
    for (uint32_t i = 0; i < client_interface->num_routing_planes; i++) {
        table_offset = sizeof(fabric_router_l1_config_t) * i;
        noc_async_read_one_packet(
            gk_rt_noc_addr + table_offset, routing_tables_offset + table_offset, sizeof(fabric_router_l1_config_t));
    }
    noc_async_read_barrier();
}
