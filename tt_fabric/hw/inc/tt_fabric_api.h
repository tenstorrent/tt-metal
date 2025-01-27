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

/*
inline void fabric_async_write(
    uint32_t routing_plane,      // the network plane to use for this transaction
    uint32_t src_addr,           // source address in sender’s memory
    uint64_t dst_addr,           // destination write address
    uint32_t size,               // number of bytes to write to remote destination
    uint32_t& fvc,               // fabric virtual channel. Set to –1 for automatic selection
    uint32_t return_status_addr  // TT-Fabric returns api call status at this address
) {
    uint32_t size_in_words = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
    local_pull_request.pull_request.wr_ptr = size_in_words;
    local_pull_request.pull_request.rd_ptr = 0;
    local_pull_request.pull_request.size = size;
    local_pull_request.pull_request.buffer_size = size_in_words;
    local_pull_request.pull_request.buffer_start = src_addr;
    local_pull_request.pull_request.ack_addr = -1;
    local_pull_request.pull_request.flags = FORWARD;

    uint64_t router_request_queue = NOC_XY_ADDR(2, 0, 0x19000);
    tt_fabric_send_pull_request(router_request_queue, &local_pull_request);
}
*/

inline uint32_t get_next_hop_router_noc_xy(uint32_t routing_plane, uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    if (dst_mesh_id != routing_table[routing_plane].my_mesh_id) {
        uint32_t next_port = routing_table[routing_plane].inter_mesh_table.dest_entry[dst_mesh_id];
        return eth_chan_to_noc_xy[noc_index][next_port];
    } else {
        uint32_t next_port = routing_table[routing_plane].intra_mesh_table.dest_entry[dst_dev_id];
        return eth_chan_to_noc_xy[noc_index][next_port];
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

inline void fabric_endpoint_init() {
    uint64_t noc_addr = client_interface->gk_interface_addr + offsetof(gatekeeper_info_t, ep_sync);
    client_interface->return_status[0] = 0;
    while (1) {
        noc_async_read_one_packet(noc_addr, (uint32_t)&client_interface->return_status[0], 4);
        noc_async_read_barrier();
        if (client_interface->return_status[0] != 0) {
            break;
        }
    }
}
