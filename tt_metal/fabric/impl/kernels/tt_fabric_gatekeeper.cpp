// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "debug/dprint.h"
// clang-format on

using namespace tt::tt_fabric;

constexpr uint32_t gatekeeper_info_addr = get_compile_time_arg_val(0);
constexpr uint32_t socket_info_addr = get_compile_time_arg_val(1);
constexpr uint32_t routing_table_addr = get_compile_time_arg_val(2);
constexpr uint32_t kernel_status_buf_addr = get_compile_time_arg_val(3);
constexpr uint32_t kernel_status_buf_size_bytes = get_compile_time_arg_val(4);
constexpr uint32_t timeout_cycles = get_compile_time_arg_val(5);
uint32_t sync_val;
uint32_t router_mask;

// careful, may be null
tt_l1_ptr uint32_t* const kernel_status = reinterpret_cast<tt_l1_ptr uint32_t*>(kernel_status_buf_addr);
volatile tt_l1_ptr fabric_router_l1_config_t* routing_table =
    reinterpret_cast<tt_l1_ptr fabric_router_l1_config_t*>(routing_table_addr);

volatile gatekeeper_info_t* gk_info = (volatile gatekeeper_info_t*)gatekeeper_info_addr;
volatile socket_info_t* socket_info = (volatile socket_info_t*)socket_info_addr;

uint64_t xy_local_addr;
uint64_t router_addr;
uint32_t gk_message_pending;

inline void notify_all_routers(uint32_t notification) {
    uint32_t remaining_cores = router_mask;
    constexpr uint32_t num_routers = sizeof(eth_chan_to_noc_xy[0]) / sizeof(eth_chan_to_noc_xy[0][0]);
    for (uint32_t i = 0; i < num_routers; i++) {
        if (remaining_cores == 0) {
            break;
        }
        if (remaining_cores & (0x1 << i)) {
            uint64_t dest_addr = get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], FABRIC_ROUTER_SYNC_SEM);
            noc_inline_dw_write(dest_addr, notification);
            remaining_cores &= ~(0x1 << i);
        }
    }
}

inline void get_routing_tables() {
    uint32_t temp_mask = router_mask;
    uint32_t channel = 0;
    uint32_t routing_plane = 0;
    for (uint32_t i = 0; i < 4; i++) {
        if (temp_mask & 0xF) {
            temp_mask &= 0xF;
            break;
        } else {
            temp_mask >>= 4;
        }
        channel += 4;
    }

    if (temp_mask) {
        for (uint32_t i = 0; i < 4; i++) {
            if (temp_mask & 0x1) {
                uint64_t router_config_addr = get_noc_addr_helper(
                    eth_chan_to_noc_xy[noc_index][channel], eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);
                noc_async_read_one_packet(
                    router_config_addr,
                    (uint32_t)&routing_table[routing_plane],
                    sizeof(tt::tt_fabric::fabric_router_l1_config_t));
                routing_plane++;
            }
            temp_mask >>= 1;
            channel++;
        }
    }
    gk_info->routing_planes = routing_plane;
    noc_async_read_barrier();
}

inline void sync_all_routers() {
    // wait for all device routers to have incremented the sync semaphore.
    // sync_val is equal to number of tt-fabric routers running on a device.
    while (gk_info->router_sync.val != sync_val);

    // send semaphore increment to all fabric routers on this device.
    // semaphore notifies all other routers that this router has completed
    // startup handshake with its ethernet peer.
    notify_all_routers(sync_val);
    get_routing_tables();
    gk_info->ep_sync.val = sync_val;
}

inline void gk_msg_buf_ptr_advance(chan_ptr* ptr) { ptr->ptr = (ptr->ptr + 1) & FVCC_PTR_MASK; }

inline void gk_msg_buf_advance_wrptr(ctrl_chan_msg_buf* msg_buf) { gk_msg_buf_ptr_advance(&(msg_buf->wrptr)); }

inline void gk_msg_buf_advance_rdptr(ctrl_chan_msg_buf* msg_buf) {
    // clear valid before incrementing read pointer.
    uint32_t rd_index = msg_buf->rdptr.ptr & FVCC_SIZE_MASK;
    msg_buf->msg_buf[rd_index].bytes[47] = 0;
    gk_msg_buf_ptr_advance(&(msg_buf->rdptr));
}

inline bool gk_msg_buf_ptrs_empty(uint32_t wrptr, uint32_t rdptr) { return (wrptr == rdptr); }

inline bool gk_msg_buf_ptrs_full(uint32_t wrptr, uint32_t rdptr) {
    uint32_t distance = wrptr >= rdptr ? wrptr - rdptr : wrptr + 2 * FVCC_BUF_SIZE - rdptr;
    return !gk_msg_buf_ptrs_empty(wrptr, rdptr) && (distance >= FVCC_BUF_SIZE);
}

inline bool gk_msg_buf_is_empty(const volatile ctrl_chan_msg_buf* msg_buf) {
    return gk_msg_buf_ptrs_empty(msg_buf->wrptr.ptr, msg_buf->rdptr.ptr);
}

inline bool gk_msg_buf_is_full(const volatile ctrl_chan_msg_buf* msg_buf) {
    return gk_msg_buf_ptrs_full(msg_buf->wrptr.ptr, msg_buf->rdptr.ptr);
}

inline bool gk_msg_valid(const volatile ctrl_chan_msg_buf* msg_buf) {
    uint32_t rd_index = msg_buf->rdptr.ptr & FVCC_SIZE_MASK;
    return msg_buf->msg_buf[rd_index].packet_header.routing.flags != 0;
}

inline socket_handle_t* socket_receiver_available(packet_header_t* packet) {
    socket_handle_t* handle = 0;
    bool sender_found = false;
    for (uint32_t i = 0; i < MAX_SOCKETS; i++) {
        if (socket_info->sockets[i].socket_state == SocketState::OPENING &&
            socket_info->sockets[i].socket_direction == SOCKET_DIRECTION_SEND) {
            handle = (socket_handle_t*)&socket_info->sockets[i];
            if (handle->socket_id != packet->packet_parameters.socket_parameters.socket_id) {
                continue;
            }
            if (handle->epoch_id != packet->packet_parameters.socket_parameters.epoch_id) {
                continue;
            }
            if (handle->sender_dev_id != routing_table[0].my_device_id) {
                continue;
            }
            if (handle->sender_mesh_id != routing_table[0].my_mesh_id) {
                continue;
            }
            if (handle->rcvr_dev_id != packet->routing.dst_dev_id) {
                continue;
            }
            if (handle->rcvr_mesh_id != packet->routing.dst_mesh_id) {
                continue;
            }
            sender_found = true;
            break;
        }
    }
    return sender_found ? handle : 0;
}
inline void set_socket_active(socket_handle_t* handle) {
    handle->socket_state = SocketState::ACTIVE;
    // send socket connection state to send socket opener.
    noc_async_write_one_packet(
        (uint32_t)(handle), handle->status_notification_addr, sizeof(socket_handle_t), noc_index);
}

inline void socket_open(packet_header_t* packet) {
    socket_handle_t* handle = 0;
    if (packet->packet_parameters.socket_parameters.socket_direction == SOCKET_DIRECTION_SEND) {
        handle = socket_receiver_available(packet);
        if (handle) {
            // If remote receive socket already opened,
            // set send socket state to active and return.
            handle->status_notification_addr =
                get_noc_addr_helper(packet->session.ack_offset_h, packet->session.ack_offset_l);
            set_socket_active(handle);
            DPRINT << "GK: Receiver Available " << (uint32_t)handle->socket_id << ENDL();
            return;
        }
    }

    for (uint32_t i = 0; i < MAX_SOCKETS; i++) {
        if (socket_info->sockets[i].socket_state == 0) {
            // idle socket handle.
            socket_handle_t* handle = (socket_handle_t*)&socket_info->sockets[i];
            handle->socket_id = packet->packet_parameters.socket_parameters.socket_id;
            handle->epoch_id = packet->packet_parameters.socket_parameters.epoch_id;
            handle->socket_type = packet->packet_parameters.socket_parameters.socket_type;
            handle->socket_direction = packet->packet_parameters.socket_parameters.socket_direction;
            handle->routing_plane = packet->packet_parameters.socket_parameters.routing_plane;
            handle->status_notification_addr =
                get_noc_addr_helper(packet->session.ack_offset_h, packet->session.ack_offset_l);
            handle->socket_state = SocketState::OPENING;

            if (handle->socket_direction == SOCKET_DIRECTION_RECV) {
                handle->pull_notification_adddr =
                    get_noc_addr_helper(packet->session.target_offset_h, packet->session.target_offset_l);
                handle->sender_dev_id = packet->routing.src_dev_id;
                handle->sender_mesh_id = packet->routing.src_mesh_id;
                handle->rcvr_dev_id = routing_table[0].my_device_id;
                handle->rcvr_mesh_id = routing_table[0].my_mesh_id;
                socket_info->socket_setup_pending++;
            } else {
                handle->sender_dev_id = routing_table[0].my_device_id;
                handle->sender_mesh_id = routing_table[0].my_mesh_id;
                handle->rcvr_dev_id = packet->routing.dst_dev_id;
                handle->rcvr_mesh_id = packet->routing.dst_mesh_id;
            }
            DPRINT << "GK: Socket Opened : " << (uint32_t)handle->socket_id << ENDL();
            break;
        }
    }
}

inline void socket_close(packet_header_t* packet) {
    for (uint32_t i = 0; i < MAX_SOCKETS; i++) {
        bool found = socket_info->sockets[i].socket_id == packet->packet_parameters.socket_parameters.socket_id &&
                     socket_info->sockets[i].epoch_id == packet->packet_parameters.socket_parameters.epoch_id;
        if (found) {
            socket_handle_t* handle = (socket_handle_t*)&socket_info->sockets[i];
            // handle->socket_id = 0;
            // handle->epoch_id = 0;
            // handle->socket_type = 0;
            // handle->socket_direction = 0;
            handle->pull_notification_adddr = 0;
            handle->socket_state = SocketState::CLOSING;
            socket_info->socket_setup_pending++;
            break;
        }
    }
}

inline void socket_open_for_connect(packet_header_t* packet) {
    // open a send socket in response to a connect message received from a
    // socket receiver.
    // Local device has not yet opened a send socket, but we need to update
    // local socket state.
    for (uint32_t i = 0; i < MAX_SOCKETS; i++) {
        if (socket_info->sockets[i].socket_state == 0) {
            // idle socket handle.
            socket_handle_t* handle = (socket_handle_t*)&socket_info->sockets[i];
            handle->socket_id = packet->packet_parameters.socket_parameters.socket_id;
            handle->epoch_id = packet->packet_parameters.socket_parameters.epoch_id;
            handle->socket_type = packet->packet_parameters.socket_parameters.socket_type;
            handle->socket_direction = SOCKET_DIRECTION_SEND;
            handle->pull_notification_adddr =
                get_noc_addr_helper(packet->session.target_offset_h, packet->session.target_offset_l);
            handle->routing_plane = packet->packet_parameters.socket_parameters.routing_plane;
            handle->socket_state = SocketState::OPENING;
            handle->sender_dev_id = routing_table[0].my_device_id;
            handle->sender_mesh_id = routing_table[0].my_mesh_id;
            handle->rcvr_dev_id = packet->routing.src_dev_id;
            handle->rcvr_mesh_id = packet->routing.src_mesh_id;
            DPRINT << "GK: Opened Send Socket for Remote " << (uint32_t)handle->socket_id << ENDL();

            break;
        }
    }
}

inline void socket_connect(packet_header_t* packet) {
    // connect messsage is initiated by a receiving socket.
    // find a matching send socket in local state.
    bool found_sender = false;
    for (uint32_t i = 0; i < MAX_SOCKETS; i++) {
        if (socket_info->sockets[i].socket_state == SocketState::OPENING &&
            socket_info->sockets[i].socket_direction == SOCKET_DIRECTION_SEND) {
            socket_handle_t* handle = (socket_handle_t*)&socket_info->sockets[i];
            if (handle->socket_id != packet->packet_parameters.socket_parameters.socket_id) {
                continue;
            }
            if (handle->epoch_id != packet->packet_parameters.socket_parameters.epoch_id) {
                continue;
            }
            if (handle->sender_dev_id != packet->routing.dst_dev_id) {
                continue;
            }
            if (handle->sender_mesh_id != packet->routing.dst_mesh_id) {
                continue;
            }
            if (handle->rcvr_dev_id != packet->routing.src_dev_id) {
                continue;
            }
            if (handle->rcvr_mesh_id != packet->routing.src_mesh_id) {
                continue;
            }
            found_sender = true;
            handle->pull_notification_adddr =
                get_noc_addr_helper(packet->session.target_offset_h, packet->session.target_offset_l);
            handle->socket_state = SocketState::ACTIVE;
            set_socket_active(handle);
            DPRINT << "GK: Found Send Socket " << (uint32_t)handle->socket_id << ENDL();
            break;
        }
    }

    if (!found_sender) {
        // No send socket opened yet.
        // Device has not made it to the epoch where a send socket is opened.
        // Log the receive socket connect request.
        socket_open_for_connect(packet);
    }
}

uint32_t get_next_hop_router_noc_xy(packet_header_t* current_packet_header, uint32_t routing_plane) {
    uint32_t dst_mesh_id = current_packet_header->routing.dst_mesh_id;
    if (dst_mesh_id != routing_table->my_mesh_id) {
        uint32_t next_port = routing_table[routing_plane].inter_mesh_table.dest_entry[dst_mesh_id];
        return eth_chan_to_noc_xy[noc_index][next_port];
    } else {
        uint32_t dst_device_id = current_packet_header->routing.dst_dev_id;
        uint32_t next_port = routing_table[routing_plane].intra_mesh_table.dest_entry[dst_device_id];
        return eth_chan_to_noc_xy[noc_index][next_port];
    }
}

inline bool send_gk_message(uint64_t dest_addr, packet_header_t* packet) {
    uint64_t noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, wrptr);
    noc_fast_atomic_increment<DM_DEDICATED_NOC, true>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        noc_addr,
        NOC_UNICAST_WRITE_VC,
        1,
        FVCC_BUF_LOG_SIZE,
        false,
        false,
        (uint32_t)&socket_info->wrptr.ptr);
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));

    uint32_t wrptr = socket_info->wrptr.ptr;
    noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, rdptr);

    noc_async_read_one_packet(noc_addr, (uint32_t)(&socket_info->rdptr.ptr), 4);
    noc_async_read_barrier();
    if (fvcc_buf_ptrs_full(wrptr, socket_info->rdptr.ptr)) {
        return false;
    }

    uint32_t dest_wr_index = wrptr & FVCC_SIZE_MASK;
    noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, msg_buf) + dest_wr_index * sizeof(packet_header_t);
    noc_async_write_one_packet((uint32_t)(packet), noc_addr, sizeof(packet_header_t), noc_index);
    return true;
}

inline bool retry_gk_message(uint64_t dest_addr, packet_header_t* packet) {
    uint32_t wrptr = socket_info->wrptr.ptr;
    uint64_t noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, rdptr);

    noc_async_read_one_packet(noc_addr, (uint32_t)(&socket_info->rdptr.ptr), 4);
    noc_async_read_barrier();
    if (fvcc_buf_ptrs_full(wrptr, socket_info->rdptr.ptr)) {
        return false;
    }

    uint32_t dest_wr_index = wrptr & FVCC_SIZE_MASK;
    noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, msg_buf) + dest_wr_index * sizeof(packet_header_t);
    noc_async_write_one_packet((uint32_t)(packet), noc_addr, sizeof(packet_header_t), noc_index);
    gk_message_pending = 0;
    return true;
}

inline void process_pending_socket() {
    if (socket_info->socket_setup_pending == 0) {
        // there is no pending socket setup work.
        // or there is a pending message to be sent out
        return;
    }

    // Check write flush of previous gk message write.
    // TODO.

    chan_request_entry_t* message = (chan_request_entry_t*)&socket_info->gk_message;

    if (gk_message_pending == 1) {
        if (retry_gk_message(router_addr, &message->packet_header)) {
            // decrement pending count.
            // if there is more pending socket work, will pick up on next iteration.
            socket_info->socket_setup_pending--;
        }
    } else {
        for (uint32_t i = 0; i < MAX_SOCKETS; i++) {
            if (socket_info->sockets[i].socket_state == SocketState::OPENING &&
                socket_info->sockets[i].socket_direction == SOCKET_DIRECTION_RECV) {
                // send a message on fvcc to socket data sender.
                socket_handle_t* handle = (socket_handle_t*)&socket_info->sockets[i];
                message->packet_header.routing.dst_dev_id = handle->sender_dev_id;
                message->packet_header.routing.dst_mesh_id = handle->sender_mesh_id;
                message->packet_header.routing.src_dev_id = handle->rcvr_dev_id;
                message->packet_header.routing.src_mesh_id = handle->rcvr_mesh_id;
                message->packet_header.routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
                message->packet_header.routing.flags = SYNC;
                message->packet_header.session.command = SOCKET_CONNECT;
                message->packet_header.session.target_offset_h = handle->pull_notification_adddr >> 32;
                message->packet_header.session.target_offset_l = (uint32_t)handle->pull_notification_adddr;
                message->packet_header.packet_parameters.socket_parameters.socket_id = handle->socket_id;
                message->packet_header.packet_parameters.socket_parameters.epoch_id = handle->epoch_id;
                message->packet_header.packet_parameters.socket_parameters.socket_type = handle->socket_type;
                message->packet_header.packet_parameters.socket_parameters.socket_direction = SOCKET_DIRECTION_RECV;
                message->packet_header.packet_parameters.socket_parameters.routing_plane = handle->routing_plane;
                tt_fabric_add_header_checksum((packet_header_t*)&message->packet_header);

                router_addr = get_noc_addr_helper(
                    get_next_hop_router_noc_xy(&message->packet_header, handle->routing_plane), FVCC_OUT_BUF_START);

                if (send_gk_message(router_addr, &message->packet_header)) {
                    DPRINT << "GK: Sending Connect to " << (uint32_t)handle->sender_dev_id << ENDL();
                    handle->socket_state = SocketState::ACTIVE;
                    socket_info->socket_setup_pending--;
                } else {
                    DPRINT << "GK: Pending Connect to " << (uint32_t)handle->sender_dev_id << ENDL();
                    // gatekeeper message buffer is full. need to retry on next iteration.
                    gk_message_pending = 1;
                }
            }
        }
    }
}

void kernel_main() {
    sync_val = get_arg_val<uint32_t>(0);
    router_mask = get_arg_val<uint32_t>(1);
    gk_message_pending = 0;
    router_addr = 0;

    tt_fabric_init();

    write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_STARTED);
    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX, 0xff000000);
    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX + 1, 0xbb000000);
    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX + 2, 0xAABBCCDD);
    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX + 3, 0xDDCCBBAA);

    zero_l1_buf((tt_l1_ptr uint32_t*)&gk_info->gk_msg_buf, FVCC_BUF_SIZE_BYTES);
    zero_l1_buf((tt_l1_ptr uint32_t*)socket_info, sizeof(socket_info_t));

    sync_all_routers();
    uint64_t start_timestamp = get_timestamp();

    uint32_t loop_count = 0;
    uint32_t total_messages_procesed = 0;
    volatile ctrl_chan_msg_buf* msg_buf = &gk_info->gk_msg_buf;
    while (1) {
        if (!gk_msg_buf_is_empty(msg_buf) && gk_msg_valid(msg_buf)) {
            uint32_t msg_index = msg_buf->rdptr.ptr & FVCC_SIZE_MASK;
            chan_request_entry_t* req = (chan_request_entry_t*)msg_buf->msg_buf + msg_index;
            packet_header_t* packet = &req->packet_header;
            if (tt_fabric_is_header_valid(packet)) {
                total_messages_procesed++;
                DPRINT << "GK: Message Received " << (uint32_t)packet->session.command
                       << " msg num = " << total_messages_procesed << ENDL();
                if (packet->routing.flags == SYNC) {
                    if (packet->session.command == SOCKET_OPEN) {
                        DPRINT << "GK: Socket Open " << ENDL();
                        socket_open(packet);
                    } else if (packet->session.command == SOCKET_CLOSE) {
                        socket_close(packet);
                    } else if (packet->session.command == SOCKET_CONNECT) {
                        DPRINT << "GK: Socket Connect " << ENDL();
                        socket_connect(packet);
                    }
                }

                // clear the flags field to invalidate pull request slot.
                // flags will be set to non-zero by next requestor.
                gk_msg_buf_advance_rdptr((ctrl_chan_msg_buf*)msg_buf);
                loop_count = 0;
            } else {
                write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_BAD_HEADER);
                return;
            }
        }

        loop_count++;

        process_pending_socket();

        if (gk_info->router_sync.val == 0) {
            // terminate signal from host sw.
            if (loop_count >= 0x1000) {
                // send terminate to all chip routers.
                notify_all_routers(0);
                break;
            }
        }
    }

    DPRINT << "Gatekeeper messages processed " << total_messages_procesed << ENDL();

    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX, 0xff000002);

    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX, 0xff000003);

    write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_PASS);

    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX, 0xff00005);
}
