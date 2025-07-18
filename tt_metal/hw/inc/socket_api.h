// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "circular_buffer.h"
#include "debug/assert.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "socket.h"
#include "utils/utils.h"

#ifndef COMPILE_FOR_TRISC
#include <type_traits>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

static_assert(offsetof(sender_socket_md, bytes_acked) % L1_ALIGNMENT == 0);
static_assert(offsetof(receiver_socket_md, bytes_sent) % L1_ALIGNMENT == 0);

template <typename T>
constexpr bool always_false = false;

template <typename SocketT>
void fabric_set_unicast_route(volatile tt_l1_ptr PACKET_HEADER_TYPE* fabric_header_addr, const SocketT& socket) {
#if defined(DYNAMIC_ROUTING_ENABLED)
    if constexpr (std::is_same_v<SocketT, SocketSenderInterface>) {
        fabric_set_unicast_route(
            (MeshPacketHeader*)fabric_header_addr,
            eth_chan_directions::COUNT,
            0,
            socket.downstream_chip_id,
            socket.downstream_mesh_id,
            0);
    } else if constexpr (std::is_same_v<SocketT, SocketReceiverInterface>) {
        fabric_set_unicast_route(
            (MeshPacketHeader*)fabric_header_addr,
            eth_chan_directions::COUNT,
            0,
            socket.upstream_chip_id,
            socket.upstream_mesh_id,
            0);
    } else {
        static_assert(always_false<SocketT>, "Unsupported socket type passed to set_fabric_unicast_route");
    }
#else
    if constexpr (std::is_same_v<SocketT, SocketSenderInterface>) {
        fabric_header_addr->to_chip_unicast(static_cast<uint8_t>(socket.downstream_chip_id));
    } else if constexpr (std::is_same_v<SocketT, SocketReceiverInterface>) {
        fabric_header_addr->to_chip_unicast(static_cast<uint8_t>(socket.upstream_chip_id));
    } else {
        static_assert(always_false<SocketT>, "Unsupported socket type passed to fabric_set_unicast_route");
    }
#endif
}
#endif

SocketSenderInterface create_sender_socket_interface(uint32_t config_addr) {
    tt_l1_ptr sender_socket_md* socket_config = reinterpret_cast<tt_l1_ptr sender_socket_md*>(config_addr);
    SocketSenderInterface socket;
    socket.config_addr = config_addr;
    socket.write_ptr = socket_config->write_ptr;
    socket.bytes_sent = socket_config->bytes_sent;
    socket.bytes_acked_addr = config_addr + offsetof(sender_socket_md, bytes_acked);
    socket.downstream_mesh_id = socket_config->downstream_mesh_id;
    socket.downstream_chip_id = socket_config->downstream_chip_id;
    socket.downstream_noc_x = socket_config->downstream_noc_x;
    socket.downstream_noc_y = socket_config->downstream_noc_y;
    socket.downstream_fifo_addr = socket_config->downstream_fifo_addr;
    socket.downstream_bytes_sent_addr = socket_config->downstream_bytes_sent_addr;
    socket.downstream_fifo_total_size = socket_config->downstream_fifo_total_size;

    return socket;
}

void set_sender_socket_page_size(SocketSenderInterface& socket, uint32_t page_size) {
    // TODO: DRAM
    ASSERT(page_size % L1_ALIGNMENT == 0);
    uint32_t fifo_start_addr = socket.downstream_fifo_addr;
    uint32_t fifo_total_size = socket.downstream_fifo_total_size;
    ASSERT(page_size <= fifo_total_size);
    uint32_t& fifo_wr_ptr = socket.write_ptr;
    uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
    uint32_t fifo_page_aligned_size = fifo_total_size - fifo_total_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + fifo_page_aligned_size;
    if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
        socket.bytes_sent += fifo_start_addr + fifo_total_size - next_fifo_wr_ptr;
        next_fifo_wr_ptr = fifo_start_addr;
    }
    fifo_wr_ptr = next_fifo_wr_ptr;
    socket.page_size = page_size;
    socket.downstream_fifo_curr_size = fifo_page_aligned_size;
}

void socket_reserve_pages(const SocketSenderInterface& socket, uint32_t num_pages) {
    uint32_t num_bytes = num_pages * socket.page_size;
    volatile tt_l1_ptr uint32_t* bytes_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_addr);
    uint32_t bytes_free;
    do {
        invalidate_l1_cache();
        // bytes_acked will never be ahead of bytes_sent, so this is safe
        bytes_free = socket.downstream_fifo_total_size - (socket.bytes_sent - *bytes_acked_ptr);
    } while (bytes_free < num_bytes);
}

void socket_push_pages(SocketSenderInterface& socket, uint32_t num_pages) {
    uint32_t num_bytes = num_pages * socket.page_size;
    ASSERT(num_bytes <= socket.downstream_fifo_curr_size);
    if (socket.write_ptr + num_bytes >= socket.downstream_fifo_curr_size + socket.downstream_fifo_addr) {
        socket.write_ptr = socket.write_ptr + num_bytes - socket.downstream_fifo_curr_size;
        socket.bytes_sent += num_bytes + socket.downstream_fifo_total_size - socket.downstream_fifo_curr_size;
    } else {
        socket.write_ptr += num_bytes;
        socket.bytes_sent += num_bytes;
    }
}

#ifndef COMPILE_FOR_TRISC
void socket_notify_receiver(const SocketSenderInterface& socket) {
    // TODO: Store noc encoding in struct?
    auto downstream_bytes_sent_noc_addr =
        get_noc_addr(socket.downstream_noc_x, socket.downstream_noc_y, socket.downstream_bytes_sent_addr);
    noc_inline_dw_write(downstream_bytes_sent_noc_addr, socket.bytes_sent);
}

void fabric_socket_notify_receiver(
    const SocketSenderInterface& socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* fabric_header_addr) {
    auto downstream_bytes_sent_noc_addr =
        get_noc_addr(socket.downstream_noc_x, socket.downstream_noc_y, socket.downstream_bytes_sent_addr);
    fabric_set_unicast_route(fabric_header_addr, socket);
    fabric_header_addr->to_noc_unicast_inline_write(
        NocUnicastInlineWriteCommandHeader{downstream_bytes_sent_noc_addr, socket.bytes_sent});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)fabric_header_addr, sizeof(PACKET_HEADER_TYPE));
}
#endif

void socket_barrier(const SocketSenderInterface& socket) {
    volatile tt_l1_ptr uint32_t* bytes_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_addr);
    while (socket.bytes_sent != *bytes_acked_ptr);
}

void update_socket_config(const SocketSenderInterface& socket) {
    volatile tt_l1_ptr sender_socket_md* socket_config =
        reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(socket.config_addr);
    socket_config->bytes_sent = socket.bytes_sent;
    socket_config->write_ptr = socket.write_ptr;
}

SocketReceiverInterface create_receiver_socket_interface(uint32_t config_addr) {
    SocketReceiverInterface socket;
#if !(defined TRISC_PACK || defined TRISC_MATH)
    tt_l1_ptr receiver_socket_md* socket_config = reinterpret_cast<tt_l1_ptr receiver_socket_md*>(config_addr);
    socket.config_addr = config_addr;
    socket.read_ptr = socket_config->read_ptr;
    socket.bytes_acked = socket_config->bytes_acked;
    socket.bytes_sent_addr = config_addr + offsetof(receiver_socket_md, bytes_sent);
    socket.fifo_addr = socket_config->fifo_addr;
    socket.fifo_total_size = socket_config->fifo_total_size;
    socket.upstream_mesh_id = socket_config->upstream_mesh_id;
    socket.upstream_chip_id = socket_config->upstream_chip_id;
    socket.upstream_noc_x = socket_config->upstream_noc_x;
    socket.upstream_noc_y = socket_config->upstream_noc_y;
    socket.upstream_bytes_acked_addr = socket_config->upstream_bytes_acked_addr;
#endif
    return socket;
}

void set_receiver_socket_page_size(SocketReceiverInterface& socket, uint32_t page_size) {
#if !(defined TRISC_PACK || defined TRISC_MATH)
    uint32_t fifo_start_addr = socket.fifo_addr;
    uint32_t fifo_total_size = socket.fifo_total_size;
    ASSERT(page_size <= fifo_total_size);
    uint32_t& fifo_rd_ptr = socket.read_ptr;
    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
    uint32_t fifo_page_aligned_size = fifo_total_size - fifo_total_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + fifo_page_aligned_size;
    if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        uint32_t bytes_adjustment = fifo_start_addr + fifo_total_size - next_fifo_rd_ptr;
        volatile tt_l1_ptr uint32_t* bytes_sent_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_sent_addr);
        uint32_t bytes_recv;
        do {
            invalidate_l1_cache();
            bytes_recv = *bytes_sent_ptr - socket.bytes_acked;
        } while (bytes_recv < bytes_adjustment);
        socket.bytes_acked += bytes_adjustment;
        next_fifo_rd_ptr = fifo_start_addr;
    }
    fifo_rd_ptr = next_fifo_rd_ptr;
    socket.page_size = page_size;
    socket.fifo_curr_size = fifo_page_aligned_size;
#endif
}

void socket_wait_for_pages(const SocketReceiverInterface& socket, uint32_t num_pages) {
#if !(defined TRISC_PACK || defined TRISC_MATH)
    uint32_t num_bytes = num_pages * socket.page_size;
    if (socket.read_ptr + num_bytes >= socket.fifo_curr_size + socket.fifo_addr) {
        num_bytes += socket.fifo_total_size - socket.fifo_curr_size;
    }
    volatile tt_l1_ptr uint32_t* bytes_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_sent_addr);
    uint32_t bytes_recv;
    do {
        invalidate_l1_cache();
        bytes_recv = *bytes_sent_ptr - socket.bytes_acked;
    } while (bytes_recv < num_bytes);
#endif
}

void socket_pop_pages(SocketReceiverInterface& socket, uint32_t num_pages) {
#if !(defined TRISC_PACK || defined TRISC_MATH)
    uint32_t num_bytes = num_pages * socket.page_size;
    ASSERT(num_bytes <= socket.fifo_curr_size);
    if (socket.read_ptr + num_bytes >= socket.fifo_curr_size + socket.fifo_addr) {
        socket.read_ptr = socket.read_ptr + num_bytes - socket.fifo_curr_size;
        socket.bytes_acked += num_bytes + socket.fifo_total_size - socket.fifo_curr_size;
    } else {
        socket.read_ptr += num_bytes;
        socket.bytes_acked += num_bytes;
    }
#endif
}

void assign_local_cb_to_socket(const SocketReceiverInterface& socket, uint32_t cb_id) {
#if !(defined TRISC_PACK || defined TRISC_MATH)
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = socket.fifo_curr_size >> cb_addr_shift;
    uint32_t fifo_limit = (socket.fifo_addr >> cb_addr_shift) + fifo_size;
    uint32_t fifo_ptr = socket.read_ptr >> cb_addr_shift;
    ASSERT(fifo_size % local_cb.fifo_page_size == 0);
    uint32_t fifo_num_pages = fifo_size / local_cb.fifo_page_size;
    local_cb.fifo_limit = fifo_limit;
    local_cb.fifo_size = fifo_size;
    local_cb.fifo_num_pages = fifo_num_pages;
    local_cb.fifo_wr_ptr = fifo_ptr;
    local_cb.fifo_rd_ptr = fifo_ptr;
#endif
}

#ifndef COMPILE_FOR_TRISC
void socket_notify_sender(const SocketReceiverInterface& socket) {
    // TODO: Store noc encoding in struct?
    auto upstream_bytes_acked_noc_addr =
        get_noc_addr(socket.upstream_noc_x, socket.upstream_noc_y, socket.upstream_bytes_acked_addr);
    noc_inline_dw_write(upstream_bytes_acked_noc_addr, socket.bytes_acked);
}

void fabric_socket_notify_sender(
    const SocketReceiverInterface& socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* fabric_header_addr) {
    auto upstream_bytes_acked_noc_addr =
        get_noc_addr(socket.upstream_noc_x, socket.upstream_noc_y, socket.upstream_bytes_acked_addr);
    fabric_set_unicast_route(fabric_header_addr, socket);
    fabric_header_addr->to_noc_unicast_inline_write(
        NocUnicastInlineWriteCommandHeader{upstream_bytes_acked_noc_addr, socket.bytes_acked});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)fabric_header_addr, sizeof(PACKET_HEADER_TYPE));
}
#endif

void update_socket_config(const SocketReceiverInterface& socket) {
    volatile tt_l1_ptr receiver_socket_md* socket_config =
        reinterpret_cast<volatile tt_l1_ptr receiver_socket_md*>(socket.config_addr);
    socket_config->bytes_acked = socket.bytes_acked;
    socket_config->read_ptr = socket.read_ptr;
}
