// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/debug/dprint.h"

constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t sender_page_size = get_compile_time_arg_val(2);
constexpr uint32_t upstream_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_whole_fabric_packets_link_0 = get_compile_time_arg_val(4);
constexpr uint32_t num_whole_fabric_packets_link_1 = get_compile_time_arg_val(5);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(6);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(7);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(8);
constexpr bool use_fabric_on_receiver = get_compile_time_arg_val(9);
constexpr bool use_fabric_on_sender = get_compile_time_arg_val(10);
constexpr uint32_t num_upstream_sockets = get_compile_time_arg_val(11);
constexpr uint32_t upstream_socket_0_config_addr = get_compile_time_arg_val(12);
constexpr uint32_t upstream_socket_1_config_addr = get_compile_time_arg_val(13);
constexpr uint32_t upstream_socket_2_config_addr = get_compile_time_arg_val(14);
constexpr uint32_t upstream_socket_3_config_addr = get_compile_time_arg_val(15);
constexpr uint32_t upstream_socket_4_config_addr = get_compile_time_arg_val(16);
constexpr uint32_t upstream_socket_5_config_addr = get_compile_time_arg_val(17);
constexpr uint32_t upstream_socket_6_config_addr = get_compile_time_arg_val(18);
constexpr uint32_t upstream_socket_7_config_addr = get_compile_time_arg_val(19);

FORCE_INLINE bool socket_wait_for_pages_with_termination(
    const SocketReceiverInterface& socket, uint32_t num_pages, volatile tt_l1_ptr uint32_t* termination_semaphore) {
    constexpr uint32_t termination_value = 1;
    while (!socket_wait_for_pages(socket, num_pages, 1000)) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == termination_value) {
            return false;
        }
    }
    return true;
}

FORCE_INLINE void write_data_to_local_core_with_ack(
    SocketSenderInterface& sender_socket, uint32_t l1_read_addr, uint64_t dst_addr, uint32_t write_size) {
    noc_async_write(l1_read_addr, dst_addr, write_size);
    noc_async_writes_flushed();
}

FORCE_INLINE void write_data_to_remote_core_with_ack(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint64_t downstream_bytes_sent_noc_addr,
    uint32_t packet_size) {
    packet_header_addr->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{dst_addr, downstream_bytes_sent_noc_addr, packet_size}, packet_size);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, packet_size);
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void send_pages_over_socket(
    SocketSenderInterface& sender_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& downstream_fabric_connection,
    tt::tt_fabric::WorkerToFabricEdmSender& downstream_fabric_connection_2,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr_2,
    uint64_t downstream_bytes_sent_noc_addr,
    uint32_t l1_read_addr,
    uint64_t dst_addr) {
    if constexpr (use_fabric_on_sender) {
        for (uint32_t i = 0; i < num_whole_fabric_packets_link_0; ++i) {
            write_data_to_remote_core_with_ack(
                downstream_fabric_connection,
                downstream_data_packet_header_addr,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);
            l1_read_addr += whole_packet_size;
            dst_addr += whole_packet_size;
        }

        for (uint32_t i = 0; i < num_whole_fabric_packets_link_1; ++i) {
            write_data_to_remote_core_with_ack(
                downstream_fabric_connection_2,
                downstream_data_packet_header_addr_2,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);
            l1_read_addr += whole_packet_size;
            dst_addr += whole_packet_size;
        }

        if constexpr (partial_packet_size > 0) {
            write_data_to_remote_core_with_ack(
                downstream_fabric_connection_2,
                downstream_data_packet_header_addr_2,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                partial_packet_size);
        }
    } else {
        write_data_to_local_core_with_ack(sender_socket, l1_read_addr, dst_addr, upstream_page_size);
    }
}

void kernel_main() {
    // Build Fabric Connections
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection_2;

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        downstream_fabric_connection_2 =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }

    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);
    set_sender_socket_page_size(sender_socket, sender_page_size);
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);

    // Create receiver socket interfaces for all upstream sockets
    SocketReceiverInterface receiver_sockets[8];
    receiver_sockets[0] = create_receiver_socket_interface(upstream_socket_0_config_addr);
    receiver_sockets[1] = create_receiver_socket_interface(upstream_socket_1_config_addr);
    receiver_sockets[2] = create_receiver_socket_interface(upstream_socket_2_config_addr);
    receiver_sockets[3] = create_receiver_socket_interface(upstream_socket_3_config_addr);
    receiver_sockets[4] = create_receiver_socket_interface(upstream_socket_4_config_addr);
    receiver_sockets[5] = create_receiver_socket_interface(upstream_socket_5_config_addr);
    receiver_sockets[6] = create_receiver_socket_interface(upstream_socket_6_config_addr);
    receiver_sockets[7] = create_receiver_socket_interface(upstream_socket_7_config_addr);

    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        set_receiver_socket_page_size(receiver_sockets[i], upstream_page_size);
    }

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    // Store just the L1 base address for downstream FIFO - we'll add offsets and create NOC addr later
    uint32_t downstream_fifo_l1_addr = sender_socket.downstream_fifo_addr;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr_2 = nullptr;

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    if constexpr (use_fabric_on_sender) {
        downstream_data_packet_header_addr =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
        downstream_data_packet_header_addr_2 = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

        downstream_fabric_connection.open();
        downstream_fabric_connection_2.open();

        fabric_set_unicast_route(downstream_data_packet_header_addr, downstream_enc);
        fabric_set_unicast_route(downstream_data_packet_header_addr_2, downstream_enc);
    }

    uint32_t bytes_accumulated = 0;

    socket_reserve_pages(sender_socket, 1);

    // Collect data from all upstream sockets into a single larger page
    // Process num_upstream_sockets pages (one from each worker) for reduce-to-one
    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        // Wait for pages in current upstream socket with termination checks
        if (!socket_wait_for_pages_with_termination(receiver_sockets[i], 1, termination_semaphore)) {
            break;
        }

        auto l1_read_addr = receiver_sockets[i].read_ptr;
        // Calculate offset within the downstream buffer for this socket's data
        uint32_t skt_offset = bytes_accumulated;
        uint32_t dst_l1_addr = downstream_fifo_l1_addr + sender_socket.write_ptr + skt_offset;
        uint64_t dst_addr =
            get_noc_addr(downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, dst_l1_addr);

        send_pages_over_socket(
            sender_socket,
            downstream_fabric_connection,
            downstream_fabric_connection_2,
            downstream_data_packet_header_addr,
            downstream_data_packet_header_addr_2,
            downstream_bytes_sent_noc_addr,
            l1_read_addr,
            dst_addr);

        socket_pop_pages(receiver_sockets[i], 1);

        socket_notify_sender(receiver_sockets[i]);

        invalidate_l1_cache();

        // Update accumulation
        bytes_accumulated += upstream_page_size;
    }

    // Push the aggregated page after collecting all worker data
    if (bytes_accumulated >= sender_page_size) {
        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
    }

    invalidate_l1_cache();

    update_socket_config(sender_socket);
    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        update_socket_config(receiver_sockets[i]);
    }

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.close();
        downstream_fabric_connection_2.close();
    }
}
