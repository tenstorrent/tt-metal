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
constexpr uint32_t page_size = get_compile_time_arg_val(2);
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
        DPRINT << "Waiting for pages in receiver socket with termination checks...\n";
        if (termination_semaphore[0] == termination_value) {
            return false;
        }
    }
    return true;
}

FORCE_INLINE void write_data_to_local_core_with_ack(
    SocketSenderInterface& sender_socket, uint32_t l1_read_addr, uint64_t dst_addr, uint32_t write_size) {
    noc_async_write(l1_read_addr, dst_addr, write_size);
    // Flush here to ensure that NOC has picked up data before we pop pages in receiver socket.
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
    uint64_t dst_addr,
    uint32_t bytes_offset) {
    DPRINT << "use fabric on sender: " << (uint32_t)use_fabric_on_sender << "\n";
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
        DPRINT << "Sending data over socket without using fabric...\n";
        write_data_to_local_core_with_ack(sender_socket, l1_read_addr, dst_addr + bytes_offset, upstream_page_size);
    }
}

void kernel_main() {
    // Build Fabric Connections
    DPRINT << "start of d2d exchange d2d_0 kernel main\n";
    DPRINT << "My NOC coordinates: x=" << (uint32_t)my_x[0] << ", y=" << (uint32_t)my_y[0] << "\n";
    DPRINT << "CT ARGS for d2d 0\n";
    DPRINT << "sender_socket_config_addr: " << (uint32_t)sender_socket_config_addr << "\n";
    DPRINT << "termination_semaphore_addr: " << (uint32_t)termination_semaphore_addr << "\n";
    DPRINT << "page_size: " << (uint32_t)page_size << "\n";
    DPRINT << "upstream_page_size: " << (uint32_t)upstream_page_size << "\n";
    DPRINT << "num_whole_fabric_packets_link_0: " << (uint32_t)num_whole_fabric_packets_link_0 << "\n";
    DPRINT << "num_whole_fabric_packets_link_1: " << (uint32_t)num_whole_fabric_packets_link_1 << "\n";
    DPRINT << "whole_packet_size: " << (uint32_t)whole_packet_size << "\n";
    DPRINT << "partial_packet_size: " << (uint32_t)partial_packet_size << "\n";
    DPRINT << "fabric_packet_header_cb_id: " << (uint32_t)fabric_packet_header_cb_id << "\n";
    DPRINT << "use_fabric_on_receiver: " << (uint32_t)use_fabric_on_receiver << "\n";
    DPRINT << "use_fabric_on_sender: " << (uint32_t)use_fabric_on_sender << "\n";
    DPRINT << "num_upstream_sockets: " << (uint32_t)num_upstream_sockets << "\n";
    DPRINT << "upstream_socket_0_config_addr: " << (uint32_t)upstream_socket_0_config_addr << "\n";
    DPRINT << "upstream_socket_1_config_addr: " << (uint32_t)upstream_socket_1_config_addr << "\n";
    DPRINT << "upstream_socket_2_config_addr: " << (uint32_t)upstream_socket_2_config_addr << "\n";
    DPRINT << "upstream_socket_3_config_addr: " << (uint32_t)upstream_socket_3_config_addr << "\n";
    DPRINT << "upstream_socket_4_config_addr: " << (uint32_t)upstream_socket_4_config_addr << "\n";
    DPRINT << "upstream_socket_5_config_addr: " << (uint32_t)upstream_socket_5_config_addr << "\n";
    DPRINT << "upstream_socket_6_config_addr: " << (uint32_t)upstream_socket_6_config_addr << "\n";
    DPRINT << "upstream_socket_7_config_addr: " << (uint32_t)upstream_socket_7_config_addr << "\n";
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection_2;
    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection;

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        downstream_fabric_connection_2 =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }
    DPRINT << "after building fabric connections on sender\n";

    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);

    // Debug: Print downstream encoding (where D2D_0 will send to D2D_1)
    DPRINT << "D2D_0 Sender Socket downstream encoding:\n";
    DPRINT << "  downstream_noc_x: " << (uint32_t)downstream_enc.d2d.downstream_noc_x << "\n";
    DPRINT << "  downstream_noc_y: " << (uint32_t)downstream_enc.d2d.downstream_noc_y << "\n";
    DPRINT << "  downstream_bytes_sent_addr: " << (uint32_t)sender_socket.downstream_bytes_sent_addr << "\n";
    DPRINT << "  downstream_fifo_addr: " << (uint32_t)sender_socket.downstream_fifo_addr << "\n";

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
    DPRINT << "after creating receiver socket interfaces\n";

    // Debug: print receiver socket FIFO addresses
    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        DPRINT << "Receiver socket " << i << ":\n";
        DPRINT << "  config_addr: " << receiver_sockets[i].config_addr << "\n";
        DPRINT << "  fifo_addr: " << receiver_sockets[i].fifo_addr << "\n";
        DPRINT << "  fifo_total_size: " << receiver_sockets[i].fifo_total_size << "\n";
    }

    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        set_receiver_socket_page_size(receiver_sockets[i], upstream_page_size);
    }

    DPRINT << "Starting d2d exchange kernel with " << (uint32_t)num_upstream_sockets << " upstream sockets" << ENDL();

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    // Store just the L1 base address for downstream FIFO - we'll add offsets and create NOC addr later
    uint32_t downstream_fifo_l1_addr = sender_socket.downstream_fifo_addr;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr_2 = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_socket_packet_header_addrs[8] = {nullptr};

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
        DPRINT << "after opening downstream fabric connections and setting routes\n";
    }

    uint64_t upstream_bytes_acked_noc_addrs[8];
    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        upstream_bytes_acked_noc_addrs[i] = get_noc_addr(
            receiver_sockets[i].d2d.upstream_noc_x,
            receiver_sockets[i].d2d.upstream_noc_y,
            receiver_sockets[i].d2d.upstream_bytes_acked_addr);
    }
    DPRINT << "after calculating NOC addresses for upstream bytes acked\n";

    uint32_t current_socket_idx = 0;
    uint32_t bytes_accumulated = 0;
    bool data_pushed = false;

    socket_reserve_pages(sender_socket, 1);
    DPRINT << "after reserving page on sender socket\n";

    // Collect data from all upstream sockets into a single larger page
    while (true) {
        // Wait for pages in current upstream socket with termination checks
        if (!socket_wait_for_pages_with_termination(receiver_sockets[current_socket_idx], 1, termination_semaphore)) {
            DPRINT << "Termination signal received. Ending kernel main loop.\n";
            break;
        }

        auto l1_read_addr = receiver_sockets[current_socket_idx].read_ptr;
        // Calculate offset within the downstream buffer for this socket's data
        uint32_t skt_offset = bytes_accumulated;
        // Calculate the L1 address first, then create NOC address
        uint32_t dst_l1_addr = downstream_fifo_l1_addr + sender_socket.write_ptr + skt_offset;
        uint64_t dst_addr =
            get_noc_addr(downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, dst_l1_addr);

        DPRINT << "Socket " << current_socket_idx << " writing to offset " << skt_offset << "\n";

        send_pages_over_socket(
            sender_socket,
            downstream_fabric_connection,
            downstream_fabric_connection_2,
            downstream_data_packet_header_addr,
            downstream_data_packet_header_addr_2,
            downstream_bytes_sent_noc_addr,
            l1_read_addr,
            dst_addr,
            0);  // Pass 0 as offset since we already added it to dst_addr
        DPRINT << "after sending pages over socket\n";

        socket_pop_pages(receiver_sockets[current_socket_idx], 1);

        socket_notify_sender(receiver_sockets[current_socket_idx]);
        DPRINT << "after notifying sender\n";

        invalidate_l1_cache();

        // Update accumulation
        bytes_accumulated += upstream_page_size;
        current_socket_idx = (current_socket_idx + 1) % num_upstream_sockets;
        DPRINT << "current socket idx: " << (uint32_t)current_socket_idx
               << ", bytes_accumulated: " << (uint32_t)bytes_accumulated << ENDL();

        // Push when we've accumulated a full downstream page
        DPRINT << "Checking push condition: bytes_accumulated=" << (uint32_t)bytes_accumulated
               << ", page_size=" << (uint32_t)page_size << ENDL();
        if (bytes_accumulated >= page_size) {
            DPRINT << "PUSHING pages to D2D_1! Calling socket_push_pages...\n";
            socket_push_pages(sender_socket, 1);
            DPRINT << "Called socket_push_pages, now calling socket_notify_receiver...\n";
            socket_notify_receiver(sender_socket);
            DPRINT << "Notified D2D_1 receiver socket!\n";
            data_pushed = true;
            bytes_accumulated = 0;

            // Reserve next page if continuing
            break;
            socket_reserve_pages(sender_socket, 1);
        }
        DPRINT << "after reserving page on sender socket\n";
    }

    // Push any remaining data if we broke out of loop before filling a complete page
    DPRINT << "Final check: bytes_accumulated=" << (uint32_t)bytes_accumulated
           << ", data_pushed=" << (uint32_t)data_pushed << ENDL();
    if (bytes_accumulated > 0 && !data_pushed) {
        DPRINT << "PUSHING remaining data to D2D_1!\n";
        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
        DPRINT << "Pushed and notified for remaining data!\n";
    }
    DPRINT << "after pushing remaining data if any\n";

    invalidate_l1_cache();

    update_socket_config(sender_socket);
    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        update_socket_config(receiver_sockets[i]);
    }

    DPRINT << "after updating socket configs\n";
    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.close();
        downstream_fabric_connection_2.close();
    }
    DPRINT << "end of d2d exchange 0 kernel main\n";
}
