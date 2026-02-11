// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Rate-mode forwarding kernel for pipeline throughput benchmarking.
// Forwards data from recv_socket to send_socket for NUM_ITERATIONS.
// No initial handshake — pipeline synchronization is handled purely by socket flow control.

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(1);
constexpr uint32_t aligned_partial_packet_size = get_compile_time_arg_val(2);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(3);
constexpr uint32_t num_whole_packets_link_0 = get_compile_time_arg_val(4);
constexpr uint32_t num_whole_packets_link_1 = get_compile_time_arg_val(5);
constexpr uint32_t num_iterations = get_compile_time_arg_val(6);
// Send cumulative ack upstream every N iterations (e.g. fifo_size_in_pages/2 for half-buffer acks).
constexpr uint32_t notify_sender_every_n_iterations = get_compile_time_arg_val(7);

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

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t send_socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t recv_socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t downstream_bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection_2 =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // Three packet headers: one for upstream acks, two for dual-link downstream forwarding
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr_2 =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + 2 * sizeof(PACKET_HEADER_TYPE));

    upstream_fabric_connection.open();
    downstream_fabric_connection.open();
    downstream_fabric_connection_2.open();

    SocketSenderInterface send_socket = create_sender_socket_interface(send_socket_config_addr);
    SocketReceiverInterface recv_socket = create_receiver_socket_interface(recv_socket_config_addr);

    set_sender_socket_page_size(send_socket, socket_block_size);
    set_receiver_socket_page_size(recv_socket, socket_block_size);

    // Only one downstream in this op
    sender_downstream_encoding downstream_enc = get_downstream_encoding(send_socket, 0);

    fabric_set_unicast_route(downstream_data_packet_header_addr, downstream_enc);
    fabric_set_unicast_route(downstream_data_packet_header_addr_2, downstream_enc);
    fabric_set_unicast_route(upstream_socket_packet_header_addr, recv_socket);

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        send_socket.downstream_bytes_sent_addr);
    uint64_t upstream_bytes_acked_noc_addr = get_noc_addr(
        recv_socket.d2d.upstream_noc_x, recv_socket.d2d.upstream_noc_y, recv_socket.d2d.upstream_bytes_acked_addr);
    uint64_t receiver_noc_coord_addr = get_noc_addr_from_bank_id<false>(
        downstream_bank_id, 0, tt::tt_fabric::connection_interface::edm_fabric_write_noc_index);

    // No handshake — socket flow control synchronizes the pipeline naturally
    for (uint32_t i = 0; i < num_iterations; ++i) {
        socket_reserve_pages(send_socket, 1);
        socket_wait_for_pages(recv_socket, 1);
        auto l1_read_addr = recv_socket.read_ptr;
        uint64_t dst_addr = receiver_noc_coord_addr + send_socket.write_ptr + send_socket.downstream_fifo_addr;

        // Forward data to downstream via dual links
        for (uint32_t j = 0; j < num_whole_packets_link_0; ++j) {
            write_data_to_remote_core_with_ack(
                downstream_fabric_connection,
                downstream_data_packet_header_addr,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);
            dst_addr += whole_packet_size;
            l1_read_addr += whole_packet_size;
        }
        for (uint32_t j = 0; j < num_whole_packets_link_1; ++j) {
            write_data_to_remote_core_with_ack(
                downstream_fabric_connection_2,
                downstream_data_packet_header_addr_2,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);
            dst_addr += whole_packet_size;
            l1_read_addr += whole_packet_size;
        }
        if constexpr (aligned_partial_packet_size) {
            write_data_to_remote_core_with_ack(
                downstream_fabric_connection_2,
                downstream_data_packet_header_addr_2,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                aligned_partial_packet_size);
        }

        // Notify upstream and downstream that data has been consumed/produced
        socket_push_pages(send_socket, 1);
        socket_pop_pages(recv_socket, 1);

        // Ack upstream periodically to free sender FIFO space
        if (notify_sender_every_n_iterations != 0 && ((i + 1) % notify_sender_every_n_iterations) == 0) {
            fabric_socket_notify_sender_stateful(
                recv_socket,
                upstream_fabric_connection,
                upstream_socket_packet_header_addr,
                upstream_bytes_acked_noc_addr);
        }
    }

    update_socket_config(send_socket);
    update_socket_config(recv_socket);
    upstream_fabric_connection.close();
    downstream_fabric_connection.close();
    downstream_fabric_connection_2.close();
}
