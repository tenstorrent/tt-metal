// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t data_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(2);  // This is assumed to be aligned
constexpr uint32_t aligned_partial_packet_size = get_compile_time_arg_val(3);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(4);
constexpr uint32_t num_whole_packets_link_0 = get_compile_time_arg_val(5);
constexpr uint32_t num_whole_packets_link_1 = get_compile_time_arg_val(6);
constexpr uint32_t input_page_size = get_compile_time_arg_val(7);
constexpr uint32_t credit_address = get_compile_time_arg_val(8);
constexpr uint32_t num_iterations = get_compile_time_arg_val(9);
constexpr uint32_t enable_correctness_check = get_compile_time_arg_val(10);
// Send cumulative ack upstream every N iterations (e.g. fifo_size_in_pages/2 for half-buffer acks).
constexpr uint32_t notify_sender_every_n_iterations = get_compile_time_arg_val(11);

constexpr uint32_t input_args_cta_idx = 12;
constexpr uint32_t input_args_crta_idx = 0;

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
    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t input_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t recv_socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_tensor_addr, input_page_size);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection_2 =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // This kernel relies on two fabric headers stored in fabric_packet_header_cb:
    //  - data_packet_header: Used for issuing writes to downstream data cores
    //  - socket_packet_header: Used by socket APIs for control flow
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr_2 =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + 2 * sizeof(PACKET_HEADER_TYPE));

    fabric_connection.open();
    fabric_connection_2.open();
    upstream_fabric_connection.open();
    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(recv_socket_config_addr);

    set_sender_socket_page_size(sender_socket, socket_block_size);
    set_receiver_socket_page_size(receiver_socket, socket_block_size);

    // Only one downstream in this op
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    fabric_set_unicast_route(data_packet_header_addr_2, downstream_enc);
    fabric_set_unicast_route(upstream_socket_packet_header_addr, receiver_socket);

    uint64_t receiver_noc_coord_addr =
        get_noc_addr_from_bank_id<false>(bank_id, 0, tt::tt_fabric::connection_interface::edm_fabric_write_noc_index);

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    uint64_t upstream_bytes_acked_noc_addr = get_noc_addr(
        receiver_socket.d2d.upstream_noc_x,
        receiver_socket.d2d.upstream_noc_y,
        receiver_socket.d2d.upstream_bytes_acked_addr);

    // Initial read of data into CB. This is not profiled, since the expectation is that
    // in a real workload,
    auto noc_read_addr = input_addr_gen.get_noc_addr(0);
    auto cb_addr = get_write_ptr(data_cb_id);
    noc_async_read<input_page_size>(noc_read_addr, cb_addr, input_page_size);
    noc_async_read_barrier();

    // Initial Handshake:
    // Each kernel sends credits to its downstream peer
    // This kernel (the sender) waits for a credit to be issued by its upstream
    // At the end of this handshake, all kernels have started
    // This allows us to accurately measure loopback latency

    // 1. Send credit downstream
    uint64_t remote_credit_addr =
        get_noc_addr(downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, credit_address);
    volatile tt_l1_ptr uint32_t* credit_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_address);
    data_packet_header_addr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{remote_credit_addr, 1});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    // 2. Wait for credit from producer of this kernel
    while (*credit_addr == 0) {
        invalidate_l1_cache();
    }
    // Measure roundtrip latency: Time it takes for the data to be picked up from L1 + pipeline latency
    // Latency measurements are written to the same buffer used for credit synchronization
    uint32_t measurement_addr = credit_address;
    auto l1_read_addr_base = get_read_ptr(data_cb_id);
    for (uint32_t i = 0; i < num_iterations; ++i) {
        auto l1_read_addr = l1_read_addr_base;
        uint64_t start_timestamp = get_timestamp();
        socket_reserve_pages(sender_socket, 1);
        uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr + sender_socket.downstream_fifo_addr;
        for (uint32_t j = 0; j < num_whole_packets_link_0; ++j) {
            write_data_to_remote_core_with_ack(
                fabric_connection,
                data_packet_header_addr,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);
            dst_addr += whole_packet_size;
            l1_read_addr += whole_packet_size;
        }

        for (uint32_t j = 0; j < num_whole_packets_link_1; ++j) {
            write_data_to_remote_core_with_ack(
                fabric_connection_2,
                data_packet_header_addr_2,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);
            dst_addr += whole_packet_size;
            l1_read_addr += whole_packet_size;
        }
        if constexpr (aligned_partial_packet_size) {
            write_data_to_remote_core_with_ack(
                fabric_connection_2,
                data_packet_header_addr_2,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                aligned_partial_packet_size);
        }
        socket_push_pages(sender_socket, 1);

        socket_wait_for_pages(receiver_socket, 1);
        if constexpr (enable_correctness_check) {
            uint32_t socket_read_addr = receiver_socket.read_ptr;
            uint32_t val = 0;
            for (uint32_t j = 0; j < input_page_size / 4; j += 4) {
                if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket_read_addr + j) != val) {
                    while (true);
                }
                val++;
            }
        }
        socket_pop_pages(receiver_socket, 1);
        if (notify_sender_every_n_iterations != 0 && (i % notify_sender_every_n_iterations) == 0) {
            fabric_socket_notify_sender_stateful(
                receiver_socket,
                upstream_fabric_connection,
                upstream_socket_packet_header_addr,
                upstream_bytes_acked_noc_addr);
        }
        uint64_t end_timestamp = get_timestamp();
        uint64_t latency = end_timestamp - start_timestamp;
        *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(measurement_addr) = latency;
        measurement_addr += sizeof(uint64_t);
    }
    update_socket_config(sender_socket);
    update_socket_config(receiver_socket);
    fabric_connection.close();
    fabric_connection_2.close();
    upstream_fabric_connection.close();
}
