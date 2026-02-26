// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Rate-mode sender kernel for pipeline throughput benchmarking.
// Sends data downstream for num_iterations without waiting for loopback acks.
// num_iterations is a runtime arg so that warmup and timed runs share the same compiled binary.

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t data_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(2);  // Aligned page size
constexpr uint32_t aligned_partial_packet_size = get_compile_time_arg_val(3);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(4);
constexpr uint32_t num_whole_packets_link_0 = get_compile_time_arg_val(5);
constexpr uint32_t num_whole_packets_link_1 = get_compile_time_arg_val(6);
constexpr uint32_t input_page_size = get_compile_time_arg_val(7);

constexpr uint32_t input_args_cta_idx = 8;
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
    size_t rt_args_idx = 0;
    uint32_t input_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_tensor_addr, input_page_size);

    // Two fabric connections for dual-link forwarding to downstream
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection_2 =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // num_iterations is a runtime arg (after fabric connection args) so compilation is shared
    uint32_t num_iterations = get_arg_val<uint32_t>(rt_args_idx++);

    // Two packet headers for dual-link forwarding
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr_2 =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

    fabric_connection.open();
    fabric_connection_2.open();

    // Create Sender Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_block_size);

    // Only one downstream in this op
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    fabric_set_unicast_route(data_packet_header_addr_2, downstream_enc);

    uint64_t receiver_noc_coord_addr =
        get_noc_addr_from_bank_id<false>(bank_id, 0, tt::tt_fabric::connection_interface::edm_fabric_write_noc_index);

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);

    // Read data from DRAM into CB once (data is reused across all iterations)
    auto noc_read_addr = input_addr_gen.get_noc_addr(0);
    auto cb_addr = get_write_ptr(data_cb_id);
    noc_async_read<input_page_size>(noc_read_addr, cb_addr, input_page_size);
    noc_async_read_barrier();

    auto l1_read_addr_base = get_read_ptr(data_cb_id);

    // Main loop: push data downstream as fast as possible
    for (uint32_t i = 0; i < num_iterations; ++i) {
        auto l1_read_addr = l1_read_addr_base;
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
    }

    update_socket_config(sender_socket);
    fabric_connection.close();
    fabric_connection_2.close();
}
