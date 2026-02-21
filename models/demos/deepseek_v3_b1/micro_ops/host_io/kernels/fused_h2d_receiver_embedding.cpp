// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "pcie_noc_utils.h"

// Get this value from MeshSocket struct on host
constexpr uint32_t recv_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t token_page_size = get_compile_time_arg_val(2);
constexpr bool pull_from_host = get_compile_time_arg_val(3);
constexpr bool loopback_mode = get_compile_time_arg_val(4);
constexpr uint32_t downstream_interface_index = get_compile_time_arg_val(5);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(6);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(7);
constexpr uint32_t num_whole_fabric_packets_link_0 = get_compile_time_arg_val(8);
constexpr uint32_t num_whole_fabric_packets_link_1 = get_compile_time_arg_val(9);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(10);
constexpr bool use_fabric = get_compile_time_arg_val(11);
constexpr uint32_t embedding_cb_index = get_compile_time_arg_val(12);
constexpr uint32_t embedding_page_size = get_compile_time_arg_val(13);
constexpr uint32_t embedding_addr = get_compile_time_arg_val(14);
// TensorAccessorArgs for embedding tensor at CT arg index 15
constexpr auto embedding_args = TensorAccessorArgs<15>();

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
    SocketSenderInterface& sender_socket, uint32_t l1_read_addr, uint64_t dst_addr, uint32_t page_size) {
    noc_async_write(l1_read_addr, dst_addr, page_size);
    socket_push_pages(sender_socket, 1);
    socket_notify_receiver(sender_socket);
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
    uint64_t dst_addr) {
    if constexpr (use_fabric) {
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
        socket_push_pages(sender_socket, 1);
    } else {
        write_data_to_local_core_with_ack(sender_socket, l1_read_addr, dst_addr, embedding_page_size);
    }
}

void kernel_main() {
    size_t rt_args_idx = 0;

    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection_2;

    if constexpr (use_fabric) {
        downstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        downstream_fabric_connection_2 =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }

    auto embedding_accessor = TensorAccessor(embedding_args, embedding_addr, embedding_page_size);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(recv_socket_config_addr);
    SocketSenderInterface sender_socket = {};

    sender_downstream_encoding downstream_enc;

    if constexpr (!loopback_mode) {
        sender_socket = create_sender_socket_interface(downstream_interface_index);
        set_sender_socket_page_size(sender_socket, embedding_page_size);
        downstream_enc = get_downstream_encoding(sender_socket, 0);
    }

    set_receiver_socket_page_size(receiver_socket, token_page_size);

    uint32_t read_addr_hi = receiver_socket.h2d.data_addr_hi;
    uint32_t read_addr_lo = receiver_socket.h2d.data_addr_lo;
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr_2 = nullptr;

    if constexpr (use_fabric) {
        // Safe to use downstream_enc here: Fabric being enabled means that a socket will be used for downstream
        // communication
        downstream_data_packet_header_addr =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
        downstream_data_packet_header_addr_2 = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

        downstream_fabric_connection.open();
        downstream_fabric_connection_2.open();

        fabric_set_unicast_route(downstream_data_packet_header_addr, downstream_enc);
        fabric_set_unicast_route(downstream_data_packet_header_addr_2, downstream_enc);
    }

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    uint64_t downstream_bytes_sent_noc_addr = 0;
    uint64_t downstream_data_addr = 0;

    if constexpr (!loopback_mode) {
        downstream_bytes_sent_noc_addr = get_noc_addr(
            downstream_enc.d2d.downstream_noc_x,
            downstream_enc.d2d.downstream_noc_y,
            sender_socket.downstream_bytes_sent_addr);

        downstream_data_addr = get_noc_addr(
            downstream_enc.d2d.downstream_noc_x,
            downstream_enc.d2d.downstream_noc_y,
            sender_socket.downstream_fifo_addr);
    }

    while (true) {
        // Wait for pages in H2D socket
        if (!socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
            break;
        }
        if constexpr (pull_from_host) {
            // Pages available in H2D socket - read over PCIe
            noc_async_wide_read_any_len_with_state(
                NOC_INDEX,
                pcie_xy_enc,
                ((static_cast<uint64_t>(read_addr_hi) << 32) | read_addr_lo) + receiver_socket.read_ptr -
                    receiver_socket.fifo_addr,
                receiver_socket.read_ptr,
                token_page_size);
            noc_async_read_barrier();
        }

        // TODO: Add and assert that token id is within vocab size
        volatile tt_l1_ptr uint32_t* token_id_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr);
        // Embedding CB is a scratch pad for now. We only read into the first slot of the CB.
        // TODO: Setup separate reader to pipeline reads.
        uint32_t l1_write_addr = get_write_ptr(embedding_cb_index);
        uint64_t noc_addr = embedding_accessor.get_noc_addr(*token_id_ptr);
        noc_async_read(noc_addr, l1_write_addr, embedding_page_size);
        noc_async_read_barrier();

        if constexpr (loopback_mode) {
            cb_reserve_back(downstream_interface_index, 1);
            noc_async_write(
                get_noc_addr(get_read_ptr(embedding_cb_index)),
                get_noc_addr(get_write_ptr(downstream_interface_index)),
                embedding_page_size);
            noc_async_write_barrier();
            cb_push_back(downstream_interface_index, 1);
        } else {
            auto l1_read_addr = get_read_ptr(embedding_cb_index);
            uint64_t dst_addr = downstream_data_addr + sender_socket.write_ptr;

            socket_reserve_pages(sender_socket, 1);
            send_pages_over_socket(
                sender_socket,
                downstream_fabric_connection,
                downstream_fabric_connection_2,
                downstream_data_packet_header_addr,
                downstream_data_packet_header_addr_2,
                downstream_bytes_sent_noc_addr,
                l1_read_addr,
                dst_addr);
        }
        socket_pop_pages(receiver_socket, 1);
        // Notify Host that pages were popped from H2D socket
        socket_notify_sender(receiver_socket);
        invalidate_l1_cache();
    }

    update_socket_config(receiver_socket);
    if constexpr (!loopback_mode) {
        socket_barrier(sender_socket);
    }

    noc_async_write_barrier();
    noc_async_read_barrier();

    if constexpr (use_fabric) {
        downstream_fabric_connection.close();
        downstream_fabric_connection_2.close();
    }
}
