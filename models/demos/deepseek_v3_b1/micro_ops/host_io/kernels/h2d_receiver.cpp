// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

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

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t recv_socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr bool pull_from_host = get_compile_time_arg_val(3);
    constexpr bool loopback_mode = get_compile_time_arg_val(4);
    constexpr uint32_t downstream_interface_index = get_compile_time_arg_val(5);
    constexpr bool has_embedding = get_compile_time_arg_val(6);
    constexpr uint32_t embedding_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t embedding_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t embedding_addr = get_compile_time_arg_val(9);
    // TensorAccessorArgs for embedding tensor at CT arg index 10
    constexpr auto embedding_args = TensorAccessorArgs<10>();

    auto embedding_accessor = TensorAccessor(embedding_args, embedding_addr, embedding_page_size);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(recv_socket_config_addr);
    SocketSenderInterface sender_socket = {};

    if constexpr (!loopback_mode) {
        sender_socket = create_sender_socket_interface(downstream_interface_index);
        set_sender_socket_page_size(sender_socket, page_size);
    }
    set_receiver_socket_page_size(receiver_socket, page_size);

    // Read first page of embedding tensor from DRAM into CB

    uint32_t read_addr_hi = receiver_socket.h2d.data_addr_hi;
    uint32_t read_addr_lo = receiver_socket.h2d.data_addr_lo;
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    while (true) {
        // Wait for pages in H2D socket
        if (!socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
            break;
        }
        if constexpr (pull_from_host) {
            // Pages available in H2D socket - read over PCIe
            noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                NOC_INDEX,
                pcie_xy_enc,
                ((static_cast<uint64_t>(read_addr_hi) << 32) | read_addr_lo) + receiver_socket.read_ptr -
                    receiver_socket.fifo_addr,
                receiver_socket.read_ptr,
                page_size);
            noc_async_read_barrier();
        }

        if constexpr (has_embedding) {
            // DPRINT << "Has embedding" << ENDL();
            volatile tt_l1_ptr uint32_t* token_id_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr);
            // DPRINT << "GOt token ID: " << token_id_ptr[0] << ENDL();
            uint32_t l1_write_addr = get_write_ptr(embedding_cb_index);
            uint64_t noc_addr = embedding_accessor.get_noc_addr(*token_id_ptr);
            // DPRINT << "Embedding Noc Address: " << noc_addr << ENDL();
            // DPRINT << "Read " << embedding_page_size << " bytes" << ENDL();
            noc_async_read(noc_addr, l1_write_addr, embedding_page_size);
            noc_async_read_barrier();
        }

        if constexpr (loopback_mode) {
            // DPRINT << "Reserve CB" << ENDL();
            cb_reserve_back(downstream_interface_index, 1);
            // DPRINT << "Write to CB" << ENDL();
            noc_async_write(
                get_noc_addr(get_read_ptr(embedding_cb_index)),
                get_noc_addr(get_write_ptr(downstream_interface_index)),
                embedding_page_size);
            noc_async_write_barrier();
            // DPRINT << "Done write" << ENDL();
            cb_push_back(downstream_interface_index, 1);
        } else {
            socket_reserve_pages(sender_socket, 1);
            sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
            noc_async_write(
                receiver_socket.read_ptr,
                get_noc_addr(
                    downstream_enc.d2d.downstream_noc_x,
                    downstream_enc.d2d.downstream_noc_y,
                    sender_socket.write_ptr + sender_socket.downstream_fifo_addr),
                page_size);
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
            noc_async_writes_flushed();
        }
        // DPRINT << "Pop pages" << ENDL();
        socket_pop_pages(receiver_socket, 1);
        // Notify Host that pages were popped from H2D socket
        // DPRINT << "Notify sender" << ENDL();
        socket_notify_sender(receiver_socket);
        invalidate_l1_cache();
    }

    update_socket_config(receiver_socket);
    if constexpr (!loopback_mode) {
        socket_barrier(sender_socket);
    }

    noc_async_write_barrier();
    noc_async_read_barrier();
}
