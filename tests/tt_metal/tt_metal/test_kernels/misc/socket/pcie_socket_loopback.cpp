// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t recv_socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(4);
    constexpr bool pull_from_host = get_compile_time_arg_val(5);
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(recv_socket_config_addr);
    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);

    set_receiver_socket_page_size(receiver_socket, page_size);
    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t read_addr_hi = receiver_socket.h2d.data_addr_hi;
    uint32_t read_addr_lo = receiver_socket.h2d.data_addr_lo;
    uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t outstanding_data_size = data_size;
        while (outstanding_data_size) {
            // Wait for pages in H2D socket
            socket_wait_for_pages(receiver_socket, 1);

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

            // Wait for space in D2H socket
            socket_reserve_pages(sender_socket, 1);
            // Space available in D2H socket - write to host over PCIe
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                NOC_INDEX,
                receiver_socket.read_ptr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                page_size,
                1);

            socket_push_pages(sender_socket, 1);
            // Notify Host that pages were pushed to D2H socket
            socket_notify_receiver(sender_socket);
            socket_pop_pages(receiver_socket, 1);
            noc_async_writes_flushed();
            // Notify Host that pages were popped from H2D socket
            socket_notify_sender(receiver_socket);

            outstanding_data_size -= page_size;
        }
    }

    update_socket_config(receiver_socket);
    update_socket_config(sender_socket);
    socket_barrier(sender_socket);

    noc_async_write_barrier();
    noc_async_read_barrier();
}
