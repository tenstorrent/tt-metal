// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(4);
    SocketReceiverInterface2 receiver_socket = create_receiver_socket_interface_2(socket_config_addr);

    set_receiver_socket_page_size(receiver_socket, page_size);

    // Reads one page at a time and sends ack to sender, can be optimized to notify
    // receiver after reading larger chunks
    // In this example the local_cb isn't actually needed (we can just get the rptr)
    // from the socket itself
    // Its there to illustrate how a local CB can be connected to the socket interface
    // If the local CB was being used for compute:
    //  - The reader (this kernel) would:
    //     - Handshake with sender
    //     - Call cb_reserve_back on the recv_cb_id (shared between reader and compute)
    //     - Call cb_push_back to notify the compute kernel of data
    //  - The compute kernel would:
    //     - Call cb_wait_front
    //     - Call cb_pop_front to notify the receiver that pages are available
    //  - In this case, an additional CB would be needed to synchronize the receiver and
    //    compute kernels (socket_notify_sender cannot be called until compute is done with
    //    pages in the local CB)
    DPRINT << "Number of iterations: " << num_iterations << ENDL();
    for (uint32_t i = 0; i < num_iterations; i++) {
        DPRINT << "Running iteration " << i << ENDL();
        uint32_t outstanding_data_size = data_size;
        uint64_t dst_noc_addr = get_noc_addr(local_l1_buffer_addr);
        while (outstanding_data_size) {
            DPRINT << "Waiting for pages" << ENDL();
            socket_wait_for_pages(receiver_socket, 1);
            DPRINT << "Pages received" << ENDL();
            noc_async_write(receiver_socket.base.read_ptr, dst_noc_addr, page_size);
            dst_noc_addr += page_size;
            outstanding_data_size -= page_size;
            noc_async_write_barrier();
            socket_pop_pages(receiver_socket, 1);
            socket_notify_sender(receiver_socket);
            // socket_notify_sender(receiver_socket);
        }
    }
    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
