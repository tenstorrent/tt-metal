// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t data_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_per_page = get_compile_time_arg_val(5);
    constexpr uint32_t num_pages = data_size / page_size;
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);

    uint64_t dst_noc_addr = get_noc_addr(local_l1_buffer_addr);

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

    for (uint32_t p = 0; p < num_pages; ++p) {
        cb_wait_front(output_cb_index, num_tiles_per_page);
        // Compute has processed the input, can signal sender
        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);
        uint32_t src_addr = get_read_ptr(output_cb_index);
        noc_async_write(src_addr, dst_noc_addr, page_size);
        dst_noc_addr += page_size;
        noc_async_writes_flushed();
        cb_pop_front(output_cb_index, num_tiles_per_page);
    }
    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
