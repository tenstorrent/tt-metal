// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t output_core_x = get_compile_time_arg_val(4);
    constexpr uint32_t output_core_y = get_compile_time_arg_val(5);
    constexpr uint32_t output_addr = get_compile_time_arg_val(6);

    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    fabric_connection.open_start();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, page_size);
    uint64_t dst_noc_addr = get_noc_addr(output_core_x, output_core_y, output_addr);

    fabric_connection.open_finish();

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
    constexpr uint32_t num_pages = data_size / page_size;
    for (uint32_t i = 0; i < num_pages; ++i) {
        socket_wait_for_pages(receiver_socket, 1);
        noc_async_write(receiver_socket.read_ptr, dst_noc_addr, page_size);
        dst_noc_addr += page_size;
        socket_pop_pages(receiver_socket, 1);
        noc_async_writes_flushed();
        fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
    }
    update_socket_config(receiver_socket);
    fabric_connection.close();
}
