// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from mesh_socket_t struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);

    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    constexpr uint32_t fabric_packet_header_cb_id = 0;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));

    fabric_connection.open();
    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    uint32_t outstanding_data_size = data_size;
    set_receiver_socket_page_size(receiver_socket, page_size);

    uint64_t dst_noc_addr = get_noc_addr(local_l1_buffer_addr);

    // Reads one page at a time and sends ack to sender, can be optimized to notify
    // receiver after reading larger chunks
    while (outstanding_data_size) {
        socket_wait_for_pages(receiver_socket, 1);
        noc_async_write(receiver_socket.read_ptr, dst_noc_addr, page_size);
        dst_noc_addr += page_size;
        outstanding_data_size -= page_size;
        socket_pop_pages(receiver_socket, 1);
        noc_async_write_barrier();
        fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
    }
    update_socket_config(receiver_socket);
    fabric_connection.close();
}
