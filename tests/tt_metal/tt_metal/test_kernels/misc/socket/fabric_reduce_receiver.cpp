// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket0_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t socket1_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(5);

    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender receiver0_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender receiver1_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    receiver0_fabric_connection.open_start();
    receiver1_fabric_connection.open_start();
    // Sanity
    cb_reserve_back(fabric_packet_header_cb_id, 2);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket0_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket1_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

    // Create Socket Interface
    SocketReceiverInterface receiver0_socket = create_receiver_socket_interface(socket0_config_addr);
    set_receiver_socket_page_size(receiver0_socket, page_size);
    SocketReceiverInterface receiver1_socket = create_receiver_socket_interface(socket1_config_addr);
    set_receiver_socket_page_size(receiver1_socket, page_size);
    receiver0_fabric_connection.open_finish();
    receiver1_fabric_connection.open_finish();

    // Reads one page at a time and sends ack to sender, can be optimized to notify
    // receiver after reading larger chunks
    constexpr uint32_t num_pages = data_size / page_size;
    for (uint32_t i = 0; i < num_pages; ++i) {
        socket_wait_for_pages(receiver0_socket, 1);
        socket_wait_for_pages(receiver1_socket, 1);
        cb_reserve_back(out_cb_id, 1);
        volatile tt_l1_ptr uint32_t* out_cb_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(out_cb_id));
        volatile tt_l1_ptr uint32_t* in0_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver0_socket.read_ptr);
        volatile tt_l1_ptr uint32_t* in1_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver1_socket.read_ptr);
        for (uint32_t j = 0; j < page_size / sizeof(uint32_t); ++j) {
            out_cb_ptr[j] = in0_ptr[j] + in1_ptr[j];
        }
        cb_push_back(out_cb_id, 1);
        socket_pop_pages(receiver0_socket, 1);
        socket_pop_pages(receiver1_socket, 1);
        fabric_socket_notify_sender(receiver0_socket, receiver0_fabric_connection, socket0_packet_header_addr);
        fabric_socket_notify_sender(receiver1_socket, receiver1_fabric_connection, socket1_packet_header_addr);
    }
    update_socket_config(receiver0_socket);
    update_socket_config(receiver1_socket);
    receiver0_fabric_connection.close();
    receiver1_fabric_connection.close();
}
