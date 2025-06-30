// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t data_size = get_compile_time_arg_val(2);
    constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(4);

    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender sender_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    sender_fabric_connection.open_start();

    // Sanity
    cb_reserve_back(fabric_packet_header_cb_id, 2);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    uint64_t receiver_noc_coord_addr = get_noc_addr(sender_socket.downstream_noc_x, sender_socket.downstream_noc_y, 0);

    sender_fabric_connection.open_finish();

    // Reads one page at a time and sends ack to sender, can be optimized to notify
    // receiver after reading larger chunks
    constexpr uint32_t num_pages = data_size / page_size;
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(out_cb_id, 1);
        socket_reserve_pages(sender_socket, 1);
        // Write Data over Fabric
        uint32_t data_addr = get_read_ptr(out_cb_id);
        fabric_set_unicast_route(data_packet_header_addr, sender_socket);
        data_packet_header_addr->to_noc_unicast_write(
            NocUnicastCommandHeader{receiver_noc_coord_addr | sender_socket.write_ptr}, page_size);
        sender_fabric_connection.wait_for_empty_write_slot();
        sender_fabric_connection.send_payload_without_header_non_blocking_from_address(data_addr, page_size);
        sender_fabric_connection.send_payload_blocking_from_address(
            (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
        socket_push_pages(sender_socket, 1);
        fabric_socket_notify_receiver(sender_socket, sender_fabric_connection, socket_packet_header_addr);
        noc_async_writes_flushed();
        cb_pop_front(out_cb_id, 1);
    }
    update_socket_config(sender_socket);
    sender_fabric_connection.close();
}
