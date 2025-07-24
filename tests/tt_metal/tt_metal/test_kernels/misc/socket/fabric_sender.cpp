// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void fabric_write_any_len(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    uint32_t src_addr,
    uint64_t dst_addr,
    uint32_t xfer_size,
    SocketSenderInterface& sender_socket) {
    fabric_set_unicast_route(data_packet_header_addr, sender_socket);
    while (xfer_size > FABRIC_MAX_PACKET_SIZE) {
        data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, FABRIC_MAX_PACKET_SIZE);
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_without_header_non_blocking_from_address(src_addr, FABRIC_MAX_PACKET_SIZE);
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
        dst_addr += FABRIC_MAX_PACKET_SIZE;
        src_addr += FABRIC_MAX_PACKET_SIZE;
        xfer_size -= FABRIC_MAX_PACKET_SIZE;
    }
    data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, xfer_size);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(src_addr, xfer_size);
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // This kernel relies on two fabric headers stored in fabric_packet_header_cb:
    //  - data_packet_header: Used for issuing writes to downstream data cores
    //  - socket_packet_header: Used by socket APIs for control flow
    constexpr uint32_t fabric_packet_header_cb_id = 0;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    fabric_connection.open();

    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t data_addr = local_l1_buffer_addr;
    uint64_t receiver_noc_coord_addr = get_noc_addr(sender_socket.downstream_noc_x, sender_socket.downstream_noc_y, 0);

    uint32_t outstanding_data_size = data_size;

    // Sends 1 page at a time and does handshake with receiver, can be optimized
    // to notify receiver after writing larger chunks
    while (outstanding_data_size) {
        socket_reserve_pages(sender_socket, 1);
        // Write Data over Fabric
        fabric_write_any_len(
            data_packet_header_addr,
            fabric_connection,
            data_addr,
            receiver_noc_coord_addr | sender_socket.write_ptr,
            page_size,
            sender_socket);
        data_addr += page_size;
        outstanding_data_size -= page_size;
        socket_push_pages(sender_socket, 1);

        fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
    }
    socket_barrier(sender_socket);
    // Write updated socket configs to the L1 config buffer (were cached on stack during kernel execution)
    update_socket_config(sender_socket);
    fabric_connection.close();
}
