// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

void kernel_main() {
    // Get this value from mesh_socket_t struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t downstream_noc_x = get_compile_time_arg_val(4);
    constexpr uint32_t downstream_noc_y = get_compile_time_arg_val(5);
    constexpr uint32_t downstream_fifo_addr = get_compile_time_arg_val(6);
    constexpr uint32_t downstream_sem_addr = get_compile_time_arg_val(7);
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    constexpr uint32_t fabric_packet_header_cb_id = 0;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* write_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* atomic_inc_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

    constexpr uint32_t packet_size_bytes = 1024;
    uint64_t noc_dest_addr = get_noc_addr(downstream_noc_x, downstream_noc_y, downstream_fifo_addr);
    uint64_t noc_dest_sem_addr = get_noc_addr(downstream_noc_x, downstream_noc_y, downstream_sem_addr);
    fabric_connection.open();

    write_packet_header_addr->to_chip_unicast(static_cast<uint8_t>(1));
    write_packet_header_addr->to_noc_unicast_write(
        NocUnicastCommandHeader{noc_dest_addr}, packet_size_bytes);  // supply dest addr here

    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(local_l1_buffer_addr, packet_size_bytes);
    fabric_connection.send_payload_blocking_from_address(
        (uint32_t)write_packet_header_addr, sizeof(PACKET_HEADER_TYPE));

    atomic_inc_packet_header_addr->to_chip_unicast(static_cast<uint8_t>(1));
    // Set dest sem addr here

    atomic_inc_packet_header_addr->to_noc_unicast_inline_write(
        NocUnicastInlineWriteCommandHeader{noc_dest_sem_addr, 1});  // supply dest addr here
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_blocking_from_address(
        (uint32_t)atomic_inc_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    fabric_connection.close();

    // SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    // uint32_t outstanding_data_size = data_size;
    // set_sender_socket_page_size(sender_socket, page_size);

    // uint32_t data_addr = local_l1_buffer_addr;
    // uint64_t receiver_noc_coord_addr = get_noc_addr(sender_socket.downstream_noc_x, sender_socket.downstream_noc_y,
    // 0);

    // // Sends 1 page at a time and does handshake with receiver, can be optimized
    // // to notify receiver after writing larger chunks
    // while (outstanding_data_size) {
    //     socket_reserve_pages(sender_socket, 1);
    //     // noc_async_read into local CB + barrier
    //     // fabric write from local CB to data cores + barrier
    //     // Issuing the write requires the wptr from the socket itself
    //     // The user can get the wptr directly from the sender_socket, or
    //     // we can add wrappers issue the write itself
    //     noc_async_write(data_addr, receiver_noc_coord_addr | sender_socket.write_ptr, page_size);
    //     data_addr += page_size;
    //     outstanding_data_size -= page_size;
    //     socket_push_pages(sender_socket, 1);
    //     socket_notify_receiver(sender_socket);
    // }
    // socket_barrier(sender_socket);
    // // Write updated socket configs to the L1 config buffer (were cached on stack during kernel execution)
    // update_socket_config(sender_socket);
}
