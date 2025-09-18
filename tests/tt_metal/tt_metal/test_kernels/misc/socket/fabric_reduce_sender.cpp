// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t data_size = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender sender_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    sender_fabric_connection.open_start();

    // Sanity
    auto* data_packet_header_addr = PacketHeaderPool::allocate_header();
    auto* socket_packet_header_addr = PacketHeaderPool::allocate_header();

    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    sender_fabric_connection.open_finish();

    // Reads one page at a time and sends ack to sender, can be optimized to notify
    // receiver after reading larger chunks
    constexpr uint32_t num_pages = data_size / page_size;
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(out_cb_id, 1);
        socket_reserve_pages(sender_socket, 1);
        // Write Data over Fabric
        uint32_t data_addr = get_read_ptr(out_cb_id);

        for (uint32_t i = 0; i < sender_socket.num_downstreams; i++) {
            sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, i);
            uint64_t receiver_noc_coord_addr =
                get_noc_addr(downstream_enc.downstream_noc_x, downstream_enc.downstream_noc_y, sender_socket.write_ptr);
            fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
            data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{receiver_noc_coord_addr}, page_size);
            sender_fabric_connection.wait_for_empty_write_slot();
            sender_fabric_connection.send_payload_without_header_non_blocking_from_address(data_addr, page_size);
            sender_fabric_connection.send_payload_blocking_from_address(
                (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
        }
        socket_push_pages(sender_socket, 1);
        fabric_socket_notify_receiver(sender_socket, sender_fabric_connection, socket_packet_header_addr);
        noc_async_writes_flushed();
        cb_pop_front(out_cb_id, 1);
    }
    update_socket_config(sender_socket);
    sender_fabric_connection.close();
}
