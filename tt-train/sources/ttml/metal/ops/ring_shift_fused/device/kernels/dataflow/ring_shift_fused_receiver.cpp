// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver core: take the sender core's handshake page from the previous
// chip, answer with every output tensor's base address and the valid flag
// (inline fabric writes into the sender's page), then wait for the
// completion token. The data itself lands in the tensors without this
// core's help.

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/debug/waypoint.h"

constexpr uint32_t cb_headers = get_compile_time_arg_val(0);
constexpr uint32_t handshake_page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_tensors = get_compile_time_arg_val(2);

constexpr uint32_t kValidOffset = 0;
constexpr uint32_t kAddressesOffset = 4;

FORCE_INLINE void inline_write_upstream(
    const SocketReceiverInterface& receiver_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint64_t dst_noc_addr,
    uint32_t value) {
    fabric_set_unicast_route(header, receiver_socket);
    header->to_noc_unicast_inline_write(NocUnicastInlineWriteCommandHeader{dst_noc_addr, value});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    size_t arg = 0;
    const uint32_t socket_config_addr = get_arg_val<uint32_t>(arg++);
    uint32_t out_base[num_tensors];
    for (uint32_t t = 0; t < num_tensors; ++t) {
        out_base[t] = get_arg_val<uint32_t>(arg++);
    }
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(cb_headers));

    WAYPOINT("FOPN");
    fabric_connection.open();
    WAYPOINT("FOPD");
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, handshake_page_size);

    WAYPOINT("PGW1");
    socket_wait_for_pages(receiver_socket, 1);
    WAYPOINT("ANSW");
    const uint32_t sender_handshake_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr)[0];
    socket_pop_pages(receiver_socket, 1);
    fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_header);

    const uint32_t upstream_x = receiver_socket.d2d.upstream_noc_x;
    const uint32_t upstream_y = receiver_socket.d2d.upstream_noc_y;
    for (uint32_t t = 0; t < num_tensors; ++t) {
        inline_write_upstream(
            receiver_socket, fabric_connection, socket_header,
            get_noc_addr(upstream_x, upstream_y, sender_handshake_addr + kAddressesOffset + 4u * t), out_base[t]);
    }
    inline_write_upstream(
        receiver_socket, fabric_connection, socket_header,
        get_noc_addr(upstream_x, upstream_y, sender_handshake_addr + kValidOffset), 1u);

    WAYPOINT("PGW2");
    socket_wait_for_pages(receiver_socket, 1);
    WAYPOINT("CMPL");
    socket_pop_pages(receiver_socket, 1);
    fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_header);
    update_socket_config(receiver_socket);
    fabric_connection.close();
}
