// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender core, writer: advertise this core's handshake page to the receiver
// core on the next chip, wait for it to publish every output tensor's base
// address, then write this core's share of every tensor's pages straight
// into those tensors through the fabric (the two chips lay their DRAM out
// alike, so a page's address on the neighbour is the one it would have
// here), and finish with one completion token on the socket.

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/debug/waypoint.h"

constexpr uint32_t cb_data = get_compile_time_arg_val(0);
constexpr uint32_t cb_headers = get_compile_time_arg_val(1);
constexpr uint32_t cb_handshake = get_compile_time_arg_val(2);
constexpr uint32_t handshake_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_banks = get_compile_time_arg_val(4);
constexpr uint32_t num_tensors = get_compile_time_arg_val(5);

constexpr uint32_t kValidOffset = 0;
constexpr uint32_t kAddressesOffset = 4;  // output base address of tensor t at 4 * (1 + t)

FORCE_INLINE void fabric_write(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint32_t l1_read_addr,
    uint64_t dst_noc_addr,
    uint32_t size_bytes) {
    header->to_noc_unicast_write(NocUnicastCommandHeader{dst_noc_addr}, size_bytes);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, size_bytes);
    fabric_connection.send_payload_flush_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    size_t arg = 0;
    const uint32_t socket_config_addr = get_arg_val<uint32_t>(arg++);
    uint32_t page_size[num_tensors];
    uint32_t pages[num_tensors];
    uint32_t start[num_tensors];
    uint32_t ppp[num_tensors];
    uint32_t packing[num_tensors];
    for (uint32_t t = 0; t < num_tensors; ++t) {
        page_size[t] = get_arg_val<uint32_t>(arg++);
        pages[t] = get_arg_val<uint32_t>(arg++);
        start[t] = get_arg_val<uint32_t>(arg++);
        ppp[t] = get_arg_val<uint32_t>(arg++);
        packing[t] = get_arg_val<uint32_t>(arg++);
    }
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(cb_headers));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
        get_write_ptr(cb_headers) + sizeof(PACKET_HEADER_TYPE));

    WAYPOINT("FOPN");
    fabric_connection.open();
    WAYPOINT("FOPD");
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, handshake_page_size);
    const sender_downstream_encoding downstream = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_header, downstream);

    // ---- the handshake: our page's address to the receiver, its answers back into it.
    const uint32_t handshake_addr = get_write_ptr(cb_handshake);
    const uint32_t advertise_addr = handshake_addr + handshake_page_size;
    volatile tt_l1_ptr uint32_t* handshake = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr);
    handshake[kValidOffset / 4] = 0;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(advertise_addr)[0] = handshake_addr;
    WAYPOINT("ADVW");
    {
        socket_reserve_pages(sender_socket, 1);
        const uint64_t advertise_dst = get_noc_addr(
            downstream.d2d.downstream_noc_x,
            downstream.d2d.downstream_noc_y,
            sender_socket.downstream_fifo_addr + sender_socket.write_ptr);
        fabric_write(fabric_connection, data_header, advertise_addr, advertise_dst, handshake_page_size);
        socket_push_pages(sender_socket, 1);
        fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_header);
    }
    WAYPOINT("VALW");
    do {
        invalidate_l1_cache();
    } while (handshake[kValidOffset / 4] == 0);
    WAYPOINT("DATA");

    // ---- every tensor, this core's pages, in the reader's order.
    for (uint32_t t = 0; t < num_tensors; ++t) {
        const uint32_t out_base = handshake[kAddressesOffset / 4 + t];
        const InterleavedAddrGen<true> gen{.bank_base_address = out_base, .page_size = page_size[t]};
        const uint32_t end = start[t] + pages[t];
        if (packing[t] != 0u) {
            const uint32_t super_block = num_banks * ppp[t];
            const uint32_t region = ppp[t] * page_size[t];
            for (uint32_t sb = start[t]; sb < end; sb += super_block) {
                cb_wait_front(cb_data, 1);
                const uint32_t l1 = get_read_ptr(cb_data);
                for (uint32_t b = 0; b < num_banks; ++b) {
                    const uint32_t head = sb + b;
                    if (head >= end) {
                        break;
                    }
                    uint32_t count = 0;
                    for (uint32_t pp = head; count < ppp[t] && pp < end; pp += num_banks) {
                        ++count;
                    }
                    fabric_write(
                        fabric_connection, data_header, l1 + b * region, gen.get_noc_addr(head),
                        count * page_size[t]);
                }
                cb_pop_front(cb_data, 1);
            }
        } else {
            for (uint32_t p = start[t]; p < end; p += ppp[t]) {
                cb_wait_front(cb_data, 1);
                const uint32_t l1 = get_read_ptr(cb_data);
                const uint32_t n = (end - p < ppp[t]) ? end - p : ppp[t];
                for (uint32_t j = 0; j < n; ++j) {
                    fabric_write(
                        fabric_connection, data_header, l1 + j * page_size[t], gen.get_noc_addr(p + j),
                        page_size[t]);
                }
                cb_pop_front(cb_data, 1);
            }
        }
    }

    // ---- done: one token, then the socket's state for the next launch.
    WAYPOINT("DONE");
    socket_reserve_pages(sender_socket, 1);
    socket_push_pages(sender_socket, 1);
    fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_header);
    update_socket_config(sender_socket);
    fabric_connection.close();
}
