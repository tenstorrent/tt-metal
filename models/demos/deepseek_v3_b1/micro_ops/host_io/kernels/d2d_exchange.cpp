// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/debug/dprint.h"

constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t receiver_socket_config_addr = get_compile_time_arg_val(1);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_whole_fabric_packets = get_compile_time_arg_val(4);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(5);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(6);

FORCE_INLINE bool socket_wait_for_pages_with_termination(
    const SocketReceiverInterface& socket, uint32_t num_pages, volatile tt_l1_ptr uint32_t* termination_semaphore) {
    constexpr uint32_t termination_value = 1;
    while (!socket_wait_for_pages(socket, num_pages, 1000)) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == termination_value) {
            return false;
        }
    }
    return true;
}

void kernel_main() {
    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(receiver_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);
    set_receiver_socket_page_size(receiver_socket, page_size);
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    uint64_t upstream_bytes_acked_noc_addr = get_noc_addr(
        receiver_socket.d2d.upstream_noc_x,
        receiver_socket.d2d.upstream_noc_y,
        receiver_socket.d2d.upstream_bytes_acked_addr);
    uint64_t downstream_data_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, sender_socket.downstream_fifo_addr);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    DPRINT << "Whole Packet Size: " << whole_packet_size << ENDL();
    DPRINT << "Partial Packet Size: " << partial_packet_size << ENDL();
    DPRINT << "Num Whole Fabric Packets: " << num_whole_fabric_packets << ENDL();
    DPRINT << "Page Size: " << page_size << ENDL();
    while (true) {
        socket_reserve_pages(sender_socket, 1);
        if (!socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
            break;
        }

        auto l1_read_addr = receiver_socket.read_ptr;
        uint64_t dst_addr = downstream_data_addr + sender_socket.write_ptr;

        for (uint32_t i = 0; i < num_whole_fabric_packets; ++i) {
            noc_async_write(l1_read_addr, dst_addr, whole_packet_size);
            l1_read_addr += whole_packet_size;
            dst_addr += whole_packet_size;
        }

        if constexpr (partial_packet_size > 0) {
            noc_async_write(l1_read_addr, dst_addr, partial_packet_size);
        }

        socket_push_pages(sender_socket, 1);
        socket_pop_pages(receiver_socket, 1);

        socket_notify_receiver(sender_socket);
        noc_async_writes_flushed();
        socket_notify_sender(receiver_socket);
    }
}
