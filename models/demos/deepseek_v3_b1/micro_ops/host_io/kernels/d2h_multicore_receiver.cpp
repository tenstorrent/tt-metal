// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core D2H receiver kernel.
// Single core reads from 8 upstream sockets in round-robin fashion and sends to D2H socket.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "pcie_noc_utils.h"

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
    constexpr uint32_t send_socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t upstream_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_upstream_sockets = get_compile_time_arg_val(4);
    constexpr uint32_t upstream_socket_0_config_addr = get_compile_time_arg_val(5);
    constexpr uint32_t upstream_socket_1_config_addr = get_compile_time_arg_val(6);
    constexpr uint32_t upstream_socket_2_config_addr = get_compile_time_arg_val(7);
    constexpr uint32_t upstream_socket_3_config_addr = get_compile_time_arg_val(8);
    constexpr uint32_t upstream_socket_4_config_addr = get_compile_time_arg_val(9);
    constexpr uint32_t upstream_socket_5_config_addr = get_compile_time_arg_val(10);
    constexpr uint32_t upstream_socket_6_config_addr = get_compile_time_arg_val(11);
    constexpr uint32_t upstream_socket_7_config_addr = get_compile_time_arg_val(12);

    SocketSenderInterface sender_socket = create_sender_socket_interface(send_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    // Create receiver socket interfaces for all upstream sockets
    SocketReceiverInterface receiver_sockets[8];
    receiver_sockets[0] = create_receiver_socket_interface(upstream_socket_0_config_addr);
    receiver_sockets[1] = create_receiver_socket_interface(upstream_socket_1_config_addr);
    receiver_sockets[2] = create_receiver_socket_interface(upstream_socket_2_config_addr);
    receiver_sockets[3] = create_receiver_socket_interface(upstream_socket_3_config_addr);
    receiver_sockets[4] = create_receiver_socket_interface(upstream_socket_4_config_addr);
    receiver_sockets[5] = create_receiver_socket_interface(upstream_socket_5_config_addr);
    receiver_sockets[6] = create_receiver_socket_interface(upstream_socket_6_config_addr);
    receiver_sockets[7] = create_receiver_socket_interface(upstream_socket_7_config_addr);

    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        set_receiver_socket_page_size(receiver_sockets[i], upstream_page_size);
    }

    uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    uint32_t current_socket_idx = 0;
    uint32_t bytes_accumulated = 0;
    bool data_pushed = false;

    socket_reserve_pages(sender_socket, 1);

    // Collect data from all upstream sockets into a single 14KB page
    while (true) {
        // Wait for pages in current upstream socket with termination checks
        if (!socket_wait_for_pages_with_termination(receiver_sockets[current_socket_idx], 1, termination_semaphore)) {
            break;
        }

        uint32_t read_addr = receiver_sockets[current_socket_idx].read_ptr;
        uint32_t skt_offset = bytes_accumulated;

        // Write to D2H socket at the appropriate offset
        noc_async_wide_write_any_len_with_state(
            NOC_INDEX,
            read_addr,
            pcie_xy_enc,
            ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                sender_socket.write_ptr + skt_offset,
            upstream_page_size);

        // Pop from current upstream socket
        socket_pop_pages(receiver_sockets[current_socket_idx], 1);
        noc_async_writes_flushed();
        socket_notify_sender(receiver_sockets[current_socket_idx]);

        invalidate_l1_cache();

        // Update accumulation
        bytes_accumulated += upstream_page_size;
        current_socket_idx = (current_socket_idx + 1) % num_upstream_sockets;

        // Push when we've accumulated a full D2H page
        if (bytes_accumulated >= page_size) {
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
            data_pushed = true;
            bytes_accumulated = 0;

            // Reserve next page if continuing
            socket_reserve_pages(sender_socket, 1);
        }
    }

    // Push any remaining data if we broke out of loop before filling a complete page
    if (bytes_accumulated > 0 && !data_pushed) {
        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
    }

    invalidate_l1_cache();

    update_socket_config(sender_socket);
    socket_barrier(sender_socket);

    noc_async_write_barrier();
    noc_async_read_barrier();
}
