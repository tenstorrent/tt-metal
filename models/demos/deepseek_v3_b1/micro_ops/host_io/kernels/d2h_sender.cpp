// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/debug/dprint.h"

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

FORCE_INLINE bool cb_wait_for_pages_with_termination(
    uint32_t cb_index, uint32_t num_pages, volatile tt_l1_ptr uint32_t* termination_semaphore) {
    constexpr uint32_t termination_value = 1;
    while (!cb_pages_available_at_front(cb_index, num_pages)) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == termination_value) {
            return false;
        }
    }
    return true;
}

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t send_socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr bool loopback_mode = get_compile_time_arg_val(3);
    constexpr uint32_t upstream_interface_index = get_compile_time_arg_val(4);

    SocketSenderInterface sender_socket = create_sender_socket_interface(send_socket_config_addr);
    SocketReceiverInterface receiver_socket = {};

    if constexpr (!loopback_mode) {
        receiver_socket = create_receiver_socket_interface(upstream_interface_index);
        set_receiver_socket_page_size(receiver_socket, page_size);
    }
    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    while (true) {
        // Wait for space in D2H socket
        socket_reserve_pages(sender_socket, 1);
        if constexpr (loopback_mode) {
            // Wait for data in CB with termination checks
            if (!cb_wait_for_pages_with_termination(upstream_interface_index, 1, termination_semaphore)) {
                break;
            }
            uint32_t read_addr = get_read_ptr(upstream_interface_index);
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                NOC_INDEX,
                read_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                page_size,
                1);
            noc_async_writes_flushed();
            cb_pop_front(upstream_interface_index, 1);
        } else {
            // Wait for pages in receiver socket with timeout and termination checks
            if (!socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
                break;
            }
            uint32_t read_addr = receiver_socket.read_ptr;
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                NOC_INDEX,
                read_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                page_size,
                1);
            socket_pop_pages(receiver_socket, 1);
            noc_async_writes_flushed();
            socket_notify_sender(receiver_socket);
        }

        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
        invalidate_l1_cache();
    }

    update_socket_config(sender_socket);
    socket_barrier(sender_socket);

    noc_async_write_barrier();
    noc_async_read_barrier();
    DPRINT << "End D2H Main Loop" << ENDL();
}
