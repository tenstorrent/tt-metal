// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core H2D sender kernel.
// Each core reads from its local CB (sharded tensor) and sends to a D2H socket via D2D socket.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

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
    constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t downstream_socket_config_addr = get_compile_time_arg_val(3);
    DPRINT << "ct args:\n";
    DPRINT << "termination_semaphore_addr: " << (uint32_t)termination_semaphore_addr << "\n";
    DPRINT << "page_size: " << (uint32_t)page_size << "\n";
    DPRINT << "input_cb_index: " << (uint32_t)input_cb_index << "\n";
    DPRINT << "downstream_socket_config_addr: " << (uint32_t)downstream_socket_config_addr << "\n";

    SocketSenderInterface sender_socket = create_sender_socket_interface(downstream_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);
    DPRINT << "after sender socket init\n";

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    DPRINT << "before main loop\n";

    while (true) {
        // Wait for data in CB with termination checks
        DPRINT << "BEFORE if statement to wait for pages in current upstream socket with termination checks\n";
        if (!cb_wait_for_pages_with_termination(input_cb_index, 1, termination_semaphore)) {
            DPRINT << "Termination semaphore set. Exiting main loop.\n";
            break;
        }
        DPRINT << "Data available in CB\n";

        uint32_t read_addr = get_read_ptr(input_cb_index);

        // Reserve space in downstream socket
        DPRINT << "Reserving pages in downstream socket\n";
        socket_reserve_pages(sender_socket, 1);
        DPRINT << "Pages reserved in downstream socket\n";

        // Get downstream encoding
        sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);

        // Write to downstream socket
        noc_async_write(
            read_addr,
            get_noc_addr(
                downstream_enc.d2d.downstream_noc_x,
                downstream_enc.d2d.downstream_noc_y,
                sender_socket.write_ptr + sender_socket.downstream_fifo_addr),
            page_size);
        DPRINT << "after noc_async_write\n";

        // Pop from CB
        cb_pop_front(input_cb_index, 1);
        DPRINT << "after cb_pop_front\n";

        // Push to downstream and notify
        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
        noc_async_writes_flushed();
        DPRINT << "after socket_push_pages and notify\n";

        invalidate_l1_cache();
        DPRINT << "after invalidate_l1_cache\n";
    }

    socket_barrier(sender_socket);
    DPRINT << "after socket_barrier\n";
    noc_async_write_barrier();
    noc_async_read_barrier();
    DPRINT << "after noc_async_barriers\n";
}
