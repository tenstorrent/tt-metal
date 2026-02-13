// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);  // unused, kept for API compat
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);  // unused
    constexpr uint32_t measurement_buffer_addr = get_compile_time_arg_val(4);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(5);

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);

    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t measurement_start = measurement_buffer_addr;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    // Warmup
    constexpr uint32_t WARMUP_ITERS = 5;
    for (uint32_t w = 0; w < WARMUP_ITERS; w++) {
        socket_reserve_pages(sender_socket, 1);
        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
        socket_barrier(sender_socket);
    }

    // Timed iterations
    for (uint32_t i = 0; i < num_iterations; i++) {
        socket_reserve_pages(sender_socket, 1);
        socket_push_pages(sender_socket, 1);

        uint64_t start_timestamp = get_timestamp();

        socket_notify_receiver(sender_socket);
        socket_barrier(sender_socket);

        uint64_t end_timestamp = get_timestamp();

        *reinterpret_cast<volatile uint64_t*>(measurement_start + i * sizeof(uint64_t)) =
            end_timestamp - start_timestamp;
    }
    socket_barrier(sender_socket);
    noc_async_write_barrier();

    update_socket_config(sender_socket);
}
