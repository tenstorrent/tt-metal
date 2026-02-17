// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t measurement_buffer_addr = get_compile_time_arg_val(3);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(4);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, page_size);

    uint64_t dst_noc_addr = get_noc_addr(local_l1_buffer_addr);

    // Warmup
    constexpr uint32_t WARMUP_ITERS = 5;
    for (uint32_t w = 0; w < WARMUP_ITERS; w++) {
        socket_wait_for_pages(receiver_socket, 1);
        noc_async_write(receiver_socket.read_ptr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);
    }

    // Timed iterations
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint64_t start_timestamp = get_timestamp();

        socket_wait_for_pages(receiver_socket, 1);
        noc_async_write(receiver_socket.read_ptr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);

        uint64_t end_timestamp = get_timestamp();

        *reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr + i * sizeof(uint64_t)) =
            end_timestamp - start_timestamp;
    }

    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
