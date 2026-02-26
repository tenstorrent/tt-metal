// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "socket_benchmark_defs.h"

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    // arg 1 (local_l1_buffer_addr) reserved for ABI compatibility
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    // arg 3 (measurement_buffer_addr) reserved for ABI compatibility; latency measured on host
    constexpr uint32_t num_iterations = get_compile_time_arg_val(4);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, page_size);

    for (uint32_t i = 0; i < WARMUP_ITERS + num_iterations; i++) {
        socket_wait_for_pages(receiver_socket, 1);
        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);
    }

    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
