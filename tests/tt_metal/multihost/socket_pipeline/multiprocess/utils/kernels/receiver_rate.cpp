// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Rate-mode receiver kernel for pipeline throughput benchmarking.
// Drains data from recv_socket for num_iterations, acking upstream periodically.
// num_iterations is a runtime arg so that warmup and timed runs share the same compiled binary.
// When enable_correctness_check is set, validates that each received page contains the
// expected pattern (sequential uint32_t values: 0, 1, 2, ..., page_size/4 - 1).

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(1);
// Send cumulative ack upstream every N iterations (e.g. fifo_size_in_pages/2 for half-buffer acks).
constexpr uint32_t notify_sender_every_n_iterations = get_compile_time_arg_val(2);
constexpr uint32_t enable_correctness_check = get_compile_time_arg_val(3);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t recv_socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // num_iterations is a runtime arg (after fabric connection args) so compilation is shared
    uint32_t num_iterations = get_arg_val<uint32_t>(rt_args_idx++);

    // Single packet header for upstream acks
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));

    upstream_fabric_connection.open();

    SocketReceiverInterface recv_socket = create_receiver_socket_interface(recv_socket_config_addr);
    set_receiver_socket_page_size(recv_socket, socket_block_size);

    fabric_set_unicast_route(upstream_socket_packet_header_addr, recv_socket);

    uint64_t upstream_bytes_acked_noc_addr = get_noc_addr(
        recv_socket.d2d.upstream_noc_x, recv_socket.d2d.upstream_noc_y, recv_socket.d2d.upstream_bytes_acked_addr);

    // Drain data from the pipeline
    for (uint32_t i = 0; i < num_iterations; ++i) {
        socket_wait_for_pages(recv_socket, 1);

        if constexpr (enable_correctness_check) {
            // Validate received data matches expected pattern: sequential uint32_t values
            volatile tt_l1_ptr uint32_t* data_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_socket.read_ptr);
            constexpr uint32_t num_elems = socket_block_size / sizeof(uint32_t);
            for (uint32_t j = 0; j < num_elems; j++) {
                if (data_ptr[j] != j) {
                    // Hang on mismatch to signal correctness failure
                    while (true);
                }
            }
        }

        socket_pop_pages(recv_socket, 1);

        // Ack upstream periodically to free sender FIFO space
        if (notify_sender_every_n_iterations != 0 && ((i + 1) % notify_sender_every_n_iterations) == 0) {
            fabric_socket_notify_sender_stateful(
                recv_socket,
                upstream_fabric_connection,
                upstream_socket_packet_header_addr,
                upstream_bytes_acked_noc_addr);
        }
    }

    update_socket_config(recv_socket);
    upstream_fabric_connection.close();
}
