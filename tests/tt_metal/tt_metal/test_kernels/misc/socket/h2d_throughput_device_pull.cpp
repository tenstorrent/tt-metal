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
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    // arg 4 (measurement_buffer_addr) reserved for ABI compatibility; throughput measured on host
    constexpr uint32_t num_iterations = get_compile_time_arg_val(5);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, page_size);

    // Set up PCIe read addresses for host pinned RAM
    uint64_t pcie_data_addr = (static_cast<uint64_t>(receiver_socket.h2d.data_addr_hi) << 32) |
                              (static_cast<uint64_t>(receiver_socket.h2d.data_addr_lo));
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t outstanding_data_size = data_size;
        while (outstanding_data_size) {
            socket_wait_for_pages(receiver_socket, 1);
            noc_read_page_chunked(
                pcie_xy_enc,
                pcie_data_addr + receiver_socket.read_ptr - receiver_socket.fifo_addr,
                receiver_socket.read_ptr,
                page_size);
            outstanding_data_size -= page_size;
            socket_pop_pages(receiver_socket, 1);
            noc_async_read_barrier();
            socket_notify_sender(receiver_socket);
        }
    }

    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
