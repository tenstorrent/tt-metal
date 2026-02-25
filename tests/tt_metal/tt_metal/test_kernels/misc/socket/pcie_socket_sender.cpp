// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "socket_benchmark_defs.h"

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t measurement_buffer_addr = get_compile_time_arg_val(4);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(5);

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);

    uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;
    uint32_t data_addr_hi = sender_socket.d2h.data_addr_hi;

    set_sender_socket_page_size(sender_socket, page_size);

    uint64_t pcie_base_addr = (static_cast<uint64_t>(data_addr_hi) << 32) | sender_socket.downstream_fifo_addr;

    uint64_t start_timestamp = get_timestamp();

    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t data_addr = local_l1_buffer_addr;
        uint32_t outstanding_data_size = data_size;
        while (outstanding_data_size) {
            socket_reserve_pages(sender_socket, 1);
            // write_ptr is a relative offset into the downstream FIFO
            noc_write_page_chunked(pcie_xy_enc, data_addr, pcie_base_addr + sender_socket.write_ptr, page_size);
            data_addr += page_size;
            outstanding_data_size -= page_size;
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
        }
    }
    socket_barrier(sender_socket);

    uint64_t end_timestamp = get_timestamp();
    *reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr) = end_timestamp - start_timestamp;

    update_socket_config(sender_socket);
}
