// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transport-agnostic socket sender. SOCKET_MODE picks only the payload move; the
// flow-control calls are identical either way.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "pcie_noc_utils.h"

constexpr uint32_t kModeHostTransport = 1;  // downstream FIFO is pinned host RAM (D2H leg)
constexpr uint32_t kModeD2D = 2;            // downstream FIFO is another device's L1

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t src_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t socket_mode = get_compile_time_arg_val(4);
    // Bytes to cycle through: data_size to make every page distinct, else one page.
    constexpr uint32_t src_buffer_size = get_compile_time_arg_val(5);

    static_assert(socket_mode == kModeHostTransport || socket_mode == kModeD2D);
    static_assert(data_size % page_size == 0, "data_size must be a whole number of pages");
    static_assert(src_buffer_size % page_size == 0, "src_buffer_size must be a whole number of pages");

    SocketSenderInterface socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(socket, page_size);

    uint32_t src = src_l1_addr;
    uint32_t remaining = data_size;

    if constexpr (socket_mode == kModeHostTransport) {
        const uint32_t pcie_xy_enc = socket.d2h.pcie_xy_enc;
        const uint64_t fifo_base = (static_cast<uint64_t>(socket.d2h.data_addr_hi) << 32) | socket.downstream_fifo_addr;
        while (remaining) {
            socket_reserve_pages(socket, 1);
            // Chunked: a single NOC write past the burst size is silently dropped.
            noc_write_page_chunked(pcie_xy_enc, src, fifo_base + socket.write_ptr, page_size);
            // Must retire before bytes_sent advertises it.
            noc_async_write_barrier();
            socket_push_pages(socket, 1);
            socket_notify_receiver(socket);
            src += page_size;
            if (src - src_l1_addr >= src_buffer_size) {
                src = src_l1_addr;
            }
            remaining -= page_size;
        }
    } else {
        while (remaining) {
            socket_reserve_pages(socket, 1);
            for (uint32_t i = 0; i < socket.num_downstreams; i++) {
                sender_downstream_encoding enc = get_downstream_encoding(socket, i);
                noc_async_write(
                    src,
                    get_noc_addr(
                        enc.d2d.downstream_noc_x,
                        enc.d2d.downstream_noc_y,
                        socket.downstream_fifo_addr + socket.write_ptr),
                    page_size);
            }
            noc_async_write_barrier();
            socket_push_pages(socket, 1);
            socket_notify_receiver(socket);
            src += page_size;
            if (src - src_l1_addr >= src_buffer_size) {
                src = src_l1_addr;
            }
            remaining -= page_size;
        }
    }

    socket_barrier(socket);
    update_socket_config(socket);
}
