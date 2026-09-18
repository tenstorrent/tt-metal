// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Counterpart to host_socket_sender.cpp. SOCKET_MODE picks only the payload move.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "pcie_noc_utils.h"

constexpr uint32_t kModeHostTransport = 1;  // FIFO is pinned host RAM, pulled over PCIe (H2D leg)
constexpr uint32_t kModeD2D = 2;            // FIFO is local L1, written by the upstream device

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t socket_mode = get_compile_time_arg_val(4);
    // Bytes to cycle through; data_size for a correctness run.
    constexpr uint32_t dst_buffer_size = get_compile_time_arg_val(5);

    static_assert(socket_mode == kModeHostTransport || socket_mode == kModeD2D);
    static_assert(data_size % page_size == 0, "data_size must be a whole number of pages");
    static_assert(dst_buffer_size % page_size == 0, "dst_buffer_size must be a whole number of pages");

    SocketReceiverInterface socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(socket, page_size);

    uint32_t dst = dst_l1_addr;
    uint32_t remaining = data_size;

    if constexpr (socket_mode == kModeHostTransport) {
        const uint32_t pcie_xy_enc = socket.h2d.pcie_xy_enc;
        const uint64_t fifo_base =
            (static_cast<uint64_t>(socket.h2d.data_addr_hi) << 32) | static_cast<uint64_t>(socket.h2d.data_addr_lo);
        while (remaining) {
            socket_wait_for_pages(socket, 1);
            // DEVICE_PULL: read_ptr and fifo_addr are offsets into the host ring.
            noc_read_page_chunked(pcie_xy_enc, fifo_base + socket.read_ptr - socket.fifo_addr, dst, page_size);
            // Must land in L1 before the ring slot is released.
            noc_async_read_barrier();
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            dst += page_size;
            if (dst - dst_l1_addr >= dst_buffer_size) {
                dst = dst_l1_addr;
            }
            remaining -= page_size;
        }
    } else {
        while (remaining) {
            socket_wait_for_pages(socket, 1);
            noc_async_write(socket.read_ptr, get_noc_addr(dst), page_size);
            noc_async_write_barrier();
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            dst += page_size;
            if (dst - dst_l1_addr >= dst_buffer_size) {
                dst = dst_l1_addr;
            }
            remaining -= page_size;
        }
    }

    update_socket_config(socket);
}
