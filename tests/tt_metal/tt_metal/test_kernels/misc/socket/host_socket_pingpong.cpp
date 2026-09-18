// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Round-trip latency initiator; peer runs host_socket_echo.cpp. Timed on this
// core's own clock, so no cross-host clock sync is needed. RTT/2 is the one-way
// estimate, assuming the directions are symmetric.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "pcie_noc_utils.h"

constexpr uint32_t kModeHostTransport = 1;
constexpr uint32_t kModeD2D = 2;

void kernel_main() {
    constexpr uint32_t send_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t recv_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t l1_buffer_addr = get_compile_time_arg_val(3);
    constexpr uint32_t measurement_addr = get_compile_time_arg_val(4);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(5);
    constexpr uint32_t socket_mode = get_compile_time_arg_val(6);
    static_assert(socket_mode == kModeHostTransport || socket_mode == kModeD2D);

    SocketSenderInterface tx = create_sender_socket_interface(send_config_addr);
    SocketReceiverInterface rx = create_receiver_socket_interface(recv_config_addr);
    set_sender_socket_page_size(tx, page_size);
    set_receiver_socket_page_size(rx, page_size);

    const uint32_t tx_pcie_enc = tx.d2h.pcie_xy_enc;
    const uint64_t tx_base = (static_cast<uint64_t>(tx.d2h.data_addr_hi) << 32) | tx.downstream_fifo_addr;
    const uint32_t rx_pcie_enc = rx.h2d.pcie_xy_enc;
    const uint64_t rx_base =
        (static_cast<uint64_t>(rx.h2d.data_addr_hi) << 32) | static_cast<uint64_t>(rx.h2d.data_addr_lo);

    auto send_one = [&]() {
        socket_reserve_pages(tx, 1);
        if constexpr (socket_mode == kModeHostTransport) {
            noc_write_page_chunked(tx_pcie_enc, l1_buffer_addr, tx_base + tx.write_ptr, page_size);
        } else {
            sender_downstream_encoding enc = get_downstream_encoding(tx, 0);
            noc_async_write(
                l1_buffer_addr,
                get_noc_addr(
                    enc.d2d.downstream_noc_x, enc.d2d.downstream_noc_y, tx.downstream_fifo_addr + tx.write_ptr),
                page_size);
        }
        noc_async_write_barrier();
        socket_push_pages(tx, 1);
        socket_notify_receiver(tx);
    };

    auto recv_one = [&]() {
        socket_wait_for_pages(rx, 1);
        if constexpr (socket_mode == kModeHostTransport) {
            noc_read_page_chunked(rx_pcie_enc, rx_base + rx.read_ptr - rx.fifo_addr, l1_buffer_addr, page_size);
            noc_async_read_barrier();
        } else {
            noc_async_write(rx.read_ptr, get_noc_addr(l1_buffer_addr), page_size);
            noc_async_write_barrier();
        }
        socket_pop_pages(rx, 1);
        socket_notify_sender(rx);
    };

    // Untimed warmup: cold state.
    for (uint32_t w = 0; w < WARMUP_ITERS; w++) {
        send_one();
        recv_one();
    }

    for (uint32_t i = 0; i < num_iterations; i++) {
        const uint64_t start = get_timestamp();
        send_one();
        recv_one();
        const uint64_t end = get_timestamp();
        *reinterpret_cast<volatile uint64_t*>(measurement_addr + i * sizeof(uint64_t)) = end - start;
    }

    noc_async_write_barrier();
    update_socket_config(tx);
    update_socket_config(rx);
}
