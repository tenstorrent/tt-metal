// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Burst latency responder: consume each burst page by page, then send one ack
// page back over the reverse socket. The initiator owns the clock. The burst
// list and iteration count must match the initiator's.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "pcie_noc_utils.h"

constexpr uint32_t kModeHostTransport = 1;
constexpr uint32_t kModeD2D = 2;
constexpr uint32_t kModeHostDma = 3;  // rx FIFO is local L1 filled by the host's DMA engine; tx is the D2H leg

void kernel_main() {
    constexpr uint32_t recv_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t send_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t ack_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t l1_buffer_addr = get_compile_time_arg_val(4);
    constexpr uint32_t ack_buffer_addr = get_compile_time_arg_val(5);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(6);
    constexpr uint32_t num_bursts = get_compile_time_arg_val(7);
    constexpr uint32_t socket_mode = get_compile_time_arg_val(8);
    static_assert(socket_mode == kModeHostTransport || socket_mode == kModeD2D || socket_mode == kModeHostDma);

    SocketReceiverInterface rx = create_receiver_socket_interface(recv_config_addr);
    SocketSenderInterface tx = create_sender_socket_interface(send_config_addr);
    set_receiver_socket_page_size(rx, page_size);
    set_sender_socket_page_size(tx, ack_page_size);

    const uint32_t rx_pcie_enc = rx.h2d.pcie_xy_enc;
    const uint64_t rx_base =
        (static_cast<uint64_t>(rx.h2d.data_addr_hi) << 32) | static_cast<uint64_t>(rx.h2d.data_addr_lo);
    const uint32_t tx_pcie_enc = tx.d2h.pcie_xy_enc;
    const uint64_t tx_base = (static_cast<uint64_t>(tx.d2h.data_addr_hi) << 32) | tx.downstream_fifo_addr;

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

    auto send_ack = [&]() {
        socket_reserve_pages(tx, 1);
        if constexpr (socket_mode == kModeHostTransport || socket_mode == kModeHostDma) {
            noc_write_page_chunked(tx_pcie_enc, ack_buffer_addr, tx_base + tx.write_ptr, ack_page_size);
        } else {
            sender_downstream_encoding enc = get_downstream_encoding(tx, 0);
            noc_async_write(
                ack_buffer_addr,
                get_noc_addr(
                    enc.d2d.downstream_noc_x, enc.d2d.downstream_noc_y, tx.downstream_fifo_addr + tx.write_ptr),
                ack_page_size);
        }
        noc_async_write_barrier();
        socket_push_pages(tx, 1);
        socket_notify_receiver(tx);
    };

    for (uint32_t b = 0; b < num_bursts; b++) {
        const uint32_t burst = get_arg_val<uint32_t>(b);
        // Initiator does WARMUP_ITERS extra laps per burst.
        const uint32_t laps = num_iterations + WARMUP_ITERS;
        for (uint32_t lap = 0; lap < laps; lap++) {
            for (uint32_t i = 0; i < burst; i++) {
                recv_one();
            }
            send_ack();
        }
    }

    socket_barrier(tx);
    update_socket_config(rx);
    update_socket_config(tx);
}
