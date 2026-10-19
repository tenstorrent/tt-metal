// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Burst latency initiator; peer runs host_socket_burst_receiver.cpp.
//
// For each burst size B (runtime args), repeatedly: stamp, send B pages back to
// back (each one reserved, written, pushed and notified on its own, exactly as a
// streaming sender does), then wait for two credits and stamp after each:
//
//   local   bytes_acked covers the last page: the local host relay has released
//           the ring slot, i.e. the NIC has finished reading it. This is the credit
//           a sender kernel actually blocks on.
//   remote  the receiver kernel, having consumed the last page, has sent one ack
//           page back over the reverse socket and it has arrived here. This is the
//           end-to-end "credit for the last message".
//
// Both are timed on this core's own clock. The remote wait also drains the
// pipeline, so every burst starts from an idle socket.

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
    constexpr uint32_t ack_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t l1_buffer_addr = get_compile_time_arg_val(4);
    constexpr uint32_t ack_buffer_addr = get_compile_time_arg_val(5);
    constexpr uint32_t measurement_addr = get_compile_time_arg_val(6);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(7);
    constexpr uint32_t num_bursts = get_compile_time_arg_val(8);
    constexpr uint32_t socket_mode = get_compile_time_arg_val(9);
    static_assert(socket_mode == kModeHostTransport || socket_mode == kModeD2D);

    SocketSenderInterface tx = create_sender_socket_interface(send_config_addr);
    SocketReceiverInterface rx = create_receiver_socket_interface(recv_config_addr);
    set_sender_socket_page_size(tx, page_size);
    set_receiver_socket_page_size(rx, ack_page_size);

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
        // Must retire before bytes_sent advertises it.
        noc_async_write_barrier();
        socket_push_pages(tx, 1);
        socket_notify_receiver(tx);
    };

    auto recv_ack = [&]() {
        socket_wait_for_pages(rx, 1);
        if constexpr (socket_mode == kModeHostTransport) {
            noc_read_page_chunked(rx_pcie_enc, rx_base + rx.read_ptr - rx.fifo_addr, ack_buffer_addr, ack_page_size);
            noc_async_read_barrier();
        } else {
            noc_async_write(rx.read_ptr, get_noc_addr(ack_buffer_addr), ack_page_size);
            noc_async_write_barrier();
        }
        socket_pop_pages(rx, 1);
        socket_notify_sender(rx);
    };

    // Layout: [burst_index][iteration][2] of uint64 cycles: {to_local, to_remote}.
    volatile tt_l1_ptr uint64_t* out = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(measurement_addr);

    for (uint32_t b = 0; b < num_bursts; b++) {
        const uint32_t burst = get_arg_val<uint32_t>(b);

        // Untimed warmup: cold state, and the first laps over a fresh ring differ.
        for (uint32_t w = 0; w < WARMUP_ITERS; w++) {
            for (uint32_t i = 0; i < burst; i++) {
                send_one();
            }
            socket_barrier(tx);
            recv_ack();
        }

        for (uint32_t it = 0; it < num_iterations; it++) {
            const uint64_t start = get_timestamp();
            for (uint32_t i = 0; i < burst; i++) {
                send_one();
            }
            // bytes_acked == bytes_sent: the last page's slot has been released.
            socket_barrier(tx);
            const uint64_t local = get_timestamp();
            recv_ack();
            const uint64_t remote = get_timestamp();
            out[(b * num_iterations + it) * 2 + 0] = local - start;
            out[(b * num_iterations + it) * 2 + 1] = remote - start;
        }
    }

    noc_async_write_barrier();
    update_socket_config(tx);
    update_socket_config(rx);
}
