// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "timestamp.hpp"

#define ARGS(X)                     \
    X(uint32_t, channel0)           \
    X(uint32_t, channel1)           \
    X(uint32_t, iter_l1_address)    \
    X(uint32_t, num_bytes_per_send) \
    X(uint32_t, transfer_size)      \
    X(uint32_t, transfer_count)     \
    X(uint32_t, send_delta_addr)    \
    X(uint32_t, send_l1_address)    \
    X(uint32_t, recv_l1_address)

void kernel_main() {
    ARG_INIT(ARGS);
    volatile uint32_t* iter = (volatile uint32_t*)iter_l1_address;
    volatile uint32_t* sendbuf = (volatile uint32_t*)send_l1_address;

    *iter = 0;

    uint64_t start = timestamp();
    for (uint32_t i = 0; i < transfer_count; i++) {
        eth_send_bytes_over_channel(
            send_l1_address, recv_l1_address, transfer_size, channel0, num_bytes_per_send, num_bytes_per_send >> 4);

        eth_wait_for_bytes_on_channel(transfer_size, channel1);

        eth_receiver_channel_done(channel1);

        eth_wait_for_receiver_channel_done(channel0);

        *iter = i;
    }
    uint64_t delta = timestamp() - start;

    *(uint64_t*)send_delta_addr = delta;
    *iter = transfer_count;
}
