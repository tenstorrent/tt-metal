// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "sync.hpp"
#include "timestamp.hpp"

#define ARGS(X)                     \
    X(uint32_t, num_bytes_per_send) \
    X(uint32_t, transfer_size)      \
    X(uint32_t, transfer_count)     \
    X(uint32_t, send_delta_addr)    \
    X(uint32_t, send_l1_address)    \
    X(uint32_t, recv_l1_address)    \
    X(uint32_t, barrier_address)

void kernel_main() {
    ARG_INIT(ARGS);

    struct barrier* b = (struct barrier*)barrier_address;
    barrier_init(b, 2);

    uint64_t start = timestamp();
    for (uint32_t i = 0; i < transfer_count; i++) {
        eth_send_bytes(send_l1_address, recv_l1_address, transfer_size, num_bytes_per_send, num_bytes_per_send >> 4);
        barrier_wait(b);
        eth_wait_for_receiver_done();
        barrier_wait(b);
    }
    uint64_t delta = timestamp() - start;

    *(uint64_t*)send_delta_addr = delta;
}
