// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "timestamp.hpp"

#define ARGS(X)           \
    X(num_bytes_per_send) \
    X(transfer_size)      \
    X(transfer_count)     \
    X(send_delta_addr)    \
    X(send_l1_address)    \
    X(recv_l1_address)

void kernel_main() {
    ARG_INIT(ARGS);

    uint64_t start = timestamp();
    for (uint32_t i = 0; i < transfer_count; i++) {
        eth_send_bytes(send_l1_address, recv_l1_address, transfer_size, num_bytes_per_send, num_bytes_per_send >> 4);
        eth_wait_for_receiver_done();
    }
    uint64_t delta = timestamp() - start;

    *(uint64_t*)send_delta_addr = delta;
}
