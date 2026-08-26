// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "sync.hpp"

static inline void delay(uint64_t loops) {
    for (uint64_t i = 0; i < loops; i++) {
        asm("");
    }
}

#define ARGS(X)                 \
    X(uint32_t, transfer_size)  \
    X(uint32_t, transfer_count) \
    X(uint32_t, barrier_address)

void kernel_main() {
    ARG_INIT(ARGS);

    struct barrier* b = (struct barrier*)barrier_address;

    for (uint32_t i = 0; i < transfer_count; i++) {
        eth_wait_for_bytes(transfer_size);
        barrier_wait(b);
        delay(1000);
        eth_receiver_done();
        barrier_wait(b);
    }
}
