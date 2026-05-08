// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"

#define ARGS(X)              \
    X(uint32_t, buffer_addr) \
    X(uint32_t, transfer_size)

#define RUNTIME_ARGS(X)     \
    X(uint32_t, kernel_id)  \
    X(uint32_t, bank_id)    \
    X(uint32_t, start_addr) \
    X(uint32_t, end_addr)

void kernel_main() {
    ARG_INIT(ARGS);
    ARG_RUNTIME_INIT(RUNTIME_ARGS);

    uint32_t* buf = (uint32_t*)buffer_addr;

    uint32_t remaining = end_addr - start_addr;
    uint32_t curr_addr = start_addr;

    while (remaining) {
        // TODO double buffer?
        for (uint32_t i = 0; i < transfer_size / 4; i++) {
            buf[i] = curr_addr + i * sizeof buf[0] + 1;
        }

        const uint32_t to_send = transfer_size < remaining ? transfer_size : remaining;

        uint64_t noc_addr = get_noc_addr_from_bank_id<true>(bank_id, curr_addr);
        noc_async_write(buffer_addr, noc_addr, to_send);
        noc_async_write_barrier();

        remaining -= to_send;
        curr_addr += to_send;
    }
}
