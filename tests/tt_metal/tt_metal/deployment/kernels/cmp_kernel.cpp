// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"

#include "api/debug/dprint.h"

#define ARGS(X)                     \
    X(uint32_t, buffer_addr0)       \
    X(uint32_t, buffer_addr1)       \
    X(uint32_t, transfer_size)      \
    X(uint32_t, error_counter_addr) \
    X(uint32_t, first_error_addr_p) \
    X(uint32_t, last_error_addr_p)

#define RUNTIME_ARGS(X)     \
    X(uint32_t, kernel_id)  \
    X(uint32_t, bank_id0)   \
    X(uint32_t, bank_id1)   \
    X(uint32_t, start_addr) \
    X(uint32_t, end_addr)

void kernel_main() {
    ARG_INIT(ARGS);
    ARG_RUNTIME_INIT(RUNTIME_ARGS);

    uint32_t* buff0 = (uint32_t*)buffer_addr0;
    uint32_t* buff1 = (uint32_t*)buffer_addr1;
    uint32_t* errorcnt = (uint32_t*)error_counter_addr;
    uint32_t* first_error_addr = (uint32_t*)first_error_addr_p;
    uint32_t* last_error_addr = (uint32_t*)last_error_addr_p;

    *errorcnt = 0;
    *first_error_addr = -1;
    *last_error_addr = 0;

    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        buff0[i] = i;
        buff1[i] = i + transfer_size;
    }

    uint32_t remaining = end_addr - start_addr;
    uint32_t curr_addr = start_addr;

    while (remaining) {
        const uint32_t to_read = transfer_size < remaining ? transfer_size : remaining;

        uint64_t noc_addr0 = get_noc_addr_from_bank_id<true>(bank_id0, curr_addr);
        uint64_t noc_addr1 = get_noc_addr_from_bank_id<true>(bank_id1, curr_addr);

        noc_async_read(noc_addr0, buffer_addr0, to_read);
        noc_async_read(noc_addr1, buffer_addr1, to_read);
        noc_async_read_barrier();

        for (uint32_t i = 0; i < to_read / 4; i++) {
            if (buff0[i] != buff1[i]) {
                uint32_t addr = i * sizeof buff0[0] + curr_addr;

                if (addr < *first_error_addr) {
                    *first_error_addr = addr;
                }
                if (addr > *last_error_addr) {
                    *last_error_addr = addr;
                }

                (*errorcnt)++;
            }
        }

        remaining -= to_read;
        curr_addr += to_read;
    }
}
