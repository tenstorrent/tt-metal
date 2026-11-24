// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads a word of another core's L1 over and over and watches that core's L1 ECC counts for DBEs in the word of a
// logged SBE: the counts show only an SBE, and the very next read clears them through the handler. Stops after the
// given number of those in a row, or when the host sets the stop flag.
// arg 0/1: target NoC x/y, arg 2: target word, arg 3: cycles to wait after each read for the handler, arg 4: target
// ECC_STATUS address, arg 5: SBE-then-DBEs to see in a row, arg 6: local scratch (32 B), arg 7: local stop flag,
// arg 8: local result (longest run, reads).

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "hostdev/l1_ecc_halt_constants.h"

void kernel_main() {
    uint32_t x = get_arg_val<uint32_t>(0);
    uint32_t y = get_arg_val<uint32_t>(1);
    uint32_t word = get_arg_val<uint32_t>(2);
    uint32_t handler_wait = get_arg_val<uint32_t>(3);
    uint32_t status_addr = get_arg_val<uint32_t>(4);
    uint32_t wanted = get_arg_val<uint32_t>(5);
    uint32_t scratch = get_arg_val<uint32_t>(6);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(7));
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(8));

    Noc noc;
    UnicastEndpoint target;
    CoreLocalMem<uint32_t> data(scratch);
    // A NoC read keeps the source's offset within 16 B.
    const uint32_t status_dst = scratch + 16 + status_addr % 16;
    CoreLocalMem<uint32_t> status(status_dst);
    volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_dst);

    // The counts cover every error since the handler last cleared them, so only an SBE counted after a clear is from
    // these reads.
    enum { WAIT_CLEAR, WAIT_SBE, SBE_LOGGED } state = WAIT_CLEAR;
    uint32_t run = 0, longest = 0, reads = 0;
    while (longest < wanted && *stop == 0) {
        noc.async_read(target, data, 16, {.noc_x = x, .noc_y = y, .addr = word}, {});
        noc.async_read_barrier();
        riscv_wait(handler_wait);
        noc.async_read(target, status, 4, {.noc_x = x, .noc_y = y, .addr = status_addr}, {});
        noc.async_read_barrier();
        reads++;
        const uint32_t counts = *counts_ptr;
        const bool sbe_only =
            (counts & (ECC_DBE_COUNT_MASK | ECC_DBE_OVERFLOW_MASK)) == 0 && (counts & ECC_SBE_COUNT_MASK) != 0;
        if (state == SBE_LOGGED) {
            if (counts == 0) {
                run++;
                longest = run > longest ? run : longest;
                state = WAIT_SBE;
            } else {
                run = 0;
                state = WAIT_CLEAR;
            }
        } else if (state == WAIT_CLEAR && counts == 0) {
            state = WAIT_SBE;
        } else if (state == WAIT_SBE && sbe_only) {
            state = SBE_LOGGED;
        }
    }
    result[1] = reads;
    result[0] = longest;
}
