// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Loads a word of this core's L1 over and over and watches the core's L1 ECC counts for DBEs in the word of a logged
// SBE: the counts show only an SBE, and the very next load clears them through BRISC's handler. Stops after the given
// number of those in a row, or when the host sets the stop flag.
// arg 0: word, arg 1: cycles to wait after each load for the handler, arg 2: ECC_STATUS address, arg 3: SBE-then-DBEs
// to see in a row, arg 4: stop flag, arg 5: result (longest run, loads).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/l1_ecc_halt_constants.h"

void kernel_main() {
    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    uint32_t handler_wait = get_arg_val<uint32_t>(1);
    volatile tt_reg_ptr uint32_t* status = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    uint32_t wanted = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));

    bool sbe_logged = false;
    uint32_t run = 0, longest = 0, loads = 0;
    while (longest < wanted && *stop == 0) {
        // The fence empties the L0 data cache, so the load reads the word from L1.
        asm volatile("fence" ::: "memory");
        (void)*word;
        riscv_wait(handler_wait);
        loads++;
        const uint32_t counts = *status;
        if (sbe_logged) {
            run = counts == 0 ? run + 1 : 0;
            longest = run > longest ? run : longest;
        }
        sbe_logged =
            (counts & (ECC_DBE_COUNT_MASK | ECC_DBE_OVERFLOW_MASK)) == 0 && (counts & ECC_SBE_COUNT_MASK) != 0;
    }
    result[1] = loads;
    result[0] = longest;
}
