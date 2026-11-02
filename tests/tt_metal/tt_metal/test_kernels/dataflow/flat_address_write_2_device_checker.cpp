// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric unicast write — checker side (runs on chip 1).
//
// 1. Spins on recv_l1_addr[0] until non-zero (the sender first word arrives).
// 2. Verifies all num_words words match base_value+i.
// 3. Writes 1u (pass) or 0xDEAD0000|i (fail at word i) to result_l1_addr.
// 4. Exits; host calls Finish(chip1) which waits for this point.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

// DPRINT is not available on ROCC (dm2, -mcpu=tt-qsr64-rocc); guard by RISC type.
#if defined(COMPILE_FOR_TRISC) || (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM < 2)
#include "debug/dprint.h"
#define HAS_DPRINT 1
#endif

void kernel_main() {
    const uint32_t recv_l1_addr = get_arg(args::recv_l1_addr);
    const uint32_t num_words = get_arg(args::num_words);
    const uint32_t base_value = get_arg(args::base_value);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    volatile uint32_t* recv_uncached = reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + recv_l1_addr);
    volatile uint32_t* result_uncached = reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + result_l1_addr);

#ifdef HAS_DPRINT
    DPRINT << "C: recv_l1=" << HEX() << recv_l1_addr << " num_words=" << num_words << ENDL();
#endif

    while (recv_uncached[0] == 0) {
    }

#ifdef HAS_DPRINT
    DPRINT << "C: got data recv_uncached[0]=" << HEX() << recv_uncached[0] << ENDL();
#endif

    uint32_t pass = 1u;
    for (uint32_t i = 0; i < num_words; i++) {
        if (recv_uncached[i] != base_value + i) {
            pass = 0xDEAD0000u | i;
            break;
        }
    }

    *result_uncached = pass;

#ifdef HAS_DPRINT
    DPRINT << "C: result=" << HEX() << pass << ENDL();
#endif
}
