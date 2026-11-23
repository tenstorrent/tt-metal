// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric unicast write — checker side.
// Spins until the first received word is non-zero, verifies all num_words words match
// base_value+i, then writes 1u (pass) or 0xDEAD0000|i (fail) to result_l1_addr.
// Uses the uncached TL1 window for all reads/writes so the host can read the result
// via cluster.read_core without a cache flush.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t recv_l1_addr = get_arg(args::recv_l1_addr);
    const uint32_t num_words = get_arg(args::num_words);
    const uint32_t base_value = get_arg(args::base_value);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    volatile uint32_t* recv_uncached = reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + recv_l1_addr);
    volatile uint32_t* result_uncached = reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + result_l1_addr);

    while (recv_uncached[0] == 0) {
    }

    uint32_t pass = 1u;
    for (uint32_t i = 0; i < num_words; i++) {
        if (recv_uncached[i] != base_value + i) {
            pass = 0xDEAD0000u | i;
            break;
        }
    }

    *result_uncached = pass;
}
