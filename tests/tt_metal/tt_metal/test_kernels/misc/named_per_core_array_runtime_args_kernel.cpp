// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for per-core array named runtime args.
// Reads a per-core ArrayArg (weights[NUM_ELEMENTS]) with PER_CORE dispatch,
// sums all elements, writes the sum to L1 at WRITE_ADDRESS.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    // Per-core array named RT arg — each core receives its own array
    uint32_t sum = 0;
    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        sum += rt_args::get<rt_args::my_kernel::weights>(i);
    }

    l1_ptr[0] = sum;
}
