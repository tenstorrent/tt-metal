// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for array named runtime args.
// Reads a scalar Arg (prefix) and an ArrayArg (data[NUM_ELEMENTS]),
// sums prefix + all data elements, writes the sum to L1 at WRITE_ADDRESS.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    // Scalar named RT arg
    uint32_t prefix = rt_args::get<rt_args::my_kernel::prefix>();

    // Array named RT arg — read NUM_ELEMENTS values and sum them
    uint32_t sum = prefix;
    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        sum += rt_args::get<rt_args::my_kernel::data>(i);
    }

    l1_ptr[0] = sum;
}
