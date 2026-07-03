// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for mixed positional + named runtime args.
// Reads positional per-core RT arg, positional common RT arg,
// named per-core RT arg, and named common RT arg — writes all to L1
// at WRITE_ADDRESS so the host can verify the index mapping is correct.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    // Positional per-core RT arg at index 0
    l1_ptr[0] = get_arg_val<uint32_t>(0);

    // Positional common RT arg at index 0
    l1_ptr[1] = get_common_arg_val<uint32_t>(0);

    // Named per-core RT arg (appended after positional per-core args)
    l1_ptr[2] = rt_args::get<ct_args::my_kernel::named_per_core>();

    // Named common RT arg (appended after positional common args)
    l1_ptr[3] = rt_args::get<ct_args::my_kernel::named_common>();
}
