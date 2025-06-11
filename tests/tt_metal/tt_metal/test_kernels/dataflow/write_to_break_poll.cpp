// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    uint32_t poll_addr = get_arg_val<uint32_t>(0);
    uint32_t value_to_write = get_arg_val<uint32_t>(1);
    auto sem_addr = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(2)));

    volatile tt_l1_ptr uint32_t* poll_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_addr);

    // don't call noc_semaphore_wait here because this kernel is run with `poll_l1.cpp` and it is meant to test the l1
    // cache invalidation from pov of the polling kernel
    while (*sem_addr != 1) {
        invalidate_l1_cache();
    }

    poll_addr_ptr[0] = value_to_write;
}
