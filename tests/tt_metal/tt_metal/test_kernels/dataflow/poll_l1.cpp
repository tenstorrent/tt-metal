// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    uint32_t poll_addr = get_arg_val<uint32_t>(0);
    uint32_t expected_value = get_arg_val<uint32_t>(1);
    auto sem_addr = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(2)));
    uint32_t invalidate_cache =
        get_arg_val<uint32_t>(3);  // For CI this will be true but for debug this can be modified

    volatile tt_l1_ptr uint32_t* poll_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_addr);
    uint32_t base_value = *poll_addr_ptr;  // Read once to cache it

    noc_semaphore_set(sem_addr, 1);

    while (*poll_addr_ptr != expected_value) {
        if (invalidate_cache) {
            invalidate_l1_cache();
        }
    }
}
