// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    set_l1_data_cache<true>();
    uint32_t poll_addr = get_arg_val<uint32_t>(0);
    uint32_t expected_value = get_arg_val<uint32_t>(1);
    uint32_t semaphore_id = get_arg_val<uint32_t>(2);
    uint32_t invalidate_cache =
        get_arg_val<uint32_t>(3);  // For CI this will be true but for debug this can be modified

    experimental::CoreLocalMem<uint32_t> poll_value(poll_addr);
    uint32_t base_value = poll_value[0];  // Read once to cache it

    experimental::Semaphore sem(semaphore_id);
    sem.set(1);

    while (poll_value[0] != expected_value) {
        if (invalidate_cache) {
            invalidate_l1_cache();
        }
    }
    set_l1_data_cache<false>();
}
