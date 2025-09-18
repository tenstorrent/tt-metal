// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

// CT (compile-time) args:
//   none
// RT (runtime) args:
//   0: completion_sem_addr   (u32)  // L1 address of the global semaphore on receiver
//   1: expected_value        (u32)  // e.g. number of pages, or just 1

void kernel_main() {
    size_t idx = 0;
    const uint32_t sem_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t expected_value = get_arg_val<uint32_t>(idx++);

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    noc_semaphore_wait(sem_ptr, expected_value);
}
