// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "debug/dprint.h"

// RT args:
//   0: completion_sem_addr (u32)
//   1: expected_value      (u32)

void kernel_main() {
    size_t idx = 0;
    const uint32_t sem_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t expected_value = get_arg_val<uint32_t>(idx++);

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    const uint32_t cur0 = *sem_ptr;
    DPRINT << "rx_wait: enter sem@0x" << sem_addr << " exp=" << expected_value << " cur=" << cur0 << ENDL();

    // Block here until the writer's atomic_inc arrives.
    noc_semaphore_wait(sem_ptr, expected_value);

    const uint32_t cur1 = *sem_ptr;
    DPRINT << "rx_wait: PASSED cur=" << cur1 << " -> resetting to 0" << ENDL();

    // Reset so subsequent iterations see a fresh transition.
    noc_semaphore_set(sem_ptr, 0);

    const uint32_t cur2 = *sem_ptr;
    DPRINT << "rx_wait: reset done cur=" << cur2 << ENDL();
}
