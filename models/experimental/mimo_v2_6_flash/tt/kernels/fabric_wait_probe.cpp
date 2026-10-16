// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Probe for fabric latency experiments: wait until a global semaphore reaches `expected`, then reset it.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t sem_addr = get_arg_val<uint32_t>(0);
    const uint32_t expected = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, expected);
    noc_semaphore_set(sem, 0);
}
