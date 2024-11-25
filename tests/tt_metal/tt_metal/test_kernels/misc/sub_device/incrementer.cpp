// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t sem_addr = get_arg_val<uint32_t>(0);
    uint32_t waiter_core_x = get_arg_val<uint32_t>(1);
    uint32_t waiter_core_y = get_arg_val<uint32_t>(2);

    uint64_t noc_remote_sem_addr = get_noc_addr(waiter_core_x, waiter_core_y, sem_addr);
    noc_semaphore_inc(noc_remote_sem_addr, 1);
    noc_async_atomic_barrier();
}
