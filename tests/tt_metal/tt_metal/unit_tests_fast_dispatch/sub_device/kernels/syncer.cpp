// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t sem_addr = get_arg_val<uint32_t>(0);

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, 1);
    uint64_t noc_local_sem_addr = get_noc_addr(sem_addr);
    noc_semaphore_inc(noc_local_sem_addr, -1);
    noc_async_atomic_barrier();
}
