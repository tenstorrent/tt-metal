// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t sem_addr = get_arg_val<uint32_t>(0);
    uint32_t remote_x = get_arg_val<uint32_t>(1);
    uint32_t remote_y = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(local_sem, 1);
    uint64_t noc_local_sem_addr = get_noc_addr(sem_addr);
    noc_semaphore_inc(noc_local_sem_addr, -1);
    uint64_t noc_remote_sem_addr = get_noc_addr(remote_x, remote_y, sem_addr);
    noc_semaphore_inc(noc_remote_sem_addr, 1);
    noc_async_atomic_barrier();
}
