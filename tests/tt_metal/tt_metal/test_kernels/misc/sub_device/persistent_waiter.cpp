// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

void kernel_main() {
    uint32_t sem_addr = get_arg_val<uint32_t>(0);
    uint32_t num_inc = get_arg_val<uint32_t>(1);
    uint32_t sync_core_x = get_arg_val<uint32_t>(2);
    uint32_t sync_core_y = get_arg_val<uint32_t>(3);

    uint64_t noc_remote_sem_addr = get_noc_addr(sync_core_x, sync_core_y, sem_addr);
    noc_semaphore_inc(noc_remote_sem_addr, 1);

    // Quasar's invalidate_l1_cache() is a no-op, so polling through the L1 D$ would spin on a stale
    // line and never observe the incrementers. Read the uncached alias, as dispatch does.
#ifdef ARCH_QUASAR
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr + MEM_L1_UNCACHED_BASE);
#else
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
#endif
    noc_semaphore_wait(sem, num_inc);
#ifndef ARCH_QUASAR
    // Reset the semaphore so callers can reuse it across iterations. Skipped on Quasar, where
    // noc_async_atomic_barrier() never retires, and where no caller reuses the semaphore.
    noc_semaphore_inc(get_noc_addr(sem_addr), -num_inc);
    noc_async_atomic_barrier();
#endif
}
