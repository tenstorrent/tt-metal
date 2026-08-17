// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

void kernel_main() {
    uint32_t sem_addr = get_arg_val<uint32_t>(0);

    // Quasar's invalidate_l1_cache() is a no-op, so polling through the L1 D$ would spin on a stale
    // line and never observe the remote increment. Read the uncached alias, as dispatch does.
#ifdef ARCH_QUASAR
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr + MEM_L1_UNCACHED_BASE);
#else
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
#endif
    noc_semaphore_wait(sem, 1);
#ifndef ARCH_QUASAR
    // Reset the semaphore so callers can reuse it across iterations. Skipped on Quasar, where
    // noc_async_atomic_barrier() never retires, and where no caller reuses the semaphore.
    noc_semaphore_inc(get_noc_addr(sem_addr), -1);
    noc_async_atomic_barrier();
#endif
}
