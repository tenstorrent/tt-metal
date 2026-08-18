// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

void kernel_main() {
    uint32_t sem_addr = get_arg_val<uint32_t>(0);

#ifdef ARCH_QUASAR
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr + MEM_L1_UNCACHED_BASE);
#else
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
#endif
    noc_semaphore_wait(sem, 1);
#ifndef ARCH_QUASAR
    noc_semaphore_inc(get_noc_addr(sem_addr), -1);
    noc_async_atomic_barrier();
#endif
}
