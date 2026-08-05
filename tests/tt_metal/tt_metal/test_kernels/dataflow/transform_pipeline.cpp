// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/core_local_mem.h"
#include "api/dataflow/noc_semaphore.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_elements = get_arg(args::num_elements);
    const uint32_t buf_a = get_arg(args::buf_a);
    const uint32_t buf_b = get_arg(args::buf_b);

#if defined(INCOMING_SEM)
    Semaphore sem_in(sem::sem_in);
#endif
#if defined(OUTGOING_SEM)
    Semaphore sem_out(sem::sem_out);
#endif

    CoreLocalMem<volatile uint32_t> a(buf_a + MEM_L1_UNCACHED_BASE);
    CoreLocalMem<volatile uint32_t> b(buf_b + MEM_L1_UNCACHED_BASE);

    for (uint32_t i = 0; i < num_elements; i++) {
#if defined(INCOMING_SEM)
        sem_in.down(1);
#endif
        const uint32_t val = a[i];
        const uint32_t new_val = val + 1;
        b[i] = new_val;
        DPRINT("Read the value {} and wrote the value {}\n", val, new_val);
#if defined(OUTGOING_SEM)
        sem_out.up(1);
#endif
    }
}
