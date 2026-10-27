// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Grid all-to-one barrier. Every node except the target increments the target's barrier semaphore
// once over the NoC. The target waits for all of them to arrive, then writes a "released" marker to
// L1 so the host can confirm the barrier completed. Exercises max semaphore fan-in onto one counter.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t target_noc_x = get_arg(args::remote_noc_x);
    const uint32_t target_noc_y = get_arg(args::remote_noc_y);
    const uint32_t is_target = get_arg(args::is_target);
    const uint32_t num_signalers = get_arg(args::num_elements);
    const uint32_t result_addr = get_arg(args::result_addr);

    Noc noc;
    Semaphore barrier_sem(sem::barrier_sem);

    if (is_target) {
        // Wait for all N signalers.
        barrier_sem.down(num_signalers);
        barrier_sem.wait(0);
        // Write the "released" marker for the host to read.
        CoreLocalMem<uint32_t> result(result_addr);
        result[0] = 0xC0DEBA11u;
        flush_l2_cache_line(result_addr);
    } else {
        // Signal the target once.
        barrier_sem.up(noc, target_noc_x, target_noc_y, 1);
    }
}
