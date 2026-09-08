// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Self-targeted ("loopback") NoC atomic increment: EXTERNAL-scope semaphores
// route even a core's own increment through the NoC so local and remote writers
// go through the same physical path and stay atomic. Every user DM thread bumps
// the same word `increment_times` times, and the host expects num_user_dms * increment_times.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t sem_addr = get_arg(args::sem_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

#ifdef REMOTE_TARGET
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);
    const uint64_t target_noc_addr = get_noc_addr(remote_noc_x, remote_noc_y, sem_addr);
#else
    const uint64_t target_noc_addr = get_noc_addr(sem_addr);
#endif

    for (uint32_t i = 0; i < increment_times; i++) {
        noc_semaphore_inc(target_noc_addr, 1);
        noc_async_atomic_barrier();
    }
}
