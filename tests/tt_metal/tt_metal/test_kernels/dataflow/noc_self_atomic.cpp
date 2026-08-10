// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Self-targeted ("loopback") NoC atomic increment probe.
//
// EXTERNAL-scope semaphores route even a core's own increment through a NoC atomic
// (NOC_AT_INS_INCR_GET) so local and remote writers serialize at one NIU point;
// RISC-V AMOs hang on the uncached alias (dev_mem_map.h), so there is no fallback.
// get_noc_addr(addr) (single-arg) encodes THIS core's own coordinates, so
// noc_semaphore_inc(get_noc_addr(sem_addr)) is a loopback atomic RMW at this
// node's TL1.
//
// Every user DM thread increments the SAME word `increment_times` times; the host
// expects exactly num_user_dms * increment_times. The word is only ever touched by
// NoC atomics (which land at TL1), so no cache flush is needed.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t sem_addr = get_arg(args::sem_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

#ifdef REMOTE_TARGET
    // Remote target: increment a word on ANOTHER node (coords passed in).
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);
    const uint64_t target_noc_addr = get_noc_addr(remote_noc_x, remote_noc_y, sem_addr);
#else
    // Self target (loopback): single-arg get_noc_addr encodes this core's own coords.
    const uint64_t target_noc_addr = get_noc_addr(sem_addr);
#endif

    for (uint32_t i = 0; i < increment_times; i++) {
        noc_semaphore_inc(target_noc_addr, 1);
        // Drain after each atomic: bounds outstanding atomics and makes completion unambiguous.
        noc_async_atomic_barrier();
    }
}
