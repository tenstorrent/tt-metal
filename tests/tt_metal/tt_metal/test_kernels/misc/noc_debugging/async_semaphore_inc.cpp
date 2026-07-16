// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Repeatedly issue non-posted remote atomic increments to a semaphore on another core (unicast) or a
    // rectangle of cores (multicast). Without an atomic (or full) barrier the increments are still outstanding at
    // kernel end, which the NOC debug tool reports as an unflushed semaphore issue. A barrier drains them.
#if defined(USE_MULTICAST)
    const uint64_t sem_noc_addr =
        get_noc_multicast_addr(MCAST_START_X, MCAST_START_Y, MCAST_END_X, MCAST_END_Y, DST_ADDR);
    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc_semaphore_inc_multicast(sem_noc_addr, 1, NUM_DEST_CORES);
    }
#else
    const uint64_t sem_noc_addr = get_noc_addr(OTHER_CORE_X, OTHER_CORE_Y, DST_ADDR);
    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc_semaphore_inc(sem_noc_addr, 1);
    }
#endif

#if defined(USE_ATOMIC_BARRIER)
    noc_async_atomic_barrier();
#elif defined(USE_FULL_BARRIER)
    noc_async_full_barrier();
#endif
}
