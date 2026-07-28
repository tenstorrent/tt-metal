// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Coexistence proof for the dedicated cached-only semaphore pool (Solution #1): a
// DM_LOCAL_CACHED semaphore (which lives in the pool, written via the cached alias + RISC-V
// AMO) and an EXTERNAL semaphore (which lives in the kernel_config ring, written via the NoC
// atomic) are hammered CONCURRENTLY by every DM in the SAME program. Because the pool is
// physically disjoint from the ring (MEM_DM_CACHED_SEM_BASE < MEM_MAP_END <= kernel_config
// ring base), the cached semaphore's dirty-line write-back can never overlap the NoC-written
// ring word -> neither clobbers the other. Both final counts must be exact.
//
// Scopes are host-baked (SemaphoreSpec.scope) and picked up via CTAD; the kernel is
// scope-agnostic. Quasar-only (roles gated by mhartid; the pool + AMO are Quasar features).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

static inline void report_value(uint32_t report_addr, uint32_t v) {
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    *r = v;
#if defined(ARCH_QUASAR)
    flush_l2_cache_line(report_addr);
#endif
}

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t num_threads = get_arg(args::num_threads);

    // NOTE: the cached-only pool slot for sem::cached is seeded by sem::init_dm_cached(), which the
    // build AUTO-INJECTS at kernel entry (before any thread's first up()) -- no call is needed here.

    Semaphore cached(sem::cached);      // CTAD -> DM_LOCAL_CACHED (pool, cached AMO)
    Semaphore external(sem::external);  // CTAD -> EXTERNAL (ring, NoC atomic)
    Semaphore done(sem::done);          // EXTERNAL: cross-thread completion barrier

    // Every thread hammers BOTH semaphores, interleaved, so the cached AMOs on the pool word and
    // the NoC atomics on the ring word are maximally concurrent.
    for (uint32_t i = 0; i < increment_times; i++) {
        cached.up(1);
        external.up(1);
    }
    done.up(1);

    uint64_t hart = 2;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {  // lowest user DM: wait for all, then report both final counts
        done.wait_min(num_threads);
        report_value(report_addr, cached.value());        // expect num_threads * increment_times
        report_value(report_addr + sizeof(uint32_t), external.value());  // expect num_threads * increment_times
    }
}
