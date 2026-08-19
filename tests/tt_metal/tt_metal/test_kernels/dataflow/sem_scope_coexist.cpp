// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Coexistence proof for the cached-only semaphore pool: every DM concurrently hammers a
// DM_LOCAL_CACHED semaphore and an EXTERNAL one, and both final
// counts must be exact. Scopes are host-picked. Quasar-only.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

static inline void report_value(uint32_t report_addr, uint32_t v) {
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    *r = v;
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    flush_l2_cache_line(report_addr);
#endif
}

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t num_threads = get_arg(args::num_threads);

    Semaphore cached(sem::cached);      // host-picked DM_LOCAL_CACHED
    Semaphore external(sem::external);  // host-picked EXTERNAL
    Semaphore done(sem::done);          // EXTERNAL: cross-thread completion barrier

    // Interleave the two so both are maximally concurrent.
    for (uint32_t i = 0; i < increment_times; i++) {
        cached.up(1);
        external.up(1);
    }
    done.up(1);

    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {
        done.wait_min(num_threads);
        report_value(report_addr, cached.value());                       // expect num_threads * increment_times
        report_value(report_addr + sizeof(uint32_t), external.value());  // expect num_threads * increment_times
        report_value(report_addr + 2 * sizeof(uint32_t), static_cast<uint32_t>(sem_scope_of(sem::cached)));
        report_value(report_addr + 3 * sizeof(uint32_t), static_cast<uint32_t>(sem_scope_of(sem::external)));
    }
}
