// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Census / AUTO-classifier probe: reports the scope the host baked for sem::counter, so a test
// can assert the classifier's decision (counts alone can't tell mechanisms apart). Also
// exercises the mechanism: every thread increments `counter` increment_times, then the reporter
// thread publishes the result.
//
// report[0] = baked SemScope of sem::counter (0=LOCAL_NONATOMIC, 1=DM_LOCAL_CACHED, 2=EXTERNAL)
// report[1] = counter.value() after this kernel's threads have all finished incrementing.
//             Only meaningful if no OTHER kernel writes this semaphore; tests where one does
//             assert the reported scope only.
// report[2] = this semaphore's RING slot (uncached alias). For a cached semaphore it must still
//             hold the initial value: proves the count lives in the pool, not the ring.
// report[3] = baked SemAccess value (mirrors the host AccessType; OBSERVE=3 for read-only)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t is_reporter = get_arg(args::is_reporter);
    // This kernel's kernel-barrier slot: co-resident kernels with different thread counts must
    // not share one (mixed groups hang), so the host hands each kernel its own.
    const uint32_t barrier_idx = get_arg(args::barrier_idx);

    // A DM_LOCAL_CACHED semaphore's pool slot is seeded by the auto-injected sem::init_dm_cached().
    Semaphore counter(sem::counter);  // CTAD deduces the host-baked scope AND access rights

    // OBSERVE makes up() a compile error -- so guard on the baked access itself. (up() is
    // legal under every writer label, so all writer-shape tests reuse this probe.)
    if constexpr (sem::counter.access != SemAccess::OBSERVE) {
        for (uint32_t i = 0; i < increment_times; i++) {
            counter.up(1);
        }
    }

    // Barrier across THIS kernel's threads, so the reporter sees all of their increments.
    // (No-op when the kernel is single-threaded.)
    sync_threads(barrier_idx);

    if (is_reporter != 0 && get_my_thread_id() == 0u) {
        volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
        report[0] = static_cast<uint32_t>(sem::counter.scope);
        report[1] = counter.value();
        // Read the baked access back so OBSERVE tests can assert the host actually emitted it.
        report[3] = static_cast<uint32_t>(sem::counter.access);
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
        // Residency check: the ring slot (uncached alias = TL1 truth) must be untouched for a
        // cached semaphore -- its count lives in the pool.
        report[2] =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(::get_semaphore(sem::counter.id) + MEM_L1_UNCACHED_BASE);
        flush_l2_cache_line(report_addr);
        flush_l2_cache_line(report_addr + sizeof(uint32_t));
        flush_l2_cache_line(report_addr + 2 * sizeof(uint32_t));
        flush_l2_cache_line(report_addr + 3 * sizeof(uint32_t));
#else
        report[2] = 0;
#endif
    }
}
