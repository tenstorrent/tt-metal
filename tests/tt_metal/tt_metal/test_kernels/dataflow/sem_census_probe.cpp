// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reports the scope the host baked for sem::counter, so the census tests can check the
// classifier's decision directly.
// Every thread also up()s the counter increment_times through whichever mechanism the
// host chose. Report: [0] baked scope, [1] final count, [2] ring slot.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t is_reporter = get_arg(args::is_reporter);
    // Kernels with different thread counts must not share a barrier slot.
    const uint32_t barrier_idx = get_arg(args::barrier_idx);
    const uint32_t wait_min_total = get_arg(args::wait_min_total);

    // The mechanism comes from the host's scope table
    Semaphore counter(sem::counter);

    for (uint32_t i = 0; i < increment_times; i++) {
        counter.up(1);
    }

    // Barrier across this kernel's own threads, so the reporter sees all of their increments.
    sync_threads(barrier_idx);

    if (is_reporter != 0 && get_my_thread_id() == 0u) {
        if (wait_min_total != 0u) {
            // Bounded wait for the other binder kernels' increments.
            constexpr uint32_t kSpinCap = 1u << 20;
            for (uint32_t spin = 0; counter.value() < wait_min_total && spin < kSpinCap; spin++) {
            }
        }
        volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
        report[0] = static_cast<uint32_t>(sem_scope_of(sem::counter));
        report[1] = counter.value();
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
        report[2] =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(::get_semaphore(sem::counter) + MEM_L1_UNCACHED_BASE);
        flush_l2_cache_line(report_addr);
        flush_l2_cache_line(report_addr + sizeof(uint32_t));
        flush_l2_cache_line(report_addr + 2 * sizeof(uint32_t));
#else
        report[2] = 0;
#endif
    }
}
