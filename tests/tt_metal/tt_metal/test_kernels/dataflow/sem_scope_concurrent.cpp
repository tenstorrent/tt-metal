// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Concurrency proof for the scoped Semaphore class: up()/down() must be atomic under real
// multi-DM contention. Scope is host-baked and picked up via CTAD, so the same source runs
// under any scope. Quasar-only (roles gated by mhartid).
//
// Modes (via -D):
//   MODE_CONCURRENT_UP     : all user DMs up(1)*iters a shared sem; the lowest DM waits on a
//                            'done' sem, reports value(). Expect num_threads*iters (a
//                            non-atomic up() loses updates -> less).
//   MODE_PRODUCER_CONSUMER : (num_threads-1) producers up(1)*iters; the lowest DM drains them
//                            all with down(1), reports value(). Expect 0 (a non-atomic down()
//                            loses units -> the consumer blocks -> timeout).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

static inline void report_value(uint32_t report_addr, uint32_t v) {
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    *r = v;
#if defined(ARCH_QUASAR)
    flush_l2_cache_line(report_addr);  // make the report visible to the host readback of TL1
#endif
}

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t num_threads = get_arg(args::num_threads);

    // Quasar user DM harts are 2..(2+num_threads-1); the lowest (2) is reporter/consumer.
    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    const bool is_lowest = (hart == 2);

    // A DM_LOCAL_CACHED semaphore's pool slot is seeded by the auto-injected sem::init_dm_cached().
    Semaphore work(sem::counter);  // CTAD deduces the host-baked scope

#if defined(MODE_PRODUCER_CONSUMER)
    if (is_lowest) {
        // Single consumer: drain every producer increment. Terminates at 0 iff the
        // decrement is atomic vs the concurrent producer increments (else it blocks).
        const uint32_t total = (num_threads - 1) * increment_times;
        for (uint32_t i = 0; i < total; i++) {
            work.down(1);
        }
        report_value(report_addr, work.value());  // expect 0
    } else {
        for (uint32_t i = 0; i < increment_times; i++) {
            work.up(1);
        }
    }
#else  // MODE_CONCURRENT_UP
    Semaphore done(sem::done);  // CTAD deduces the host-baked scope
    for (uint32_t i = 0; i < increment_times; i++) {
        work.up(1);
    }
    done.up(1);
    if (is_lowest) {
        done.wait_min(num_threads);  // barrier: every thread finished its up() loop
        report_value(report_addr, work.value());  // expect num_threads * increment_times
    }
#endif
}
