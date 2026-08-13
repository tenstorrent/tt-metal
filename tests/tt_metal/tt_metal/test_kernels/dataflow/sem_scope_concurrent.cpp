// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Concurrency proof for the scoped Semaphore class: up()/down() must be atomic under real
// multi-DM contention. The scope is host-picked (invisible table), so the same source runs
// under any scope. Quasar-only (roles gated by mhartid).
//
// Modes (via -D):
//   MODE_CONCURRENT_UP     : all user DMs up(1)*iters a shared sem; the lowest DM waits on a
//                            'done' sem, reports value(). Expect num_threads*iters (a
//                            non-atomic up() loses updates).
//   MODE_PRODUCER_CONSUMER : (num_threads-1) producers up(1)*iters; the lowest DM drains them
//                            all with down(1), reports value(). Expect 0 (a non-atomic down()
//                            loses units and the consumer blocks).
//   MODE_MULTI_CONSUMER    : the lowest DM up(1)s (num_threads-2)*iters credits; hart 3 is a
//                            WATCHDOG; every other DM concurrently drains its share as single
//                            down(1)s, then bumps 'done'. Expect 0 -- but the count alone is
//                            conservation-blind (each consumer issues a fixed number of
//                            decrements, so a double-spend wraps the word only TRANSIENTLY and
//                            the modular sum still lands on 0). The watchdog latches the max
//                            value it ever observes into report[16] (own 64B line): under a
//                            correct lock it can never exceed the credits issued, while a
//                            double-spend leaves a ~2^32 wrap it latches.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

static inline void report_value(uint32_t report_addr, uint32_t v) {
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    *r = v;
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    flush_l2_cache_line(report_addr);  // make the report visible to the host readback of TL1
#endif
}

// Report the final count AND which mechanism the host actually baked for each semaphore, so a
// census change can't silently demote these shapes to a different (still-exact) mechanism while
// the tests stay green. Layout: [0]=count, [1]=scope(counter), [2]=scope(done).
static inline void report(uint32_t report_addr, uint32_t count) {
    report_value(report_addr, count);
    report_value(report_addr + sizeof(uint32_t), static_cast<uint32_t>(sem_scope_of(sem::counter)));
    report_value(report_addr + 2 * sizeof(uint32_t), static_cast<uint32_t>(sem_scope_of(sem::done)));
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
    Semaphore work(sem::counter);  // mechanism comes from the host's scope table

#if defined(MODE_PRODUCER_CONSUMER)
    if (is_lowest) {
        // Single consumer: drain every producer increment. Terminates at 0 iff the
        // decrement is atomic vs the concurrent producer increments (else it blocks).
        const uint32_t total = (num_threads - 1) * increment_times;
        for (uint32_t i = 0; i < total; i++) {
            work.down(1);
        }
        report(report_addr, work.value());  // expect 0
    } else {
        for (uint32_t i = 0; i < increment_times; i++) {
            work.up(1);
        }
    }
#elif defined(MODE_MULTI_CONSUMER)
    // ONE producer (hart 2), ONE watchdog (hart 3), (num_threads-2) CONCURRENT consumers taking
    // their credits as single down(1)s for maximal lock contention (failure modes: see header).
    Semaphore done(sem::done);  // mechanism comes from the host's scope table
    const uint32_t num_consumers = num_threads - 2;
    if (is_lowest) {
        // Producer + reporter: single-credit ups for maximal interleave with the racing consumers.
        const uint32_t total = num_consumers * increment_times;
        for (uint32_t i = 0; i < total; i++) {
            work.up(1);
        }
        done.wait_min(num_consumers);       // every consumer drained its share
        report(report_addr, work.value());  // expect exactly 0
    } else if (hart == 3) {
        // Watchdog: sample until every consumer has finished, latch the max observed value.
        uint32_t max_seen = 0;
        while (done.value() < num_consumers) {
            const uint32_t v = work.value();
            if (v > max_seen) {
                max_seen = v;
            }
        }
        report_value(report_addr + 64u, max_seen);  // own 64B line: no clobber vs the reporter's
    } else {
        for (uint32_t i = 0; i < increment_times; i++) {
            work.down(1);
        }
        done.up(1);
    }
#else  // MODE_CONCURRENT_UP
    Semaphore done(sem::done);  // mechanism comes from the host's scope table
    for (uint32_t i = 0; i < increment_times; i++) {
        work.up(1);
    }
    done.up(1);
    if (is_lowest) {
        done.wait_min(num_threads);         // barrier: every thread finished its up() loop
        report(report_addr, work.value());  // expect num_threads * increment_times
    }
#endif
}
