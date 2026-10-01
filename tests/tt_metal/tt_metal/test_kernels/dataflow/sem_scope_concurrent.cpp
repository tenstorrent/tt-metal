// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Proves up()/down() stay atomic under real multi-DM contention. Four shapes, one per
// -D mode: every thread up()s a shared count, one consumer drains many producers, many
// concurrent consumers run under a watchdog, and every thread bumps through the
// self-targeted remote-up form. The host picks the scope, so the same source runs under
// any mechanism. Quasar-only.

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

// Reports the final count plus the mechanism the host actually baked for each semaphore.
static inline void report(uint32_t report_addr, uint32_t count) {
    report_value(report_addr, count);
    report_value(report_addr + sizeof(uint32_t), static_cast<uint32_t>(sem::counter.scope));
    report_value(report_addr + 2 * sizeof(uint32_t), static_cast<uint32_t>(sem::done.scope));
}

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t num_threads = get_arg(args::num_threads);

    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    const bool is_lowest = (hart == 2);

    // Every wait below polls under this cap, so a starved wait ends in a wrong reported
    // count (which the host fails on) instead of hanging the test binary. It is a
    // iteration count that is ~100x any healthy wait.
    constexpr uint32_t kSpinCap = 1u << 20;

    // The mechanism comes from the binding token the host emitted.
    Semaphore work(sem::counter);

#if defined(MODE_PRODUCER_CONSUMER)
    if (is_lowest) {
        // One consumer drains every producer increment.
        const uint32_t total = (num_threads - 1) * increment_times;
        uint32_t drained = 0;
        for (uint32_t spin = 0; drained < total && spin < kSpinCap; spin++) {
            if (work.value() >= 1) {
                work.down(1);
                drained++;
            }
        }
        report(report_addr, work.value());  // expect 0
    } else {
        for (uint32_t i = 0; i < increment_times; i++) {
            work.up(1);
        }
    }
#elif defined(MODE_MULTI_CONSUMER)
    // One producer (hart 2), one watchdog (hart 3), and (num_threads-2) consumers taking
    // their credits as single down(1)s for maximum lock contention. The final count alone
    // cannot catch a broken lock: every consumer decrements a fixed number of times, so a
    // double-spend wraps the word only briefly and the sum still ends at 0. The watchdog
    // instead latches the highest value it ever sees: a working lock keeps that at or
    // below the credits issued, while a double-spend wraps it near 2^32.
    Semaphore done(sem::done);
    const uint32_t num_consumers = num_threads - 2;
    if (is_lowest) {
        const uint32_t total = num_consumers * increment_times;
        for (uint32_t i = 0; i < total; i++) {
            work.up(1);
        }
        for (uint32_t spin = 0; done.value() < num_consumers && spin < kSpinCap; spin++) {
        }
        report(report_addr, work.value());  // expect exactly 0
    } else if (hart == 3) {
        // Watchdog: sample until every consumer has finished, latch the max observed value.
        uint32_t max_seen = 0;
        for (uint32_t spin = 0; done.value() < num_consumers && spin < kSpinCap; spin++) {
            const uint32_t v = work.value();
            if (v > max_seen) {
                max_seen = v;
            }
        }
        report_value(report_addr + 64u, max_seen);
    } else {
        // Poll for a credit before each down(1).
        uint32_t drained = 0;
        for (uint32_t spin = 0; drained < increment_times && spin < kSpinCap; spin++) {
            if (work.value() >= 1) {
                work.down(1);
                drained++;
            }
        }
        done.up(1);
    }
#elif defined(MODE_SELF_NOC_UP)
    // The WH/BH pattern: bump the local semaphore through the remote-up form aimed
    // at this node's own coordinates. On a cached semaphore it must be served by the local
    // AMO (a NoC atomic must never touch the cached pool) and keep exact counts.
    Semaphore done(sem::done);
    Noc noc;
    const uint32_t self_noc_x = get_arg(args::self_noc_x);
    const uint32_t self_noc_y = get_arg(args::self_noc_y);
    for (uint32_t i = 0; i < increment_times; i++) {
        work.up(noc, self_noc_x, self_noc_y, 1);
    }
    noc.async_atomic_barrier();
    done.up(1);
    if (is_lowest) {
        // Barrier: every thread finished its up() loop (bounded, see kSpinCap).
        for (uint32_t spin = 0; done.value() < num_threads && spin < kSpinCap; spin++) {
        }
        report(report_addr, work.value());
    }
#else  // MODE_CONCURRENT_UP
    Semaphore done(sem::done);
    for (uint32_t i = 0; i < increment_times; i++) {
        work.up(1);
    }
    done.up(1);
    if (is_lowest) {
        for (uint32_t spin = 0; done.value() < num_threads && spin < kSpinCap; spin++) {
        }
        report(report_addr, work.value());  // expect num_threads * increment_times
    }
#endif
}
