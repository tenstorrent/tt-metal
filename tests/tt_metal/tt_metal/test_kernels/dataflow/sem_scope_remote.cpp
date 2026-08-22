// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exercises the remote Semaphore::up(noc, x, y, v) with exact counts. One source, two
// roles per -D: REMOTE_SENDER bumps sem::counter on the semaphore's node, and the
// default receiver waits for the exact total, then reports the baked scope and value.
// The host picks the scope, so the same source runs under any mechanism.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
#ifdef REMOTE_SENDER
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);

    Semaphore counter(sem::counter);
    Noc noc;
    for (uint32_t i = 0; i < increment_times; i++) {
        counter.up(noc, remote_noc_x, remote_noc_y, 1);
    }
    noc.async_atomic_barrier();
#else
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t expected = get_arg(args::expected);

    Semaphore counter(sem::counter);
    // Bounded wait for the sender's increments.
    constexpr uint32_t kSpinCap = 1u << 20;
    for (uint32_t spin = 0; counter.value() < expected && spin < kSpinCap; spin++) {
    }

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = static_cast<uint32_t>(sem_scope_of(sem::counter));
    report[1] = counter.value();
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    flush_l2_cache_line(report_addr);
    flush_l2_cache_line(report_addr + sizeof(uint32_t));
#endif
#endif
}
