// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A single DM thread up()s sem::counter increment_times and reports
// value() to a scratch L1 word for the host to check. The host picks
// the scope, so the same source runs under any mechanism.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

    // The mechanism comes from the host's scope table
    Semaphore s(sem::counter);
#if defined(SEM_SCOPE_SENTINEL_DOWN)
    // Sentinel collision: with the word at 0xFFFFFFFF, EXTERNAL down()'s pre-op return
    // looks exactly like the CAS-return sentinel, so its bounded poll must give up
    // instead of hanging. The second down(1) proves the ret slot and lock still work.
    (void)increment_times;
    s.set(0xFFFFFFFFu);
    s.down(1);
    s.down(1);  // expect value 0xFFFFFFFD reported below
#else
    for (uint32_t i = 0; i < increment_times; i++) {
        s.up(1);
    }
#endif
#if defined(SEM_SCOPE_UPDOWN)
    // Single writer: up(N) then down(N) must leave the semaphore at 0.
    for (uint32_t i = 0; i < increment_times; i++) {
        s.down(1);
    }
#endif
    const uint32_t observed = s.value();

    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    *r = observed;
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    flush_l2_cache_line(report_addr);
#endif
}
