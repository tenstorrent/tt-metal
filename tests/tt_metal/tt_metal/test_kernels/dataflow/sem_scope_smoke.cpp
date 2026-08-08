// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SemScope smoke kernel: a single DM thread up()s sem::counter increment_times, reads it back
// with value(), and reports the result to a scratch L1 word for the host to check. Scope is
// host-baked and picked up via CTAD, so the same source runs under EXTERNAL, DM_LOCAL_CACHED,
// or LOCAL_NONATOMIC.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

    // A DM_LOCAL_CACHED semaphore's pool slot is seeded by the auto-injected sem::init_dm_cached().
    Semaphore s(sem::counter);  // CTAD deduces the host-baked scope
    for (uint32_t i = 0; i < increment_times; i++) {
        s.up(1);
    }
#if defined(SEM_SCOPE_UPDOWN)
    // Exercise down() per scope (EXTERNAL uses the atomic NoC decrement). Single
    // writer: up(N) then down(N) must leave the semaphore at 0.
    for (uint32_t i = 0; i < increment_times; i++) {
        s.down(1);
    }
#endif
    const uint32_t observed = s.value();

    // Report the observed value to a scratch word for the host to verify.
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    *r = observed;
#if defined(ARCH_QUASAR)
    flush_l2_cache_line(report_addr);  // make the write visible to the host readback of TL1
#endif
}
