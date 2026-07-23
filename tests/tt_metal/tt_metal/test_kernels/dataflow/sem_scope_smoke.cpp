// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SemScope smoke kernel: exercises the scoped Semaphore class end-to-end.
//
// A single DM thread constructs a Semaphore over a bound semaphore (sem::counter),
// increments it `increment_times` via up(), reads it back with value(), and reports the
// observed value to a scratch L1 word for host checking. This validates that the selected
// scope's up()/value() paths compile (templates only compile when instantiated) and produce
// the correct single-writer count.
//
// The physical scope is baked host-side (SemaphoreSpec.scope) into the sem::counter token and
// picked up here via CTAD, so this kernel is scope-agnostic: the same source runs under
// EXTERNAL (self-targeted NoC atomic), DM_LOCAL_CACHED (32-bit RISC-V AMO on the cached alias),
// or LOCAL_NONATOMIC (legacy plain RMW) depending on how the host built the SemaphoreSpec.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

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
