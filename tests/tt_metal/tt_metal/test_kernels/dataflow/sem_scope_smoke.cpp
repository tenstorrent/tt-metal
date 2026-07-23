// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase-1 SemScope smoke kernel: exercises the scoped Semaphore class end-to-end.
//
// A single DM thread constructs a Semaphore<TENSIX, kScope> over a bound semaphore
// (sem::counter), increments it `increment_times` via up(), reads it back with
// value(), and reports the observed value to a scratch L1 word for host checking.
// This validates that the selected scope's up()/value() paths compile (templates
// only compile when instantiated) and produce the correct single-writer count.
//
// Scope is chosen at compile time via a -D define from the host:
//   SEM_SCOPE_EXTERNAL         -> EXTERNAL         (self-targeted NoC atomic increment)
//   SEM_SCOPE_DM_LOCAL_CACHED  -> DM_LOCAL_CACHED  (32-bit RISC-V AMO on cached alias)
//   (neither)                  -> LOCAL_NONATOMIC  (legacy plain RMW)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

#if defined(SEM_SCOPE_EXTERNAL)
constexpr SemScope kScope = SemScope::EXTERNAL;
#elif defined(SEM_SCOPE_DM_LOCAL_CACHED)
constexpr SemScope kScope = SemScope::DM_LOCAL_CACHED;
#else
constexpr SemScope kScope = SemScope::LOCAL_NONATOMIC;
#endif

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

#if defined(SEM_TOKEN_CTAD)
    // Phase-2 S1: construct from the baked SemAccessor token; CTAD deduces
    // Semaphore<TENSIX, kScope> with no explicit template args (sem::counter is the
    // bare id today; from S2 it becomes the token itself).
    Semaphore s(SemAccessor<sem::counter, kScope>{});
#else
    Semaphore<ProgrammableCoreType::TENSIX, kScope> s(sem::counter);
#endif
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
