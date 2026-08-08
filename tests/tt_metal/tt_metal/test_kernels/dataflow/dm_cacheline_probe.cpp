// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM write-back cache line width probe: the minimum separation at which a cached-AMO
// word (wc) and a NoC-atomic word (wn) stop sharing a cache line -- i.e. the DM's
// dirty-line write-back stops clobbering the NoC word. Single-thread and sequenced to
// FORCE the worst-case ordering, so a "safe" result means the hazard is impossible at
// that separation, not that a race was won.
//
// Controls (a platform that does not model the write-back cache would read "safe"
// everywhere and yield a bogus small width; the host rejects such runs):
//   CONTROL A (residency): a cached write must stay invisible via the uncached alias
//     until flushed (write-back cache exists; uncached alias bypasses it).
//   CONTROL B (landed): wn is read via the uncached alias BEFORE the flush and must
//     equal NOC_ADD, so a post-flush 0 is a genuine clobber, not "never landed".
//   POSITIVE CONTROL (host): sep=4 (guaranteed same line) must read CLOBBERED, else
//     the clobber mechanism is not live and the width is INVALID.
//
// Per sub-test i:  wc = base + i*STRIDE  (cached-AMO word),  wn = wc + sep[i]  (NoC word)
//   1. zero wc, wn via cached alias; flush both lines           -> TL1=0, lines clean
//   2. cached AMO wc += CACHED_ADD                              -> wc's line dirty; wn's
//                                                                 OLD (0) cached if same line
//   3. NoC atomic wn += NOC_ADD; atomic barrier                 -> TL1 wn = NOC_ADD
//   4. read wn via uncached (pre-flush)                         -> Control B: must be NOC_ADD
//   5. flush wc's line                                          -> clobbers TL1 wn iff same line
//   6. read wc, wn via uncached                                 -> report
//
// Ceiling: flush_l2_cache_line = FLUSH64 (fixed 64B window), so a width > 64B is not
// detectable here; 64B matches the documented L1 D$ / L2 line. Quasar-only.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
#if defined(ARCH_QUASAR)
    const uint32_t base_addr = get_arg(args::base_addr);          // line-aligned sweep region
    const uint32_t report_addr = get_arg(args::report_addr);      // disjoint scratch (2 + 3*NUM_SEPS words)
    const uint32_t residency_addr = get_arg(args::residency_addr);  // disjoint scratch for Control A

    constexpr uint32_t CACHED_ADD = 5;
    constexpr uint32_t NOC_ADD = 7;
    constexpr uint32_t STRIDE = 512;  // per-sub-test region; keeps each wc line-aligned
    constexpr uint32_t RES_OLD = 0x1111u;
    constexpr uint32_t RES_NEW = 0x2222u;
    // First entry (4) is the positive control: guaranteed same-line for a line-aligned
    // base and any plausible line >= 8B, so it MUST clobber if the hazard is live.
    constexpr uint32_t NUM_SEPS = 6;
    const uint32_t seps[NUM_SEPS] = {4u, 8u, 16u, 32u, 64u, 128u};

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);

    // ---- Control A: write-back residency proof ----
    {
        volatile tt_l1_ptr uint32_t* r_unc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(residency_addr) + MEM_L1_UNCACHED_BASE);
        volatile uint32_t* r_cached = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(residency_addr));
        *r_unc = RES_OLD;                             // TL1 = OLD (uncached alias bypasses the cache)
        *r_cached = RES_NEW;                          // cached write: dirty, NOT flushed
        report[0] = *r_unc;                           // expect OLD (NEW is stuck in the write-back cache)
        flush_l2_cache_line(static_cast<uintptr_t>(residency_addr));
        report[1] = *r_unc;                           // expect NEW (flush wrote it back)
    }

    // ---- Per-separation clobber sweep ----
    for (uint32_t i = 0; i < NUM_SEPS; i++) {
        const uint32_t wc = base_addr + i * STRIDE;  // cached-AMO word
        const uint32_t wn = wc + seps[i];            // NoC-atomic word

        uint32_t* wc_cached = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(wc));
        volatile tt_l1_ptr uint32_t* wc_unc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(wc) + MEM_L1_UNCACHED_BASE);
        volatile tt_l1_ptr uint32_t* wn_unc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(wn) + MEM_L1_UNCACHED_BASE);

        // 1. Zero both via the cached alias, write back so TL1 starts clean at 0.
        *reinterpret_cast<volatile uint32_t*>(wc_cached) = 0;
        *reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(wn)) = 0;
        flush_l2_cache_line(static_cast<uintptr_t>(wc));
        flush_l2_cache_line(static_cast<uintptr_t>(wn));

        // 2. Cached AMO on wc: dirties wc's line; wn's old 0 is cached if same line.
        __atomic_add_fetch(wc_cached, CACHED_ADD, __ATOMIC_SEQ_CST);

        // 3. NoC atomic on wn -> TL1 (bypasses DM cache). Barrier: committed before flush.
        noc_semaphore_inc(get_noc_addr(wn), NOC_ADD);
        noc_async_atomic_barrier();

        // 4. Control B: pre-flush uncached read proves the atomic landed at TL1.
        const uint32_t wn_pre = *wn_unc;

        // 5. Flush wc's dirty line: clobbers TL1 wn to the stale cached 0 iff same line.
        flush_l2_cache_line(static_cast<uintptr_t>(wc));

        // 6. Read TL1 truth via the uncached alias.
        report[2 + 3 * i + 0] = *wc_unc;   // liveness: expect CACHED_ADD
        report[2 + 3 * i + 1] = wn_pre;    // Control B: expect NOC_ADD
        report[2 + 3 * i + 2] = *wn_unc;   // NOC_ADD => safe (different line); 0 => clobbered
    }

    // Publish the whole report region to TL1 for the host readback.
    const uint32_t total = 2 + 3 * NUM_SEPS;
    for (uint32_t i = 0; i < total; i++) {
        flush_l2_cache_line(static_cast<uintptr_t>(report_addr) + i * sizeof(uint32_t));
    }
#endif
}
