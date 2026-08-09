// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 32-bit RISC-V conditional-decrement CAS (lr.w/sc.w) on the CACHED L1 alias.
//
// The multi-consumer DM_LOCAL_CACHED down() commits its >=value check and the
// subtract with ONE strong CAS on the cached alias (noc_semaphore.h): a losing
// CAS refreshes `observed` in place and re-enters the >= check, so two consumers
// can never both pass the check on the same credit. This kernel runs that EXACT
// production shape (value = 1).
//
// Every user DM thread decrements the SAME word `increment_times` times; host
// preloads num_user_dms * increment_times and expects exactly 0. The guarded CAS
// can never take the word below 0, so: nonzero => a decrement was LOST; a hang
// => a decrement was OVER-COMMITTED (word drained early, surplus threads spin)
// or LR/SC is unsupported on the cached alias.

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Cached alias (plain L1 address): LR/SC, like AMOs, hangs on the uncached alias (dev_mem_map.h).
    uint32_t* word = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(get_arg(args::sem_addr)));
    const uint32_t increment_times = get_arg(args::increment_times);

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Rerun safety: a previous run's flushed line may still be cache-resident and would hide
    // the host's fresh TL1 preload. ONE thread discards it, then all rendezvous before loading
    // (a per-thread invalidate could discard another thread's committed decrement mid-run).
    if (get_my_thread_id() == 0u) {
        invalidate_l2_cache_line(reinterpret_cast<uintptr_t>(word));
    }
    sync_threads(0);
#endif

    for (uint32_t i = 0; i < increment_times; i++) {
        // Exact production down(1) shape (noc_semaphore.h, DM_LOCAL_CACHED).
        uint32_t observed = __atomic_load_n(word, __ATOMIC_RELAXED);
        do {
            while (observed < 1) {  // DM cores are mutually coherent; no invalidate needed
                observed = __atomic_load_n(word, __ATOMIC_RELAXED);
            }
        } while (!__atomic_compare_exchange_n(
            word, &observed, observed - 1, /*weak=*/false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Flush the write-back cache so the host readback of TL1 sees the drained word.
    // (Quasar-only kernel: lr.w/sc.w needs Zalrsc, which Blackhole lacks.)
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(word));
#endif
}
