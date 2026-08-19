// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 32-bit guarded-decrement CAS (lr.w/sc.w) on the cached L1 alias, the exact multi-consumer
// DM_LOCAL_CACHED down(1) shape. Every user DM thread drains one shared
// word `increment_times` times; the host preloads the total and expects 0. The guard cannot
// underflow: nonzero means a lost decrement. (Quasar-only kernel)

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t* word = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(get_arg(args::sem_addr)));
    const uint32_t increment_times = get_arg(args::increment_times);

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    if (get_my_thread_id() == 0u) {
        invalidate_l2_cache_line(reinterpret_cast<uintptr_t>(word));
    }
    sync_threads(0);
#endif

    for (uint32_t i = 0; i < increment_times; i++) {
        uint32_t observed = __atomic_load_n(word, __ATOMIC_RELAXED);
        do {
            while (observed < 1) {  // DM cores are mutually coherent, no invalidate needed.
                observed = __atomic_load_n(word, __ATOMIC_RELAXED);
            }
        } while (!__atomic_compare_exchange_n(
            word, &observed, observed - 1, /*weak=*/false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Flush the write-back cache so the host readback of TL1 sees the drained word.
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(word));
#endif
}
