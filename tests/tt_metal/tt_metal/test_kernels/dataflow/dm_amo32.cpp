// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 32-bit RISC-V atomic add on the cached L1 alias, the DM_LOCAL_CACHED fast path's
// increment, at a width only 64-bit tests had proven. Every
// user DM thread bumps one shared word `increment_times` times; a short count means
// lost updates.

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t* counter = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(get_arg(args::sem_addr)));
    const uint32_t increment_times = get_arg(args::increment_times);

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    if (get_my_thread_id() == 0u) {
        invalidate_l2_cache_line(reinterpret_cast<uintptr_t>(counter));
    }
    sync_threads(0);
#endif

    for (uint32_t i = 0; i < increment_times; i++) {
        __atomic_add_fetch(counter, 1u, __ATOMIC_SEQ_CST);
    }

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Flush the write-back line so the host readback sees the result
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(counter));
#endif
}
