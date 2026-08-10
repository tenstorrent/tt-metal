// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 32-bit RISC-V atomic add (amoadd.w) on the CACHED L1 alias.
//
// The DM_LOCAL_CACHED fast path increments a semaphore with a plain RISC-V AMO on
// the cached alias (DM cores are cache-coherent, so no NoC round-trip). Only the
// 64-bit AMO width is proven on Quasar (test_riscv_atomics.cpp); semaphore words
// are 32-bit, so this checks amoadd.w on a 4-byte L1 word.
//
// Every user DM thread increments the SAME word `increment_times` times; host
// expects num_user_dms * increment_times. Short count => 32-bit AMO unreliable;
// hang/fault => amoadd.w unsupported on L1.

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Cached alias (plain L1 address): RISC-V AMOs hang on the uncached alias (dev_mem_map.h).
    uint32_t* counter = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(get_arg(args::sem_addr)));
    const uint32_t increment_times = get_arg(args::increment_times);

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Rerun safety: discard any cache-resident copy so the host's fresh TL1 preload is seen.
    // One thread invalidates, all rendezvous (see dm_cas32.cpp).
    if (get_my_thread_id() == 0u) {
        invalidate_l2_cache_line(reinterpret_cast<uintptr_t>(counter));
    }
    sync_threads(0);
#endif

    for (uint32_t i = 0; i < increment_times; i++) {
        __atomic_add_fetch(counter, 1u, __ATOMIC_SEQ_CST);  // 32-bit amoadd.w
    }

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Flush the write-back cache so the host readback of TL1 sees the updated word.
    // (Blackhole's L1 cache is write-through, so no flush is needed there.)
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(counter));
#endif
}
