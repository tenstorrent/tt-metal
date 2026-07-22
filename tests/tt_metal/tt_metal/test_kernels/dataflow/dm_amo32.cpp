// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FAST-PATH keystone kernel: 32-bit RISC-V atomic add on the CACHED L1 alias.
//
// The DM_LOCAL_CACHED fast path of the auto-path semaphore design increments a
// semaphore with a plain RISC-V AMO on the cached alias (DM cores are mutually
// cache-coherent, so this is atomic among them without any NoC round-trip). But
// the only AMO width PROVEN on Quasar today is 64-bit (amoadd.d, see
// test_riscv_atomics.cpp which uses uint64_t on Quasar), while semaphore words
// are 32-bit. This kernel checks whether a 32-bit __atomic_add_fetch (amoadd.w)
// on a 4-byte L1 word is correct on the target arch -- the open question the fast
// path depends on (P0-B in the analysis).
//
// Every user DM thread increments the SAME 32-bit word `increment_times` times.
// Host verifies final count == num_user_dms * increment_times.
//   - wrong/short count -> 32-bit AMO on L1 is not reliable -> fast path must use
//     a 32-bit CAS loop or widen the word to 64-bit.
//   - hang/fault        -> amoadd.w unsupported on L1.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // CACHED alias (plain L1 address, NOT the uncached alias): RISC-V AMOs require
    // the cache-coherence system and hang on the uncached alias (dev_mem_map.h).
    uint32_t* counter = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(get_arg(args::sem_addr)));
    const uint32_t increment_times = get_arg(args::increment_times);

    for (uint32_t i = 0; i < increment_times; i++) {
        __atomic_add_fetch(counter, 1u, __ATOMIC_SEQ_CST);  // 32-bit amoadd.w
    }

#if defined(ARCH_QUASAR)
    // Flush the write-back cache so the host readback of TL1 sees the updated word.
    // (Blackhole's L1 cache is write-through, so no flush is needed there.)
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(counter));
#endif
}
