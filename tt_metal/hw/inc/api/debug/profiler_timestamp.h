// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Clock-source seam for the device kernel profiler (kernel_profiler.hpp).
//
// The profiler timestamps every zone by reading the tile WALL_CLOCK debug register
// (RISCV_DEBUG_REG_WALL_CLOCK_L, as two 32-bit words: low at index 0, high at index 1). On Quasar
// the RTL emulator HANGS when that register is read (empirically verified; the reference
// quasar_core_shift sender_bw.cpp warns "get_timestamp() crashes the simulator"). The
// emulator-safe clock is the per-core RISC-V `rdcycle` CSR.
//
// This header isolates the clock read behind one inline returning {hi, lo} so the rest of the
// profiler is unchanged:
//   - ARCH_QUASAR  -> rdcycle (low 32 bits) + a high word (true high on the 64-bit DM core, a
//                     software wrap-tracked high word on the 32-bit TRISC; see Phase 1).
//   - otherwise    -> the exact same RISCV_DEBUG_REG_WALL_CLOCK_L reads the profiler did before,
//                     so silicon (Wormhole/Blackhole) compiles to byte-identical instructions.
//
// ARCH_QUASAR is injected into the kernel JIT build by qa_hal.cpp, so the #if resolves correctly in
// JIT'd kernels. Host-side reconstruction is unchanged: (uint64_t(hi) << 32) | lo.

#include <cstdint>

#include "internal/risc_attribs.h"  // tt_reg_ptr

// 64-bit timestamp split into the profiler's wire format (two 32-bit words).
struct profiler_ts_t {
    uint32_t hi;  // high 32 bits (only low 12 are kept in the packed marker word)
    uint32_t lo;  // low 32 bits
};

#if defined(ARCH_QUASAR)

// Per-core RISC-V cycle counter. 32-bit on TRISC (tt-qsr32), 64-bit on DM (tt-qsr64). Reading
// rdcycle is emulator-safe; reading WALL_CLOCK is not.
inline __attribute__((always_inline)) uint32_t profiler_rdcycle_lo() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c)::"memory");
    return c;
}

inline __attribute__((always_inline)) profiler_ts_t profiler_read_timestamp() {
    // Phase 0: low word from rdcycle, high word 0. Phase 1 replaces the high word with a
    // per-hardware-thread wrap-tracked value (TRISC) / true rdcycleh (DM) so same-core deltas across
    // a 32-bit wrap stay exact. Same-core deltas are already correct here for intervals < 2^32.
    profiler_ts_t ts;
    ts.lo = profiler_rdcycle_lo();
    ts.hi = 0;
    return ts;
}

#else

// Silicon path: identical reads to the original profiler code (RISCV_DEBUG_REG_WALL_CLOCK_L, low at
// index 0 / high at index 1). Kept here so the seam is a pure refactor on Wormhole/Blackhole.
inline __attribute__((always_inline)) profiler_ts_t profiler_read_timestamp() {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    profiler_ts_t ts;
    ts.lo = p_reg[0];  // WALL_CLOCK_LOW_INDEX
    ts.hi = p_reg[1];  // WALL_CLOCK_HIGH_INDEX
    return ts;
}

#endif
