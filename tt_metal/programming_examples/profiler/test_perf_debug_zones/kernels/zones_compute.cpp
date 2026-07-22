// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: compute RISCs (TRISC0/1/2). Same 10-zone sweep as zones_dm.cpp, tagged per
// TRISC (T0_/T1_/T2_) so each of the 3 compute RISCs emits its own 10 distinctly-named zones.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef N_ITERS
#define N_ITERS 50u
#endif

#if COMPILE_FOR_TRISC == 0
#define ZTAG "T0"
#elif COMPILE_FOR_TRISC == 1
#define ZTAG "T1"
#else
#define ZTAG "T2"
#endif

// One named zone whose body busy-waits CYC wall-clock spin-counts. CYC is EMPIRICALLY calibrated so the
// zone displays ~CYC/2500 us in Tracy: at the ~1.35 GHz boosted aiclk the profiler records ~0.55 timestamp
// tick per spin-count and the context period is ~0.741 ns/tick, so displayed_ns ~= CYC * 0.41. LOW-register-
// only read with unsigned-wrap subtraction is tear-free for spins << 2^32.
#define ZONE(NAME, CYC)                                                                            \
    {                                                                                              \
        DeviceZoneScopedN(NAME);                                                                   \
        volatile tt_reg_ptr uint32_t* _zwc =                                                       \
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);         \
        uint32_t _zt0 = _zwc[kernel_profiler::WALL_CLOCK_LOW_INDEX];                               \
        while ((uint32_t)(_zwc[kernel_profiler::WALL_CLOCK_LOW_INDEX] - _zt0) < (uint32_t)(CYC)) { \
            asm volatile("nop");                                                                   \
        }                                                                                          \
    }

void kernel_main() {
    // Durations span ~1..100 us (typical ~10 us). CYC = us * 2500 (see ZONE calibration note above).
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
        ZONE(ZTAG "_Zone0", 2500u);    // ~1 us
        ZONE(ZTAG "_Zone1", 5000u);    // ~2 us
        ZONE(ZTAG "_Zone2", 7500u);    // ~3 us
        ZONE(ZTAG "_Zone3", 12500u);   // ~5 us
        ZONE(ZTAG "_Zone4", 20000u);   // ~8 us
        ZONE(ZTAG "_Zone5", 30000u);   // ~12 us
        ZONE(ZTAG "_Zone6", 50000u);   // ~20 us
        ZONE(ZTAG "_Zone7", 100000u);  // ~40 us
        ZONE(ZTAG "_Zone8", 175000u);  // ~70 us
        ZONE(ZTAG "_Zone9", 250000u);  // ~100 us
    }
}
