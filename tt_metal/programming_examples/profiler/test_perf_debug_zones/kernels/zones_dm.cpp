// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: data-movement RISCs (BRISC = RISCV_0, NCRISC = RISCV_1). Each iteration
// enters 10 DIFFERENTLY-NAMED DeviceZoneScopedN scopes with INCREASING durations, so the perf-debug (X280)
// profiler captures a variety of named zones across all RISCs. The name carries a per-RISC tag (BR_/NC_)
// so each RISC's 10 zones are distinct. N_ITERS controls how many times the 10-zone sweep repeats.
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"

#ifndef N_ITERS
#define N_ITERS 50u
#endif

// ZONE_CYC: UNIFORM per-zone spin count, for producer-rate (knee) sweeps. 0 = keep the graduated
// ~1..100 us table below, which is what you want for a representative Tracy capture. A non-zero value
// makes every zone the same length, which is what you want for a knee: the marker rate per lane becomes
// a single number (2 markers per zone, so ~2 * aiclk / ZONE_CYC markers/s), and every lane runs at that
// same rate. Graduated durations would smear the knee, because the instantaneous aggregate rate would
// swing through the sweep of durations instead of holding at the peak.
#ifndef ZONE_CYC
#define ZONE_CYC 0u
#endif

#if ZONE_CYC
#define ZCYC(graduated) ((uint32_t)(ZONE_CYC))
#else
#define ZCYC(graduated) ((uint32_t)(graduated))
#endif

#if defined(COMPILE_FOR_BRISC)
#define ZTAG "BR"
#else
#define ZTAG "NC"
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
        ZONE(ZTAG "_Zone0", ZCYC(2500u));    // ~1 us
        ZONE(ZTAG "_Zone1", ZCYC(5000u));    // ~2 us
        ZONE(ZTAG "_Zone2", ZCYC(7500u));    // ~3 us
        ZONE(ZTAG "_Zone3", ZCYC(12500u));   // ~5 us
        ZONE(ZTAG "_Zone4", ZCYC(20000u));   // ~8 us
        ZONE(ZTAG "_Zone5", ZCYC(30000u));   // ~12 us
        ZONE(ZTAG "_Zone6", ZCYC(50000u));   // ~20 us
        ZONE(ZTAG "_Zone7", ZCYC(100000u));  // ~40 us
        ZONE(ZTAG "_Zone8", ZCYC(175000u));  // ~70 us
        ZONE(ZTAG "_Zone9", ZCYC(250000u));  // ~100 us
    }
}
