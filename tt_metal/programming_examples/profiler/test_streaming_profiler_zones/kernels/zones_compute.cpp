// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streaming-profiler workload for the compute RISCs (TRISC0/1/2). Same 10-zone sweep as zones_dm.cpp, tagged
// per TRISC (T0_/T1_/T2_) so each compute RISC emits its own 10 distinctly-named zones.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef N_ITERS
#define N_ITERS 50u
#endif

// ZONE_MODE selects the zone body; ZONE_CYC is the nop-iteration count used when ZONE_MODE == 1. They are
// separate because ZONE_CYC == 0 is a legitimate producer-rate point (max rate, no spin at all) and so
// cannot double as "use the graduated table". Uniform durations give a single marker rate per lane
// (2 markers per zone); graduated ones would smear a rate sweep across the table of durations.
#ifndef ZONE_MODE
#define ZONE_MODE 0  // 0 = graduated wall-clock durations, 1 = uniform nop spin (knee sweeps)
#endif
#ifndef ZONE_CYC
#define ZONE_CYC 0u
#endif

#if COMPILE_FOR_TRISC == 0
#define ZTAG "T0"
#elif COMPILE_FOR_TRISC == 1
#define ZTAG "T1"
#else
#define ZTAG "T2"
#endif

// One named zone whose body busy-waits CYC wall-clock spin-counts. CYC is calibrated so the zone displays
// ~CYC/2500 us in Tracy: at the ~1.35 GHz boosted aiclk the profiler records ~0.55 timestamp tick per
// spin-count at a ~0.741 ns context period, so displayed_ns ~= CYC * 0.41. Reading the low register only,
// with unsigned-wrap subtraction, is tear-free for spins << 2^32.
// Not kernel_profiler::WALL_CLOCK_LOW_INDEX: that constant lives inside kernel_profiler.hpp's
// `#if defined(PROFILE_KERNEL) && ...` block, so referencing it breaks the build when the profiler is
// compiled out, even though DeviceZoneScopedN() expands to nothing there.
static constexpr int kWallClockLowIdx = 0;

#define ZONE_WALL(NAME, CYC)                                                               \
    {                                                                                      \
        DeviceZoneScopedN(NAME);                                                           \
        volatile tt_reg_ptr uint32_t* _zwc =                                               \
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); \
        uint32_t _zt0 = _zwc[kWallClockLowIdx];                                            \
        while ((uint32_t)(_zwc[kWallClockLowIdx] - _zt0) < (uint32_t)(CYC)) {              \
            asm volatile("nop");                                                           \
        }                                                                                  \
    }

// Uniform-spin body (ZONE_MODE == 1). The counter must stay `volatile`: that is what forces a
// load/increment/store/compare per iteration, giving the calibrated 10 cycles rather than a single nop.
#define ZONE_NOPS(NAME, ITERS)                                            \
    {                                                                     \
        DeviceZoneScopedN(NAME);                                          \
        for (volatile uint32_t _zj = 0; _zj < (uint32_t)(ITERS); _zj++) { \
            asm volatile("nop");                                          \
        }                                                                 \
    }

// Empty body (ZONE_MODE == 3): the pure-overhead microbenchmark; zones_dm.cpp documents the duration/gap
// decomposition it measures.
#define ZONE_EMPTY(NAME)         \
    {                            \
        DeviceZoneScopedN(NAME); \
    }

// ZONE_MODE == 4: the empty zone plus one extra latched wall-clock read pair in the body, so
// duration(mode 4) - duration(mode 3) is the cost of one wall-clock read on this RISC.
#define ZONE_PRICE_CLOCK(NAME)                                                             \
    {                                                                                      \
        DeviceZoneScopedN(NAME);                                                           \
        volatile tt_reg_ptr uint32_t* _pwc =                                               \
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); \
        uint32_t _plo = _pwc[0];                                                           \
        uint32_t _phi = _pwc[1];                                                           \
        asm volatile("" ::"r"(_plo), "r"(_phi));                                           \
    }

// ZONE_MODE == 0 keeps the wall-clock spin: it needs durations calibrated in microseconds, which a
// nop-iteration count cannot express.
#if ZONE_MODE == 4
#define ZONE(NAME, GRADUATED) ZONE_PRICE_CLOCK(NAME)
#elif ZONE_MODE == 3
#define ZONE(NAME, GRADUATED) ZONE_EMPTY(NAME)
#elif ZONE_MODE
#define ZONE(NAME, GRADUATED) ZONE_NOPS(NAME, ZONE_CYC)
#else
#define ZONE(NAME, GRADUATED) ZONE_WALL(NAME, GRADUATED)
#endif

// ZONE_MODE == 2: the DeviceZoneScopedN microbench, same shape as zones_dm.cpp. Slots 2..4 of BENCH_ADDR, so
// all five lanes of a core report side by side.
#if ZONE_MODE == 2
void kernel_main() {
    volatile tt_reg_ptr uint32_t* wc = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    constexpr uint32_t kSlot = 2;
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1
    constexpr uint32_t kSlot = 3;
#else
    constexpr uint32_t kSlot = 4;
#endif
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(BENCH_ADDR) + kSlot * 2u;
    constexpr uint32_t kBurst = 100;
    uint32_t cycles = 0, zones = 0;
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
        const uint32_t t0 = wc[kWallClockLowIdx];
        for (uint32_t i = 0; i < kBurst; i++) {
            DeviceZoneScopedN(ZTAG "_BENCH");
        }
        cycles += (uint32_t)(wc[kWallClockLowIdx] - t0);
        zones += kBurst;
    }
    out[0] = cycles;
    out[1] = zones;
}
#else
void kernel_main() {
    // Durations span ~1..100 us. CYC = us * 2500, per the ZONE_WALL calibration above.
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
#endif  // ZONE_MODE == 2
