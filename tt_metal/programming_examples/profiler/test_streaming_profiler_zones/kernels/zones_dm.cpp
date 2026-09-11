// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streaming-profiler workload for the data-movement RISCs (BRISC = RISCV_0, NCRISC = RISCV_1). Each
// iteration enters 10 differently-named DeviceZoneScopedN scopes with increasing durations. The name carries
// a per-RISC tag (BR_/NC_) so each RISC's 10 zones are distinct, and N_ITERS repeats the sweep.
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"

#ifndef N_ITERS
#define N_ITERS 50u
#endif

// ZONE_CYC == 0 is a legitimate rate point (max rate, no spin), so it cannot double as "use the graduated
// table"; ZONE_MODE selects the body instead.
#ifndef ZONE_MODE
#define ZONE_MODE 0  // 0 = graduated wall-clock durations, 1 = uniform nop spin (knee sweeps), 2 = marker-cost bench
#endif
#ifndef ZONE_CYC
#define ZONE_CYC 0u
#endif

#if defined(COMPILE_FOR_BRISC)
#define ZTAG "BR"
#else
#define ZTAG "NC"
#endif

// Body busy-waits CYC spin-counts, calibrated so the zone displays ~CYC/2500 us in Tracy (displayed_ns ~=
// CYC * 0.41 at the 1.35 GHz aiclk). Low register only with wrap-safe subtraction, tear-free for spins << 2^32.
// Not kernel_profiler::WALL_CLOCK_LOW_INDEX: it lives inside the PROFILE_KERNEL block and breaks the
// profiler-off build.
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

// `volatile` forces load/increment/store/compare per iteration, the calibrated 10 cycles.
#define ZONE_NOPS(NAME, ITERS)                                            \
    {                                                                     \
        DeviceZoneScopedN(NAME);                                          \
        for (volatile uint32_t _zj = 0; _zj < (uint32_t)(ITERS); _zj++) { \
            asm volatile("nop");                                          \
        }                                                                 \
    }

// ZONE_MODE == 0 keeps the wall-clock spin: it needs durations calibrated in microseconds, which a
// nop-iteration count cannot express.
#if ZONE_MODE
#define ZONE(NAME, GRADUATED) ZONE_NOPS(NAME, ZONE_CYC)
#else
#define ZONE(NAME, GRADUATED) ZONE_WALL(NAME, GRADUATED)
#endif

// Marker-cost microbench (--bench): bursts of one marker kind, each burst timed against the wall clock, the totals
// left in L1 at BENCH_ADDR (slot per RISC; DPRINT and the profiler are mutually exclusive). A burst stays under the
// 512-word ring so it never blocks, and BENCH_DELAY paces it so the relay keeps up: with a stall anywhere the
// number is the stall, not the marker.
#if ZONE_MODE == 2
void kernel_main() {
    volatile tt_reg_ptr uint32_t* wc = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t kSlot = 0;
#else
    constexpr uint32_t kSlot = 1;
#endif
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(BENCH_ADDR) + kSlot * 2u;
    // BENCH_KIND 0 = spin only, 1 = empty zone (3 words), 2 = DeviceRecordEvent (3 words), 3 = DeviceTimestampedData
    // (6 words), 4 / 5 = DeviceZoneScopedNIf with an opaque runtime true / false, 6 = DeviceZoneSetCounter (the
    // launch path's STICKY_PROG + publish); the burst stays under the 512-word ring so it never blocks.
    constexpr uint32_t kBurst = BENCH_KIND == 3 ? 64 : 100;
    uint32_t cycles = 0, markers = 0;
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
        const uint32_t t0 = wc[kWallClockLowIdx];
        for (uint32_t i = 0; i < kBurst; i++) {
#if BENCH_KIND == 3
            DeviceTimestampedData(ZTAG "_BENCH", (uint64_t)i);
#elif BENCH_KIND == 2
            DeviceRecordEvent(ZTAG "_BENCH");
#elif BENCH_KIND == 1
            DeviceZoneScopedN(ZTAG "_BENCH");
#elif BENCH_KIND == 4 || BENCH_KIND == 5
            uint32_t on = BENCH_KIND == 4;
            asm volatile("" : "+r"(on));
            DeviceZoneScopedNIf(ZTAG "_BENCH", on != 0);
#elif BENCH_KIND == 6
            DeviceZoneSetCounter(i + 1);  // the launch path's runtime-id sticky and publish
#endif
            // BENCH_DELAY nop iterations keep the ring from filling, so the marker's own cost is what is timed;
            // kind 0 (no marker) prices the loop and spin for subtraction.
            for (uint32_t d = 0; d < (uint32_t)BENCH_DELAY; d++) {
                asm volatile("nop");
            }
        }
        cycles += (uint32_t)(wc[kWallClockLowIdx] - t0);
        markers += kBurst;
    }
    out[0] = cycles;
    out[1] = markers;
}
#else
void kernel_main() {
    // Durations span ~1..100 us. CYC = us * 2500, per the ZONE_WALL calibration above.
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
// Opt-in (--markers 1): exercises every point-marker shape but adds wire volume a rate sweep does not want.
#if defined(EMIT_MARKERS) && EMIT_MARKERS
        DeviceRecordEvent(ZTAG "_Event");
        DeviceTimestampedData(ZTAG "_Data", ((uint64_t)0xF00D << 32) | it);
        DeviceTimestampedData(ZTAG "_Iter", it);
#endif
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
