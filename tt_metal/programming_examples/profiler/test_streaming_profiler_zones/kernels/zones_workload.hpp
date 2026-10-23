// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming-profiler workload zones_dm.cpp and zones_compute.cpp build, after defining ZTAG, the per-RISC tag each
// zone's name carries, and kBenchSlot, the RISC's slot of the bench totals. Each iteration enters 10 differently-named
// DeviceZoneScopedN scopes with increasing durations, and N_ITERS repeats the sweep.
#pragma once

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"

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

// Opt-in (--external-every N) on the data-movement RISCs: every N iterations, a counters snapshot for the external
// process.
#if defined(EXTERNAL_EVERY) && EXTERNAL_EVERY && !defined(COMPILE_FOR_TRISC)
#include "api/debug/device_publish.h"
struct ExternalCounters {
    uint32_t iteration;
    uint32_t counters[7];
};
struct ExternalName {
    tt::tt_metal::experimental::streaming_profiler::String name;
};
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
#include "api/debug/device_publish.h"

// The structs --bench 7 to 12 publish.
struct Name {
    tt::tt_metal::experimental::streaming_profiler::String name;
};
struct CoreXY {
    uint16_t x, y;
};
struct NocEvent : CoreXY {
    uint32_t bytes;
};
struct LaneFlags {
    uint32_t kind : 3;
    uint32_t lane : 5;
    uint32_t count : 24;
};
struct TightBits {
    uint32_t x;
    uint8_t a;
    uint16_t b : 8;
};
struct Padded {
    uint8_t a;
    uint32_t b;
};

void kernel_main() {
    volatile tt_reg_ptr uint32_t* wc = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(BENCH_ADDR) + kBenchSlot * 2u;
    // BENCH_KIND as in test_streaming_profiler_zones.cpp. Kinds 3 and 7 to 10 write records of up to 6 words (the
    // timer's high word when it changes, a header, the time's low word and up to three payload words), so 64 of them
    // stay under the 512-word ring.
    constexpr uint32_t kBurst = BENCH_KIND == 3 || BENCH_KIND >= 7 ? 64 : 100;
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
#elif BENCH_KIND == 6 && !defined(COMPILE_FOR_TRISC)
            DeviceZoneSetCounter(i + 1);  // the launch path's runtime-id sticky and publish
#elif BENCH_KIND == 7
            DevicePublish<Name{"plain"_s}>(NocEvent{{uint16_t(i), uint16_t(i >> 1)}, i});
#elif BENCH_KIND == 8
            DevicePublish<Name{"bits"_s}>(LaneFlags{i & 7, i & 31, i});
#elif BENCH_KIND == 9
            DevicePublish<Name{"tight"_s}>(TightBits{i, uint8_t(i), uint16_t(i >> 1)});
#elif BENCH_KIND == 10
            DevicePublish<Name{"padded"_s}>(i, Padded{uint8_t(i), i});
#elif BENCH_KIND == 11
            DevicePublish<Name{"routed"_s}, tt::tt_metal::experimental::streaming_profiler::kSendToExternalProcess>(
                NocEvent{{uint16_t(i), uint16_t(i >> 1)}, i});
#elif BENCH_KIND == 12
            DevicePublish<Name{"immediate"_s}, tt::tt_metal::experimental::streaming_profiler::kPublishImmediately>(
                NocEvent{{uint16_t(i), uint16_t(i >> 1)}, i});
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
// Opt-in (--markers 1) on the data-movement RISCs: exercises every point-marker shape but adds wire volume a rate sweep
// does not want.
#if defined(EMIT_MARKERS) && EMIT_MARKERS && !defined(COMPILE_FOR_TRISC)
        DeviceRecordEvent(ZTAG "_Event");
        DeviceTimestampedData(ZTAG "_Data", ((uint64_t)0xF00D << 32) | it);
        DeviceTimestampedData(ZTAG "_Iter", it);
#endif
#if defined(EXTERNAL_EVERY) && EXTERNAL_EVERY && !defined(COMPILE_FOR_TRISC)
        if (it % EXTERNAL_EVERY == EXTERNAL_PHASE) {
#if EXTERNAL_TO_METAL
            // The control: the same records, published as immediately, that stay in this process.
            DevicePublish<
                ExternalName{"zones counters"_s},
                tt::tt_metal::experimental::streaming_profiler::kPublishImmediately>(
                ExternalCounters{it, {it, it + 1, it + 2, it + 3, it + 4, it + 5, it + 6}});
#else
            DevicePublish<
                ExternalName{"zones counters"_s},
                tt::tt_metal::experimental::streaming_profiler::kSendToExternalProcess>(
                ExternalCounters{it, {it, it + 1, it + 2, it + 3, it + 4, it + 5, it + 6}});
#endif
        }
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
