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

// ZONE_CYC: UNIFORM per-zone spin count, for producer-rate (knee) sweeps. 0 = keep the graduated
// ~1..100 us table below, which is what you want for a representative Tracy capture. A non-zero value
// makes every zone the same length, which is what you want for a knee: the marker rate per lane becomes
// a single number (2 markers per zone, so ~2 * aiclk / ZONE_CYC markers/s), and every lane runs at that
// same rate. Graduated durations would smear the knee, because the instantaneous aggregate rate would
// swing through the sweep of durations instead of holding at the peak.
// ZONE_MODE selects the zone body; ZONE_CYC is the nop-iteration count used when ZONE_MODE == 1.
// These are SEPARATE on purpose: ZONE_CYC == 0 is a legitimate knee point (max rate, no spin at all), the
// same as the standalone drain harness --proddelay 0, so it must not double as "use the graduated table".
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

// One named zone whose body busy-waits CYC wall-clock spin-counts. CYC is EMPIRICALLY calibrated so the
// zone displays ~CYC/2500 us in Tracy: at the ~1.35 GHz boosted aiclk the profiler records ~0.55 timestamp
// tick per spin-count and the context period is ~0.741 ns/tick, so displayed_ns ~= CYC * 0.41. LOW-register-
// only read with unsigned-wrap subtraction is tear-free for spins << 2^32.
#define ZONE_WALL(NAME, CYC)                                                                       \
    {                                                                                              \
        DeviceZoneScopedN(NAME);                                                                   \
        volatile tt_reg_ptr uint32_t* _zwc =                                                       \
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);         \
        uint32_t _zt0 = _zwc[kernel_profiler::WALL_CLOCK_LOW_INDEX];                               \
        while ((uint32_t)(_zwc[kernel_profiler::WALL_CLOCK_LOW_INDEX] - _zt0) < (uint32_t)(CYC)) { \
            asm volatile("nop");                                                                   \
        }                                                                                          \
    }

// KNEE body (ZONE_MODE == 1): byte-identical to the standalone drain harness's producer loop (realprof_dm.cpp,
// WORK_SIZE), so ZONE_CYC and that test's --proddelay are the SAME UNIT and the two knees are directly
// comparable. The counter MUST stay `volatile`: that is what forces a load/increment/store/compare per
// iteration, so one iteration costs several cycles rather than a single nop. Do NOT turn this into a
// wall-clock spin -- that is exactly what made ZONE_CYC 30000 produce 22.2 us zones (30000 / 1.35 GHz)
// while --proddelay 950 produces ~5-6 us, i.e. two knees on different axes that looked like a regression.
#define ZONE_NOPS(NAME, ITERS)                                            \
    {                                                                     \
        DeviceZoneScopedN(NAME);                                          \
        for (volatile uint32_t _zj = 0; _zj < (uint32_t)(ITERS); _zj++) { \
            asm volatile("nop");                                          \
        }                                                                 \
    }

// GRADUATED (ZONE_MODE == 0) keeps the wall-clock spin (ZONE_WALL above): its point is durations calibrated in
// microseconds for a representative capture, which a nop-iteration count cannot express.
#if ZONE_MODE
#define ZONE(NAME, GRADUATED) ZONE_NOPS(NAME, ZONE_CYC)
#else
#define ZONE(NAME, GRADUATED) ZONE_WALL(NAME, GRADUATED)
#endif

// ZONE_SCALE stretches every graduated duration (host --scale): more wall time per zone WITHOUT more
// zones, which is what a live-GUI viewing session needs (connect time) and a knee sweep must not have.
#ifndef ZONE_SCALE
#define ZONE_SCALE 1u
#endif

// Bare graduated spin (no zone): a parent's own work before/after its children, so parent spans are
// not merely the sum of their children.
#define ZONE_BODY(CYC)                                                                                          \
    {                                                                                                           \
        volatile tt_reg_ptr uint32_t* _zwb =                                                                    \
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);                      \
        uint32_t _zb0 = _zwb[kernel_profiler::WALL_CLOCK_LOW_INDEX];                                            \
        while ((uint32_t)(_zwb[kernel_profiler::WALL_CLOCK_LOW_INDEX] - _zb0) < (uint32_t)(CYC) * ZONE_SCALE) { \
            asm volatile("nop");                                                                                \
        }                                                                                                       \
    }

#if defined(ZONE_SELFTIME)
// Self-timed DeviceZoneScopedN overhead. Two device-timed brackets (zones, so the durations ride the
// normal pipeline -- DPRINT and the profiler are mutually exclusive): N_ITERS iterations of a nop spin
// alone, then the same spin with one zone per iteration. Overhead per zone = (STTOT - STBASE) / N.
// The spin (ZONE_CYC, --delay) appears in BOTH brackets so it cancels; its job is to hold the zone
// rate below the ring-stall knee so no stall time lands inside the brackets.
#define SELFTIME_SPIN()                                             \
    for (volatile uint32_t _j = 0; _j < (uint32_t)ZONE_CYC; _j++) { \
        asm volatile("nop");                                        \
    }

namespace stv {
constexpr uint32_t kHash = 0x1234u;
using namespace kernel_profiler;
// Clone of the atomic close path with one line removed per variant (see brackets below).
inline __attribute__((always_inline)) void emit_variant(uint64_t start, bool do_dur, bool do_sticky, bool do_credit) {
    if (do_credit) {
        ring_ensure_room(4);
    }
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    uint32_t d32 = 7u;
    if (do_dur) {
        const uint64_t d = ((static_cast<uint64_t>(hi) << 32) | lo) - start;
        if (d >> 32) {
            d32 = 0xFFFFFFFEu;
        } else {
            d32 = static_cast<uint32_t>(d);
        }
    }
    if (do_sticky && hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::w0(ppfmt::T_ZONE_ATOMIC, kHash));
    ring_write_word(lo);
    ring_write_word(d32);
    if (wIndex - g_last_pub >= kPublishBatchWords) {
        publish_tail();
        g_last_pub = wIndex;
    }
}
inline __attribute__((always_inline)) uint64_t rd64() {
    uint32_t hi, lo;
    kernel_profiler::read_wall_clock(hi, lo);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
}  // namespace stv
struct ZCtorOnly {
    uint64_t s;
    __attribute__((always_inline)) ZCtorOnly() { s = stv::rd64(); }
    __attribute__((always_inline)) ~ZCtorOnly() {
        uint32_t hi, lo;
        kernel_profiler::read_wall_clock(hi, lo);
        asm volatile("" ::"r"((uint32_t)s), "r"(hi), "r"(lo));
    }
};
struct ZNoDur {
    uint64_t s;
    __attribute__((always_inline)) ZNoDur() { s = stv::rd64(); }
    __attribute__((always_inline)) ~ZNoDur() { stv::emit_variant(s, false, true, true); }
};
struct ZNoSticky {
    uint64_t s;
    __attribute__((always_inline)) ZNoSticky() { s = stv::rd64(); }
    __attribute__((always_inline)) ~ZNoSticky() { stv::emit_variant(s, true, false, true); }
};
struct ZNoCredit {
    uint64_t s;
    __attribute__((always_inline)) ZNoCredit() { s = stv::rd64(); }
    __attribute__((always_inline)) ~ZNoCredit() { stv::emit_variant(s, true, true, false); }
};

void kernel_main() {
    {
        DeviceZoneScopedN(ZTAG "_STBASE");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            SELFTIME_SPIN();
        }
    }
    {
        DeviceZoneScopedN(ZTAG "_STTOT");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            DeviceZoneScopedN(ZTAG "_S");
            SELFTIME_SPIN();
        }
    }
    // Component brackets, same spin in each so (X - STBASE) / N prices one piece. The variant emits
    // below are clones of mark_zone_atomic each missing exactly ONE line, so that line's cost is
    // (STTOT - variant). All variants still write wire-valid records and keep the batched publish
    // (the drainer must keep flowing).
    {
        DeviceZoneScopedN(ZTAG "_STCLK");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            uint32_t h, l;
            kernel_profiler::read_wall_clock(h, l);
            asm volatile("" ::"r"(h), "r"(l));
            SELFTIME_SPIN();
        }
    }
    {
        DeviceZoneScopedN(ZTAG "_STFEN");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            asm volatile("fence" ::: "memory");
            SELFTIME_SPIN();
        }
    }
    {  // scope machinery only: both clock reads + held start, NO emit
        DeviceZoneScopedN(ZTAG "_STCTOR");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            ZCtorOnly z{};
            SELFTIME_SPIN();
        }
    }
    {  // real emit minus the 64-bit dur math (constant dur)
        DeviceZoneScopedN(ZTAG "_STNODUR");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            ZNoDur z{};
            SELFTIME_SPIN();
        }
    }
    {  // real emit minus the sticky-timer check
        DeviceZoneScopedN(ZTAG "_STNOSTK");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            ZNoSticky z{};
            SELFTIME_SPIN();
        }
    }
    {  // real emit minus the room/credit check
        DeviceZoneScopedN(ZTAG "_STNOCRD");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            ZNoCredit z{};
            SELFTIME_SPIN();
        }
    }
}
#else
void kernel_main() {
#if ZONE_MODE
    // KNEE: 10 FLAT sequential zones. This shape is load-bearing: the marker-wire GB/s numbers and the
    // knee sweeps were all measured against a pure adjacent START/END train -- do not nest it.
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
        ZONE(ZTAG "_Zone0", 2500u);
        ZONE(ZTAG "_Zone1", 5000u);
        ZONE(ZTAG "_Zone2", 7500u);
        ZONE(ZTAG "_Zone3", 12500u);
        ZONE(ZTAG "_Zone4", 20000u);
        ZONE(ZTAG "_Zone5", 30000u);
        ZONE(ZTAG "_Zone6", 50000u);
        ZONE(ZTAG "_Zone7", 100000u);
        ZONE(ZTAG "_Zone8", 175000u);
        ZONE(ZTAG "_Zone9", 250000u);
    }
#else
    // GRADUATED: the representative capture, so it nests the way real kernels do. Still exactly 10
    // zones per iteration (the host prints that). Under the KERNEL wrapper:
    //   Outer { Prep, Pipe { Load, Math { Inner }, Store }, Post }, Solo, Tail
    // -- 4 nesting levels, siblings at several depths, parents with their own trailing work, and two
    // flat zones between nests. Durations ~1..20 us x ZONE_SCALE.
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
        {
            DeviceZoneScopedN(ZTAG "_Outer");
            ZONE(ZTAG "_Prep", 5000u * ZONE_SCALE);
            {
                DeviceZoneScopedN(ZTAG "_Pipe");
                ZONE(ZTAG "_Load", 12500u * ZONE_SCALE);
                {
                    DeviceZoneScopedN(ZTAG "_Math");
                    ZONE(ZTAG "_Inner", 20000u * ZONE_SCALE);
                    ZONE_BODY(12500u);
                }
                ZONE(ZTAG "_Store", 7500u * ZONE_SCALE);
            }
            ZONE(ZTAG "_Post", 2500u * ZONE_SCALE);
        }
        ZONE(ZTAG "_Solo", 30000u * ZONE_SCALE);
        ZONE(ZTAG "_Tail", 50000u * ZONE_SCALE);
    }
#endif
}
#endif
