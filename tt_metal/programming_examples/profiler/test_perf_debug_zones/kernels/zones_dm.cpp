// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: data-movement RISCs (BRISC = RISCV_0, NCRISC = RISCV_1). Each iteration
// enters 10 DIFFERENTLY-NAMED DeviceZoneScopedN scopes with INCREASING durations, so the perf-debug (drainer)
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
// ZONE_MODE selects the zone body; ZONE_CYC is the nop-iteration count used when ZONE_MODE == 1.
// These are SEPARATE on purpose: ZONE_CYC == 0 is a legitimate knee point (max rate, no spin at all), the
// same as the standalone drain harness --proddelay 0, so it must not double as "use the graduated table".
#ifndef ZONE_MODE
#define ZONE_MODE 0  // 0 = graduated wall-clock durations, 1 = uniform nop spin (knee sweeps)
#endif
#ifndef ZONE_CYC
#define ZONE_CYC 0u
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
// Self-timed DeviceZoneScopedN overhead. Device-timed brackets (zones, so the durations ride the
// normal pipeline -- DPRINT and the profiler are mutually exclusive): N_ITERS iterations of a nop spin
// alone (STBASE), then the same spin with one zone per iteration (STTOT); per-zone overhead =
// (STTOT - STBASE) / N. STCLK prices the clock-read primitive; STOLD and STDRAM are faithful clones
// of the pre-atomic split emit and the DRAM-push backend's emit, as same-capture reference points.
// The spin (ZONE_CYC, --delay) appears in EVERY bracket so it cancels; its job is to hold the zone
// rate below the ring-stall knee so no stall time lands inside the brackets.
#define SELFTIME_SPIN()                                             \
    for (volatile uint32_t _j = 0; _j < (uint32_t)ZONE_CYC; _j++) { \
        asm volatile("nop");                                        \
    }

namespace stv {
static uint32_t g_scratch_w = 0;
using namespace kernel_profiler;
inline __attribute__((always_inline)) void dram_marker(uint32_t timer_id) {
    volatile uint32_t* scr = reinterpret_cast<volatile uint32_t*>(0x80000);
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    if (g_scratch_w + 2u < 512u) {
        scr[g_scratch_w] =
            0x80000000u | ((timer_id & 0x7FFFFu) << 12) | (p_reg[kernel_profiler::WALL_CLOCK_HIGH_INDEX] & 0xFFFu);
        scr[g_scratch_w + 1] = p_reg[kernel_profiler::WALL_CLOCK_LOW_INDEX];
        g_scratch_w += 2u;
    } else {
        g_scratch_w = 0;
    }
}
// Faithful clone of the PRE-ATOMIC mark_time (Mo's shipped baseline): per-marker room check on a
// DIRECT head load, retry-loop clock read, sticky check, 2 ring words, per-marker fence + TAIL.
// Writes real split markers, so the host pairing path decodes them like any legacy pair.
inline __attribute__((always_inline)) void old_mark(uint32_t timer_id, bool is_end) {
    using namespace kernel_profiler;
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_CAPACITY - 3u)) {
    }
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t hi, lo;
    do {
        hi = p_reg[WALL_CLOCK_HIGH_INDEX];
        lo = p_reg[WALL_CLOCK_LOW_INDEX];
    } while (hi != p_reg[WALL_CLOCK_HIGH_INDEX]);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::zone_w0(timer_id, is_end ? ZoneKind::End : ZoneKind::Start));
    ring_write_word(lo);
    publish_tail();
}
}  // namespace stv
struct ZOldSplit {
    __attribute__((always_inline)) ZOldSplit() { stv::old_mark(0x333u, false); }
    __attribute__((always_inline)) ~ZOldSplit() { stv::old_mark(0x333u, true); }
};
struct ZDramClone {
    __attribute__((always_inline)) ZDramClone() { stv::dram_marker(0x111u); }
    __attribute__((always_inline)) ~ZDramClone() { stv::dram_marker(0x222u); }
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
    {
        DeviceZoneScopedN(ZTAG "_STCLK");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            uint32_t h, l;
            kernel_profiler::read_wall_clock(h, l);
            asm volatile("" ::"r"(h), "r"(l));
            SELFTIME_SPIN();
        }
    }
    {  // faithful clone of the PRE-ATOMIC split emit (Mo's shipped baseline)
        DeviceZoneScopedN(ZTAG "_STOLD");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            ZOldSplit z{};
            SELFTIME_SPIN();
        }
    }
    {  // faithful clone of the DRAM-push backend's zone (2 markers), into scratch L1
        DeviceZoneScopedN(ZTAG "_STDRAM");
        for (uint32_t i = 0; i < (uint32_t)N_ITERS; i++) {
            ZDramClone z{};
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
