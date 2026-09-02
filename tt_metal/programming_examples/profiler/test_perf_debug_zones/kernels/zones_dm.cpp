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
// _zwc points AT the wall-clock LOW register (RISCV_DEBUG_REG_WALL_CLOCK_L), so the low word is index 0.
// Deliberately NOT kernel_profiler::WALL_CLOCK_LOW_INDEX: that constant is declared inside the
// `#if defined(PROFILE_KERNEL) && ...` block of kernel_profiler.hpp, i.e. it is internal state of the
// STREAMING backend and does not exist when the profiler is compiled out. Referencing it made this kernel
// fail to COMPILE in the unprofiled build ("'WALL_CLOCK_LOW_INDEX' is not a member of 'kernel_profiler'")
// even though DeviceZoneScopedN() expands to nothing there -- and that build failure looks exactly like a
// card wedge from the host side. The spin must work with or without the profiler compiled in.
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

// EMPTY body (ZONE_MODE == 3): the pure-overhead microbenchmark. Ten fully unrolled back-to-back empty
// zones per iteration, nothing between them, so the profiler measures ITSELF: each zone's recorded
// DURATION = open's clock read -> close's clock read with an empty body (the in-zone overhead: the
// close's ring room check + the wall-clock read), and the GAP between one zone's end and the next
// zone's start = the close's post-clock work (sticky check + 3 ring stores + publish) plus the next
// open's clock read. duration + gap = the full cost one zone adds at max rate. The host workload
// (--empty 1) registers a consumer that computes exactly those two numbers per RISC. (--bench, mode 2,
// is the DEVICE-side twin: same empty scopes, but timed by the kernel itself against the wall clock --
// mode 3 prices what the CAPTURE observes, mode 2 what the PRODUCER pays.)
#define ZONE_EMPTY(NAME)         \
    {                            \
        DeviceZoneScopedN(NAME); \
    }

// PRICE-CLOCK body (ZONE_MODE == 4): the EMPTY zone plus ONE extra latched wall-clock read pair in
// the body. duration(mode 4) - duration(mode 3) = the cost of one read_wall_clock on this RISC.
#define ZONE_PRICE_CLOCK(NAME)                                                             \
    {                                                                                      \
        DeviceZoneScopedN(NAME);                                                           \
        volatile tt_reg_ptr uint32_t* _pwc =                                               \
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); \
        uint32_t _plo = _pwc[0];                                                           \
        uint32_t _phi = _pwc[1];                                                           \
        asm volatile("" ::"r"(_plo), "r"(_phi));                                           \
    }

// GRADUATED (ZONE_MODE == 0) keeps the wall-clock spin (ZONE_WALL above): its point is durations calibrated in
// microseconds for a representative capture, which a nop-iteration count cannot express.
#if ZONE_MODE == 4
#define ZONE(NAME, GRADUATED) ZONE_PRICE_CLOCK(NAME)
#elif ZONE_MODE == 3
#define ZONE(NAME, GRADUATED) ZONE_EMPTY(NAME)
#elif ZONE_MODE
#define ZONE(NAME, GRADUATED) ZONE_NOPS(NAME, ZONE_CYC)
#else
#define ZONE(NAME, GRADUATED) ZONE_WALL(NAME, GRADUATED)
#endif

// ZONE_MODE == 2: DEDICATED MICROBENCH of DeviceZoneScopedN itself. Times bursts of empty scopes with
// the wall clock on-device, so nothing host-side, no nop padding and no drain wall is in the number.
// kBurst * 3 words stays under the 512-word ring so a burst never blocks on ring room; the drainer
// empties it between bursts, which is also why the burst is timed rather than the whole loop.
// ZONE_MODE == 2: DEDICATED MICROBENCH of DeviceZoneScopedN. The kernel does NOTHING but enter and
// leave empty scopes, so the host's launch->done wall divided by the zone count IS the per-zone cost:
// no nop padding, no point markers, no other work. Reported by the host (DPRINT cannot be used -- it and
// the profiler are mutually exclusive). Bursts of kBurst keep 3*kBurst words under the 512-word ring so
// the producer never blocks on ring room; the drainer empties it between bursts.
// ZONE_MODE == 2: DEDICATED MICROBENCH of DeviceZoneScopedN. The kernel does nothing but enter and leave
// empty scopes; it times every burst with the wall clock and leaves the totals in L1 for the host to read
// (DPRINT cannot be used -- it and the profiler are mutually exclusive). kBurst*3 words stays under the
// 512-word ring so a burst never blocks on ring room, and the drainer empties it between bursts.
#if ZONE_MODE == 2
void kernel_main() {
    volatile tt_reg_ptr uint32_t* wc = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t kSlot = 0;
#else
    constexpr uint32_t kSlot = 1;
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
    // Durations span ~1..100 us (typical ~10 us). CYC = us * 2500 (see ZONE calibration note above).
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
// The marker trio is OPT-IN (--markers 1 -> EMIT_MARKERS=1): it exists to exercise every point-marker
// shape on the wire (PP_EVENT has no other emitter in the tree), but it costs ~21% of wire volume /
// ~45% of onset delay, so knee sweeps and onset captures want a pure zone stream. EMPTY overhead mode
// (ZONE_MODE >= 3) never emits it regardless.
#if defined(EMIT_MARKERS) && EMIT_MARKERS && ZONE_MODE < 3
        DeviceFlag(ZTAG "_Flag");
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
