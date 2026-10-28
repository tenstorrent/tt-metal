// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISC-V hardware performance monitor for the Tensix baby cores (BRISC, NCRISC, TRISC0/1/2).
//
// Each baby core carries its own RISC-V HPM inside the core: a free-running mcycle and minstret
// plus two programmable counters, mhpmcounter3 and mhpmcounter4, steered by mhpmevent3/4. This is
// unrelated to the Tensix MMIO perf-counter banks in perf_counters.hpp; see
// tech_reports/PerfCounters/riscv-pmc.md for the CSR map, the full event catalog and the interpretation traps.
//
// An mhpmevent value is eight 4-bit nibbles, one per pipeline domain (BP, IF, DE, DE1, EX, LS,
// TTCSR, MEM low-to-high). Each nibble selects one event index from that domain's 16-bit event bus.
// Index 0 in every domain is a hardwired tie-off, so a zero nibble disables the domain. The eight
// selected events are OR-ed before the increment, so a counter advances by at most 1 per cycle.
//
// Usage: firmware brackets the kernel invocation with RiscvPmcBegin() / RiscvPmcEnd(). Counters are
// read before and after and the deltas emitted as three profiler markers per core per launch.
//
// Enable with TT_METAL_PROFILE_RISCV_PMC=<mhpmevent3>[,<mhpmevent4>]. The selects are baked in at
// compile time, so they are folded into RunTimeOptions::get_compile_hash_string() and therefore
// into the firmware build key: changing them forces a firmware rebuild instead of silently reusing
// the previous pass's image. That is what makes a multi-run event sweep trustworthy.

#pragma once

// Marker ids. 9090 and 9091 belong to the Tensix MMIO perf counters.
//
// Three separate TS_DATA markers rather than one wider TS_DATA_16B packet, deliberately. A
// TS_DATA_16B trailer lands in TTDeviceMarker::data_high, and data_high has no column in
// profile_log_device.csv -- it only ever reaches a reader through host-side meta_data enrichment,
// which in turn only runs for markers emitted inside an open profiler zone. Keeping each value in
// the plain `data` field means every row survives into the CSV with no host-side change at all.
constexpr uint16_t RISCV_PMC_COUNTS_PROFILER_ID = 9092;  // (pmc4 << 32) | pmc3
constexpr uint16_t RISCV_PMC_CYCLES_PROFILER_ID = 9093;  // (minstret << 32) | mcycle
constexpr uint16_t RISCV_PMC_EVENTS_PROFILER_ID = 9094;  // (mhpmevent4 << 32) | mhpmevent3

// This header must be included directly by the firmware TU, not relied upon transitively through
// kernel_profiler.hpp: the includes there sit inside its `#if defined(PROFILE_KERNEL)` block, so on
// a profiler-off build they never run and the null macros below would never be defined. The streaming
// profiler has no TS_DATA path, so the hooks stay empty under it.
//
// FW_BUILD restricts the body to the firmware images. The hooks live in brisc.cc / ncrisc.cc /
// trisc.cc, so kernel builds need none of it and stay the size they were. PROFILE_KERNEL is
// required because the marker path lives in kernel_profiler.hpp; rtoptions forces the profiler on
// whenever the PMC is enabled, so this only guards against a hand-rolled define combination.
#if defined(PROFILE_RISCV_PMC) && defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING) && defined(ARCH_BLACKHOLE) && \
    defined(FW_BUILD) && (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_TRISC))

#include "api/debug/kernel_profiler.hpp"

#if !defined(PROFILE_RISCV_PMC_EV3)
#define PROFILE_RISCV_PMC_EV3 0
#endif
#if !defined(PROFILE_RISCV_PMC_EV4)
#define PROFILE_RISCV_PMC_EV4 0
#endif

namespace kernel_profiler {
namespace riscv_pmc {

// Blackhole briscv CSR numbers, from ws-tensix src/hardware/common/rtl/briscv_csr_regs.h.
constexpr uint32_t CSR_MCYCLE = 0xB00;
constexpr uint32_t CSR_MINSTRET = 0xB02;
constexpr uint32_t CSR_MHPMCOUNTER3 = 0xB03;
constexpr uint32_t CSR_MHPMCOUNTER4 = 0xB04;
constexpr uint32_t CSR_MHPMEVENT3 = 0x323;
constexpr uint32_t CSR_MHPMEVENT4 = 0x324;

struct Snapshot {
    uint32_t pmc3;
    uint32_t pmc4;
    uint32_t mcycle;
    uint32_t minstret;
};

// One live snapshot per core. Each baby core is single threaded with one launch in flight, so a
// plain global is enough and avoids any scoping constraint on where the hooks can be placed.
static Snapshot snap_begin;

// mcountinhibit is deliberately never written. Its reset value of 0 runs both counters, and in RTL
// the shared event enable tests bit 3 for both counters (tt_csr.sv:326), so writing bit 4 does not
// pause counter 4 while writing bit 3 silently freezes both. Leaving it alone sidesteps that.
//
// cfgPmcClrOnRd is likewise not used: it lives in cfg CSR 0x7C0 alongside the L1-data-cache,
// gathering and memory-ordering bits that firmware_common.h programs at boot, so touching it would
// mean composing read-modify-writes with those sequences. Deltas give the same answer for free.
inline __attribute__((always_inline)) void arm() {
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT3), "r"((uint32_t)PROFILE_RISCV_PMC_EV3));
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT4), "r"((uint32_t)PROFILE_RISCV_PMC_EV4));
}

// The memory clobber keeps these reads from drifting across the kernel call or the surrounding
// stores; without it the compiler is free to hoist a volatile asm past non-volatile memory ops.
inline __attribute__((always_inline)) Snapshot read() {
    Snapshot s;
    asm volatile("csrr %0, %1" : "=r"(s.pmc3) : "i"(CSR_MHPMCOUNTER3) : "memory");
    asm volatile("csrr %0, %1" : "=r"(s.pmc4) : "i"(CSR_MHPMCOUNTER4) : "memory");
    asm volatile("csrr %0, %1" : "=r"(s.mcycle) : "i"(CSR_MCYCLE) : "memory");
    asm volatile("csrr %0, %1" : "=r"(s.minstret) : "i"(CSR_MINSTRET) : "memory");
    return s;
}

inline __attribute__((always_inline)) void begin() {
    arm();
    snap_begin = read();
}

// Three markers per core per launch. Re-emitting the event selects every launch costs 4 words and
// makes each capture self-describing, which is what keeps a multi-run event sweep honest: a row can
// never be mis-attributed to the wrong event pair during the offline merge.
//
// 32-bit deltas are sufficient. mcycle wraps 32 bits after ~3.2 s at 1.35 GHz and a launch is
// microseconds, so a delta cannot overflow in practice.
//
// Deliberately no flush_to_dram_if_full() here. That would pull quick_push() into the reference
// graph, and quick_push() touches noc_index, which only brisc.cc defines -- on NCRISC firmware it
// is dead code that LTO drops, so calling the flush breaks the NCRISC link with an undefined
// reference to noc_index. timeStampedData already checks bufferHasRoom, so a full buffer drops
// these three markers instead of displacing zone markers, and the loader tolerates short streams.
inline __attribute__((always_inline)) void end() {
    const Snapshot t1 = read();
    const Snapshot t0 = snap_begin;

    const uint64_t counts = ((uint64_t)(t1.pmc4 - t0.pmc4) << 32) | (t1.pmc3 - t0.pmc3);
    const uint64_t timing = ((uint64_t)(t1.minstret - t0.minstret) << 32) | (t1.mcycle - t0.mcycle);
    const uint64_t events = ((uint64_t)(uint32_t)PROFILE_RISCV_PMC_EV4 << 32) | (uint32_t)PROFILE_RISCV_PMC_EV3;

    timeStampedData<RISCV_PMC_COUNTS_PROFILER_ID, DoingDispatch::DISPATCH>(counts);
    timeStampedData<RISCV_PMC_CYCLES_PROFILER_ID, DoingDispatch::DISPATCH>(timing);
    timeStampedData<RISCV_PMC_EVENTS_PROFILER_ID, DoingDispatch::DISPATCH>(events);
}

}  // namespace riscv_pmc
}  // namespace kernel_profiler

#define RiscvPmcBegin() kernel_profiler::riscv_pmc::begin()
#define RiscvPmcEnd() kernel_profiler::riscv_pmc::end()

#else

#define RiscvPmcBegin()
#define RiscvPmcEnd()

#endif
