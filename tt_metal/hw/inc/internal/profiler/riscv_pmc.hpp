// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISC-V hardware performance monitor of the Tensix baby cores: BRISC, NCRISC and TRISC0/1/2 on
// Blackhole, TRISC0-3 of each Neo on Quasar.
//
// Each of these cores has a free-running mcycle and minstret plus two programmable counters,
// mhpmcounter3 and mhpmcounter4, steered by mhpmevent3/4. This is unrelated to the Tensix MMIO
// perf-counter banks in perf_counters.hpp; see tech_reports/PerfCounters/riscv-pmc.md for the CSR
// map, the event catalog and the support matrix.
//
// An mhpmevent value is eight 4-bit nibbles, one per pipeline domain (BP, IF, DE, DE1, EX, LS,
// TTCSR, MEM low-to-high). Each nibble selects one event index from that domain's 16-bit event bus.
// Index 0 in every domain is a tie-off, so a zero nibble disables the domain. The selected events
// are OR-ed before the increment, so a counter advances by at most 1 per cycle.
//
// Firmware brackets the kernel invocation with RiscvPmcBegin() / RiscvPmcEnd(). Counters are read
// before and after and the deltas emitted as three profiler markers per core per launch.
//
// Enable with TT_METAL_PROFILE_RISCV_PMC=<mhpmevent3>[,<mhpmevent4>]. The selects are compile-time
// constants and are part of the firmware build key, so changing them rebuilds the firmware.

#pragma once

// 9090 and 9091 belong to the Tensix MMIO perf counters. Each value goes in the plain TS_DATA field so
// every row reaches profile_log_device.csv; a TS_DATA_16B trailer would not.
constexpr uint16_t RISCV_PMC_COUNTS_PROFILER_ID = 9092;  // (pmc4 << 32) | pmc3
constexpr uint16_t RISCV_PMC_CYCLES_PROFILER_ID = 9093;  // (minstret << 32) | mcycle
constexpr uint16_t RISCV_PMC_EVENTS_PROFILER_ID = 9094;  // (mhpmevent4 << 32) | mhpmevent3

// Cores that carry the counters and have a firmware hook.
#if defined(ARCH_BLACKHOLE) && (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_TRISC))
#define RISCV_PMC_CORE 1
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
#define RISCV_PMC_CORE 1
#endif

// Included directly by the firmware TUs: the kernel_profiler.hpp includes sit inside its
// PROFILE_KERNEL block, so the empty macros below would be missing on a profiler-off build.
// The streaming profiler has no TS_DATA path.
#if defined(PROFILE_RISCV_PMC) && defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING) && defined(FW_BUILD) && \
    defined(RISCV_PMC_CORE)

#include "api/debug/kernel_profiler.hpp"

#if !defined(PROFILE_RISCV_PMC_EV3)
#define PROFILE_RISCV_PMC_EV3 0
#endif
#if !defined(PROFILE_RISCV_PMC_EV4)
#define PROFILE_RISCV_PMC_EV4 0
#endif

namespace kernel_profiler {
namespace riscv_pmc {

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

// One launch in flight per core, so a plain global is enough. Per-thread on Quasar, where the four
// TRISCs of a Neo share the image.
#if defined(ARCH_QUASAR)
static thread_local Snapshot snap_begin;
#else
static Snapshot snap_begin;
#endif

// mcountinhibit is never written: its reset value runs both counters. Deltas are used instead of
// clear-on-read so the cfg CSR that firmware programs at boot is left alone.
inline __attribute__((always_inline)) void arm() {
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT3), "r"((uint32_t)PROFILE_RISCV_PMC_EV3));
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT4), "r"((uint32_t)PROFILE_RISCV_PMC_EV4));
}

// The memory clobber keeps the reads from drifting across the kernel call.
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

// Re-emitting the selects every launch keeps each capture self-describing. 32-bit deltas cannot
// overflow within a launch. No flush_to_dram_if_full(): timeStampedData drops the markers when the
// buffer is full, and the flush would pull noc_index into firmware images that do not define it.
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
