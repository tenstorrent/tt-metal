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

// mcycle and minstret are never written, so they are read as deltas; the programmable counters are
// zeroed at begin. One launch is in flight per core, so a plain global is enough (per thread on
// Quasar, where the four TRISCs of a Neo share the image).
#if defined(ARCH_QUASAR)
static thread_local uint32_t cycle_begin, instret_begin;
#else
static uint32_t cycle_begin, instret_begin;
#endif

// mcountinhibit is never written: its reset value runs both counters.
inline __attribute__((always_inline)) void begin() {
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT3), "r"((uint32_t)PROFILE_RISCV_PMC_EV3));
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT4), "r"((uint32_t)PROFILE_RISCV_PMC_EV4));
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER3));
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER4));
    asm volatile("csrr %0, %1" : "=r"(cycle_begin) : "i"(CSR_MCYCLE) : "memory");
    asm volatile("csrr %0, %1" : "=r"(instret_begin) : "i"(CSR_MINSTRET) : "memory");
}

// The three markers share one buffer-room check and one timestamp, which keeps the BRISC image
// inside its region. Re-emitting the selects every launch keeps each capture self-describing.
// No flush_to_dram_if_full(): a full buffer drops the markers, and the flush would pull noc_index
// into firmware images that do not define it.
inline __attribute__((always_inline)) void end() {
    uint32_t pmc3, pmc4, mcycle, minstret;
    asm volatile("csrr %0, %1" : "=r"(pmc3) : "i"(CSR_MHPMCOUNTER3) : "memory");
    asm volatile("csrr %0, %1" : "=r"(pmc4) : "i"(CSR_MHPMCOUNTER4) : "memory");
    asm volatile("csrr %0, %1" : "=r"(mcycle) : "i"(CSR_MCYCLE) : "memory");
    asm volatile("csrr %0, %1" : "=r"(minstret) : "i"(CSR_MINSTRET) : "memory");

    constexpr uint32_t marker_words = PROFILER_L1_MARKER_UINT32_SIZE + 2;
    if (!bufferHasRoom<DoingDispatch::DISPATCH>(3 * marker_words - 1)) {
        return;
    }

#if defined(ARCH_QUASAR)
    const uint64_t now = quasar_read_wall_clock_64();
    const uint32_t time_low = static_cast<uint32_t>(now);
    const uint32_t time_high = static_cast<uint32_t>(now >> 32);
#else
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t time_high = p_reg[WALL_CLOCK_HIGH_INDEX];
    const uint32_t time_low = p_reg[WALL_CLOCK_LOW_INDEX];
#endif

    auto& data = profiler_data_buffer[myRiscID].data;
    const uint32_t stamp_high = time_high & PROFILER_MARKER_TS_HIGH_MASK;
    uint32_t w = wIndex;
    auto put = [&](uint32_t id, uint32_t value_high, uint32_t value_low) __attribute__((always_inline)) {
        data[w] = PROFILER_MARKER_VALID |
                  ((get_const_id(id, PacketTypes::TS_DATA) & PROFILER_MARKER_TIMER_ID_MASK)
                   << PROFILER_MARKER_TIMER_ID_SHIFT) |
                  stamp_high;
        data[w + 1] = time_low;
        data[w + 2] = value_high;
        data[w + 3] = value_low;
        w += marker_words;
    };
    put(RISCV_PMC_COUNTS_PROFILER_ID, pmc4, pmc3);
    put(RISCV_PMC_CYCLES_PROFILER_ID, minstret - instret_begin, mcycle - cycle_begin);
    put(RISCV_PMC_EVENTS_PROFILER_ID, (uint32_t)PROFILE_RISCV_PMC_EV4, (uint32_t)PROFILE_RISCV_PMC_EV3);
    wIndex = w;
}

}  // namespace riscv_pmc
}  // namespace kernel_profiler

#define RiscvPmcBegin() kernel_profiler::riscv_pmc::begin()
#define RiscvPmcEnd() kernel_profiler::riscv_pmc::end()

#else

#define RiscvPmcBegin()
#define RiscvPmcEnd()

#endif
