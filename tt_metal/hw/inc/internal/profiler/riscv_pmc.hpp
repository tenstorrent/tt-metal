// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISC-V performance monitor capture for the Tensix baby cores, the Blackhole ethernet cores and the Quasar
// data-movement cores. The hardware
// description (CSRs, select formats, event tables) is in tt-llk perf_counters/riscv_pmc_*.h; this header adds the
// profiler marker format and emission.
//
// Enable with TT_METAL_PROFILE_RISCV_PMC=<mhpmevent3>[,<mhpmevent4>] (Quasar DMs:
// TT_METAL_PROFILE_RISCV_PMC_DM=<mhpmevent3>[,<mhpmevent4>[,<mhpmevent5>[,<mhpmevent6>]]]).
// The selects are compile-time constants and part of the firmware build key.

#pragma once

// 9090 and 9091 belong to the Tensix MMIO perf counters. Each value goes in the plain TS_DATA field so
// every row reaches profile_log_device.csv; a TS_DATA_16B trailer would not.
constexpr uint16_t RISCV_PMC_COUNTS_PROFILER_ID = 9092;  // (pmc4 << 32) | pmc3
constexpr uint16_t RISCV_PMC_CYCLES_PROFILER_ID = 9093;  // (minstret << 32) | mcycle
constexpr uint16_t RISCV_PMC_EVENTS_PROFILER_ID = 9094;  // (mhpmevent4 << 32) | mhpmevent3
// Quasar data-movement cores with a programmed counter 5 or 6 add these two markers.
constexpr uint16_t RISCV_PMC_COUNTS_UPPER_PROFILER_ID = 9095;  // (pmc6 << 32) | pmc5
constexpr uint16_t RISCV_PMC_EVENTS_UPPER_PROFILER_ID = 9096;  // (mhpmevent6 << 32) | mhpmevent5

// Cores that carry the counters and have a firmware hook.
#if defined(ARCH_BLACKHOLE) && (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || \
                                defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_AERISC))
#define RISCV_PMC_CORE 1
#elif defined(ARCH_QUASAR) && (defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_DM))
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
#if !defined(PROFILE_RISCV_PMC_DM_EV3)
#define PROFILE_RISCV_PMC_DM_EV3 0
#endif
#if !defined(PROFILE_RISCV_PMC_DM_EV4)
#define PROFILE_RISCV_PMC_DM_EV4 0
#endif
#if !defined(PROFILE_RISCV_PMC_DM_EV5)
#define PROFILE_RISCV_PMC_DM_EV5 0
#endif
#if !defined(PROFILE_RISCV_PMC_DM_EV6)
#define PROFILE_RISCV_PMC_DM_EV6 0
#endif

// The Quasar data-movement cores take (event mask << 8) | event set; every other core takes eight nibbles.
#if defined(COMPILE_FOR_DM)
#define RISCV_PMC_SEL3 ((uint32_t)PROFILE_RISCV_PMC_DM_EV3)
#define RISCV_PMC_SEL4 ((uint32_t)PROFILE_RISCV_PMC_DM_EV4)
#else
#define RISCV_PMC_SEL3 ((uint32_t)PROFILE_RISCV_PMC_EV3)
#define RISCV_PMC_SEL4 ((uint32_t)PROFILE_RISCV_PMC_EV4)
#endif

// Only the data-movement cores have counters 5 and 6, and only a programmed one is captured.
#if defined(COMPILE_FOR_DM) && (PROFILE_RISCV_PMC_DM_EV5 != 0 || PROFILE_RISCV_PMC_DM_EV6 != 0)
#define RISCV_PMC_UPPER 1
#endif

#include "perf_counters/riscv_pmc_hw.h"

namespace kernel_profiler {
namespace riscv_pmc {

// mcycle and minstret are never written, so they are read as deltas; the programmable counters are zeroed at
// begin. One launch is in flight per core, so a plain global is enough (per thread on Quasar, where cores share
// the image).
#if defined(ARCH_QUASAR)
static thread_local uint32_t cycle_begin, instret_begin;
#else
static uint32_t cycle_begin, instret_begin;
#endif

#if defined(RISCV_PMC_UPPER)
constexpr uint32_t marker_count = 5;
#else
constexpr uint32_t marker_count = 3;
#endif
constexpr uint32_t marker_words = PROFILER_L1_MARKER_UINT32_SIZE + 2;
constexpr uint32_t reserved_words = marker_count * marker_words + DISPATCH_HEADROOM_SIZE;

// The markers are written after the kernel, so their room (and the room check's headroom) is held back from the
// kernel's own zones; a kernel that fills the buffer would otherwise drop them.
inline __attribute__((always_inline)) void begin() {
    stackSize += reserved_words;
    llk::perf::riscv::select(RISCV_PMC_SEL3, RISCV_PMC_SEL4);
    llk::perf::riscv::zero_counters();
#if defined(RISCV_PMC_UPPER)
    llk::perf::riscv::select_upper(PROFILE_RISCV_PMC_DM_EV5, PROFILE_RISCV_PMC_DM_EV6);
    llk::perf::riscv::zero_counters_upper();
#endif
    cycle_begin = llk::perf::riscv::read_mcycle();
    instret_begin = llk::perf::riscv::read_minstret();
}

// The markers share one buffer-room check and one timestamp, which keeps the BRISC image
// inside its region. Re-emitting the selects every launch keeps each capture self-describing.
// No flush_to_dram_if_full(): a full buffer drops the markers, and the flush would pull noc_index
// into firmware images that do not define it.
inline __attribute__((always_inline)) void end() {
    const uint32_t pmc3 = llk::perf::riscv::read_counter3();
    const uint32_t pmc4 = llk::perf::riscv::read_counter4();
#if defined(RISCV_PMC_UPPER)
    const uint32_t pmc5 = llk::perf::riscv::read_counter5();
    const uint32_t pmc6 = llk::perf::riscv::read_counter6();
#endif
    const uint32_t mcycle = llk::perf::riscv::read_mcycle();
    const uint32_t minstret = llk::perf::riscv::read_minstret();

    stackSize -= reserved_words;
    if (!bufferHasRoom<DoingDispatch::DISPATCH>(marker_count * marker_words - 1)) {
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
    put(RISCV_PMC_EVENTS_PROFILER_ID, RISCV_PMC_SEL4, RISCV_PMC_SEL3);
#if defined(RISCV_PMC_UPPER)
    put(RISCV_PMC_COUNTS_UPPER_PROFILER_ID, pmc6, pmc5);
    put(RISCV_PMC_EVENTS_UPPER_PROFILER_ID, PROFILE_RISCV_PMC_DM_EV6, PROFILE_RISCV_PMC_DM_EV5);
#endif
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
