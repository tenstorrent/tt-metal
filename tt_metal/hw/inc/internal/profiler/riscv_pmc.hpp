// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISC-V performance monitor capture per kernel launch, also on the Blackhole active and idle ethernet cores: marker
// format and emission over the tt-llk tables in perf_counters/riscv_pmc_*.h, selects fixed at compile time.

#pragma once

// Plain TS_DATA so every row reaches profile_log_device.csv; 9090 is the Tensix MMIO counters.
constexpr uint16_t RISCV_PMC_COUNTS_PROFILER_ID = 9092;  // (pmc4 << 32) | pmc3
constexpr uint16_t RISCV_PMC_CYCLES_PROFILER_ID = 9093;  // (minstret << 32) | mcycle
constexpr uint16_t RISCV_PMC_EVENTS_PROFILER_ID = 9094;  // (mhpmevent4 << 32) | mhpmevent3
// Quasar data-movement cores with a programmed counter 5 or 6 add these two markers.
constexpr uint16_t RISCV_PMC_COUNTS_UPPER_PROFILER_ID = 9095;  // (pmc6 << 32) | pmc5
constexpr uint16_t RISCV_PMC_EVENTS_UPPER_PROFILER_ID = 9096;  // (mhpmevent6 << 32) | mhpmevent5

// Blackhole Tensix cores run the hook in the kernel entry because their firmware regions are fixed.
#if defined(ARCH_BLACKHOLE) && !defined(FW_BUILD) && !defined(DISPATCH_KERNEL) && \
    (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_TRISC))
#define RISCV_PMC_CORE 1
#elif defined(ARCH_BLACKHOLE) && defined(FW_BUILD) && (defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_IDLE_ERISC))
#define RISCV_PMC_CORE 1
#elif defined(ARCH_QUASAR) && defined(FW_BUILD) && (defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_DM))
#define RISCV_PMC_CORE 1
#endif

// Included directly by the entry TUs so the empty macros below exist on a profiler-off build.
#if defined(PROFILE_RISCV_PMC) && defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING) && defined(RISCV_PMC_CORE)

#include "internal/profiler/ts_data_marker.hpp"

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

// mcycle and minstret are read as deltas; one launch is in flight per core (per thread on Quasar).
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
constexpr uint32_t reserved_words = marker_count * TS_DATA_MARKER_WORDS;

// Holds back room for the markers so a kernel that fills the buffer does not drop them.
inline __attribute__((always_inline)) void begin() {
    stackSize += reserved_words;
    llk::perf::riscv::select(RISCV_PMC_SEL3, RISCV_PMC_SEL4);
    llk::perf::riscv::zero_counters();
#if defined(RISCV_PMC_UPPER)
    llk::perf::riscv::select_upper(PROFILE_RISCV_PMC_DM_EV5, PROFILE_RISCV_PMC_DM_EV6);
    llk::perf::riscv::zero_counters_upper();
#endif
    cycle_begin = llk::perf::riscv::read_csr<llk::perf::riscv::CSR_MCYCLE>();
    instret_begin = llk::perf::riscv::read_csr<llk::perf::riscv::CSR_MINSTRET>();
}

// The selects are re-emitted so each capture is self-describing. No flush here: it needs noc_index.
inline __attribute__((always_inline)) void end() {
    using namespace llk::perf::riscv;
    const uint32_t pmc3 = read_csr<CSR_MHPMCOUNTER3>();
    const uint32_t pmc4 = read_csr<CSR_MHPMCOUNTER4>();
#if defined(RISCV_PMC_UPPER)
    const uint32_t pmc5 = read_csr<CSR_MHPMCOUNTER5>();
    const uint32_t pmc6 = read_csr<CSR_MHPMCOUNTER6>();
#endif
    const uint32_t mcycle = read_csr<CSR_MCYCLE>();
    const uint32_t minstret = read_csr<CSR_MINSTRET>();

    stackSize -= reserved_words;
    if (!bufferHasRoom(reserved_words - 1)) {
        return;
    }

    const TsDataStamp stamp = ts_data_stamp();
    uint32_t w = wIndex;
    put_ts_data(w, stamp, RISCV_PMC_COUNTS_PROFILER_ID, pmc4, pmc3);
    put_ts_data(w, stamp, RISCV_PMC_CYCLES_PROFILER_ID, minstret - instret_begin, mcycle - cycle_begin);
    put_ts_data(w, stamp, RISCV_PMC_EVENTS_PROFILER_ID, RISCV_PMC_SEL4, RISCV_PMC_SEL3);
#if defined(RISCV_PMC_UPPER)
    put_ts_data(w, stamp, RISCV_PMC_COUNTS_UPPER_PROFILER_ID, pmc6, pmc5);
    put_ts_data(w, stamp, RISCV_PMC_EVENTS_UPPER_PROFILER_ID, PROFILE_RISCV_PMC_DM_EV6, PROFILE_RISCV_PMC_DM_EV5);
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
