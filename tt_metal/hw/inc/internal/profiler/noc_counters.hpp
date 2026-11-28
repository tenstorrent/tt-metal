// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per-launch capture of the NoC interface counters (tables in tt-llk perf_counters/noc_counters_*.h): the BRISC
// firmware opens the window before it starts the other cores, the kernel entry closes it; on Quasar DM0 does both.

#pragma once

// One launch's markers share a timestamp and the host keeps one marker per (timestamp, core, risc, id), so each pair
// of counters has its own id.
constexpr uint16_t NOC_COUNTERS_HEADER_PROFILER_ID = 9100;     // (noc count << 32) | group mask
constexpr uint16_t NOC_COUNTERS_PAIR_PROFILER_ID_BASE = 9101;  // base + k: (counter[2k + 1] << 32) | counter[2k]
constexpr uint16_t NOC_COUNTERS_MAX_PAIRS = 16;

#if (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(COMPILE_FOR_BRISC) && defined(FW_BUILD)
#define NOC_COUNTERS_OPEN 1
#elif (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(COMPILE_FOR_BRISC) && !defined(DISPATCH_KERNEL)
#define NOC_COUNTERS_CLOSE 1
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(FW_BUILD)
#define NOC_COUNTERS_OPEN 1
#define NOC_COUNTERS_CLOSE 1
#endif

// The streaming profiler has no TS_DATA path.
#if defined(PROFILE_NOC_COUNTERS) && defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING) && \
    (defined(NOC_COUNTERS_OPEN) || defined(NOC_COUNTERS_CLOSE))
#define NOC_COUNTERS_CAPTURE 1

#include <array>

#include "internal/profiler/ts_data_marker.hpp"
#include "perf_counters/noc_counters_inventory.h"

namespace kernel_profiler {
namespace noc_counters {

constexpr uint32_t kGroups = PROFILE_NOC_COUNTERS;
constexpr auto kRegs = llk::perf::noc::selected_regs<kGroups>();
constexpr uint32_t kPerNoc = kRegs.size();
constexpr uint32_t kTotal = kPerNoc * llk::perf::noc::NIU_NOCS;
constexpr uint32_t kPairs = kTotal / 2;
constexpr uint32_t kReservedWords = (1 + kPairs) * TS_DATA_MARKER_WORDS;
static_assert(kTotal % 2 == 0, "counters are emitted in pairs");
static_assert(kPairs <= NOC_COUNTERS_MAX_PAIRS, "pair ids end at 9116");

// Word offset of every counter from the NoC 0 status file, NoC 0 counters first.
constexpr auto kOffsets = []() {
    std::array<uint16_t, kTotal> out{};
    for (uint32_t i = 0; i < kTotal; i++) {
        out[i] = (i / kPerNoc) * (llk::perf::noc::NIU_NOC_STRIDE / 4) + kRegs[i % kPerNoc];
    }
    return out;
}();

inline __attribute__((always_inline)) uint32_t read_counter(uint32_t i) {
    return *reinterpret_cast<volatile uint32_t*>(llk::perf::noc::NIU_STATUS_BASE + kOffsets[i] * 4);
}

}  // namespace noc_counters
}  // namespace kernel_profiler

// The start values live in the firmware, which the BRISC kernel links against.
#if defined(FW_BUILD)
uint32_t noc_counters_begin_values[kernel_profiler::noc_counters::kTotal] __attribute__((used));
#else
extern uint32_t noc_counters_begin_values[kernel_profiler::noc_counters::kTotal];
#endif

namespace kernel_profiler {
namespace noc_counters {

#if defined(NOC_COUNTERS_OPEN)
__attribute__((noinline)) void begin() {
    for (uint32_t i = 0; i < kTotal; i++) {
        noc_counters_begin_values[i] = read_counter(i);
    }
}
#endif

#if defined(NOC_COUNTERS_CLOSE)
// Holds back room for the markers so a kernel that fills the buffer does not drop them.
inline __attribute__((always_inline)) void reserve() { stackSize += kReservedWords; }

// One header marker and kPairs pair markers behind one room check and one timestamp; the counters are read as the
// pairs are written.
__attribute__((noinline)) void end() {
    stackSize -= kReservedWords;
    // The firmware marks a launch that runs no kernel on this core as invalid, and such a launch publishes nothing.
    if (get_profiler_zone_invalid() || !bufferHasRoom(kReservedWords - 1)) {
        return;
    }
    const TsDataStamp stamp = ts_data_stamp();
    uint32_t w = wIndex;
    put_ts_data(w, stamp, NOC_COUNTERS_HEADER_PROFILER_ID, llk::perf::noc::NIU_NOCS, kGroups);
    for (uint32_t k = 0; k < kPairs; k++) {
        const uint32_t lo = read_counter(2 * k) - noc_counters_begin_values[2 * k];
        const uint32_t hi = read_counter(2 * k + 1) - noc_counters_begin_values[2 * k + 1];
        put_ts_data(w, stamp, NOC_COUNTERS_PAIR_PROFILER_ID_BASE + k, hi, lo);
    }
    wIndex = w;
}
#endif

}  // namespace noc_counters
}  // namespace kernel_profiler

#endif

#if defined(NOC_COUNTERS_CAPTURE) && defined(NOC_COUNTERS_OPEN)
#define NocCountersBegin() kernel_profiler::noc_counters::begin()
#else
#define NocCountersBegin()
#endif

#if defined(NOC_COUNTERS_CAPTURE) && defined(NOC_COUNTERS_CLOSE)
#define NocCountersReserve() kernel_profiler::noc_counters::reserve()
#define NocCountersEnd() kernel_profiler::noc_counters::end()
#else
#define NocCountersReserve()
#define NocCountersEnd()
#endif
