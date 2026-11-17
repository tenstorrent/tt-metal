// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per-launch capture of the NoC router counters (tables in tt-llk perf_counters/router_counters_*.h). The router
// belongs to the tile, so one core captures it: on Blackhole the BRISC firmware opens the window, as for the NoC
// interface counters, and the BRISC kernel entry closes it; on Quasar DM0 does both.

#pragma once

// Ids above the NoC interface counters (9100 to 9116). One launch's markers share a timestamp and the host keeps one
// per (timestamp, core, risc, id), so each pair of values has its own id.
constexpr uint16_t ROUTER_COUNTERS_HEADER_PROFILER_ID = 9120;     // (noc count << 32) | option mask
constexpr uint16_t ROUTER_COUNTERS_PAIR_PROFILER_ID_BASE = 9121;  // base + k: (value[2k + 1] << 32) | value[2k]
constexpr uint16_t ROUTER_COUNTERS_MAX_PAIRS = 16;

#if defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_BRISC) && defined(FW_BUILD)
#define ROUTER_COUNTERS_OPEN 1
#elif defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_BRISC) && !defined(DISPATCH_KERNEL)
#define ROUTER_COUNTERS_CLOSE 1
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(FW_BUILD)
#define ROUTER_COUNTERS_OPEN 1
#define ROUTER_COUNTERS_CLOSE 1
#endif

// The streaming profiler has no TS_DATA path.
#if defined(PROFILE_ROUTER_COUNTERS) && defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING) && \
    (defined(ROUTER_COUNTERS_OPEN) || defined(ROUTER_COUNTERS_CLOSE))
#define ROUTER_COUNTERS_CAPTURE 1

#include "internal/profiler/ts_data_marker.hpp"
#include "perf_counters/router_counters_hw.h"

namespace kernel_profiler {
namespace router_counters {

constexpr uint32_t kOption = PROFILE_ROUTER_COUNTERS;
constexpr auto kBanks = llk::perf::router::selected_bank_offsets<kOption>();
constexpr uint32_t kVcMask = kOption & llk::perf::router::VC_MASK_ALL;
constexpr bool kPerVc = (kOption & llk::perf::router::PER_VC) != 0;
constexpr uint32_t kTotal = llk::perf::router::value_count(kOption);
constexpr uint32_t kPairs = (kTotal + 1) / 2;
constexpr uint32_t kReservedWords = (1 + kPairs) * TS_DATA_MARKER_WORDS;
static_assert((kOption & ~llk::perf::router::OPTION_VALID) == 0, "unknown bits in the router counter option");
static_assert(kBanks.size() != 0, "the router counter option selects no bank this architecture has");
static_assert(kVcMask != 0, "the router counter option selects no virtual channel");
static_assert(
    kTotal <= llk::perf::router::MAX_VALUES, "too many router counter values: select fewer banks or channels");

}  // namespace router_counters
}  // namespace kernel_profiler

// The start values live in the firmware, which the BRISC kernel links against; end() turns them into the deltas.
#if defined(FW_BUILD)
uint32_t router_counters_values[kernel_profiler::router_counters::kTotal] __attribute__((used));
#else
extern uint32_t router_counters_values[kernel_profiler::router_counters::kTotal];
#endif

namespace kernel_profiler {
namespace router_counters {

// The first call stores the readings, the second turns them into deltas.
__attribute__((noinline)) void scan(bool delta) {
    uint32_t i = 0;
    for (uint32_t noc = 0; noc < llk::perf::router::ROUTER_NOCS; noc++) {
        for (uint32_t b = 0; b < kBanks.size(); b++) {
            const uint32_t base =
                llk::perf::router::ROUTER_REGS_BASE + noc * llk::perf::router::ROUTER_NOC_STRIDE + kBanks[b];
            uint32_t sum = 0;
            for (uint32_t vc = 0; vc < llk::perf::router::NUM_VCS; vc++) {
                if (((kVcMask >> vc) & 1u) == 0) {
                    continue;
                }
                const uint32_t v =
                    *reinterpret_cast<volatile uint32_t*>(base + vc * llk::perf::router::ROUTER_VC_STRIDE);
                if (kPerVc) {
                    router_counters_values[i] = delta ? v - router_counters_values[i] : v;
                    i++;
                } else {
                    sum += v;
                }
            }
            if (!kPerVc) {
                router_counters_values[i] = delta ? sum - router_counters_values[i] : sum;
                i++;
            }
        }
    }
}

#if defined(ROUTER_COUNTERS_OPEN)
inline __attribute__((always_inline)) void begin() { scan(false); }
#endif

#if defined(ROUTER_COUNTERS_CLOSE)
// Holds back room for the markers so a kernel that fills the buffer does not drop them.
inline __attribute__((always_inline)) void reserve() { stackSize += kReservedWords; }

// The header and pair markers share one room check and one timestamp.
__attribute__((noinline)) void end() {
    stackSize -= kReservedWords;
    // The firmware marks a launch that runs no kernel on this core as invalid, and such a launch publishes nothing.
    if (get_profiler_zone_invalid() || !bufferHasRoom(kReservedWords - 1)) {
        return;
    }
    scan(true);
    const TsDataStamp stamp = ts_data_stamp();
    uint32_t w = wIndex;
    put_ts_data(w, stamp, ROUTER_COUNTERS_HEADER_PROFILER_ID, llk::perf::router::ROUTER_NOCS, kOption);
    for (uint32_t k = 0; k < kPairs; k++) {
        const uint32_t lo = router_counters_values[2 * k];
        const uint32_t hi = 2 * k + 1 < kTotal ? router_counters_values[2 * k + 1] : 0;
        put_ts_data(w, stamp, ROUTER_COUNTERS_PAIR_PROFILER_ID_BASE + k, hi, lo);
    }
    wIndex = w;
}
#endif

}  // namespace router_counters
}  // namespace kernel_profiler

#endif

#if defined(ROUTER_COUNTERS_CAPTURE) && defined(ROUTER_COUNTERS_OPEN)
#define RouterCountersBegin() kernel_profiler::router_counters::begin()
#else
#define RouterCountersBegin()
#endif

#if defined(ROUTER_COUNTERS_CAPTURE) && defined(ROUTER_COUNTERS_CLOSE)
#define RouterCountersReserve() kernel_profiler::router_counters::reserve()
#define RouterCountersEnd() kernel_profiler::router_counters::end()
#else
#define RouterCountersReserve()
#define RouterCountersEnd()
#endif
