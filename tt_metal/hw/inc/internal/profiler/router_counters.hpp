// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per-launch capture of the NoC router counters. The register tables are in tt-llk perf_counters/router_counters_*.h;
// this header adds the profiler marker format and emission.
//
// Enable with TT_METAL_PROFILE_ROUTER_COUNTERS=<option mask>. The router belongs to the tile, so one core captures:
// the BRISC kernel entry on Blackhole (brisck, so the BRISC firmware region keeps its size), the DM0 firmware on
// Quasar. The mask is a compile-time constant of the build.

#pragma once

// Ids above the NoC interface counters (9100 to 9116). Each value goes in the plain TS_DATA field so every row reaches
// profile_log_device.csv. One launch's markers share a timestamp, and the host keeps one marker per
// (timestamp, core, risc, id), so every pair of values has an id of its own.
constexpr uint16_t ROUTER_COUNTERS_HEADER_PROFILER_ID = 9120;     // (noc count << 32) | option mask
constexpr uint16_t ROUTER_COUNTERS_PAIR_PROFILER_ID_BASE = 9121;  // base + k: (value[2k + 1] << 32) | value[2k]
constexpr uint16_t ROUTER_COUNTERS_MAX_PAIRS = 16;

#if defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_BRISC) && !defined(FW_BUILD) && !defined(DISPATCH_KERNEL)
#define ROUTER_COUNTERS_CORE 1
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(FW_BUILD)
#define ROUTER_COUNTERS_CORE 1
#endif

// Included directly by the firmware and kernel entry TUs, like noc_counters.hpp. The streaming profiler has no TS_DATA
// path.
#if defined(PROFILE_ROUTER_COUNTERS) && defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING) && \
    defined(ROUTER_COUNTERS_CORE)
#define ROUTER_COUNTERS_CAPTURE 1

#include "api/debug/kernel_profiler.hpp"
#include "perf_counters/router_counters_hw.h"

namespace kernel_profiler {
namespace router_counters {

constexpr uint32_t kOption = PROFILE_ROUTER_COUNTERS;
constexpr auto kBanks = llk::perf::router::selected_bank_offsets<kOption>();
constexpr uint32_t kVcMask = kOption & llk::perf::router::VC_MASK_ALL;
constexpr bool kPerVc = (kOption & llk::perf::router::PER_VC) != 0;
constexpr uint32_t kTotal = llk::perf::router::value_count(kOption);
constexpr uint32_t kMarkers = 1 + (kTotal + 1) / 2;
static_assert((kOption & ~llk::perf::router::OPTION_VALID) == 0, "unknown bits in the router counter option");
static_assert(kBanks.size() != 0, "the router counter option selects no bank this architecture has");
static_assert(kVcMask != 0, "the router counter option selects no virtual channel");
static_assert(
    kTotal <= llk::perf::router::MAX_VALUES, "too many router counter values: select fewer banks or channels");

static uint32_t values[kTotal];

// First call stores the readings, the second turns them into deltas. A bank's selected channels are added up
// unless the option keeps them apart.
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
                    values[i] = delta ? v - values[i] : v;
                    i++;
                } else {
                    sum += v;
                }
            }
            if (!kPerVc) {
                values[i] = delta ? sum - values[i] : sum;
                i++;
            }
        }
    }
}

__attribute__((noinline)) void begin() { scan(false); }

// One header marker and the value pair markers behind one buffer-room check and one timestamp. A full buffer drops the
// whole capture.
__attribute__((noinline)) void end() {
    // Every worker gets a go signal; a launch without a kernel publishes nothing.
    if (get_profiler_zone_invalid()) {
        return;
    }
    constexpr uint32_t marker_words = PROFILER_L1_MARKER_UINT32_SIZE + 2;
    if (!bufferHasRoom<DoingDispatch::DISPATCH>(kMarkers * marker_words - 1)) {
        return;
    }
    scan(true);

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
    for (uint32_t m = 0; m < kMarkers; m++) {
        const uint32_t j = 2 * m - 2;
        data[w] = PROFILER_MARKER_VALID |
                  ((get_const_id(ROUTER_COUNTERS_HEADER_PROFILER_ID + m, PacketTypes::TS_DATA) &
                    PROFILER_MARKER_TIMER_ID_MASK)
                   << PROFILER_MARKER_TIMER_ID_SHIFT) |
                  stamp_high;
        data[w + 1] = time_low;
        data[w + 2] = m ? (j + 1 < kTotal ? values[j + 1] : 0) : llk::perf::router::ROUTER_NOCS;
        data[w + 3] = m ? values[j] : kOption;
        w += marker_words;
    }
    wIndex = w;
}

}  // namespace router_counters
}  // namespace kernel_profiler

#define RouterCountersBegin() kernel_profiler::router_counters::begin()
#define RouterCountersEnd() kernel_profiler::router_counters::end()

#else

#define RouterCountersBegin()
#define RouterCountersEnd()

#endif
