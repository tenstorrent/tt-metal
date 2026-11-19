// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per-launch capture of the NoC interface (NIU) event counters. The register tables are in tt-llk
// perf_counters/noc_counters_*.h; this header adds the profiler marker format and emission.
//
// Enable with TT_METAL_PROFILE_NOC_COUNTERS=<group mask>. The counters belong to the tile, so one core captures:
// the BRISC kernel entry on Wormhole and Blackhole Tensix (brisck, so the BRISC firmware region keeps its size), the
// DM0 firmware on Quasar. The mask is a compile-time constant of the build.

#pragma once

// Ids above the Tensix MMIO counters (9090, 9091) and the RISC-V monitor (9092 to 9096). Each value goes
// in the plain TS_DATA field so every row reaches profile_log_device.csv. One launch's markers share a timestamp, and
// the host keeps one marker per (timestamp, core, risc, id), so every pair of counters has an id of its own.
constexpr uint16_t NOC_COUNTERS_HEADER_PROFILER_ID = 9100;     // (noc count << 32) | group mask
constexpr uint16_t NOC_COUNTERS_PAIR_PROFILER_ID_BASE = 9101;  // base + k: (counter[2k + 1] << 32) | counter[2k]
constexpr uint16_t NOC_COUNTERS_MAX_PAIRS = 16;

#if (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(COMPILE_FOR_BRISC) && !defined(FW_BUILD) && \
    !defined(DISPATCH_KERNEL)
#define NOC_COUNTERS_CORE 1
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(FW_BUILD)
#define NOC_COUNTERS_CORE 1
#endif

// Included directly by the firmware and kernel entry TUs, like riscv_pmc.hpp. The streaming profiler has no TS_DATA
// path.
#if defined(PROFILE_NOC_COUNTERS) && defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING) && \
    defined(NOC_COUNTERS_CORE)

#include <array>

#include "api/debug/kernel_profiler.hpp"
#include "perf_counters/noc_counters_hw.h"

namespace kernel_profiler {
namespace noc_counters {

constexpr uint32_t kGroups = PROFILE_NOC_COUNTERS;
constexpr auto kRegs = llk::perf::noc::selected_regs<kGroups>();
constexpr uint32_t kPerNoc = kRegs.size();
constexpr uint32_t kTotal = kPerNoc * llk::perf::noc::NIU_NOCS;
constexpr uint32_t kMarkers = 1 + kTotal / 2;
static_assert(kTotal % 2 == 0, "counters are emitted in pairs");
static_assert(kTotal / 2 <= NOC_COUNTERS_MAX_PAIRS, "pair ids end at 9116");

// Word offset of every counter from the NoC 0 status file, NoC 0 counters first.
constexpr auto kOffsets = []() {
    std::array<uint16_t, kTotal> out{};
    for (uint32_t i = 0; i < kTotal; i++) {
        out[i] = (i / kPerNoc) * (llk::perf::noc::NIU_NOC_STRIDE / 4) + kRegs[i % kPerNoc];
    }
    return out;
}();

static uint32_t begin_values[kTotal];

inline __attribute__((always_inline)) uint32_t read_counter(uint32_t i) {
    return *reinterpret_cast<volatile uint32_t*>(llk::perf::noc::NIU_STATUS_BASE + kOffsets[i] * 4);
}

__attribute__((noinline)) void begin() {
    for (uint32_t i = 0; i < kTotal; i++) {
        begin_values[i] = read_counter(i);
    }
}

// One header marker and kTotal / 2 pair markers behind one buffer-room check and one timestamp; the counters are read
// as the pairs are written. A full buffer drops the whole capture.
__attribute__((noinline)) void end() {
#if !defined(ARCH_QUASAR)
    // The window spans the launch, so the BRISC kernel waits for NCRISC and the TRISCs as the firmware does after it.
    auto* sync = reinterpret_cast<volatile tt_l1_ptr subordinate_map_t*>(GET_MAILBOX_ADDRESS_DEV(subordinate_sync.map));
    while (sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE) {
#if defined(ARCH_WORMHOLE)
        asm volatile("nop; nop; nop; nop; nop");
#endif
        invalidate_l1_cache();
    }
#endif
    // Every worker gets a go signal; a launch without a kernel publishes nothing.
    if (get_profiler_zone_invalid()) {
        return;
    }
    constexpr uint32_t marker_words = PROFILER_L1_MARKER_UINT32_SIZE + 2;
    if (!bufferHasRoom<DoingDispatch::DISPATCH>(kMarkers * marker_words - 1)) {
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
    for (uint32_t m = 0; m < kMarkers; m++) {
        const uint32_t j = 2 * m - 2;
        data[w] =
            PROFILER_MARKER_VALID |
            ((get_const_id(NOC_COUNTERS_HEADER_PROFILER_ID + m, PacketTypes::TS_DATA) & PROFILER_MARKER_TIMER_ID_MASK)
             << PROFILER_MARKER_TIMER_ID_SHIFT) |
            stamp_high;
        data[w + 1] = time_low;
        data[w + 2] = m ? read_counter(j + 1) - begin_values[j + 1] : llk::perf::noc::NIU_NOCS;
        data[w + 3] = m ? read_counter(j) - begin_values[j] : kGroups;
        w += marker_words;
    }
    wIndex = w;
}

}  // namespace noc_counters
}  // namespace kernel_profiler

#define NocCountersBegin() kernel_profiler::noc_counters::begin()
#define NocCountersEnd() kernel_profiler::noc_counters::end()

#else

#define NocCountersBegin()
#define NocCountersEnd()

#endif
