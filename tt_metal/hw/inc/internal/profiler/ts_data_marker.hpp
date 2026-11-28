// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Plain TS_DATA markers that share one timestamp, for the counter captures written after a kernel.

#pragma once

#include "api/debug/kernel_profiler.hpp"

namespace kernel_profiler {

constexpr uint32_t TS_DATA_MARKER_WORDS = PROFILER_L1_MARKER_UINT32_SIZE + 2;

struct TsDataStamp {
    uint32_t high;
    uint32_t low;
};

inline __attribute__((always_inline)) TsDataStamp ts_data_stamp() {
#if defined(ARCH_QUASAR)
    const uint64_t now = quasar_read_wall_clock_64();
    return {static_cast<uint32_t>(now >> 32) & PROFILER_MARKER_TS_HIGH_MASK, static_cast<uint32_t>(now)};
#else
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t high = p_reg[WALL_CLOCK_HIGH_INDEX];
    const uint32_t low = p_reg[WALL_CLOCK_LOW_INDEX];
    return {high & PROFILER_MARKER_TS_HIGH_MASK, low};
#endif
}

// Writes one marker at w and moves w past it; the caller has checked the room.
inline __attribute__((always_inline)) void put_ts_data(
    uint32_t& w, TsDataStamp stamp, uint32_t id, uint32_t value_high, uint32_t value_low) {
    auto& data = profiler_data_buffer[myRiscID].data;
    data[w] =
        PROFILER_MARKER_VALID |
        ((get_const_id(id, PacketTypes::TS_DATA) & PROFILER_MARKER_TIMER_ID_MASK) << PROFILER_MARKER_TIMER_ID_SHIFT) |
        stamp.high;
    data[w + 1] = stamp.low;
    data[w + 2] = value_high;
    data[w + 3] = value_low;
    w += TS_DATA_MARKER_WORDS;
}

}  // namespace kernel_profiler
