// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "risc_common.h"
#include "hostdev/realtime_profiler_msgs.h"

// Wall clock register indices — registers are 8 bytes apart (0x1F0, 0x1F8),
// so the uint32_t array stride is 2, not 1.
constexpr uint32_t WALL_CLOCK_LOW_INDEX = 0;
constexpr uint32_t WALL_CLOCK_HIGH_INDEX = 2;

// Sync marker ID - used to identify sync packets in real-time profiler stream
constexpr uint32_t REALTIME_PROFILER_SYNC_MARKER_ID = 0xFFFFFFFF;

// Runtime ID zero means the dispatch event is intentionally unprofiled.
constexpr uint16_t REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID = 0;

#ifndef ARCH_QUASAR
FORCE_INLINE void read_realtime_wall_clock(uint32_t* time_hi, uint32_t* time_lo) {
    // LOW latches HIGH for a coherent 64-bit Blackhole wall-clock sample.
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    *time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
    *time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];
}
#else
FORCE_INLINE void read_realtime_wall_clock(uint32_t*, uint32_t*) {}
#endif
