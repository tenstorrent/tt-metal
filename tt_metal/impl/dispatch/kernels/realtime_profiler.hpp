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
// Record a real-time profiler timestamp (start or end) to the appropriate ping-pong buffer.
// Reads mailbox state to determine which buffer to use (opposite of what's being pushed).
// is_start: true for kernel start timestamp, false for kernel end timestamp
FORCE_INLINE
void record_realtime_timestamp(volatile tt_l1_ptr realtime_profiler_msg_t* msg, bool is_start) {
    // Read wall clock - LOW first to latch HIGH
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
    uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];

    // Determine buffer from profiler state: write to buffer NOT being pushed
    // PUSH_B means real-time profiler is pushing B, so write to A
    // Otherwise (IDLE, PUSH_A) write to B
    RealtimeProfilerState state = static_cast<RealtimeProfilerState>(msg->realtime_profiler_state);
    bool use_buffer_a = (state == REALTIME_PROFILER_STATE_PUSH_B);

    // Get pointer to appropriate timestamp field
    volatile realtime_profiler_timestamp_t* ts;
    if (use_buffer_a) {
        ts = is_start ? &msg->kernel_start_a : &msg->kernel_end_a;
    } else {
        ts = is_start ? &msg->kernel_start_b : &msg->kernel_end_b;
    }

    ts->time_lo = time_lo;
    ts->time_hi = time_hi;
}

// Write the program ID to the start timestamp of the current write buffer. The
// reserved-core filter and host parser consume only this ID; the end ID is not
// part of the legacy wire contract.
// For GO_SIGNAL commands: pass the ID carried in the command itself.
// For non-GO commands: pass REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID so the host filters them out.
FORCE_INLINE
void write_buffer_id(volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t id) {
    RealtimeProfilerState state = static_cast<RealtimeProfilerState>(msg->realtime_profiler_state);
    bool use_buffer_a = (state == REALTIME_PROFILER_STATE_PUSH_B);

    if (use_buffer_a) {
        msg->kernel_start_a.id = id;
    } else {
        msg->kernel_start_b.id = id;
    }
}
#else
FORCE_INLINE
void record_realtime_timestamp(volatile tt_l1_ptr realtime_profiler_msg_t*, bool) {}

FORCE_INLINE
void write_buffer_id(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t) {}
#endif
