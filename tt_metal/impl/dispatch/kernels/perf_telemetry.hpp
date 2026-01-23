// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "risc_common.h"
#include "hostdev/dev_msgs.h"

// Wall clock register indices
constexpr uint32_t WALL_CLOCK_LOW_INDEX = 0;
constexpr uint32_t WALL_CLOCK_HIGH_INDEX = 1;

// Record a telemetry timestamp (start or end) to the appropriate ping-pong buffer.
// Reads mailbox state to determine which buffer to use (opposite of what's being pushed).
// is_start: true for kernel start timestamp, false for kernel end timestamp
FORCE_INLINE
void record_telemetry_timestamp(volatile tt_l1_ptr perf_telemetry_msg_t* mailbox, bool is_start) {
    // Read wall clock - LOW first to latch HIGH
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
    uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];

    // Determine buffer from mailbox state: write to buffer NOT being pushed
    // PUSH_B means telemetry is pushing B, so write to A
    // Otherwise (IDLE, PUSH_A) write to B
    TelemetryState state = static_cast<TelemetryState>(mailbox->telemetry_state);
    bool use_buffer_a = (state == TELEMETRY_STATE_PUSH_B);

    // Get pointer to appropriate timestamp field
    volatile telemetry_timestamp_t* ts;
    if (use_buffer_a) {
        ts = is_start ? &mailbox->kernel_start_a : &mailbox->kernel_end_a;
    } else {
        ts = is_start ? &mailbox->kernel_start_b : &mailbox->kernel_end_b;
    }

    ts->time_lo = time_lo;
    ts->time_hi = time_hi;
}
