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

// Sync marker ID - used to identify sync packets in real-time profiler stream
constexpr uint32_t REALTIME_PROFILER_SYNC_MARKER_ID = 0xFFFFFFFF;

// Program ID FIFO size
constexpr uint32_t PROGRAM_ID_FIFO_SIZE = 32;

// Real-time profiler is only functional on Blackhole due to HW constraints.
// On other architectures, all functions compile to no-ops.
#if defined(ARCH_BLACKHOLE)

// Append a program ID to the circular buffer.
// Returns true if successful, false if the buffer is full.
FORCE_INLINE
bool program_id_fifo_append(volatile tt_l1_ptr realtime_profiler_msg_t* mailbox, uint32_t program_id) {
    uint32_t end = mailbox->program_id_fifo_end;
    uint32_t next_end = (end + 1) % PROGRAM_ID_FIFO_SIZE;

    // Check if buffer is full (next write position equals read position)
    if (next_end == mailbox->program_id_fifo_start) {
        return false;
    }

    mailbox->program_id_fifo[end] = program_id;
    mailbox->program_id_fifo_end = next_end;
    return true;
}

// Pop a program ID from the circular buffer.
// Returns true if successful (and stores the value in *program_id), false if the buffer is empty.
FORCE_INLINE
bool program_id_fifo_pop(volatile tt_l1_ptr realtime_profiler_msg_t* mailbox, uint32_t* program_id) {
    uint32_t start = mailbox->program_id_fifo_start;

    // Check if buffer is empty (read position equals write position)
    if (start == mailbox->program_id_fifo_end) {
        return false;
    }

    *program_id = mailbox->program_id_fifo[start];
    mailbox->program_id_fifo_start = (start + 1) % PROGRAM_ID_FIFO_SIZE;
    return true;
}

// Record a real-time profiler timestamp (start or end) to the appropriate ping-pong buffer.
// Reads mailbox state to determine which buffer to use (opposite of what's being pushed).
// is_start: true for kernel start timestamp, false for kernel end timestamp
FORCE_INLINE
void record_realtime_timestamp(volatile tt_l1_ptr realtime_profiler_msg_t* mailbox, bool is_start) {
    // Read wall clock - LOW first to latch HIGH
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
    uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];

    // Determine buffer from mailbox state: write to buffer NOT being pushed
    // PUSH_B means real-time profiler is pushing B, so write to A
    // Otherwise (IDLE, PUSH_A) write to B
    RealtimeProfilerState state = static_cast<RealtimeProfilerState>(mailbox->realtime_profiler_state);
    bool use_buffer_a = (state == REALTIME_PROFILER_STATE_PUSH_B);

    // Get pointer to appropriate timestamp field
    volatile realtime_profiler_timestamp_t* ts;
    if (use_buffer_a) {
        ts = is_start ? &mailbox->kernel_start_a : &mailbox->kernel_end_a;
    } else {
        ts = is_start ? &mailbox->kernel_start_b : &mailbox->kernel_end_b;
    }

    ts->time_lo = time_lo;
    ts->time_hi = time_hi;
}

// Switch the real-time profiler ping-pong buffer state.
// Determines which buffer was just written to and toggles the state so that the next iteration
// writes to the other buffer. Returns the new state so the caller can signal the real-time
// profiler core if needed (signaling requires dispatch-specific NOC write functions).
FORCE_INLINE
RealtimeProfilerState signal_realtime_profiler_and_switch(volatile tt_l1_ptr realtime_profiler_msg_t* mailbox) {
    // Determine which buffer we just wrote to
    RealtimeProfilerState current_state = static_cast<RealtimeProfilerState>(mailbox->realtime_profiler_state);
    bool used_buffer_a = (current_state == REALTIME_PROFILER_STATE_PUSH_B);

    // New state: push the buffer we just wrote to
    RealtimeProfilerState new_state = used_buffer_a ? REALTIME_PROFILER_STATE_PUSH_A : REALTIME_PROFILER_STATE_PUSH_B;

    // Update local mailbox state (so next iteration writes to other buffer)
    mailbox->realtime_profiler_state = new_state;

    return new_state;
}

// Set the program ID in both start and end timestamps of the appropriate ping-pong buffer.
// Pops the program ID from the FIFO and sets it in the appropriate buffer.
// If the FIFO is empty, sets the ID to zero.
// Reads mailbox state to determine which buffer to use (opposite of what's being pushed).
FORCE_INLINE
void set_program_id(volatile tt_l1_ptr realtime_profiler_msg_t* mailbox) {
    uint32_t id = 0;
    program_id_fifo_pop(mailbox, &id);

    // Determine buffer from mailbox state: write to buffer NOT being pushed
    // PUSH_B means real-time profiler is pushing B, so write to A
    // Otherwise (IDLE, PUSH_A) write to B
    RealtimeProfilerState state = static_cast<RealtimeProfilerState>(mailbox->realtime_profiler_state);
    bool use_buffer_a = (state == REALTIME_PROFILER_STATE_PUSH_B);

    if (use_buffer_a) {
        mailbox->kernel_start_a.id = id;
        mailbox->kernel_end_a.id = id;
    } else {
        mailbox->kernel_start_b.id = id;
        mailbox->kernel_end_b.id = id;
    }
}

#else  // !ARCH_BLACKHOLE — real-time profiler compiled out as no-ops

FORCE_INLINE
bool program_id_fifo_append(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t) { return false; }

FORCE_INLINE
bool program_id_fifo_pop(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t*) { return false; }

FORCE_INLINE
void record_realtime_timestamp(volatile tt_l1_ptr realtime_profiler_msg_t*, bool) {}

FORCE_INLINE
RealtimeProfilerState signal_realtime_profiler_and_switch(volatile tt_l1_ptr realtime_profiler_msg_t*) {
    return REALTIME_PROFILER_STATE_IDLE;
}

FORCE_INLINE
void set_program_id(volatile tt_l1_ptr realtime_profiler_msg_t*) {}

#endif  // ARCH_BLACKHOLE
