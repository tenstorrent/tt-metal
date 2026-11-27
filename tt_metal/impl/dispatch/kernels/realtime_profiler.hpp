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

// Dispatch-stall marker ID - the BRISC reports a dispatch_s wait for a free record slot with this ID in word 3
// (words 0-1: stall end time, word 2: stall cycles). Must match the host-side copy.
constexpr uint32_t REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID = 0xFFFFFFFE;

// CQDispatchSetWriteOffsetCmd::program_host_id and RT timestamp correlation: this value means the
// dispatch event is not tied to a profiled program (raw streams, preamble defaults). dispatch_d
// must not push it into the program-id FIFO; dispatch_s passes it to write_buffer_id for non-GO
// commands so the host can filter those records out.
constexpr uint16_t REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID = 0;

// Program ID FIFO size
constexpr uint32_t PROGRAM_ID_FIFO_SIZE = 32;

#ifndef ARCH_QUASAR
// Append a program ID to the circular buffer embedded in realtime_profiler_msg_t.
// Returns true if successful, false if the buffer is full.
// The control block (including this FIFO) lives in dispatch-core-local L1, assigned by
// CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG.
FORCE_INLINE
bool program_id_fifo_append(volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t program_id) {
    uint32_t end = msg->program_id_fifo_end;
    uint32_t next_end = (end + 1) % PROGRAM_ID_FIFO_SIZE;

    // Check if buffer is full (next write position equals read position)
    if (next_end == msg->program_id_fifo_start) {
        return false;
    }

    msg->program_id_fifo[end] = program_id;
    msg->program_id_fifo_end = next_end;
    return true;
}

// Pop a program ID from the circular buffer embedded in realtime_profiler_msg_t.
// Returns true if successful (and stores the value in *program_id), false if the buffer is empty.
FORCE_INLINE
bool program_id_fifo_pop(volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t* program_id) {
    uint32_t start = msg->program_id_fifo_start;

    // Check if buffer is empty (read position equals write position)
    if (start == msg->program_id_fifo_end) {
        return false;
    }

    *program_id = msg->program_id_fifo[start];
    msg->program_id_fifo_start = (start + 1) % PROGRAM_ID_FIFO_SIZE;
    return true;
}

// The record slot dispatch_s is currently filling (see the record ring in realtime_profiler_msgs.h).
FORCE_INLINE
volatile realtime_profiler_record_t* open_record(volatile tt_l1_ptr realtime_profiler_msg_t* msg) {
    return &msg->records[msg->record_wr_idx & (REALTIME_PROFILER_RECORD_SLOTS - 1)];
}

// Record a real-time profiler timestamp (start or end) into the open record slot.
// is_start: true for kernel start timestamp, false for kernel end timestamp
FORCE_INLINE
void record_realtime_timestamp(volatile tt_l1_ptr realtime_profiler_msg_t* msg, bool is_start) {
    // Read wall clock - LOW first to latch HIGH
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
    uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];

    volatile realtime_profiler_record_t* record = open_record(msg);
    volatile realtime_profiler_timestamp_t* ts = is_start ? &record->kernel_start : &record->kernel_end;

    ts->time_lo = time_lo;
    ts->time_hi = time_hi;
}

// Pop a program ID from the FIFO without writing it to the buffer.
// Returns the popped ID, or 0 if the FIFO was empty.
// Call this early to maintain timing (FIFO pop involves L1 reads/writes),
// then call write_buffer_id() later once the command type is known.
FORCE_INLINE
uint32_t pop_program_id(volatile tt_l1_ptr realtime_profiler_msg_t* msg) {
    uint32_t id = 0;
    program_id_fifo_pop(msg, &id);
    return id;
}

// Write a program ID to both start and end timestamps of the open record slot.
// For GO_SIGNAL commands: pass the ID from pop_program_id().
// For non-GO commands: pass REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID so the host filters them out.
FORCE_INLINE
void write_buffer_id(volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t id) {
    volatile realtime_profiler_record_t* record = open_record(msg);
    record->kernel_start.id = id;
    record->kernel_end.id = id;
}

// Low 32 bits of the wall clock, for timing short waits.
FORCE_INLINE
uint32_t realtime_profiler_wall_clock_lo() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[WALL_CLOCK_LOW_INDEX];
}
#else
FORCE_INLINE
bool program_id_fifo_append(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t) { return false; }

FORCE_INLINE
bool program_id_fifo_pop(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t*) { return false; }

FORCE_INLINE
void record_realtime_timestamp(volatile tt_l1_ptr realtime_profiler_msg_t*, bool) {}

FORCE_INLINE
uint32_t pop_program_id(volatile tt_l1_ptr realtime_profiler_msg_t*) { return 0; }

FORCE_INLINE
void write_buffer_id(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t) {}

FORCE_INLINE
uint32_t realtime_profiler_wall_clock_lo() { return 0; }
#endif
