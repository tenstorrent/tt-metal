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

// CQDispatchSetWriteOffsetCmd::program_host_id and RT timestamp correlation: this value means the
// dispatch event is not tied to a profiled program (raw streams, preamble defaults, reset and trace
// go signals). dispatch_s never publishes a record with it.
constexpr uint16_t REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID = 0;

// MEM_WORD_ADDR_WIDTH counter bits of a worker-completion stream register.
constexpr uint32_t REALTIME_PROFILER_STREAM_COUNT_MASK = (1u << 17) - 1;

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

FORCE_INLINE
uint64_t realtime_profiler_wall_clock() {
    // LOW first to latch HIGH
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
    uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];
    return (static_cast<uint64_t>(time_hi) << 32) | time_lo;
}

// 0 is a stream reset, not a completion.
FORCE_INLINE
void record_stream_done(volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t stream, uint32_t count) {
    if (count != 0) {
        uint64_t now = realtime_profiler_wall_clock();
        msg->stream_done[stream].time_hi = static_cast<uint32_t>(now >> 32);
        msg->stream_done[stream].time_lo = static_cast<uint32_t>(now);
    }
    msg->stream_done[stream].count = count;
}

// PUSH_B means B is in flight.
FORCE_INLINE
void write_realtime_record(volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t id, uint64_t start, uint64_t end) {
    bool use_buffer_a = (msg->realtime_profiler_state == REALTIME_PROFILER_STATE_PUSH_B);
    volatile realtime_profiler_timestamp_t* start_ts = use_buffer_a ? &msg->kernel_start_a : &msg->kernel_start_b;
    volatile realtime_profiler_timestamp_t* end_ts = use_buffer_a ? &msg->kernel_end_a : &msg->kernel_end_b;
    start_ts->time_hi = static_cast<uint32_t>(start >> 32);
    start_ts->time_lo = static_cast<uint32_t>(start);
    start_ts->id = id;
    end_ts->time_hi = static_cast<uint32_t>(end >> 32);
    end_ts->time_lo = static_cast<uint32_t>(end);
    end_ts->id = id;
}
#else
FORCE_INLINE
bool program_id_fifo_append(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t) { return false; }

FORCE_INLINE
bool program_id_fifo_pop(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t*) { return false; }

FORCE_INLINE
void record_stream_done(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t, uint32_t) {}

FORCE_INLINE
void write_realtime_record(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t, uint64_t, uint64_t) {}
#endif
