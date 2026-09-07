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

// Zero marks infrastructure GO commands that do not represent a profiled program.
constexpr uint16_t REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID = 0;

FORCE_INLINE
uint64_t realtime_profiler_read_timestamp() {
    volatile tt_reg_ptr uint32_t* clock = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while (true) {
        const uint32_t low = clock[WALL_CLOCK_LOW_INDEX];
        const uint32_t high = clock[WALL_CLOCK_HIGH_INDEX];
        // LOW latches HIGH, but other RISCs on this core share that latch.
        // If one samples LOW after rollover, HIGH may no longer belong to our
        // first LOW. Retry when the interval containing both reads crossed it.
        const uint32_t after = clock[WALL_CLOCK_LOW_INDEX];
        if (after >= low) {
            return (static_cast<uint64_t>(high) << 32) | low;
        }
    }
}

#ifndef ARCH_QUASAR
FORCE_INLINE
volatile tt_l1_ptr realtime_profiler_record_t* realtime_profiler_begin_record(
    volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t id, uint32_t stream, uint32_t completion_count) {
    if (id == REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID) {
        return nullptr;
    }
    static uint32_t next_slot = 0;
    auto* record = &msg->records[next_slot];
    invalidate_l1_cache();
    if (record->state != REALTIME_PROFILER_RECORD_FREE) {
        msg->dropped_records++;
        return nullptr;
    }
    next_slot = (next_slot + 1) % REALTIME_PROFILER_RECORD_CAPACITY;
    record->completion_stream = stream;
    record->completion_count = completion_count;
    record->start.id = id;
    record->end.id = id;
    record->start.header = 0;
    record->end.header = 0;
    const uint64_t timestamp = realtime_profiler_read_timestamp();
    record->start.time_lo = static_cast<uint32_t>(timestamp);
    record->start.time_hi = timestamp >> 32;
    return record;
}

FORCE_INLINE
void realtime_profiler_publish_record(volatile tt_l1_ptr realtime_profiler_record_t* record) {
    if (record != nullptr) {
        asm volatile("fence rw, rw" ::: "memory");
        record->state = REALTIME_PROFILER_RECORD_PENDING;
    }
}

// Called only after the existing worker wait, before clearing that stream's
// counter. Let the monitor observe completion before the counter is reused.
// Other subdevices remain free to execute; this never waits for their workers.
FORCE_INLINE
void realtime_profiler_retire_stream(volatile tt_l1_ptr realtime_profiler_msg_t* msg, uint32_t stream) {
    if (msg->realtime_profiler_core_noc_xy == 0) {
        return;
    }
    for (uint32_t i = 0; i < REALTIME_PROFILER_RECORD_CAPACITY; ++i) {
        auto* record = &msg->records[i];
        while (record->state == REALTIME_PROFILER_RECORD_PENDING && record->completion_stream == stream) {
            invalidate_l1_cache();
        }
    }
}

#else
FORCE_INLINE
volatile tt_l1_ptr realtime_profiler_record_t* realtime_profiler_begin_record(
    volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t, uint32_t, uint32_t) {
    return nullptr;
}
FORCE_INLINE
void realtime_profiler_publish_record(volatile tt_l1_ptr realtime_profiler_record_t*) {}
FORCE_INLINE
void realtime_profiler_retire_stream(volatile tt_l1_ptr realtime_profiler_msg_t*, uint32_t) {}
#endif
