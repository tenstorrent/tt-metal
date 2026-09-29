// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Real-time profiler L1 layout for the block carved by DispatchMemMap
// (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG). Not part of mailboxes_t.
//
// Consumed by tt_metal/llrt/hal/codegen/codegen.sh (same rules as hostdevcommon/fabric_telemetry_msgs.h:
// structs, enums, constants, 1-D arrays only).

#pragma once

#include <cstdint>

// Dispatch-core-local control word: tells the dispatch_s compute helper when to stop.
// Records are handed to the RT-profiler BRISC through the record ring below, not through this word.
enum RealtimeProfilerState : uint32_t {
    REALTIME_PROFILER_STATE_IDLE = 0,
    REALTIME_PROFILER_STATE_TERMINATE = 3,
};

// dispatch_s -> RT-profiler BRISC record ring (single producer, single consumer).
// Slots live in dispatch_s L1; the BRISC pulls them over NOC. Both indices are free-running counts:
// slot = index % REALTIME_PROFILER_RECORD_SLOTS. Each index has exactly one writer:
//   record_wr_idx: written only by dispatch_s. On the dispatch core it is the open slot that dispatch_s
//                  and its compute helper are filling; dispatch_s pushes it to the BRISC's copy on publish.
//   record_rd_idx: written only by the BRISC. It advances its dispatch-core copy after a slot's NOC read
//                  has landed, which hands the slot back to dispatch_s.
// dispatch_s keeps (record_wr_idx - record_rd_idx) < REALTIME_PROFILER_RECORD_SLOTS, so the open slot is
// never one the BRISC may still be reading. When it has to wait for a free slot, it stores the wait in cycles
// in kernel_end.header of the record it publishes next (0 means no wait); the BRISC reports it to the host and
// zeroes that word before handing the slot back.
static constexpr uint32_t REALTIME_PROFILER_RECORD_SLOTS = 4;  // must be a power of 2
// Set in the BRISC's record_wr_idx together with the final count, so terminate cannot overtake it.
static constexpr uint32_t REALTIME_PROFILER_RECORD_WR_IDX_TERMINATE = 0x80000000u;
static constexpr uint32_t REALTIME_PROFILER_RECORD_WR_IDX_MASK = 0x7FFFFFFFu;

struct realtime_profiler_timestamp_t {
    uint32_t time_hi;
    uint32_t time_lo;
    uint32_t id;
    uint32_t header;
};

struct realtime_profiler_record_t {
    struct realtime_profiler_timestamp_t kernel_start;
    struct realtime_profiler_timestamp_t kernel_end;
};

struct realtime_profiler_msg_t {
    volatile uint32_t config_buffer_addr;
    volatile uint32_t realtime_profiler_state;
    volatile uint32_t realtime_profiler_core_noc_xy;
    volatile uint32_t realtime_profiler_remote_wr_idx_addr;  // L1 addr of record_wr_idx on the profiler tensix
    // Kept at a 16 B offset: the BRISC NOC-reads slots straight into its 16 B-aligned ring.
    struct realtime_profiler_record_t records[REALTIME_PROFILER_RECORD_SLOTS];
    volatile uint32_t record_wr_idx;
    volatile uint32_t record_rd_idx;
    // Times dispatch_s found the record ring full and waited for the BRISC (dispatch core; dispatch_s only).
    volatile uint32_t record_full_wait_count;
    volatile uint32_t sync_request;
    volatile uint32_t sync_host_timestamp;
    volatile uint32_t program_id_fifo[32];
    volatile uint32_t program_id_fifo_start;
    volatile uint32_t program_id_fifo_end;
};
