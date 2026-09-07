// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Real-time profiler L1 layout for the block carved by DispatchMemMap
// (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG). Not part of mailboxes_t.
//
// Consumed by tt_metal/llrt/hal/codegen/codegen.sh (same rules as fabric_telemetry_msgs.h:
// structs, enums, constants, 1-D arrays only).

#pragma once

#include <cstdint>

enum RealtimeProfilerState : uint32_t {
    REALTIME_PROFILER_STATE_IDLE = 0,       // Waiting for initialization, skip iteration
    REALTIME_PROFILER_STATE_TERMINATE = 3,  // Signal to terminate the kernel
};

struct realtime_profiler_timestamp_t {
    uint32_t time_hi;
    uint32_t time_lo;
    uint32_t id;
    uint32_t header;
};

enum RealtimeProfilerRecordState : uint32_t {
    REALTIME_PROFILER_RECORD_FREE = 0,
    REALTIME_PROFILER_RECORD_PENDING = 1,
    REALTIME_PROFILER_RECORD_READY = 2,
};

constexpr uint32_t REALTIME_PROFILER_RECORD_CAPACITY = 64;
constexpr uint32_t REALTIME_PROFILER_DROPPED_MARKER_ID = 0xFFFFFFFE;

// Dispatch publishes PENDING immediately before GO issue. TRISC writes end and publishes
// READY only when this stream reaches this launch's target. The remote reader
// returns the slot to FREE after copying it. A full queue drops profiling data,
// never a GO command; the overflow count is delivered to host callbacks.
struct realtime_profiler_record_t {
    struct realtime_profiler_timestamp_t start;
    struct realtime_profiler_timestamp_t end;
    uint32_t completion_count;
    uint32_t completion_stream;
    volatile uint32_t state;
    uint32_t reserved[5];
};

struct realtime_profiler_msg_t {
    volatile uint32_t config_buffer_addr;
    volatile uint32_t realtime_profiler_state;
    volatile uint32_t realtime_profiler_core_noc_xy;
    volatile uint32_t realtime_profiler_remote_state_addr;  // L1 addr on profiler tensix for state NOC writes
    volatile uint32_t sync_request;
    volatile uint32_t sync_host_timestamp;
    volatile uint32_t dropped_records;
    uint32_t record_alignment[1];
    struct realtime_profiler_record_t records[64];
};
