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
    REALTIME_PROFILER_STATE_IDLE = 0,
    REALTIME_PROFILER_STATE_PUSH_B = 1,
};

// Dispatch-s writes completed program intervals into this SPSC queue and the reserved
// realtime-profiler tensix drains them over NOC. The queue decouples dispatch from the
// profiler transport and, unlike the former ping-pong notification slot, cannot overwrite a
// record when independent subdevices complete close together.
constexpr uint32_t REALTIME_PROFILER_RECORD_QUEUE_CAPACITY = 128;
constexpr uint32_t REALTIME_PROFILER_RECORD_WORDS = 8;
constexpr uint32_t REALTIME_PROFILER_MAX_STREAMS = 8;
constexpr uint32_t REALTIME_PROFILER_START_QUEUE_CAPACITY = 4;
constexpr uint32_t REALTIME_PROFILER_START_DESCRIPTOR_WORDS = 5;
constexpr uint32_t REALTIME_PROFILER_START_QUEUE_WORDS = 160;
constexpr uint32_t REALTIME_PROFILER_RECORD_QUEUE_WORDS = 1024;
constexpr uint32_t REALTIME_PROFILER_PROTOCOL_VERSION = 7;
constexpr uint32_t REALTIME_PROFILER_QUALIFICATION_SCRATCH_WORDS = 4;
#ifdef REALTIME_PROFILER_PROTOCOL_BUILD_KEY
static_assert(REALTIME_PROFILER_PROTOCOL_BUILD_KEY == REALTIME_PROFILER_PROTOCOL_VERSION);
#endif

constexpr uint32_t REALTIME_PROFILER_RECORD_SCHEMA_VERSION = 1;
constexpr uint32_t REALTIME_PROFILER_RECORD_TYPE_INTERVAL = 1;
constexpr uint32_t REALTIME_PROFILER_RECORD_TYPE_WATERMARK = 2;
constexpr uint32_t REALTIME_PROFILER_WATERMARK_MARKER_ID = 0xFFFFFFFE;
constexpr uint32_t REALTIME_PROFILER_WATERMARK_PROTOCOL_ERROR_MARKER_ID = 0xFFFFFFFD;

struct realtime_profiler_timestamp_t {
    uint32_t time_hi;
    uint32_t time_lo;
    uint32_t id;
    uint32_t header;
};

struct realtime_profiler_msg_t {
    volatile uint32_t config_buffer_addr;
    volatile uint32_t realtime_profiler_state;
    volatile uint32_t realtime_profiler_core_noc_xy;
    volatile uint32_t realtime_profiler_remote_state_addr;  // L1 addr on profiler tensix for state NOC writes
    struct realtime_profiler_timestamp_t kernel_start_b;
    struct realtime_profiler_timestamp_t kernel_end_b;
    // Dedicated termination handshake. Keep these words after the NOC payload
    // so kernel_start_b retains 16-byte NOC alignment.
    volatile uint32_t terminate_requested;
    volatile uint32_t completion_observer_stopped;
    volatile uint32_t sync_request;
    volatile uint32_t sync_host_timestamp;
    volatile uint32_t program_id_fifo[32];
    volatile uint32_t program_id_fifo_start;
    volatile uint32_t program_id_fifo_end;

    // Blackhole dispatch_s NCRISC -> TRISC0 start-descriptor queues. Each
    // stream is an independent SPSC queue so completion order across
    // sub-devices does not create head-of-line blocking.
    volatile uint32_t start_descriptor_write_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t start_descriptor_read_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t stream_reset_generation[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t start_descriptor_words[REALTIME_PROFILER_START_QUEUE_WORDS];

    // Blackhole TRISC0 -> dispatch_s NCRISC completed-interval queue.
    volatile uint32_t record_write_index;
    volatile uint32_t record_read_index;
    volatile uint32_t record_words[REALTIME_PROFILER_RECORD_QUEUE_WORDS];
    volatile uint32_t successful_record_sequence;
    volatile uint32_t start_descriptor_drop_count;
    volatile uint32_t unsupported_launch_drop_count;
    volatile uint32_t reset_descriptor_drop_count;
    volatile uint32_t completion_observer_drop_count;
    volatile uint32_t stuck_descriptor_head_count;
    volatile uint32_t completed_record_drop_count;
    volatile uint32_t terminal_descriptor_drop_count;
    volatile uint32_t terminal_record_drop_count;
    volatile uint32_t completion_observer_timeout_count;

    // One nonblocking watermark request and one completed watermark slot per
    // stream. Natural producer/consumer indices distinguish an empty slot from
    // a pending request without reserving a batch ID value.
    volatile uint32_t watermark_request_write_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_request_read_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_request_id[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_request_target[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_request_generation[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_write_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_read_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_id[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_sequence[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_descriptor_drop_count[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_observer_drop_count[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_record_drop_count[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_record_write_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_ready_protocol_error[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t watermark_request_drop_count;
    volatile uint32_t watermark_protocol_error_count;
};
