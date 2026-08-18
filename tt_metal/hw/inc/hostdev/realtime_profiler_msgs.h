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
    REALTIME_PROFILER_STATE_PUSH_A = 1,     // Push real-time profiler data from buffer A
    REALTIME_PROFILER_STATE_PUSH_B = 2,     // Push real-time profiler data from buffer B
    REALTIME_PROFILER_STATE_TERMINATE = 3,  // Signal to terminate the kernel
};

// Blackhole concurrent-profiler protocol. The capacities are deliberately
// compile-time constants: no producer may allocate, resize, or wait for a
// consumer in the dispatch path.
constexpr uint32_t REALTIME_PROFILER_PROTOCOL_VERSION = 3;
constexpr uint32_t REALTIME_PROFILER_MAX_STREAMS = 8;
constexpr uint32_t REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY = 4;
constexpr uint32_t REALTIME_PROFILER_DESCRIPTOR_WORDS = 5;
constexpr uint32_t REALTIME_PROFILER_DESCRIPTOR_QUEUE_WORDS = 160;
constexpr uint32_t REALTIME_PROFILER_COMPLETED_QUEUE_CAPACITY = 64;
constexpr uint32_t REALTIME_PROFILER_COMPLETED_RECORD_WORDS = 8;
constexpr uint32_t REALTIME_PROFILER_COMPLETED_QUEUE_WORDS = 512;
constexpr uint32_t REALTIME_PROFILER_RECORD_SCHEMA_VERSION = 1;
constexpr uint32_t REALTIME_PROFILER_RECORD_TYPE_INTERVAL = 1;
constexpr uint32_t REALTIME_PROFILER_RECORD_TYPE_RESET_OBSERVED = 2;
constexpr uint32_t REALTIME_PROFILER_PUBLICATION_EPOCH_MASK = 0x00FFFFFF;
constexpr uint32_t REALTIME_PROFILER_SHUTDOWN_CYCLE_BUDGET = 1000000;
constexpr uint32_t REALTIME_PROFILER_SHUTDOWN_ITEM_BUDGET = REALTIME_PROFILER_COMPLETED_QUEUE_CAPACITY;

// Host-published terminal capability marker. A dispatch_s image can contain the
// observer path before D2H socket construction succeeds; this value latches the
// late failure without leaving a per-command activation poll.
constexpr uint32_t REALTIME_PROFILER_REMOTE_STATE_DISABLED = 0xFFFFFFFF;

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
    struct realtime_profiler_timestamp_t kernel_start_a;
    struct realtime_profiler_timestamp_t kernel_end_a;
    struct realtime_profiler_timestamp_t kernel_start_b;
    struct realtime_profiler_timestamp_t kernel_end_b;
    volatile uint32_t sync_request;
    volatile uint32_t sync_host_timestamp;

    // Concurrent device path. The host zeros this whole block before dispatch
    // kernels launch. At runtime every field has one writer as documented in
    // docs/realtime_profiler_clean_room_protocol.md.
    volatile uint32_t dispatch_d_ready;
    volatile uint32_t dispatch_s_ready;
    volatile uint32_t observer_ready;
    volatile uint32_t observer_stop_requested;

    // dispatch_d-owned reset state.
    volatile uint32_t stream_generation[REALTIME_PROFILER_MAX_STREAMS];

    // dispatch_s NCRISC -> TRISC0 per-stream descriptor rings. Descriptor word
    // order is runtime ID, start hi, start lo, completion target, generation.
    volatile uint32_t descriptor_write_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t descriptor_read_index[REALTIME_PROFILER_MAX_STREAMS];
    volatile uint32_t descriptor_words[REALTIME_PROFILER_DESCRIPTOR_QUEUE_WORDS];

    // TRISC0 -> dispatch_s NCRISC completed-record ring. The producer publishes
    // completed_write_index before its register-space epoch. dispatch_s publishes
    // its bounded low-24-bit consumer index directly in register space.
    volatile uint32_t completed_write_index;
    volatile uint32_t completed_words[REALTIME_PROFILER_COMPLETED_QUEUE_WORDS];

    // Sole writers: dispatch_s owns descriptor_full, unsupported_launch,
    // terminal_descriptor, terminal_record, and observer_stop_timeout;
    // TRISC0 owns reset_descriptor, observer_coalesced, stuck_head, and
    // completed_record; the reserved profiler BRISC owns device_ring.
    volatile uint32_t loss_descriptor_full;
    volatile uint32_t loss_unsupported_launch;
    volatile uint32_t loss_terminal_descriptor;
    volatile uint32_t loss_reset_descriptor;
    volatile uint32_t loss_observer_coalesced;
    volatile uint32_t loss_stuck_head;
    volatile uint32_t loss_completed_record;
    volatile uint32_t loss_terminal_record;
    volatile uint32_t loss_observer_stop_timeout;
    volatile uint32_t loss_device_ring;

    // Dedicated fault-injection and measurement storage. Production firmware
    // never reads or writes these words; test defines are added only to the
    // separately hashed firmware used by the two adversarial tests.
    // The reset-pause and observer-cycle defines are mutually exclusive. The
    // former uses word 0; the latter uses totals [0..2] and samples [3..5].
    volatile uint32_t test_protocol_words[6];
};
