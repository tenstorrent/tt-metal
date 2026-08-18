// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "hostdev/realtime_profiler_protocol_common.h"

static_assert(
    REALTIME_PROFILER_DESCRIPTOR_QUEUE_WORDS ==
    REALTIME_PROFILER_MAX_STREAMS * REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY * REALTIME_PROFILER_DESCRIPTOR_WORDS);
static_assert(
    REALTIME_PROFILER_COMPLETED_QUEUE_WORDS ==
    REALTIME_PROFILER_COMPLETED_QUEUE_CAPACITY * REALTIME_PROFILER_COMPLETED_RECORD_WORDS);
static_assert(
    REALTIME_PROFILER_COMPLETED_QUEUE_CAPACITY >=
    2 * REALTIME_PROFILER_MAX_STREAMS * REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY);
static_assert(sizeof(realtime_profiler_msg_t) <= 8 * 1024);

template <uint32_t Capacity>
FORCE_INLINE bool realtime_profiler_queue_full(uint32_t write_index, uint32_t read_index) {
    static_assert((Capacity & (Capacity - 1)) == 0);
    return static_cast<uint32_t>(write_index - read_index) >= Capacity;
}

FORCE_INLINE bool realtime_profiler_generation_after(uint32_t value, uint32_t reference) {
    return static_cast<int32_t>(value - reference) > 0;
}

FORCE_INLINE uint32_t realtime_profiler_next_publication_epoch(uint32_t epoch) {
    return (epoch + 1) & REALTIME_PROFILER_PUBLICATION_EPOCH_MASK;
}

FORCE_INLINE uint32_t realtime_profiler_observer_loss(volatile tt_l1_ptr realtime_profiler_msg_t* msg) {
    // stuck_head is a diagnostic for a queue that may still make progress; it
    // is not itself a discarded interval and must not inflate source loss.
    return msg->loss_reset_descriptor + msg->loss_observer_coalesced + msg->loss_completed_record;
}

FORCE_INLINE uint32_t realtime_profiler_dispatch_loss(volatile tt_l1_ptr realtime_profiler_msg_t* msg) {
    return msg->loss_descriptor_full + msg->loss_unsupported_launch + msg->loss_terminal_descriptor;
}
