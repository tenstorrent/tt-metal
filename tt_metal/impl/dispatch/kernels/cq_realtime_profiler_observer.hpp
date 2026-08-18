// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "hostdev/realtime_profiler_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_protocol.hpp"

constexpr uint32_t realtime_profiler_first_stream = FIRST_STREAM_INDEX;
constexpr uint32_t realtime_profiler_num_streams = NUM_STREAMS_TO_MONITOR;
constexpr uint32_t realtime_profiler_scratch_stream = 8;
static uint32_t realtime_profiler_successful_sequence = 0;
static uint32_t realtime_profiler_completed_epoch = 0;

static_assert(realtime_profiler_num_streams <= REALTIME_PROFILER_MAX_STREAMS);

__attribute__((noinline, noclone)) void publish_realtime_profiler_completed_record(
    volatile tt_l1_ptr realtime_profiler_msg_t* msg,
    volatile tt_l1_ptr uint32_t* descriptor,
    uint32_t stream_index,
    uint32_t record_type,
    uint32_t end_hi,
    uint32_t end_lo) {
    const uint32_t write_index = msg->completed_write_index;
    const uint32_t consumer_low = NOC_STREAM_READ_REG(realtime_profiler_scratch_stream, STREAM_SCRATCH_2_REG_INDEX) &
                                  REALTIME_PROFILER_PUBLICATION_EPOCH_MASK;
    const uint32_t occupancy = ((write_index & REALTIME_PROFILER_PUBLICATION_EPOCH_MASK) - consumer_low) &
                               REALTIME_PROFILER_PUBLICATION_EPOCH_MASK;
    if (occupancy >= REALTIME_PROFILER_COMPLETED_QUEUE_CAPACITY) {
        msg->loss_completed_record++;
        return;
    }

    const uint32_t slot = write_index & (REALTIME_PROFILER_COMPLETED_QUEUE_CAPACITY - 1);
    volatile tt_l1_ptr uint32_t* record = &msg->completed_words[slot * REALTIME_PROFILER_COMPLETED_RECORD_WORDS];
    const uint32_t sequence = ++realtime_profiler_successful_sequence;
    record[0] = descriptor[1];
    record[1] = descriptor[2];
    record[2] = (descriptor[0] & 0xFFFF) | ((descriptor[4] & 0xFFFF) << 16);
    record[3] =
        (REALTIME_PROFILER_RECORD_SCHEMA_VERSION & 0xFF) | ((record_type & 0xF) << 8) | ((stream_index & 0xFF) << 16);
    record[4] = end_hi;
    record[5] = end_lo;
    record[6] = sequence;
    record[7] = realtime_profiler_observer_loss(msg);
    asm volatile("fence w, w" ::: "memory");
    msg->completed_write_index = write_index + 1;
    asm volatile("fence iorw, iorw" ::: "memory");
    realtime_profiler_completed_epoch = realtime_profiler_next_publication_epoch(realtime_profiler_completed_epoch);
    NOC_STREAM_WRITE_REG(
        realtime_profiler_scratch_stream, STREAM_SCRATCH_4_REG_INDEX, realtime_profiler_completed_epoch);
}

FORCE_INLINE void observe_realtime_profiler_stream(
    volatile tt_l1_ptr realtime_profiler_msg_t* msg,
    uint32_t stream_index,
    uint32_t write_index,
    uint32_t adopted_generation,
    uint32_t* active_mask,
    bool* stuck_head_reported) {
    uint32_t read_index = msg->descriptor_read_index[stream_index];
    uint32_t scan_index = read_index;
    const uint32_t stream_id = realtime_profiler_first_stream + stream_index;
    const uint32_t current_count = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    volatile tt_l1_ptr uint32_t* newest_descriptor = nullptr;
    uint32_t satisfied = 0;

    while (scan_index != write_index) {
        const uint32_t slot = scan_index & (REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY - 1);
        volatile tt_l1_ptr uint32_t* descriptor =
            &msg->descriptor_words
                 [(stream_index * REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY + slot) *
                  REALTIME_PROFILER_DESCRIPTOR_WORDS];
        const uint32_t generation = descriptor[4];
        if (generation != adopted_generation) {
            if (realtime_profiler_generation_after(generation, adopted_generation)) {
                break;
            }
            msg->loss_reset_descriptor++;
            scan_index++;
            continue;
        }
        if (!realtime_profiler_modular_ge<MEM_WORD_ADDR_WIDTH>(current_count, descriptor[3])) {
            break;
        }
        newest_descriptor = descriptor;
        satisfied++;
        scan_index++;
    }

    if (satisfied != 0) {
        if (satisfied > 1) {
            msg->loss_observer_coalesced += satisfied - 1;
        }
        uint32_t end_hi = 0;
        uint32_t end_lo = 0;
        read_realtime_wall_clock(&end_hi, &end_lo);
        publish_realtime_profiler_completed_record(
            msg, newest_descriptor, stream_index, REALTIME_PROFILER_RECORD_TYPE_INTERVAL, end_hi, end_lo);
        *stuck_head_reported = false;
    } else {
        const bool full_with_unsatisfied_head =
            read_index != write_index && scan_index == read_index &&
            realtime_profiler_queue_full<REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY>(write_index, read_index);
        if (full_with_unsatisfied_head && !*stuck_head_reported) {
            msg->loss_stuck_head++;
            *stuck_head_reported = true;
        } else if (!full_with_unsatisfied_head) {
            *stuck_head_reported = false;
        }
    }

    if (scan_index != read_index) {
        msg->descriptor_read_index[stream_index] = scan_index;
    }
    if (scan_index == write_index) {
        *active_mask &= ~(1u << stream_index);
    }
}

FORCE_INLINE void observe_realtime_profiler_reset(
    volatile tt_l1_ptr realtime_profiler_msg_t* msg,
    uint32_t stream_index,
    uint32_t new_generation,
    uint32_t* active_mask,
    bool* stuck_head_reported) {
    uint32_t read_index = msg->descriptor_read_index[stream_index];
    const uint32_t write_index = msg->descriptor_write_index[stream_index];
    const uint32_t previous_generation = new_generation - 1;
    volatile tt_l1_ptr uint32_t* newest_descriptor = nullptr;
    uint32_t satisfied = 0;

    while (read_index != write_index) {
        const uint32_t slot = read_index & (REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY - 1);
        volatile tt_l1_ptr uint32_t* descriptor =
            &msg->descriptor_words
                 [(stream_index * REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY + slot) *
                  REALTIME_PROFILER_DESCRIPTOR_WORDS];
        const uint32_t generation = descriptor[4];
        if (generation == new_generation || realtime_profiler_generation_after(generation, new_generation)) {
            break;
        }
        if (generation == previous_generation) {
            newest_descriptor = descriptor;
            satisfied++;
        } else {
            msg->loss_reset_descriptor++;
        }
        read_index++;
    }

    if (satisfied != 0) {
        if (satisfied > 1) {
            msg->loss_observer_coalesced += satisfied - 1;
        }
        uint32_t end_hi = 0;
        uint32_t end_lo = 0;
        read_realtime_wall_clock(&end_hi, &end_lo);
        publish_realtime_profiler_completed_record(
            msg, newest_descriptor, stream_index, REALTIME_PROFILER_RECORD_TYPE_RESET_OBSERVED, end_hi, end_lo);
    }
    msg->descriptor_read_index[stream_index] = read_index;
    *stuck_head_reported = false;
    if (read_index == write_index) {
        *active_mask &= ~(1u << stream_index);
    } else {
        *active_mask |= 1u << stream_index;
    }
}

FORCE_INLINE void dispatch_subordinate_realtime_profiler_observer() {
    volatile tt_l1_ptr realtime_profiler_msg_t* msg =
        reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

#pragma GCC unroll 1
    for (uint32_t i = 0; i < realtime_profiler_num_streams; ++i) {
        msg->descriptor_read_index[i] = 0;
    }
    msg->completed_write_index = 0;
#if REALTIME_PROFILER_TEST_OBSERVER_CYCLES
    for (uint32_t i = 0; i < 3; ++i) {
        msg->test_protocol_words[i] = 0;
        msg->test_protocol_words[3 + i] = 0;
    }
#endif
    NOC_STREAM_WRITE_REG(realtime_profiler_scratch_stream, STREAM_SCRATCH_4_REG_INDEX, 0);
    asm volatile("fence iorw, iorw" ::: "memory");
    msg->observer_ready = REALTIME_PROFILER_PROTOCOL_VERSION;

    while (msg->realtime_profiler_core_noc_xy == 0) {
        invalidate_l1_cache();
        if (msg->observer_stop_requested != 0 ||
            msg->realtime_profiler_remote_state_addr == REALTIME_PROFILER_REMOTE_STATE_DISABLED) {
            msg->observer_ready = 0;
            asm volatile("fence w, w" ::: "memory");
            return;
        }
    }

    // Every element is assigned from shared state before the first scan. Do
    // not zero-initialize this array: that emits eight redundant TRISC stores
    // in the observer's 2 KiB text budget.
    uint32_t adopted_generation[REALTIME_PROFILER_MAX_STREAMS];
    bool stuck_head_reported[REALTIME_PROFILER_MAX_STREAMS] = {false};
    uint32_t active_mask = 0;
    uint32_t observed_descriptor_epoch =
        NOC_STREAM_READ_REG(realtime_profiler_scratch_stream, STREAM_SCRATCH_5_REG_INDEX);
    uint32_t observed_reset_epoch = NOC_STREAM_READ_REG(realtime_profiler_scratch_stream, STREAM_SCRATCH_3_REG_INDEX);
    invalidate_l1_cache();
#pragma GCC unroll 1
    for (uint32_t i = 0; i < realtime_profiler_num_streams; ++i) {
        adopted_generation[i] = msg->stream_generation[i];
        if (msg->descriptor_read_index[i] != msg->descriptor_write_index[i]) {
            active_mask |= 1u << i;
        }
    }

    while (msg->observer_stop_requested == 0) {
#if REALTIME_PROFILER_TEST_OBSERVER_CYCLES
        uint32_t test_start_hi = 0;
        uint32_t test_start_lo = 0;
        const uint32_t test_active_count = __builtin_popcount(active_mask);
        read_realtime_wall_clock(&test_start_hi, &test_start_lo);
#endif
        if (active_mask != 0) {
            const uint32_t reset_epoch =
                NOC_STREAM_READ_REG(realtime_profiler_scratch_stream, STREAM_SCRATCH_3_REG_INDEX);
            if (reset_epoch != observed_reset_epoch) {
                observed_reset_epoch = reset_epoch;
                invalidate_l1_cache();
                for (uint32_t i = 0; i < realtime_profiler_num_streams; ++i) {
                    const uint32_t generation = msg->stream_generation[i];
                    if (generation != adopted_generation[i]) {
                        observe_realtime_profiler_reset(msg, i, generation, &active_mask, &stuck_head_reported[i]);
                        adopted_generation[i] = generation;
                    }
                }
            }
        }

        // Empty-path cost is exactly this descriptor-epoch register read. A
        // reset cannot require work with no active descriptor. A descriptor
        // from a newer generation remains pending for the next iteration,
        // whose reset check runs before it can be completed.
        const uint32_t descriptor_epoch =
            NOC_STREAM_READ_REG(realtime_profiler_scratch_stream, STREAM_SCRATCH_5_REG_INDEX);
        if (descriptor_epoch != observed_descriptor_epoch) {
            observed_descriptor_epoch = descriptor_epoch;
            invalidate_l1_cache();
            for (uint32_t i = 0; i < realtime_profiler_num_streams; ++i) {
                if (msg->descriptor_read_index[i] != msg->descriptor_write_index[i]) {
                    active_mask |= 1u << i;
                }
            }
        }

        uint32_t scan_mask = active_mask;
        while (scan_mask != 0) {
            const uint32_t stream_index = __builtin_ctz(scan_mask);
            scan_mask &= scan_mask - 1;
            observe_realtime_profiler_stream(
                msg,
                stream_index,
                msg->descriptor_write_index[stream_index],
                adopted_generation[stream_index],
                &active_mask,
                &stuck_head_reported[stream_index]);
        }
#if REALTIME_PROFILER_TEST_OBSERVER_CYCLES
        uint32_t test_end_hi = 0;
        uint32_t test_end_lo = 0;
        read_realtime_wall_clock(&test_end_hi, &test_end_lo);
        uint32_t test_bucket = 3;
        if (test_active_count == 0) {
            test_bucket = 0;
        } else if (test_active_count == 1) {
            test_bucket = 1;
        } else if (test_active_count == realtime_profiler_num_streams) {
            test_bucket = 2;
        }
        if (test_bucket < 3 && msg->test_protocol_words[3 + test_bucket] != 0xFFFFFFFF) {
            msg->test_protocol_words[test_bucket] += test_end_lo - test_start_lo;
            msg->test_protocol_words[3 + test_bucket]++;
        }
#endif
    }

    msg->observer_ready = 0;
    asm volatile("fence w, w" ::: "memory");
}
