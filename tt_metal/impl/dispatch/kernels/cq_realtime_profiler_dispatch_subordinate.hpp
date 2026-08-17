// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "hostdev/realtime_profiler_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_protocol.hpp"

// Stream register definitions
#define NOC_OVERLAY_START_ADDR 0xFFB40000
#define NOC_STREAM_REG_SPACE_SIZE 0x1000
#define STREAM_REG_ADDR(stream_id, reg_id) \
    ((NOC_OVERLAY_START_ADDR) + (((uint32_t)(stream_id)) * (NOC_STREAM_REG_SPACE_SIZE)) + (((uint32_t)(reg_id)) << 2))

// For Blackhole: STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX = 297
// For Wormhole: STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX = 64
#if defined(ARCH_BLACKHOLE)
#define STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX 297
#else
#define STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX 64
#endif

// Compile-time args from host (set in dispatch_s.cpp)
constexpr uint32_t first_stream_index = FIRST_STREAM_INDEX;
constexpr uint32_t num_streams_to_monitor = NUM_STREAMS_TO_MONITOR;

FORCE_INLINE uint32_t
realtime_profiler_descriptor_drop_total(volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg) {
    return rt_profiler_msg->start_descriptor_drop_count + rt_profiler_msg->unsupported_launch_drop_count +
           rt_profiler_msg->reset_descriptor_drop_count + rt_profiler_msg->stuck_descriptor_head_count +
           rt_profiler_msg->terminal_descriptor_drop_count;
}

FORCE_INLINE void try_complete_realtime_profiler_watermark(
    volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg,
    uint32_t stream_index,
    uint32_t current_count,
    uint32_t adopted_generation) {
    const uint32_t request_read_index = rt_profiler_msg->watermark_request_read_index[stream_index];
    const uint32_t request_write_index = rt_profiler_msg->watermark_request_write_index[stream_index];
    if (request_read_index == request_write_index || rt_profiler_msg->watermark_ready_read_index[stream_index] !=
                                                         rt_profiler_msg->watermark_ready_write_index[stream_index]) {
        return;
    }

    invalidate_l1_cache();
    const uint32_t request_generation = rt_profiler_msg->watermark_request_generation[stream_index];
    if (realtime_profiler_generation_after(request_generation, adopted_generation)) {
        // dispatch_d published a newer reset epoch after this TRISC0 pass
        // sampled the stream generation. Wait for the next pass to adopt it;
        // the target is not comparable in the older epoch.
        return;
    }
    const bool generation_changed = request_generation != adopted_generation;
    if (!generation_changed && !realtime_profiler_stream_count_ge<MEM_WORD_ADDR_WIDTH>(
                                   current_count, rt_profiler_msg->watermark_request_target[stream_index])) {
        return;
    }
    if (generation_changed) {
        // The request belongs to an older epoch already adopted by TRISC0. The
        // dispatch reset protocol guarantees that epoch was quiesced, but the
        // old target is no longer observable, so complete with a protocol error.
        rt_profiler_msg->watermark_protocol_error_count++;
    }

    rt_profiler_msg->watermark_ready_id[stream_index] = rt_profiler_msg->watermark_request_id[stream_index];
    rt_profiler_msg->watermark_ready_sequence[stream_index] = rt_profiler_msg->successful_record_sequence;
    rt_profiler_msg->watermark_ready_descriptor_drop_count[stream_index] =
        realtime_profiler_descriptor_drop_total(rt_profiler_msg);
    rt_profiler_msg->watermark_ready_observer_drop_count[stream_index] =
        rt_profiler_msg->completion_observer_drop_count + rt_profiler_msg->completion_observer_timeout_count;
    rt_profiler_msg->watermark_ready_record_drop_count[stream_index] =
        rt_profiler_msg->completed_record_drop_count + rt_profiler_msg->terminal_record_drop_count;
    rt_profiler_msg->watermark_ready_record_write_index[stream_index] = rt_profiler_msg->record_write_index;
    rt_profiler_msg->watermark_ready_protocol_error[stream_index] = generation_changed;
    asm volatile("fence w, w" ::: "memory");
    rt_profiler_msg->watermark_ready_write_index[stream_index]++;
    asm volatile("fence w, w" ::: "memory");
    rt_profiler_msg->watermark_request_read_index[stream_index] = request_read_index + 1;
}

FORCE_INLINE void dispatch_subordinate_realtime_profiler() {
    // Dispatch-core-local L1 region carved by DispatchMemMap (CommandQueueDeviceAddrType::
    // REALTIME_PROFILER_MSG). Address is supplied by host via the REALTIME_PROFILER_MSG_ADDR
    // compile-time define; mirrors cq_dispatch.cpp / cq_dispatch_subordinate.cpp on the same core.
    volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
        reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

    // Clear stale RT-profiler carve-out state left in L1 from prior runs.
    rt_profiler_msg->realtime_profiler_core_noc_xy = 0;
    rt_profiler_msg->realtime_profiler_remote_state_addr = 0;
    rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;

    // Wait until host explicitly enables RT profiler, or terminate if RT is not used.
    while (rt_profiler_msg->realtime_profiler_core_noc_xy == 0) {
        invalidate_l1_cache();
        if (rt_profiler_msg->terminate_requested != 0) {
            rt_profiler_msg->completion_observer_stopped = 1;
            return;
        }
    }

    static_assert(num_streams_to_monitor <= REALTIME_PROFILER_MAX_STREAMS);
    uint32_t adopted_generation[num_streams_to_monitor];
    bool stuck_head_reported[num_streams_to_monitor];
#ifdef RT_PROFILER_QUALIFICATION_HOOK
    // The opt-in host dispatch map reserves four words immediately after the
    // production protocol message and supplies their exact address. No such
    // reservation or define exists in the production layout/image.
    volatile tt_l1_ptr uint32_t* qualification_scratch =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_PROFILER_QUALIFICATION_SCRATCH_ADDR);
    constexpr uint32_t qualification_record_cycles_low = 0;
    constexpr uint32_t qualification_record_cycles_high = 1;
    constexpr uint32_t qualification_record_count_word = 2;
    constexpr uint32_t qualification_max_scan_cycles_word = 3;
    uint64_t qualification_record_handler_cycles = 0;
    uint32_t qualification_record_handler_count = 0;
    uint32_t qualification_max_scan_cycles = 0;
    for (uint32_t word = 0; word < REALTIME_PROFILER_QUALIFICATION_SCRATCH_WORDS; ++word) {
        qualification_scratch[word] = 0;
    }
    asm volatile("fence w, w" ::: "memory");
#endif
    for (uint32_t i = 0; i < num_streams_to_monitor; i++) {
        invalidate_l1_cache();
        adopted_generation[i] = rt_profiler_msg->stream_reset_generation[i];
        stuck_head_reported[i] = false;
    }

    while (rt_profiler_msg->terminate_requested == 0) {
#ifdef RT_PROFILER_QUALIFICATION_HOOK
        const uint64_t qualification_scan_start = get_timestamp();
#endif
        invalidate_l1_cache();
        for (uint32_t i = 0; i < num_streams_to_monitor; i++) {
            uint32_t stream_id = first_stream_index + i;
            volatile uint32_t* stream_reg =
                (volatile uint32_t*)STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

            invalidate_l1_cache();
            const uint32_t generation = rt_profiler_msg->stream_reset_generation[i];
            uint32_t read_index = rt_profiler_msg->start_descriptor_read_index[i];
            const uint32_t write_index = rt_profiler_msg->start_descriptor_write_index[i];

            // The producer publishes every descriptor payload before its write
            // index. One invalidation after observing that index makes the
            // entire snapshotted batch visible; invalidating once per entry
            // only adds polling overhead.
            if (read_index != write_index) {
                invalidate_l1_cache();
            }

            if (generation != adopted_generation[i]) {
                // Drop only descriptors from the old epoch. New-epoch descriptors
                // may already have been published behind them.
                while (read_index != write_index) {
                    const uint32_t slot = read_index & (REALTIME_PROFILER_START_QUEUE_CAPACITY - 1);
                    const uint32_t descriptor_offset =
                        (i * REALTIME_PROFILER_START_QUEUE_CAPACITY + slot) * REALTIME_PROFILER_START_DESCRIPTOR_WORDS;
                    volatile tt_l1_ptr uint32_t* descriptor =
                        &rt_profiler_msg->start_descriptor_words[descriptor_offset];
                    if (descriptor[4] == generation || realtime_profiler_generation_after(descriptor[4], generation)) {
                        break;
                    }
                    rt_profiler_msg->reset_descriptor_drop_count++;
                    read_index++;
                }
                rt_profiler_msg->start_descriptor_read_index[i] = read_index;
                adopted_generation[i] = generation;
            }

            const uint32_t current_count = *stream_reg & ((1u << MEM_WORD_ADDR_WIDTH) - 1);
            uint32_t ready_count = 0;
            uint32_t start_hi = 0;
            uint32_t start_lo = 0;
            uint32_t runtime_id = REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID;
            uint32_t scan_index = read_index;
            while (scan_index != write_index) {
                const uint32_t slot = scan_index & (REALTIME_PROFILER_START_QUEUE_CAPACITY - 1);
                const uint32_t descriptor_offset =
                    (i * REALTIME_PROFILER_START_QUEUE_CAPACITY + slot) * REALTIME_PROFILER_START_DESCRIPTOR_WORDS;
                volatile tt_l1_ptr uint32_t* descriptor = &rt_profiler_msg->start_descriptor_words[descriptor_offset];
                if (descriptor[4] != adopted_generation[i]) {
                    if (realtime_profiler_generation_after(descriptor[4], adopted_generation[i])) {
                        break;
                    }
                    rt_profiler_msg->reset_descriptor_drop_count++;
                    scan_index++;
                    continue;
                }
                if (!realtime_profiler_stream_count_ge<MEM_WORD_ADDR_WIDTH>(current_count, descriptor[3])) {
                    break;
                }
                start_hi = descriptor[0];
                start_lo = descriptor[1];
                runtime_id = descriptor[2];
                ready_count++;
                scan_index++;
            }

            if (ready_count == 0) {
                const bool full_with_unmet_head =
                    scan_index == read_index && read_index != write_index &&
                    realtime_profiler_queue_full(write_index, read_index, REALTIME_PROFILER_START_QUEUE_CAPACITY);
                if (full_with_unmet_head && !stuck_head_reported[i]) {
                    // An impossible target can otherwise look exactly like a
                    // slow program. Count the head once when it blocks a full
                    // ring; do not impose a wall-clock timeout on valid work.
                    rt_profiler_msg->stuck_descriptor_head_count++;
                    stuck_head_reported[i] = true;
                } else if (!full_with_unmet_head) {
                    stuck_head_reported[i] = false;
                }
                if (scan_index != read_index) {
                    rt_profiler_msg->start_descriptor_read_index[i] = scan_index;
                }
                try_complete_realtime_profiler_watermark(rt_profiler_msg, i, current_count, adopted_generation[i]);
                continue;
            }

            stuck_head_reported[i] = false;

#ifdef RT_PROFILER_QUALIFICATION_HOOK
            const uint64_t qualification_handler_start = get_timestamp();
#endif
            DeviceZoneScopedN("TRISC0-record-end-ts");
            if (ready_count > 1) {
                // One sampled tick cannot represent multiple distinct completion
                // events. Keep only the newest satisfied descriptor.
                rt_profiler_msg->completion_observer_drop_count += ready_count - 1;
            }

            uint32_t end_hi = 0;
            uint32_t end_lo = 0;
            read_realtime_wall_clock(&end_hi, &end_lo);
            invalidate_l1_cache();
            const uint32_t record_write_index = rt_profiler_msg->record_write_index;
            const uint32_t record_read_index = rt_profiler_msg->record_read_index;
            if (realtime_profiler_queue_full(
                    record_write_index, record_read_index, REALTIME_PROFILER_RECORD_QUEUE_CAPACITY)) {
                rt_profiler_msg->completed_record_drop_count++;
            } else {
                const uint32_t record_slot = record_write_index & (REALTIME_PROFILER_RECORD_QUEUE_CAPACITY - 1);
                volatile tt_l1_ptr uint32_t* record =
                    &rt_profiler_msg->record_words[record_slot * REALTIME_PROFILER_RECORD_WORDS];
                const uint32_t sequence = rt_profiler_msg->successful_record_sequence + 1;
                rt_profiler_msg->successful_record_sequence = sequence;
                record[0] = start_hi;
                record[1] = start_lo;
                record[2] = runtime_id;
                record[3] = (REALTIME_PROFILER_RECORD_SCHEMA_VERSION << 24) |
                            (REALTIME_PROFILER_RECORD_TYPE_INTERVAL << 16) | i;
                record[4] = end_hi;
                record[5] = end_lo;
                record[6] = sequence;
                record[7] =
                    rt_profiler_msg->start_descriptor_drop_count + rt_profiler_msg->unsupported_launch_drop_count +
                    rt_profiler_msg->reset_descriptor_drop_count + rt_profiler_msg->completion_observer_drop_count +
                    rt_profiler_msg->stuck_descriptor_head_count + rt_profiler_msg->completed_record_drop_count;
                asm volatile("fence w, w" ::: "memory");
                rt_profiler_msg->record_write_index = record_write_index + 1;
            }
            rt_profiler_msg->start_descriptor_read_index[i] = scan_index;
#ifdef RT_PROFILER_QUALIFICATION_HOOK
            qualification_record_handler_cycles += get_timestamp() - qualification_handler_start;
            qualification_record_handler_count++;
            qualification_scratch[qualification_record_cycles_low] =
                static_cast<uint32_t>(qualification_record_handler_cycles);
            qualification_scratch[qualification_record_cycles_high] =
                static_cast<uint32_t>(qualification_record_handler_cycles >> 32);
            qualification_scratch[qualification_record_count_word] = qualification_record_handler_count;
            asm volatile("fence w, w" ::: "memory");
#endif
            try_complete_realtime_profiler_watermark(rt_profiler_msg, i, current_count, adopted_generation[i]);
        }
#ifdef RT_PROFILER_QUALIFICATION_HOOK
        const uint32_t qualification_scan_cycles = static_cast<uint32_t>(get_timestamp() - qualification_scan_start);
        if (qualification_scan_cycles > qualification_max_scan_cycles) {
            qualification_max_scan_cycles = qualification_scan_cycles;
            qualification_scratch[qualification_max_scan_cycles_word] = qualification_max_scan_cycles;
            asm volatile("fence w, w" ::: "memory");
        }
#endif
    }
    asm volatile("fence w, w" ::: "memory");
    rt_profiler_msg->completion_observer_stopped = 1;
}
