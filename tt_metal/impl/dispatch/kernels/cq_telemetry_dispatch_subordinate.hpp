// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "hostdevcommon/dispatch_telemetry_types.hpp"
#include "tt_metal/impl/dispatch/kernels/telemetry.hpp"
#include "risc_common.h"

constexpr uint32_t first_stream_index = FIRST_STREAM_INDEX;
constexpr uint32_t total_sub_devices = TOTAL_SUB_DEVICES;
constexpr bool telemetry_enabled = !DISPATCH_TELEMETRY_DISABLED;
constexpr uint32_t dispatch_telemetry_base = DISPATCH_TELEMETRY_ADDR;
constexpr uint32_t dispatch_telemetry_control_addr = DISPATCH_TELEMETRY_CONTROL_ADDR;

FORCE_INLINE uint32_t stream_wrap_delta(uint32_t current, uint32_t previous) {
    constexpr uint32_t shift = 32 - MEM_WORD_ADDR_WIDTH;
    return ((current - previous) << shift) >> shift;
}

// Compress work runtime into avg_work_runtime_per_worker only when needed to avoid losing cycles
FORCE_INLINE void compress_work_runtime(
    uint64_t& avg_work_runtime_per_worker, uint64_t& current_sub_device_work_runtime, uint32_t workers_per_sub_device) {
    avg_work_runtime_per_worker += current_sub_device_work_runtime / workers_per_sub_device;
    current_sub_device_work_runtime = 0;
    auto dispatch_telemetry =
        reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::DispatchCoreTelemetry*>(dispatch_telemetry_base);
    dispatch_telemetry->avg_work_runtime_per_worker = avg_work_runtime_per_worker;
}

FORCE_INLINE void dispatch_subordinate_telemetry() {
    if (!telemetry_enabled) {
        return;
    }

    bool done = false;

    uint32_t stream_sem_counter[total_sub_devices] = {0};
    for (uint32_t i = 0; i < total_sub_devices; ++i) {
        stream_sem_counter[i] =
            NOC_STREAM_READ_REG(i + first_stream_index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
        DEVICE_PRINT("starting stream_sem_counter[{}]: {}\n", i, stream_sem_counter[i]);
    }

    // Used only to cache the current sem count, cleared every loop
    uint64_t curr_stream_sem_count[total_sub_devices] = {0};

    // local telemetry copies for read access
    uint64_t last_work_launch_timestamp[total_sub_devices] = {0};
    uint64_t avg_work_runtime_per_worker = 0;
    uint64_t current_sub_device_work_runtime[total_sub_devices] = {0};
    uint32_t completion_count[total_sub_devices] = {0};
    uint32_t workers_per_sub_device[total_sub_devices] = {0};

    uint32_t sub_device_update_sem = 0;
    uint32_t stream_reset_update_sem = 0;
    auto dispatch_telemetry_control =
        reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::DispatchTelemetryControl*>(dispatch_telemetry_control_addr);
    auto sub_device_update_sem_ptr = &dispatch_telemetry_control->sub_device_worker_counts_update;
    auto stream_reset_update_sem_ptr = &dispatch_telemetry_control->worker_stream_reset_update;
    auto telemetry_compute_terminate = &dispatch_telemetry_control->compute_terminate;
    auto dispatch_telemetry =
        reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::DispatchCoreTelemetry*>(dispatch_telemetry_base);

    // Start with all workers in the complete state until work is detected
    for (uint32_t i = 0; i < total_sub_devices; ++i) {
        completion_count[i] = workers_per_sub_device[i];
        dispatch_telemetry->completion_count[i] = completion_count[i];
    }

    while (!done) {
        if (*telemetry_compute_terminate != 0) {
            done = true;
            break;
        }

        for (uint32_t i = 0; i < total_sub_devices; ++i) {
            const uint32_t latest_sub_device_update_sem = *sub_device_update_sem_ptr;
            const uint32_t latest_stream_reset_update_sem = *stream_reset_update_sem_ptr;
            const bool sub_device_update = latest_sub_device_update_sem != sub_device_update_sem;
            const bool stream_reset_update = latest_stream_reset_update_sem != stream_reset_update_sem;

            // If signals are detected, immediately reset and break
            if (sub_device_update || stream_reset_update) {
                for (uint32_t i = 0; i < total_sub_devices; ++i) {
                    if (completion_count[i] < workers_per_sub_device[i]) {
                        // Finish inflight work
                        while (completion_count[i] < workers_per_sub_device[i]) {
                            uint64_t current_timestamp = get_timestamp();
                            uint64_t delta_work_runtime = current_timestamp - last_work_launch_timestamp[i];

                            const bool will_overflow =
                                UINT64_MAX - current_sub_device_work_runtime[i] < delta_work_runtime;
                            if (will_overflow) {
                                compress_work_runtime(
                                    avg_work_runtime_per_worker,
                                    current_sub_device_work_runtime[i],
                                    workers_per_sub_device[i]);
                            }
                            current_sub_device_work_runtime[i] += delta_work_runtime;
                            completion_count[i]++;
                        }

                        compress_work_runtime(
                            avg_work_runtime_per_worker, current_sub_device_work_runtime[i], workers_per_sub_device[i]);
                        current_sub_device_work_runtime[i] = 0;
                        dispatch_telemetry->current_sub_device_work_runtime[i] = current_sub_device_work_runtime[i];
                    }

                    if (sub_device_update) {
                        workers_per_sub_device[i] = dispatch_telemetry->workers_per_sub_device[i];
                    }

                    if (stream_reset_update) {
                        stream_sem_counter[i] = 0;
                    } else {
                        stream_sem_counter[i] = NOC_STREAM_READ_REG(
                            i + first_stream_index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
                    }

                    completion_count[i] = workers_per_sub_device[i];  // triggers new workload check
                    dispatch_telemetry->completion_count[i] = completion_count[i];
                }

                dispatch_telemetry->avg_work_runtime_per_worker = avg_work_runtime_per_worker;
                sub_device_update_sem = latest_sub_device_update_sem;
                stream_reset_update_sem = latest_stream_reset_update_sem;
                break;
            }

            if (workers_per_sub_device[i] == 0) {
                continue;
            }

            // Stream count with dispatch_s is a rolling counter unless explicit reset flag is sent
            curr_stream_sem_count[i] =
                NOC_STREAM_READ_REG(i + first_stream_index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

            uint32_t delta_sem_count =
                stream_wrap_delta(static_cast<uint32_t>(curr_stream_sem_count[i]), stream_sem_counter[i]);

            if (delta_sem_count != 0) {
                DEVICE_PRINT(
                    "delta_sem_count: {}, curr_stream_sem_count: {}, stream_sem_counter: {}\n",
                    delta_sem_count,
                    curr_stream_sem_count[i],
                    stream_sem_counter[i]);
            }

            uint64_t current_timestamp = get_timestamp();
            dispatch_telemetry->current_timestamp = current_timestamp;

            // while completion count is less than total workers, use local last timestamp
            // Once completion count exceeds, update last timestamp with current timestamp if it's the latest
            // Skip everything in between
            uint64_t delta_work_runtime = current_timestamp - last_work_launch_timestamp[i];
            while (delta_sem_count > 0 && completion_count[i] < workers_per_sub_device[i]) {
                const bool will_overflow = UINT64_MAX - current_sub_device_work_runtime[i] < delta_work_runtime;
                if (will_overflow) {
                    compress_work_runtime(
                        avg_work_runtime_per_worker, current_sub_device_work_runtime[i], workers_per_sub_device[i]);
                }
                current_sub_device_work_runtime[i] += delta_work_runtime;
                completion_count[i]++;
                delta_sem_count--;
            }

            if (completion_count[i] == workers_per_sub_device[i]) {
                uint64_t current_last_work_launch_timestamp = 0;
                {
                    auto timestamp_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        &dispatch_telemetry->last_work_launch_timestamp[i]);
                    uint32_t timestamp_low_check_1 = 0;
                    uint32_t timestamp_low_check_2 = 0;
                    uint32_t timestamp_high = 0;
                    do {
                        timestamp_low_check_1 = timestamp_words[0];
                        timestamp_high = timestamp_words[1];
                        timestamp_low_check_2 = timestamp_words[0];
                    } while (timestamp_low_check_1 != timestamp_low_check_2);
                    current_last_work_launch_timestamp = timestamp_high;
                    current_last_work_launch_timestamp <<= 32;
                    current_last_work_launch_timestamp |= timestamp_low_check_1;
                }

                const bool new_workload = current_last_work_launch_timestamp != last_work_launch_timestamp[i];
                if (new_workload) {
                    // Catch up to the latest workload, drop any we missed since last_work_launch_timestamp
                    // information for those runs has been lost
                    delta_sem_count = delta_sem_count % workers_per_sub_device[i];

                    completion_count[i] = 0;
                    last_work_launch_timestamp[i] = current_last_work_launch_timestamp;
                }
            }

            stream_sem_counter[i] = curr_stream_sem_count[i] - delta_sem_count;
            dispatch_telemetry->completion_count[i] = completion_count[i];
            dispatch_telemetry->current_sub_device_work_runtime[i] = current_sub_device_work_runtime[i];
        }
    }
}
