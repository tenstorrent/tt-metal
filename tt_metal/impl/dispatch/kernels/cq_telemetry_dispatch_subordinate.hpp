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
constexpr uintptr_t dispatch_telemetry_base = DISPATCH_TELEMETRY_ADDR;
constexpr uintptr_t dispatch_telemetry_control_addr = DISPATCH_TELEMETRY_CONTROL_ADDR;

FORCE_INLINE uint32_t stream_wrap_delta(uint32_t current, uint32_t previous) {
    constexpr uint32_t shift = 32 - MEM_WORD_ADDR_WIDTH;
    return ((current - previous) << shift) >> shift;
}

// Compress work runtime into avg_work_runtime_per_worker only when needed to avoid losing cycles
FORCE_INLINE void compress_work_runtime(
    uint64_t& avg_work_runtime_per_worker, uint64_t& current_sub_device_work_runtime, uint32_t workers_per_sub_device) {
    if (workers_per_sub_device == 0) {
        return;
    }
    avg_work_runtime_per_worker += current_sub_device_work_runtime / workers_per_sub_device;
    current_sub_device_work_runtime = 0;
    auto dispatch_telemetry =
        reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::dispatch_telemetry_types::DispatchCoreTelemetry*>(
            dispatch_telemetry_base);
    dispatch_telemetry->avg_work_runtime_per_worker = avg_work_runtime_per_worker;
}

FORCE_INLINE void dispatch_subordinate_telemetry() {
    if (!telemetry_enabled) {
        return;
    }

    bool done = false;
    uint32_t working_sub_device_count = 0;
    uint64_t work_runtime_start = 0;

    // Local telemetry copies for read access
    uint64_t last_work_launch_timestamp[total_sub_devices] = {0};
    uint64_t avg_work_runtime_per_worker = 0;
    uint64_t current_sub_device_work_runtime[total_sub_devices] = {0};
    uint64_t utilization_work_runtime = 0;
    uint32_t completion_count[total_sub_devices] = {0};
    uint32_t workers_per_sub_device[total_sub_devices] = {0};

    // Local counts for control semaphores
    uint32_t local_sub_device_update_sem = 0;
    uint32_t local_stream_reset_update_sem = 0;

    uint32_t local_stream_sem_counter[total_sub_devices] = {0};
    for (uint32_t i = 0; i < total_sub_devices; ++i) {
        local_stream_sem_counter[i] =
            NOC_STREAM_READ_REG(i + first_stream_index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    }

    auto dispatch_telemetry =
        reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::dispatch_telemetry_types::DispatchCoreTelemetry*>(
            dispatch_telemetry_base);
    auto dispatch_telemetry_control =
        reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::dispatch_telemetry_types::DispatchTelemetryControl*>(
            dispatch_telemetry_control_addr);

    // Start with all workers in the complete state until work is detected
    for (uint32_t i = 0; i < total_sub_devices; ++i) {
        completion_count[i] = workers_per_sub_device[i];
        dispatch_telemetry->completion_count[i] = completion_count[i];
    }

    while (!done) {
        if (dispatch_telemetry_control->compute_terminate != 0) {
            done = true;
            break;
        }

        // Update current time every loop
        uint64_t current_timestamp = get_timestamp();
        dispatch_telemetry->current_timestamp = current_timestamp;

        const uint32_t latest_sub_device_update_sem = dispatch_telemetry_control->sub_device_worker_counts_update;
        const uint32_t latest_stream_reset_update_sem = dispatch_telemetry_control->worker_stream_reset_update;
        const bool sub_device_update = latest_sub_device_update_sem != local_sub_device_update_sem;
        const bool stream_reset_update = latest_stream_reset_update_sem != local_stream_reset_update_sem;

        if (sub_device_update || stream_reset_update) {
            for (uint32_t i = 0; i < total_sub_devices; ++i) {
                // Finish inflight work
                if (completion_count[i] < workers_per_sub_device[i]) {
                    uint64_t delta_work_runtime = current_timestamp - last_work_launch_timestamp[i];
                    while (completion_count[i] < workers_per_sub_device[i]) {
                        const bool will_overflow = UINT64_MAX - current_sub_device_work_runtime[i] < delta_work_runtime;
                        if (will_overflow) {
                            compress_work_runtime(
                                avg_work_runtime_per_worker,
                                current_sub_device_work_runtime[i],
                                workers_per_sub_device[i]);
                        }
                        current_sub_device_work_runtime[i] += delta_work_runtime;
                        completion_count[i]++;
                    }
                }

                if (sub_device_update) {
                    // If the workers_per_sub_device has changed, the cumulative work runtime must be compressed
                    if (dispatch_telemetry->workers_per_sub_device[i] != workers_per_sub_device[i]) {
                        compress_work_runtime(
                            avg_work_runtime_per_worker, current_sub_device_work_runtime[i], workers_per_sub_device[i]);
                        current_sub_device_work_runtime[i] = 0;
                        dispatch_telemetry->current_sub_device_work_runtime[i] = current_sub_device_work_runtime[i];
                        workers_per_sub_device[i] = dispatch_telemetry->workers_per_sub_device[i];
                    }
                }

                if (stream_reset_update) {
                    local_stream_sem_counter[i] = 0;
                } else {
                    local_stream_sem_counter[i] =
                        NOC_STREAM_READ_REG(i + first_stream_index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
                }

                completion_count[i] = workers_per_sub_device[i];  // triggers new workload check
                dispatch_telemetry->completion_count[i] = completion_count[i];
            }

            if (working_sub_device_count > 0) {
                working_sub_device_count = 0;
                utilization_work_runtime += current_timestamp - work_runtime_start;
                dispatch_telemetry->utilization_work_runtime = utilization_work_runtime;
                work_runtime_start = 0;
                dispatch_telemetry->work_runtime_start = work_runtime_start;
            }

            dispatch_telemetry->avg_work_runtime_per_worker = avg_work_runtime_per_worker;
            local_sub_device_update_sem = latest_sub_device_update_sem;
            local_stream_reset_update_sem = latest_stream_reset_update_sem;
        }

        for (uint32_t i = 0; i < total_sub_devices; ++i) {
            if (workers_per_sub_device[i] == 0) {
                continue;
            }

            // Check if new workload arrived if all expected workers are completed
            if (completion_count[i] == workers_per_sub_device[i]) {
                uint64_t current_last_work_launch_timestamp = 0;
                uint32_t current_launched_work_start_stream_sem = 0;
                {
                    auto timestamp_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        &dispatch_telemetry->last_work_launch_timestamp[i]);
                    uint32_t timestamp_low = 0;
                    uint32_t timestamp_high = 0;
                    uint32_t launched_work_sequence_counter_check_start = 0;
                    uint32_t launched_work_sequence_counter_check_end = 0;

                    // Ensure last work timestamp is stable across two 32 bit reads
                    do {
                        launched_work_sequence_counter_check_start =
                            dispatch_telemetry_control->launched_work_sequence_counter[i];
                        timestamp_low = timestamp_words[0];
                        timestamp_high = timestamp_words[1];
                        current_launched_work_start_stream_sem =
                            dispatch_telemetry_control->launched_work_start_stream_sem[i];
                        launched_work_sequence_counter_check_end =
                            dispatch_telemetry_control->launched_work_sequence_counter[i];
                    } while (launched_work_sequence_counter_check_start != launched_work_sequence_counter_check_end ||
                             (launched_work_sequence_counter_check_start & 1));

                    current_last_work_launch_timestamp = timestamp_high;
                    current_last_work_launch_timestamp <<= 32;
                    current_last_work_launch_timestamp |= timestamp_low;
                }

                const bool new_workload = current_last_work_launch_timestamp != last_work_launch_timestamp[i];
                if (new_workload) {
                    // Catch up to the latest workload, drop any we missed since last_work_launch_timestamp
                    // information for those runs has been lost
                    local_stream_sem_counter[i] = current_launched_work_start_stream_sem;
                    completion_count[i] = 0;
                    last_work_launch_timestamp[i] = current_last_work_launch_timestamp;

                    // Track the transition from no work running to any work running for the utilization calculation
                    if (working_sub_device_count == 0) {
                        work_runtime_start = current_last_work_launch_timestamp;
                        dispatch_telemetry->work_runtime_start = work_runtime_start;
                    }
                    working_sub_device_count++;
                }
            }
            // Use cached timing values to evaluate work runtime
            else {
                // Stream count with dispatch_s is a rolling counter unless explicit reset flag is sent
                const uint32_t curr_stream_sem_count =
                    NOC_STREAM_READ_REG(i + first_stream_index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

                uint32_t delta_sem_count =
                    stream_wrap_delta(static_cast<uint32_t>(curr_stream_sem_count), local_stream_sem_counter[i]);

                // Get latest timestamp for work runtime calculation
                uint64_t latest_current_timestamp = get_timestamp();
                dispatch_telemetry->current_timestamp = latest_current_timestamp;

                uint64_t delta_work_runtime = latest_current_timestamp - last_work_launch_timestamp[i];
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

                // If the workload is complete, add the total work runtime to utilization
                if (completion_count[i] == workers_per_sub_device[i] && working_sub_device_count > 0) {
                    working_sub_device_count--;
                    // Update the utilization calculation if transitioning to no work running
                    if (working_sub_device_count == 0) {
                        utilization_work_runtime += latest_current_timestamp - work_runtime_start;
                        dispatch_telemetry->utilization_work_runtime = utilization_work_runtime;
                        work_runtime_start = 0;
                        dispatch_telemetry->work_runtime_start = work_runtime_start;
                    }
                }

                local_stream_sem_counter[i] = curr_stream_sem_count - delta_sem_count;
                dispatch_telemetry->completion_count[i] = completion_count[i];
                dispatch_telemetry->current_sub_device_work_runtime[i] = current_sub_device_work_runtime[i];
            }
        }
    }
}
