// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>
#include <tracy/TracyTTDevice.hpp>
#include "context/context_types.hpp"
#include <tt-metalium/experimental/realtime_profiler.hpp>

namespace tt::tt_metal {

class RealtimeProfilerTracyConsumer {
public:
    explicit RealtimeProfilerTracyConsumer(ContextId context_id) : context_id_(context_id) {}

    void set_handle(experimental::ProgramRealtimeProfilerCallbackHandle handle) { handle_ = handle; }
    ~RealtimeProfilerTracyConsumer();

    RealtimeProfilerTracyConsumer(const RealtimeProfilerTracyConsumer&) = delete;
    RealtimeProfilerTracyConsumer& operator=(const RealtimeProfilerTracyConsumer&) = delete;

    void on_records(const experimental::ProgramRealtimeRecordBatch& batch);

private:
    // Creates a chip's Tracy context on its first record, and refreshes the host-clock calibration.
    void refresh_tracy_calibration(
        const experimental::ProgramRealtimeRecord& record, std::chrono::steady_clock::time_point now);
    TracyTTCtx add_device(uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency);
    void handle_record(const experimental::ProgramRealtimeRecord& record);
    void calibrate_device(uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);
    TracyTTCtx get_context(uint32_t chip_id);
    bool validate_host_clock_domain();
    // Converts a steady_clock host time into Tracy's rdtsc CPU-tick domain.
    int64_t host_time_to_tracy_cpu_ticks(std::chrono::steady_clock::time_point host_time);
    // Re-measures the CLOCK_MONOTONIC-to-rdtsc slope and re-pins its reference point.
    void refresh_host_clock_mapping();
    // A side-by-side read of both host clocks, keeping the tightest of several attempts.
    std::pair<int64_t, int64_t> correlate_host_clocks();
    void publish_device_profiler_sync_anchor(
        uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);

    ContextId context_id_;
    experimental::ProgramRealtimeProfilerCallbackHandle handle_ = 0;
    bool host_clock_checked_ = false;

    int64_t host_clock_ref_mono_ns_ = 0;
    int64_t host_clock_ref_tracy_ = 0;
    double host_clock_ticks_per_ns_ = 0.0;
    struct PerChip {
        TracyTTCtx ctx = nullptr;
        std::chrono::steady_clock::time_point last_calibrated;
    };
    std::vector<PerChip> chips_;
};

}  // namespace tt::tt_metal
