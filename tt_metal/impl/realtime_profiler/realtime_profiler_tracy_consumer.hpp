// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <tracy/TracyTTDevice.hpp>
#include "context/context_types.hpp"
#include <tt-metalium/experimental/realtime_profiler.hpp>

namespace tt::tt_metal {

class RealtimeProfilerTracyConsumer {
public:
    explicit RealtimeProfilerTracyConsumer(ContextId context_id) : context_id_(context_id) {}

    // The handle arrives after registration returns, so the first batches can race it; retirement
    // waits for it (see on_records) rather than ever unregistering a guessed handle.
    void set_handle(experimental::ProgramRealtimeProfilerCallbackHandle handle) {
        handle_.store(handle, std::memory_order_release);
    }
    ~RealtimeProfilerTracyConsumer();

    RealtimeProfilerTracyConsumer(const RealtimeProfilerTracyConsumer&) = delete;
    RealtimeProfilerTracyConsumer& operator=(const RealtimeProfilerTracyConsumer&) = delete;

    void on_records(const experimental::ProgramRealtimeRecordBatch& batch);

private:
    // Creates a chip's Tracy context on its first record, refreshes the host-clock calibration,
    // and returns the chip's context.
    TracyTTCtx refresh_tracy_calibration(
        const experimental::ProgramRealtimeRecord& record, std::chrono::steady_clock::time_point now);
    TracyTTCtx add_device(uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency);
    void handle_record(TracyTTCtx ctx, const experimental::ProgramRealtimeRecord& record);
    bool validate_host_clock_domain();
    // Converts a steady_clock host time into Tracy's rdtsc CPU-tick domain.
    int64_t host_time_to_tracy_cpu_ticks(std::chrono::steady_clock::time_point host_time);
    // Re-measures the CLOCK_MONOTONIC-to-rdtsc slope and re-pins its reference point.
    void refresh_host_clock_mapping();
    // A side-by-side read of both host clocks, keeping the tightest of several attempts.
    std::pair<int64_t, int64_t> correlate_host_clocks();
    void publish_device_profiler_sync_anchor(
        uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);

    static constexpr experimental::ProgramRealtimeProfilerCallbackHandle kHandleNotSet =
        std::numeric_limits<experimental::ProgramRealtimeProfilerCallbackHandle>::max();

    ContextId context_id_;
    std::atomic<experimental::ProgramRealtimeProfilerCallbackHandle> handle_{kHandleNotSet};
    bool host_clock_checked_ = false;

    int64_t host_clock_ref_mono_ns_ = 0;
    int64_t host_clock_ref_tracy_ = 0;
    double host_clock_ticks_per_ns_ = 0.0;
    struct PerChip {
        TracyTTCtx ctx = nullptr;
        std::chrono::steady_clock::time_point last_calibrated;
    };
    std::vector<PerChip> chips_;
    // Scratch reused across records: the callback runs per record, and rebuilding these from
    // scratch is the drop-inducing cost the batch API exists to amortize.
    std::string file_scratch_;
    std::string name_scratch_;
};

}  // namespace tt::tt_metal
