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

// Renders program records as Tracy device zones and calibrates Tracy from the host/device clock-sync stream. Every
// hook is serialized on this consumer's own thread.
class RealtimeProfilerTracyConsumer {
public:
    explicit RealtimeProfilerTracyConsumer(ContextId context_id) : context_id_(context_id) {}

    void set_handle(experimental::ProgramRealtimeProfilerCallbackHandle handle) { handle_ = handle; }
    ~RealtimeProfilerTracyConsumer();

    RealtimeProfilerTracyConsumer(const RealtimeProfilerTracyConsumer&) = delete;
    RealtimeProfilerTracyConsumer& operator=(const RealtimeProfilerTracyConsumer&) = delete;

    void on_records(const experimental::ProgramRealtimeRecordBatch& batch);

private:
    // Creates a chip's Tracy context on its first record, and refreshes the host-clock calibration on a slow timer.
    void CalibrateFromRecord(const experimental::ProgramRealtimeRecord& record);
    TracyTTCtx AddDevice(uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency);
    void HandleRecord(const experimental::ProgramRealtimeRecord& record);
    void CalibrateDevice(uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);
    TracyTTCtx GetContext(uint32_t chip_id);
    bool ValidateHostClockDomain();
    // Converts a steady_clock host time into Tracy's rdtsc CPU-tick domain, required by the Tracy calibration APIs.
    int64_t HostTimeToTracyCpuTicks(std::chrono::steady_clock::time_point host_time);
    // Re-measures the CLOCK_MONOTONIC-to-rdtsc slope and re-pins its reference point.
    void RefreshHostClockMapping();
    // A side-by-side read of both host clocks, keeping the tightest of several attempts.
    std::pair<int64_t, int64_t> CorrelateHostClocks();
    void PublishDeviceProfilerSyncAnchor(
        uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);

    ContextId context_id_;
    experimental::ProgramRealtimeProfilerCallbackHandle handle_ = 0;
    bool host_clock_checked_ = false;
    bool host_clock_valid_ = false;

    // CLOCK_MONOTONIC to rdtsc, as a measured affine relation rather than the nominal one Tracy's timer multiplier
    // describes. Host-wide, not per chip.
    int64_t host_clock_ref_mono_ns_ = 0;
    int64_t host_clock_ref_tracy_ = 0;
    int64_t host_clock_slope_mono_ns_ = 0;
    int64_t host_clock_slope_tracy_ = 0;
    double host_clock_ticks_per_ns_ = 0.0;
    // chip_ids are small and dense, so a flat vector (not a map) keeps the per-record lookup hash-free.
    struct PerChip {
        TracyTTCtx ctx = nullptr;
        std::chrono::steady_clock::time_point last_calibrated;
    };
    std::vector<PerChip> chips_;
};

}  // namespace tt::tt_metal
