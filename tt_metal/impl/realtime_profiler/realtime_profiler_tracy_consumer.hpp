// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
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
    // Creates a chip's Tracy context on its first record; recalibrates when device_cycle_offset changes (a
    // host<->device re-anchor).
    void CalibrateFromRecord(const experimental::ProgramRealtimeRecord& record);
    TracyTTCtx AddDevice(uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency);
    void HandleRecord(const experimental::ProgramRealtimeRecord& record);
    void CalibrateDevice(uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);
    TracyTTCtx GetContext(uint32_t chip_id);
    bool ValidateHostClockDomain();
    // Converts a steady_clock host time into Tracy's rdtsc CPU-tick domain, required by the Tracy calibration APIs.
    int64_t HostTimeToTracyCpuTicks(std::chrono::steady_clock::time_point host_time);
    void PublishDeviceProfilerSyncAnchor(
        uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);

    ContextId context_id_;
    experimental::ProgramRealtimeProfilerCallbackHandle handle_ = 0;
    bool host_clock_checked_ = false;
    bool host_clock_valid_ = false;
    // chip_ids are small and dense, so a flat vector (not a map) keeps the per-record lookup hash-free.
    struct PerChip {
        TracyTTCtx ctx = nullptr;
        int64_t last_seen_offset = 0;  // clock_sync.device_cycle_offset; a change signals a host<->device re-anchor
    };
    std::vector<PerChip> chips_;
};

}  // namespace tt::tt_metal
