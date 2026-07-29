// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    // Establish a chip's Tracy context on its first record, then recalibrate whenever the record's device_cycle_offset
    // moves (i.e. a host<->device re-anchor happened).
    void CalibrateFromRecord(const experimental::ProgramRealtimeRecord& record);
    // Create and calibrate a Tracy context for the given device; returns it (caller stores it in chips_).
    TracyTTCtx AddDevice(uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency);
    void HandleRecord(const experimental::ProgramRealtimeRecord& record);
    // Send a GpuCalibration event to Tracy, updating the host-device clock mapping.
    void CalibrateDevice(uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);
    TracyTTCtx GetContext(uint32_t chip_id);
    bool ValidateHostClockDomain();
    // Convert a CLOCK_MONOTONIC host timestamp (the domain of clock_sync) into Tracy's rdtsc CPU-tick domain, which the
    // Tracy context/calibration APIs require. Reads both host clocks side by side to pin the offset.
    int64_t HostMonoNsToTracyCpuTicks(int64_t host_mono_ns);
    void PublishDeviceProfilerSyncAnchor(
        uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);

    ContextId context_id_;
    experimental::ProgramRealtimeProfilerCallbackHandle handle_ = 0;
    bool host_clock_checked_ = false;
    bool host_clock_valid_ = false;
    // Per-chip Tracy state, indexed by chip_id. chip_ids are small and dense, so a flat vector keeps the per-record
    // path hash-free (a vector index + an offset compare) — see CalibrateFromRecord / GetContext.
    struct PerChip {
        TracyTTCtx ctx = nullptr;
        int64_t last_seen_offset =
            0;  // last clock_sync.device_cycle_offset seen; a change signals a host<->device re-anchor
    };
    std::vector<PerChip> chips_;
};

}  // namespace tt::tt_metal
