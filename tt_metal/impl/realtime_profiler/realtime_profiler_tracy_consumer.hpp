// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <tracy/TracyTTDevice.hpp>
#include "context/context_types.hpp"
#include "realtime_profiler_consumer.hpp"

namespace tt::tt_metal {

// Renders program records as Tracy device zones and calibrates Tracy from the host/device clock-sync stream. Every
// hook is serialized on this consumer's dedicated delivery thread.
class RealtimeProfilerTracyConsumer final : public RealtimeProfilerConsumer {
public:
    explicit RealtimeProfilerTracyConsumer(ContextId context_id) : context_id_(context_id) {}
    ~RealtimeProfilerTracyConsumer() override;

    RealtimeProfilerTracyConsumer(const RealtimeProfilerTracyConsumer&) = delete;
    RealtimeProfilerTracyConsumer& operator=(const RealtimeProfilerTracyConsumer&) = delete;

    void on_records(const tt::tt_metal::experimental::ProgramRealtimeRecordBatch& batch) override;

private:
    // Establish a chip's Tracy context on its first record, then recalibrate whenever the record's device_cycle_offset
    // moves (i.e. a host<->device re-anchor happened).
    void CalibrateFromRecord(const tt::tt_metal::experimental::ProgramRealtimeRecord& record);
    // Create and calibrate a Tracy context for the given device; returns it (caller stores it in chips_).
    TracyTTCtx AddDevice(uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency);
    void HandleRecord(const tt::tt_metal::experimental::ProgramRealtimeRecord& record);
    // Send a GpuCalibration event to Tracy, updating the host-device clock mapping.
    void CalibrateDevice(uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);
    TracyTTCtx GetContext(uint32_t chip_id);
    bool ValidateHostClockDomain();
    // Convert a CLOCK_MONOTONIC host timestamp (the domain of clock_sync) into Tracy's rdtsc CPU-tick domain, which the
    // Tracy context/calibration APIs require. Reads both host clocks side by side to pin the offset.
    int64_t HostMonoNsToTracyCpuTicks(int64_t host_mono_ns);
    void PublishDeviceProfilerSyncAnchor(
        uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);

    void RecordSkippedZoneWithEndBeforeStart(
        const tt::tt_metal::experimental::ProgramRealtimeRecord& record, int64_t delta);
    void MaybeEmitSkippedZoneSummary();

    struct SkippedEndBeforeStartStats {
        uint64_t total_skipped = 0;
        uint64_t suppressed_since_last_summary = 0;
        bool logged_first_detail = false;
        std::unordered_map<uint32_t, uint64_t> count_by_runtime_id;
        std::unordered_map<uint32_t, uint64_t> count_by_chip_id;
        std::chrono::steady_clock::time_point last_summary_time{};
        static constexpr std::chrono::seconds kSummaryInterval{30};
    };

    ContextId context_id_;
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
    SkippedEndBeforeStartStats skipped_end_before_start_stats_;
};

}  // namespace tt::tt_metal
