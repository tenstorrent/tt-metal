// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <tracy/TracyTTDevice.hpp>
#include "context/context_types.hpp"
#include "realtime_profiler_consumer.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"

namespace tt::tt_metal {

// Renders program records as Tracy device zones and calibrates Tracy from the host/device clock-sync stream. Every
// hook is serialized on this consumer's dedicated delivery thread.
class RealtimeProfilerTracyConsumer final : public RealtimeProfilerConsumer {
public:
    explicit RealtimeProfilerTracyConsumer(ContextId context_id) : context_id_(context_id) {}
    ~RealtimeProfilerTracyConsumer() override;

    RealtimeProfilerTracyConsumer(const RealtimeProfilerTracyConsumer&) = delete;
    RealtimeProfilerTracyConsumer& operator=(const RealtimeProfilerTracyConsumer&) = delete;

    void on_records(const tt::ProgramRealtimeRecordBatch& batch) override;

private:
    // Establish a chip's Tracy context on its first record, then recalibrate whenever the record's device_cycle_offset
    // moves (i.e. a host<->device re-anchor happened).
    void CalibrateFromRecord(const tt::ProgramRealtimeRecord& record);
    // Create and calibrate a Tracy context for the given device.
    void AddDevice(uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency);
    // Handle a single program record.
    void HandleRecord(const tt::ProgramRealtimeRecord& record);
    // Send a GpuCalibration event to Tracy, updating the host-device clock mapping.
    void CalibrateDevice(uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);
    TracyTTCtx GetContext(uint32_t chip_id);
    bool ValidateHostClockDomain();
    void PublishDeviceProfilerSyncAnchor(
        uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency);

    void RecordSkippedZoneWithEndBeforeStart(const tt::ProgramRealtimeRecord& record, int64_t delta);
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
#if defined(TRACY_ENABLE)
    bool host_clock_checked_ = false;
    bool host_clock_valid_ = false;
#endif
    std::unordered_map<uint32_t, TracyTTCtx> tracy_contexts_;
    // Last device_cycle_offset applied to the Tracy context per chip; a change signals a re-anchor.
    std::unordered_map<uint32_t, int64_t> last_device_cycle_offset_by_chip_;
    // When the Tracy context was last recalibrated per chip. The servo re-anchors device_cycle_offset ~20x/s, but the
    // GpuCalibration server math derives its slope from the gap between consecutive calibrations, so recalibrating that
    // often collapses the device zones. The Tracy view only needs occasional drift correction, so throttle it.
    std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> last_calibrate_at_by_chip_;
    static constexpr std::chrono::seconds kTracyRecalibrateInterval{30};
    SkippedEndBeforeStartStats skipped_end_before_start_stats_;
};

}  // namespace tt::tt_metal
