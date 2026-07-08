// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <tracy/TracyTTDevice.hpp>
#include "data_collection.hpp"

namespace tt::tt_metal {

// Manages Tracy contexts for real-time profiler.
// Registers itself as a ProgramRealtimeProfilerCallback and routes records to the correct
// Tracy context based on chip_id. Thread-safe for AddDevice/RemoveDevice.
class RealtimeProfilerTracyHandler {
public:
    RealtimeProfilerTracyHandler();
    ~RealtimeProfilerTracyHandler();

    RealtimeProfilerTracyHandler(const RealtimeProfilerTracyHandler&) = delete;
    RealtimeProfilerTracyHandler& operator=(const RealtimeProfilerTracyHandler&) = delete;

    // Create and calibrate a Tracy context for the given device.
    void AddDevice(uint32_t chip_id, int64_t host_start, double first_timestamp, double frequency);

    // Remove and destroy the Tracy context for the given device.
    void RemoveDevice(uint32_t chip_id);

    // Callback handler invoked by the real-time profiler system for each program record.
    void HandleRecord(const tt::ProgramRealtimeRecord& record);

    // Push a sync-check marker on the device GPU timeline at the given device timestamp.
    // Used to verify host-device clock sync accuracy in the Tracy GUI.
    void PushSyncCheckMarker(uint32_t chip_id, uint64_t device_timestamp, double frequency);

    // Send a GpuCalibration event to Tracy, updating the host-device clock mapping.
    void CalibrateDevice(uint32_t chip_id, int64_t host_time, uint64_t device_timestamp, double frequency);

private:
    TracyTTCtx GetContext(uint32_t chip_id);

    void RecordSkippedZoneWithEndBeforeStart(const tt::ProgramRealtimeRecord& record, int64_t delta);
    void MaybeEmitSkippedZoneSummaryLocked();

    struct SkippedEndBeforeStartStats {
        uint64_t total_skipped = 0;
        uint64_t suppressed_since_last_summary = 0;
        bool logged_first_detail = false;
        std::unordered_map<uint32_t, uint64_t> count_by_runtime_id;
        std::unordered_map<uint32_t, uint64_t> count_by_chip_id;
        std::chrono::steady_clock::time_point last_summary_time{};
        static constexpr std::chrono::seconds kSummaryInterval{30};
    };

    std::mutex mutex_;
    std::unordered_map<uint32_t, TracyTTCtx> tracy_contexts_;
    SkippedEndBeforeStartStats skipped_end_before_start_stats_;
    tt::ProgramRealtimeProfilerCallbackHandle callback_handle_;
};

}  // namespace tt::tt_metal
