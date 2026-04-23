// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_tracy_handler.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#endif

namespace tt::tt_metal {

namespace {

#if defined(TRACY_ENABLE)
tracy::TTDeviceMarker make_marker(
    const tt::ProgramRealtimeRecord& record,
    uint64_t timestamp,
    tracy::TTDeviceMarkerType type,
    const std::string& file_str) {
    constexpr uint32_t kRealtimeProfilerCore_X = 100;
    constexpr uint32_t kRealtimeProfilerCore_Y = 100;

    tracy::TTDeviceMarker marker;
    marker.chip_id = record.chip_id;
    marker.core_x = kRealtimeProfilerCore_X;
    marker.core_y = kRealtimeProfilerCore_Y;
    marker.risc = tracy::RiscType::BRISC;
    marker.timestamp = timestamp;
    marker.runtime_host_id = record.program_id;
    marker.marker_name = fmt::format("Program_{}", record.program_id);
    marker.marker_type = type;
    marker.file = file_str;
    marker.line = 0;
    marker.color = tracy::Color::LightBlue;
    return marker;
}
#endif

}  // namespace

RealtimeProfilerTracyHandler::RealtimeProfilerTracyHandler() {
    callback_handle_ = tt::RegisterProgramRealtimeProfilerCallback(
        [this](const tt::ProgramRealtimeRecord& record) { HandleRecord(record); });
}

RealtimeProfilerTracyHandler::~RealtimeProfilerTracyHandler() {
    tt::UnregisterProgramRealtimeProfilerCallback(callback_handle_);

#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : tracy_contexts_) {
        TracyTTDestroy(entry.second);
    }
    tracy_contexts_.clear();
#endif
}

void RealtimeProfilerTracyHandler::AddDevice(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] int64_t host_start,
    [[maybe_unused]] double first_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracy_contexts_.contains(chip_id)) {
        log_warning(tt::LogMetal, "RealtimeProfilerTracyHandler: device {} already added, skipping", chip_id);
        return;
    }

    TracyTTCtx ctx = TracyTTContext();
    TracyTTContextPopulate(ctx, host_start, first_timestamp, frequency);
    std::string name = fmt::format("Device {}:", chip_id);
    TracyTTContextName(ctx, name.c_str(), name.size());

    tracy_contexts_[chip_id] = ctx;
#endif
}

void RealtimeProfilerTracyHandler::RemoveDevice([[maybe_unused]] uint32_t chip_id) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tracy_contexts_.find(chip_id);
    if (it == tracy_contexts_.end()) {
        return;
    }
    TracyTTDestroy(it->second);
    tracy_contexts_.erase(it);
#endif
}

TracyTTCtx RealtimeProfilerTracyHandler::GetContext(uint32_t chip_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tracy_contexts_.find(chip_id);
    return it != tracy_contexts_.end() ? it->second : nullptr;
}

void RealtimeProfilerTracyHandler::HandleRecord([[maybe_unused]] const tt::ProgramRealtimeRecord& record) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetContext(record.chip_id);
    if (!ctx) {
        return;
    }

    if (record.end_timestamp < record.start_timestamp) {
        auto delta = static_cast<int64_t>(record.end_timestamp) - static_cast<int64_t>(record.start_timestamp);
        constexpr int64_t kStartupRaceThreshold = -100000;
        if (delta > kStartupRaceThreshold) {
            // Small negative delta from deterministic startup race: compute kernel
            // detects dispatch_d's stream-register clearing before dispatch_s records
            // the first start timestamp. Benign — silently skip.
            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Skipping startup zone with end < start: "
                "program_id={}, delta={}",
                record.program_id,
                delta);
        } else {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Skipping zone with end < start: program_id={}, chip_id={}, "
                "start_timestamp={}, end_timestamp={} (delta={})",
                record.program_id,
                record.chip_id,
                record.start_timestamp,
                record.end_timestamp,
                delta);
        }
        return;
    }

    std::string file_str = record.program_id > 0 ? tt::GetKernelSourcesForRuntimeId(record.program_id) : "";
    if (file_str.empty()) {
        file_str = "realtime_profiler";
    }

    auto start = make_marker(record, record.start_timestamp, tracy::TTDeviceMarkerType::ZONE_START, file_str);
    auto end = make_marker(record, record.end_timestamp, tracy::TTDeviceMarkerType::ZONE_END, file_str);

    TracyTTPushStartMarker(ctx, start);
    TracyTTPushEndMarker(ctx, end);
#endif
}

void RealtimeProfilerTracyHandler::PushSyncCheckMarker(
    [[maybe_unused]] uint32_t chip_id, [[maybe_unused]] uint64_t device_timestamp, [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetContext(chip_id);
    if (!ctx) {
        return;
    }

    // Use same core as program zones but a different RiscType (SYNC) so the
    // sync check zone lives on its own Tracy lane. Program and sync zones on
    // the same lane would have to be strictly nested/non-overlapping, and any
    // edge case (zone straddling sync, rounding at LOD boundaries) caused
    // program zones to visually disappear or duplicate.
    constexpr uint32_t kRealtimeProfilerCore_X = 100;
    constexpr uint32_t kRealtimeProfilerCore_Y = 100;

    tracy::TTDeviceMarker start_marker;
    start_marker.chip_id = chip_id;
    start_marker.core_x = kRealtimeProfilerCore_X;
    start_marker.core_y = kRealtimeProfilerCore_Y;
    start_marker.risc = tracy::RiscType::NCRISC;
    start_marker.timestamp = device_timestamp;
    start_marker.runtime_host_id = 0;
    start_marker.marker_name = "SYNC_CHECK";
    start_marker.marker_type = tracy::TTDeviceMarkerType::ZONE_START;
    start_marker.file = "sync_check";
    start_marker.line = 0;

    // End marker: 1µs after start (just enough to be visible in Tracy)
    uint64_t end_timestamp = device_timestamp + static_cast<uint64_t>(frequency * 1000.0);

    tracy::TTDeviceMarker end_marker = start_marker;
    end_marker.timestamp = end_timestamp;
    end_marker.marker_type = tracy::TTDeviceMarkerType::ZONE_END;

    TracyTTPushStartMarker(ctx, start_marker);
    TracyTTPushEndMarker(ctx, end_marker);
#endif
}

void RealtimeProfilerTracyHandler::CalibrateDevice(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] int64_t host_time,
    [[maybe_unused]] uint64_t device_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetContext(chip_id);
    if (!ctx) {
        return;
    }
    TracyTTContextCalibrate(ctx, host_time, static_cast<double>(device_timestamp), frequency);
#endif
}

}  // namespace tt::tt_metal
