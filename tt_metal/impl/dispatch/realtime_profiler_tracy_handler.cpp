// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_tracy_handler.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

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
    marker.runtime_host_id = record.runtime_id;
    marker.marker_name = fmt::format("Program_{}", record.runtime_id);
    marker.marker_type = type;
    marker.file = file_str;
    marker.line = 0;
    return marker;
}
#endif

constexpr size_t kMaxSummaryEntries = 8;

template <typename Key>
std::string FormatTopCounts(const std::unordered_map<Key, uint64_t>& counts) {
    if (counts.empty()) {
        return "none";
    }
    std::vector<std::pair<Key, uint64_t>> entries(counts.begin(), counts.end());
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string result;
    const size_t limit = std::min(entries.size(), kMaxSummaryEntries);
    for (size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += fmt::format("{}×{}", entries[i].first, entries[i].second);
    }
    if (entries.size() > kMaxSummaryEntries) {
        result += fmt::format(", ... (+{} more)", entries.size() - kMaxSummaryEntries);
    }
    return result;
}

}  // namespace

RealtimeProfilerTracyHandler::RealtimeProfilerTracyHandler() {
    callback_handle_ = tt::RegisterProgramRealtimeProfilerCallback(
        [this](const tt::ProgramRealtimeRecord& record) { HandleRecord(record); });
}

RealtimeProfilerTracyHandler::~RealtimeProfilerTracyHandler() {
    tt::UnregisterProgramRealtimeProfilerCallback(callback_handle_);

    std::lock_guard<std::mutex> lock(mutex_);
    MaybeEmitSkippedZoneSummaryLocked();

#if defined(TRACY_ENABLE)
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

void RealtimeProfilerTracyHandler::RecordSkippedZoneWithEndBeforeStart(
    const tt::ProgramRealtimeRecord& record, int64_t delta) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& stats = skipped_end_before_start_stats_;
    stats.total_skipped++;

    if (!stats.logged_first_detail) {
        stats.logged_first_detail = true;
        stats.last_summary_time = std::chrono::steady_clock::now();
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Skipping zone with end < start: runtime_id={}, chip_id={}, "
            "start_timestamp={}, end_timestamp={} (delta={})",
            record.runtime_id,
            record.chip_id,
            record.start_timestamp,
            record.end_timestamp,
            delta);
        return;
    }

    stats.suppressed_since_last_summary++;
    stats.count_by_runtime_id[record.runtime_id]++;
    stats.count_by_chip_id[record.chip_id]++;
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.last_summary_time >= SkippedEndBeforeStartStats::kSummaryInterval) {
        MaybeEmitSkippedZoneSummaryLocked();
    }
}

void RealtimeProfilerTracyHandler::MaybeEmitSkippedZoneSummaryLocked() {
    auto& stats = skipped_end_before_start_stats_;
    if (stats.total_skipped == 0) {
        return;
    }
    if (stats.suppressed_since_last_summary == 0) {
        return;
    }

    log_warning(
        tt::LogMetal,
        "[Real-time profiler] Skipped {} additional zones with end < start ({} total); programs: {}; chips: {}",
        stats.suppressed_since_last_summary,
        stats.total_skipped,
        FormatTopCounts(stats.count_by_runtime_id),
        FormatTopCounts(stats.count_by_chip_id));

    stats.suppressed_since_last_summary = 0;
    stats.count_by_runtime_id.clear();
    stats.count_by_chip_id.clear();
    stats.last_summary_time = std::chrono::steady_clock::now();
}

void RealtimeProfilerTracyHandler::HandleRecord(const tt::ProgramRealtimeRecord& record) {
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
                "runtime_id={}, delta={}",
                record.runtime_id,
                delta);
        } else {
            RecordSkippedZoneWithEndBeforeStart(record, delta);
        }
        return;
    }

#if defined(TRACY_ENABLE)
    if (!tracy::GetProfiler().IsConnected()) {
        return;
    }
    TracyTTCtx ctx = GetContext(record.chip_id);
    if (!ctx) {
        return;
    }

    std::string file_str;
    for (size_t i = 0; i < record.kernel_sources.size(); i++) {
        if (i > 0) {
            file_str += ",\n";
        }
        file_str += record.kernel_sources[i];
    }
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
    if (!tracy::GetProfiler().IsConnected()) {
        return;
    }
    TracyTTCtx ctx = GetContext(chip_id);
    if (!ctx) {
        return;
    }

    // Sync-check zones go on a dedicated Tracy lane (RiscType::SYNC) so they don't have to
    // strictly nest with program zones — overlap there caused zones to disappear or duplicate.
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
    if (!tracy::GetProfiler().IsConnected()) {
        return;
    }
    TracyTTCtx ctx = GetContext(chip_id);
    if (!ctx) {
        return;
    }
    TracyTTContextCalibrate(ctx, host_time, static_cast<double>(device_timestamp), frequency);
#endif
}

}  // namespace tt::tt_metal
