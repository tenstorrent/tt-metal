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
    packet_callback_handle_ = tt::tt_metal::experimental::RegisterProfilerPacketCallback(
        [this](const tt::tt_metal::experimental::WorkerZonePacket& zone) { HandleWorkerZone(zone); });
}

RealtimeProfilerTracyHandler::~RealtimeProfilerTracyHandler() {
    tt::tt_metal::experimental::UnregisterProfilerPacketCallback(packet_callback_handle_);
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
    // Record the chip's calibration; the actual Tracy contexts are created lazily, one per core, on
    // the first marker for that core (see GetOrCreateContext). A single per-chip context would put
    // every core's RISCs on one device row.
    if (chip_calibrations_.contains(chip_id)) {
        log_warning(tt::LogMetal, "RealtimeProfilerTracyHandler: device {} already added, skipping", chip_id);
        return;
    }
    chip_calibrations_[chip_id] = ChipCalibration{host_start, first_timestamp, frequency};
#endif
}

void RealtimeProfilerTracyHandler::RemoveDevice([[maybe_unused]] uint32_t chip_id) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    // Destroy every per-core context belonging to this chip (key high bits == chip_id).
    for (auto it = tracy_contexts_.begin(); it != tracy_contexts_.end();) {
        if (static_cast<uint32_t>(it->first >> 40) == chip_id) {
            TracyTTDestroy(it->second);
            it = tracy_contexts_.erase(it);
        } else {
            ++it;
        }
    }
    chip_calibrations_.erase(chip_id);
#endif
}

TracyTTCtx RealtimeProfilerTracyHandler::GetOrCreateContext(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t core_x,
    [[maybe_unused]] uint32_t core_y,
    [[maybe_unused]] const std::string& name) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t key = ContextKey(chip_id, core_x, core_y);
    if (auto it = tracy_contexts_.find(key); it != tracy_contexts_.end()) {
        return it->second;
    }
    auto calib_it = chip_calibrations_.find(chip_id);
    if (calib_it == chip_calibrations_.end()) {
        return nullptr;  // AddDevice not called for this chip yet
    }
    const ChipCalibration& c = calib_it->second;
    TracyTTCtx ctx = TracyTTContext();
    // Populate WITH the accurate sync anchor, exactly like the standard DeviceProfiler
    // (updateTracyContext Populates using device_sync_info). Using the initial AddDevice anchor and
    // then immediately Calibrating with the sync-check anchor gives Tracy two calibration points that
    // can be ~seconds apart -> a broken GPU->CPU mapping and the device zones don't render in the GUI.
    // A single consistent anchor both renders correctly AND keeps every core aligned with the program
    // row. Newer syncs still recalibrate all contexts via CalibrateDevice.
    if (c.has_calibrate) {
        TracyTTContextPopulate(ctx, c.cal_host_time, c.cal_device_timestamp, c.cal_frequency);
    } else {
        TracyTTContextPopulate(ctx, c.host_start, c.first_timestamp, c.frequency);
    }
    TracyTTContextName(ctx, name.c_str(), name.size());
    tracy_contexts_[key] = ctx;
    log_debug(
        tt::LogMetal, "[Real-time profiler] created Tracy device context ({} total): {}", tracy_contexts_.size(), name);
    return ctx;
#else
    return nullptr;
#endif
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
    // Program zones live on the fixed "program" core (100,100) that make_marker() uses; give them
    // their own device row named just after the chip.
    TracyTTCtx ctx = GetOrCreateContext(record.chip_id, 100, 100, fmt::format("Device {}", record.chip_id));
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
    // Sync-check zones ride the same program core (100,100) row as program zones.
    TracyTTCtx ctx = GetOrCreateContext(chip_id, 100, 100, fmt::format("Device {}", chip_id));
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

void RealtimeProfilerTracyHandler::HandleWorkerZone(
    [[maybe_unused]] const tt::tt_metal::experimental::WorkerZonePacket& zone) {
#if defined(TRACY_ENABLE)
    if (!tracy::GetProfiler().IsConnected()) {
        return;
    }
    // One Tracy context (device row) per core, so a core's 5 RISCs are lanes within it rather than
    // all cores collapsing into a single device. Name matches the standard DeviceProfiler so the
    // X280 view lines up with the DRAM push profiler's.
    TracyTTCtx ctx = GetOrCreateContext(
        zone.chip_id,
        zone.core_noc0_x,
        zone.core_noc0_y,
        fmt::format(
            "Device: {}, Logical ({},{}) Physical ({},{})",
            zone.chip_id,
            zone.core_logical_x,
            zone.core_logical_y,
            zone.core_noc0_x,
            zone.core_noc0_y));
    if (!ctx) {
        return;
    }

    // The enriched packet is already fully resolved by the host (NOC0 coord, deciphered name,
    // is_start). Zones arrive in ring order (== emission order == correct nest order per lane), so
    // pushing START/END in arrival order lets Tracy nest them correctly. Markers share the device
    // clock domain, so the context calibration applies directly.
    static constexpr tracy::RiscType kRisc[5] = {
        tracy::RiscType::BRISC,
        tracy::RiscType::NCRISC,
        tracy::RiscType::TRISC_0,
        tracy::RiscType::TRISC_1,
        tracy::RiscType::TRISC_2};

    tracy::TTDeviceMarker marker;
    marker.chip_id = zone.chip_id;
    marker.core_x = zone.core_noc0_x;
    marker.core_y = zone.core_noc0_y;
    marker.risc = kRisc[zone.risc % 5];
    marker.timestamp = zone.timestamp;
    marker.runtime_host_id = zone.timer_id;
    marker.marker_type = zone.is_start ? tracy::TTDeviceMarkerType::ZONE_START : tracy::TTDeviceMarkerType::ZONE_END;
    marker.marker_name = zone.name.empty() ? fmt::format("Zone_{}", zone.timer_id) : std::string(zone.name);
    marker.file = "kernel_profiler";
    marker.line = 0;

    if (zone.is_start) {
        TracyTTPushStartMarker(ctx, marker);
    } else {
        TracyTTPushEndMarker(ctx, marker);
    }
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
    // Guard 1: a failed/bogus sync read reports device_timestamp == 0. Calibrating with it slams the
    // Tracy context anchor to 0 and collapses every subsequent zone to the start of the trace (t≈0) —
    // the classic "device zones far off to the left" symptom. Drop it.
    if (device_timestamp == 0) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} skipping calibrate: device_timestamp=0 (failed sync read)",
            chip_id);
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = chip_calibrations_.find(chip_id);
    if (it == chip_calibrations_.end()) {
        return;
    }
    ChipCalibration& c = it->second;

    // Guard 2: reject calibrations whose implied device<->host clock rate is not ~1.0. Tracy sets
    // calibrationMod = host_delta_ns / device_delta_ns and maps every zone by it; a garbage sync
    // (non-monotonic device time, or a wildly off delta) yields a mod that skews or zeroes the whole
    // timeline. The previous anchor is the last accepted calibrate, else the init Populate anchor.
    const double prev_dev = c.has_calibrate ? c.cal_device_timestamp : c.first_timestamp;
    const int64_t prev_host = c.has_calibrate ? c.cal_host_time : c.host_start;
    const double prev_freq = c.has_calibrate ? c.cal_frequency : c.frequency;
    const double dev_new_ns = static_cast<double>(device_timestamp) / frequency;
    const double dev_prev_ns = (prev_freq > 0.0) ? prev_dev / prev_freq : 0.0;
    const double dev_delta_ns = dev_new_ns - dev_prev_ns;
    const double host_delta_ns = static_cast<double>(host_time - prev_host) * TracyGetTimerMul();
    if (dev_delta_ns <= 0.0) {
        log_debug(tt::LogMetal, "[Real-time profiler] Device {} skip calibrate: non-monotonic device time", chip_id);
        return;
    }
    const double rate = host_delta_ns / dev_delta_ns;
    if (rate < 0.8 || rate > 1.2) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} skipping calibrate: implied host/device rate {:.4f} out of band "
            "(bad sync sample)",
            chip_id,
            rate);
        return;
    }

    // Accepted. Record as the chip's latest calibration so contexts created LATER (lazy, per core)
    // pick it up on creation and stay aligned with the ones recalibrated below.
    c.has_calibrate = true;
    c.cal_host_time = host_time;
    c.cal_device_timestamp = static_cast<double>(device_timestamp);
    c.cal_frequency = frequency;
    // Recalibrate every per-core context belonging to this chip (they share the device clock domain).
    for (auto& [key, ctx] : tracy_contexts_) {
        if (static_cast<uint32_t>(key >> 40) == chip_id && ctx != nullptr) {
            TracyTTContextCalibrate(ctx, host_time, static_cast<double>(device_timestamp), frequency);
        }
    }
#endif
}

}  // namespace tt::tt_metal
