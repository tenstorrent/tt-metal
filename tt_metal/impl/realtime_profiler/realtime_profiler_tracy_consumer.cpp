// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_tracy_consumer.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <client/TracyProfiler.hpp>
#include <common/TracyTTDeviceData.hpp>

#include "context/metal_context.hpp"
#include "realtime_profiler_host_clock.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include "tt_metal/impl/profiler/profiler_state_manager.hpp"

namespace tt::tt_metal {

namespace {

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

}  // namespace

void RealtimeProfilerTracyConsumer::on_records(const tt::ProgramRealtimeRecordBatch& batch) {
    // Nothing consumes the markers with no server attached, and emitting one per record (millions/s under load) makes
    // Tracy buffer unboundedly, starving every other Tracy-emitting thread (see TTTracyConnected).
    if (!TTTracyConnected()) {
        return;
    }

    if (!host_clock_checked_) {
        host_clock_valid_ = ValidateHostClockDomain();
        host_clock_checked_ = true;
    }
    if (!host_clock_valid_) {
        return;
    }
    for (const auto& record : batch.records) {
        CalibrateFromRecord(record);
        HandleRecord(record);
    }
}

void RealtimeProfilerTracyConsumer::CalibrateFromRecord(const tt::ProgramRealtimeRecord& record) {
    // device_cycle_offset was anchored using our host clock's ns/tick, so recover the raw host tick with the SAME
    // multiplier — NOT Tracy's TimerMul. They differ by a few ppm, and against a multi-day absolute tick count that ppm
    // becomes seconds of placement error. The raw host tick is Tracy's CPU-tick domain (same rdtsc counter).
    const double device_cycles_per_host_tick = record.frequency * realtime_profiler_host_ns_per_tick();
    if (device_cycles_per_host_tick <= 0.0) {
        return;
    }
    const int64_t host_anchor = std::llround(
        (static_cast<double>(record.start_timestamp) - static_cast<double>(record.clock_sync.device_cycle_offset)) /
        device_cycles_per_host_tick);

    auto it = last_device_cycle_offset_by_chip_.find(record.chip_id);
    if (it == last_device_cycle_offset_by_chip_.end()) {
        AddDevice(record.chip_id, host_anchor, static_cast<double>(record.start_timestamp), record.frequency);
        PublishDeviceProfilerSyncAnchor(record.chip_id, host_anchor, record.start_timestamp, record.frequency);
        last_device_cycle_offset_by_chip_.emplace(record.chip_id, record.clock_sync.device_cycle_offset);
        last_calibrate_at_by_chip_[record.chip_id] = std::chrono::steady_clock::now();
        return;
    }
    // The servo re-anchors device_cycle_offset ~1/s; recalibrating the Tracy context that often collapses the device
    // zones (the GpuCalibration slope is derived from the gap between consecutive calibrations). Throttle to occasional
    // drift correction — the per-record record calibration is already accurate for consumers; this only steers the
    // view.
    const auto now = std::chrono::steady_clock::now();
    if (it->second != record.clock_sync.device_cycle_offset &&
        now - last_calibrate_at_by_chip_[record.chip_id] >= kTracyRecalibrateInterval) {
        CalibrateDevice(record.chip_id, host_anchor, record.start_timestamp, record.frequency);
        PublishDeviceProfilerSyncAnchor(record.chip_id, host_anchor, record.start_timestamp, record.frequency);
        it->second = record.clock_sync.device_cycle_offset;
        last_calibrate_at_by_chip_[record.chip_id] = now;
    }
}

RealtimeProfilerTracyConsumer::~RealtimeProfilerTracyConsumer() {
    MaybeEmitSkippedZoneSummary();

    for (auto& entry : tracy_contexts_) {
        TracyTTDestroy(entry.second);
    }
    tracy_contexts_.clear();
}

void RealtimeProfilerTracyConsumer::AddDevice(
    uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency) {
    if (tracy_contexts_.contains(chip_id)) {
        log_warning(tt::LogMetal, "RealtimeProfilerTracyConsumer: device {} already added, skipping", chip_id);
        return;
    }

    TracyTTCtx ctx = TracyTTContext();
    TracyTTContextPopulate(ctx, host_anchor, device_anchor, frequency);
    const std::string name = fmt::format("Device {}:", chip_id);
    TracyTTContextName(ctx, name.c_str(), name.size());
    tracy_contexts_[chip_id] = ctx;
}

TracyTTCtx RealtimeProfilerTracyConsumer::GetContext(uint32_t chip_id) {
    auto it = tracy_contexts_.find(chip_id);
    return it != tracy_contexts_.end() ? it->second : nullptr;
}

bool RealtimeProfilerTracyConsumer::ValidateHostClockDomain() {
    constexpr double kMaxClockSampleSeparationNs = 100'000.0;
    const int64_t host_before = realtime_profiler_host_timestamp();
    const int64_t tracy_timestamp = TracyGetCpuTime();
    const int64_t host_after = realtime_profiler_host_timestamp();
    const int64_t host_midpoint = host_before + (host_after - host_before) / 2;
    const double difference_ns = std::abs(static_cast<double>(tracy_timestamp - host_midpoint) * TracyGetTimerMul());
    if (difference_ns > kMaxClockSampleSeparationNs) {
        log_error(
            tt::LogMetal,
            "[Real-time profiler] Host clock does not match Tracy's CPU timer (difference {:.0f} ns); "
            "disabling Tracy real-time calibration",
            difference_ns);
        return false;
    }
    return true;
}

void RealtimeProfilerTracyConsumer::PublishDeviceProfilerSyncAnchor(
    uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency) {
    auto& metal_context = MetalContext::instance(context_id_);
    if (!metal_context.rtoptions().get_profiler_accumulate()) {
        return;
    }
    auto& profiler_state_manager = metal_context.profiler_state_manager();
    if (!profiler_state_manager) {
        return;
    }
    {
        std::lock_guard<std::recursive_mutex> lock(profiler_state_manager->device_profiler_map_mutex);
        auto it = profiler_state_manager->device_profiler_map.find(chip_id);
        if (it == profiler_state_manager->device_profiler_map.end()) {
            return;
        }
        it->second.realtime_sync_line = tt::tt_metal::DeviceProfiler::RealtimeSyncLine{
            static_cast<double>(host_anchor), static_cast<double>(device_anchor), frequency};
    }
    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device-profiler clock anchor for device {}: "
        "host_anchor={}, device_anchor={}, freq={:.6f} GHz",
        chip_id,
        host_anchor,
        device_anchor,
        frequency);
}

void RealtimeProfilerTracyConsumer::RecordSkippedZoneWithEndBeforeStart(
    const tt::ProgramRealtimeRecord& record, int64_t delta) {
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
        MaybeEmitSkippedZoneSummary();
    }
}

void RealtimeProfilerTracyConsumer::MaybeEmitSkippedZoneSummary() {
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

void RealtimeProfilerTracyConsumer::HandleRecord(const tt::ProgramRealtimeRecord& record) {
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

    const auto start = make_marker(record, record.start_timestamp, tracy::TTDeviceMarkerType::ZONE_START, file_str);
    const auto end = make_marker(record, record.end_timestamp, tracy::TTDeviceMarkerType::ZONE_END, file_str);
    TracyTTPushStartMarker(ctx, start);
    TracyTTPushEndMarker(ctx, end);
}

void RealtimeProfilerTracyConsumer::CalibrateDevice(
    uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency) {
    if (auto it = tracy_contexts_.find(chip_id); it != tracy_contexts_.end()) {
        TracyTTContextCalibrate(it->second, host_anchor, static_cast<double>(device_anchor), frequency);
    }
}

}  // namespace tt::tt_metal
