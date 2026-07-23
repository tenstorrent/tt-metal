// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_tracy_consumer.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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
    const tt::tt_metal::experimental::ProgramRealtimeRecord& record,
    uint64_t timestamp,
    tracy::TTDeviceMarkerType marker_type,
    const std::string& file) {
    tracy::TTDeviceMarker marker;
    marker.runtime_host_id = record.runtime_id;
    marker.chip_id = record.chip_id;
    marker.core_x = 0;
    marker.core_y = 0;
    marker.risc = tracy::RiscType::BRISC;
    marker.timestamp = timestamp;
    marker.marker_name = "Program";
    marker.marker_type = marker_type;
    marker.file = file;
    marker.line = 0;
    marker.color = 0xee9a00;  // Orange2, matching the previous RT-profiler zone color
    return marker;
}

}  // namespace

void RealtimeProfilerTracyConsumer::on_records(const tt::tt_metal::experimental::ProgramRealtimeRecordBatch& batch) {
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

void RealtimeProfilerTracyConsumer::CalibrateFromRecord(
    const tt::tt_metal::experimental::ProgramRealtimeRecord& record) {
    if (record.chip_id >= chips_.size()) {
        chips_.resize(record.chip_id + 1);
    }
    PerChip& s = chips_[record.chip_id];

    // Per-record fast path: once a chip is calibrated, only a device_cycle_offset change (a host<->device re-anchor,
    // ~20/s) can warrant recalibration. An unchanged offset — the vast majority of records — bails here on a vector
    // index + an int compare: no clock read, no hash, no correlation.
    const bool first = s.ctx == nullptr;
    if (!first && s.last_seen_offset == record.clock_sync.device_cycle_offset) {
        return;
    }

    if (record.frequency <= 0.0) {
        return;  // leave the chip uncalibrated so the next valid record retries
    }
    s.last_seen_offset = record.clock_sync.device_cycle_offset;

    // Re-steer the Tracy context on every offset change (the servo re-anchor, ~20/s) so its view tracks the servo
    // instead of lagging a throttle window behind it. clock_sync maps device cycles to CLOCK_MONOTONIC ns (frequency is
    // cycles/ns); recover the host anchor, then convert it into Tracy's rdtsc CPU-tick domain (only here, off the
    // per-record path — see HostMonoNsToTracyCpuTicks).
    const int64_t host_anchor_mono_ns = std::llround(
        (static_cast<double>(record.start_timestamp) - static_cast<double>(record.clock_sync.device_cycle_offset)) /
        record.frequency);
    const int64_t host_anchor = HostMonoNsToTracyCpuTicks(host_anchor_mono_ns);

    if (first) {
        s.ctx = AddDevice(record.chip_id, host_anchor, static_cast<double>(record.start_timestamp), record.frequency);
    } else {
        CalibrateDevice(record.chip_id, host_anchor, record.start_timestamp, record.frequency);
    }
    PublishDeviceProfilerSyncAnchor(record.chip_id, host_anchor, record.start_timestamp, record.frequency);
}

RealtimeProfilerTracyConsumer::~RealtimeProfilerTracyConsumer() {
    MaybeEmitSkippedZoneSummary();

    for (auto& c : chips_) {
        if (c.ctx != nullptr) {
            TracyTTDestroy(c.ctx);
        }
    }
    chips_.clear();
}

TracyTTCtx RealtimeProfilerTracyConsumer::AddDevice(
    uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency) {
    TracyTTCtx ctx = TracyTTContext();
    TracyTTContextPopulate(ctx, host_anchor, device_anchor, frequency);
    const std::string name = fmt::format("Device {}:", chip_id);
    TracyTTContextName(ctx, name.c_str(), static_cast<uint16_t>(name.size()));
    return ctx;
}

TracyTTCtx RealtimeProfilerTracyConsumer::GetContext(uint32_t chip_id) {
    return chip_id < chips_.size() ? chips_[chip_id].ctx : nullptr;
}

bool RealtimeProfilerTracyConsumer::ValidateHostClockDomain() {
    // clock_sync is CLOCK_MONOTONIC; HostMonoNsToTracyCpuTicks bridges it into Tracy's rdtsc domain, which needs a
    // usable Tracy CPU timer. Bail out of calibration if Tracy can't report one.
    if (!(TracyGetTimerMul() > 0.0)) {
        log_error(
            tt::LogMetal,
            "[Real-time profiler] Tracy CPU timer unavailable (TimerMul <= 0); disabling Tracy real-time calibration");
        return false;
    }
    return true;
}

int64_t RealtimeProfilerTracyConsumer::HostMonoNsToTracyCpuTicks(int64_t host_mono_ns) {
    const double ns_per_tick = TracyGetTimerMul();
    if (!(ns_per_tick > 0.0)) {
        return TracyGetCpuTime();
    }
    // A side-by-side read pins the CLOCK_MONOTONIC<->rdtsc offset (both are the same TSC oscillator). A hardware
    // interrupt landing between the two mono reads stretches the bracket and skews the midpoint, so keep the tightest
    // of several attempts (the NTP/PTP correlation trick) — this drops the rare ~µs excursion to a ~ns floor. The
    // anchor is recent, so applying Tracy's ns/tick over the small delta adds no long-baseline ppm error.
    constexpr int kCorrelationAttempts = 8;
    int64_t best_gap = std::numeric_limits<int64_t>::max();
    int64_t mono_now = 0;
    int64_t tracy_now = 0;
    for (int i = 0; i < kCorrelationAttempts; ++i) {
        const int64_t mono_before = realtime_profiler_host_timestamp();
        const int64_t tracy = TracyGetCpuTime();
        const int64_t mono_after = realtime_profiler_host_timestamp();
        const int64_t gap = mono_after - mono_before;
        if (gap < best_gap) {
            best_gap = gap;
            mono_now = mono_before + gap / 2;
            tracy_now = tracy;
        }
    }
    return tracy_now - std::llround(static_cast<double>(mono_now - host_mono_ns) / ns_per_tick);
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
    const tt::tt_metal::experimental::ProgramRealtimeRecord& record, int64_t delta) {
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

void RealtimeProfilerTracyConsumer::HandleRecord(const tt::tt_metal::experimental::ProgramRealtimeRecord& record) {
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

    std::string file;
    for (size_t i = 0; i < record.kernel_sources.size(); i++) {
        if (i > 0) {
            file += ",\n";
        }
        file.append(record.kernel_sources[i].data(), record.kernel_sources[i].size());
    }
    if (file.empty()) {
        file = "realtime_profiler";
    }

    TracyTTPushStartMarker(
        ctx, make_marker(record, record.start_timestamp, tracy::TTDeviceMarkerType::ZONE_START, file));
    TracyTTPushEndMarker(ctx, make_marker(record, record.end_timestamp, tracy::TTDeviceMarkerType::ZONE_END, file));
}

void RealtimeProfilerTracyConsumer::CalibrateDevice(
    uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency) {
    if (chip_id < chips_.size() && chips_[chip_id].ctx != nullptr) {
        TracyTTContextCalibrate(chips_[chip_id].ctx, host_anchor, static_cast<double>(device_anchor), frequency);
    }
}

}  // namespace tt::tt_metal
