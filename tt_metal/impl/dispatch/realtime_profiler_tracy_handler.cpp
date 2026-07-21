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
#include <tracy/Tracy.hpp>  // host CPU ZoneScoped* to pinpoint where the marker push spends time
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
    // Program-record path: this branch's RT-profiler delivers records in BATCHES
    // (ProgramRealtimeRecordBatch), so unpack the batch and route each record through
    // HandleRecord (the per-record (chip,100,100) context logic below).
    callback_handle_ = tt::RegisterProgramRealtimeProfilerCallback([this](const tt::ProgramRealtimeRecordBatch& batch) {
#if defined(TRACY_ENABLE)
        if (!tracy::GetProfiler().IsConnected()) {
            return;
        }
#endif
        for (const auto& record : batch.records) {
            HandleRecord(record);
        }
    });
    // X280 kernel-zone path: subscribe to enriched WorkerZone packets emitted by the manager's
    // X280 drainer and push them to per-core Tracy lanes.
    packet_callback_handle_ = tt::tt_metal::experimental::RegisterProfilerPacketCallback(
        [this](const tt::tt_metal::experimental::WorkerZonePacket& zone) { HandleWorkerZone(zone); });
}

RealtimeProfilerTracyHandler::~RealtimeProfilerTracyHandler() {
    tt::tt_metal::experimental::UnregisterProfilerPacketCallback(packet_callback_handle_);
    tt::UnregisterProgramRealtimeProfilerCallback(callback_handle_);

    std::lock_guard<std::mutex> lock(mutex_);
    MaybeEmitSkippedZoneSummaryLocked();

#if defined(TRACY_ENABLE)
    log_debug(
        tt::LogMetal,
        "[Real-time profiler] destroying {} per-core Tracy contexts (1 program/sync + workers)",
        tracy_contexts_.size());
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
    // Record the chip's anchor; the actual Tracy contexts are created lazily, one per core, on the
    // first marker for that core (see GetOrCreateContext). A single per-chip context would collapse
    // every core's RISCs into one device row.
    chip_anchors_[chip_id] = ChipAnchor{host_start, first_timestamp, frequency};
#endif
}

void RealtimeProfilerTracyHandler::RemoveDevice([[maybe_unused]] uint32_t chip_id) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    // Localize the residual: unmatched ENDs (orphans dropped) vs unmatched STARTs (lanes left with
    // positive depth). If starts_open ~= orphan_ends, a start was routed to the WRONG lane
    // (host enrichment mis-map). If there are orphan ends but ~no open starts, the start was
    // genuinely LOST upstream (producer ring overwrite / transport). Normal trailing-open zones give
    // a small baseline of starts_open.
    int64_t starts_open = 0;
    size_t open_lanes = 0;
    for (const auto& [lane, depth] : lane_depth_) {
        if (depth > 0) {
            starts_open += depth;
            ++open_lanes;
        }
    }
    if (orphan_end_count_ > 0 || starts_open > 0) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] SUMMARY: {} orphan ZONE_END(s) across {} lane(s) [{} benign "
            "capture-start boundary, {} MID-RUN imbalance]; {} unmatched ZONE_START(s) still open "
            "across {} lane(s). (mid-run>0 => real pairing/drain bug; boundary-only => clean)",
            orphan_end_count_,
            orphan_lanes_.size(),
            orphan_boundary_count_,
            orphan_end_count_ - orphan_boundary_count_,
            starts_open,
            open_lanes);
    }
    for (auto it = tracy_contexts_.begin(); it != tracy_contexts_.end();) {
        if ((it->first >> 40) == chip_id) {
            TracyTTDestroy(it->second);
            it = tracy_contexts_.erase(it);
        } else {
            ++it;
        }
    }
    chip_anchors_.erase(chip_id);
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
    auto ait = chip_anchors_.find(chip_id);
    if (ait == chip_anchors_.end()) {
        return nullptr;  // device was never AddDevice'd
    }
    const ChipAnchor& a = ait->second;
    TracyTTCtx ctx = TracyTTContext();
    TracyTTContextPopulate(ctx, a.host_start, a.first_timestamp, a.frequency);
    TracyTTContextName(ctx, name.c_str(), name.size());
    tracy_contexts_[key] = ctx;
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
    // Program/op + sync-check zones ride the program core (100,100) row, named "Device {chip}".
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
    // One Tracy context (row) per worker core, keyed by its NOC0 coord; the 5 RISCs are lanes within.
    TracyTTCtx ctx;
    {
        // NOTE: the per-marker CPU ZoneScopedN("HWZ-GetCtx") that used to bracket this lookup was
        // removed. It is NOT needed for device (GPU) zone capture: the GPU serial zones are captured
        // regardless (verified via GetGpuZoneCount/ctx-inspector). tracy-capture's printed "Zones:"
        // headline is GetZoneCount() = CPU zones only, which made the missing CPU brackets look like
        // lost device data. Dropping the brackets removes ~2 overhead CPU zones/marker and declutters.
        ctx = GetOrCreateContext(
            zone.chip_id,
            zone.core_noc0_x,
            zone.core_noc0_y,
            fmt::format("Device: {} Physical ({},{})", zone.chip_id, zone.core_noc0_x, zone.core_noc0_y));
    }
    if (!ctx) {
        return;
    }

    // The enriched packet is already fully resolved by the host (NOC0 coord, deciphered name,
    // is_start). Zones arrive in ring order (== emission order == correct nest order per lane), so
    // pushing START/END in arrival order lets Tracy nest them correctly. Markers share the device
    // clock domain, so the per-chip context calibration applies directly.
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

    // Mirror Tracy's own per-(context,thread) GPU zone stack depth. An unmatched ZONE_END (this lane
    // has no open zone) would pop an empty stack in tracy-capture and SEGV. Detect it here, drop it,
    // and log the offending core once so we can tell whether the unbalance is on an active core (a
    // drain/relay/pairing bug) or an idle core (an idle-core-filter leak).
    const uint64_t lane_key = (ContextKey(zone.chip_id, zone.core_noc0_x, zone.core_noc0_y) << 3) | (zone.risc & 0x7);
    if (zone.is_start) {
        lane_depth_[lane_key]++;
    } else {
        // A lane only enters lane_depth_ via a START (above). So if this END finds no entry at all,
        // it is the lane's FIRST-EVER event => the matching START predates our capture window (the
        // zone was already open when the drain began) => a benign boundary straddle, NOT a lost START.
        // An END that finds an existing entry at depth<=0 means the lane had balanced traffic and then
        // an EXTRA end => a genuine pairing/drain bug.
        auto it = lane_depth_.find(lane_key);
        const bool never_opened = (it == lane_depth_.end());
        const int32_t depth = never_opened ? 0 : it->second;
        if (depth <= 0) {
            ++orphan_end_count_;
            orphan_lanes_.insert(lane_key);
            if (never_opened) {
                ++orphan_boundary_count_;
            }
            if (orphan_end_count_ <= 25) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] orphan ZONE_END dropped ({}): chip {} noc0=({},{}) risc {} "
                    "name '{}' id 0x{:x} -- would SEGV tracy-capture; #{}",
                    never_opened ? "capture-start boundary: zone open before drain, START never seen -- benign"
                                 : "reader ring over-run under back-pressure: producer got >RING_CAP ahead of "
                                   "the busy reader, the lap guard clamped head=tail-RING_CAP and dropped the "
                                   "oldest words (the FW START) -- see reader LAP-DROPPED telemetry; benign, dropped",
                    zone.chip_id,
                    zone.core_noc0_x,
                    zone.core_noc0_y,
                    zone.risc,
                    marker.marker_name,
                    zone.timer_id,
                    orphan_end_count_);
            }
            return;
        }
        --it->second;
    }

    {
        // (per-marker CPU ZoneScopedN("HWZ-TracyPush") intentionally removed — see note above)
        if (zone.is_start) {
            TracyTTPushStartMarker(ctx, marker);
        } else {
            TracyTTPushEndMarker(ctx, marker);
        }
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
    std::lock_guard<std::mutex> lock(mutex_);
    // Deliberately does NOT emit a Tracy GpuCalibration. GPU drift-calibration is wrong for tt_device
    // contexts: the Tensix wall clock is a free-running ABSOLUTE ns counter at the host clock's rate, so a
    // drift scale (calibrationMod) derived from the anchor-vs-absolute-timestamp delta comes out ~0.11 and
    // shrinks every zone duration ~9x (see TracyTTDevice.hpp GpuNewContext flags). We keep the mapping as a
    // pure anchor (server: gpuTime = tgpu + timeDiff). This call only refreshes the stored host/frequency
    // anchor for any context created later; it never rescales existing zones.
    (void)device_timestamp;
    if (auto ait = chip_anchors_.find(chip_id); ait != chip_anchors_.end()) {
        ait->second.host_start = host_time;
        ait->second.frequency = frequency;
    }
#endif
}

void RealtimeProfilerTracyHandler::PreCreateContexts(
    [[maybe_unused]] uint32_t chip_id, [[maybe_unused]] const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0) {
#if defined(TRACY_ENABLE)
    // Program/sync zones ride the (100,100) row.
    GetOrCreateContext(chip_id, 100, 100, fmt::format("Device {}", chip_id));
    // One context per worker core, created NOW so the drain loop only ever does fast lookups.
    for (const auto& [cx, cy] : worker_noc0) {
        GetOrCreateContext(chip_id, cx, cy, fmt::format("Device: {} Physical ({},{})", chip_id, cx, cy));
    }
    log_info(
        tt::LogMetal,
        "[Real-time profiler] Device {}: pre-created {} per-core Tracy contexts (off the drain hot path)",
        chip_id,
        worker_noc0.size() + 1);
#endif
}

}  // namespace tt::tt_metal
