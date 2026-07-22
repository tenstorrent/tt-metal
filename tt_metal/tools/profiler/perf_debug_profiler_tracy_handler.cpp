// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "perf_debug_profiler_tracy_handler.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <string>
#include <utility>

#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#endif

namespace tt::tt_metal {

PerfDebugTracyHandler::PerfDebugTracyHandler() = default;

PerfDebugTracyHandler::~PerfDebugTracyHandler() {
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(TRACY_ENABLE)
    if (orphan_end_count_ > 0) {
        log_debug(
            tt::LogMetal,
            "[perf-debug profiler] dropped {} orphan ZONE_ENDs across {} lanes (capture-boundary straddles / "
            "over-run); each would SEGV tracy-capture",
            orphan_end_count_,
            orphan_lanes_.size());
    }
    for (auto& entry : tracy_contexts_) {
        TracyTTDestroy(entry.second);
    }
    tracy_contexts_.clear();
#endif
}

void PerfDebugTracyHandler::AddDevice(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] int64_t host_start,
    [[maybe_unused]] double first_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    // Record the chip anchor; per-core contexts are created (Populated with this) in PreCreateContexts
    // / GetOrCreateContext. A single per-chip context would collapse every core's RISCs into one row.
    chip_anchors_[chip_id] = ChipAnchor{host_start, first_timestamp, frequency};
#endif
}

TracyTTCtx PerfDebugTracyHandler::GetOrCreateContext(
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
    // Calibrated variant: marks the context calibrated (calibrationMod=1.0, no calibration events) so the
    // Tracy GUI does NOT show a per-context "Drift (ns/s)/Auto" control under every core. Timestamps are
    // host-rebased, so the anchor mapping is exact and no drift correction is wanted.
    TracyTTContextPopulateCalibrated(ctx, a.host_start, a.first_timestamp, a.frequency);
    TracyTTContextName(ctx, name.c_str(), name.size());
    tracy_contexts_[key] = ctx;
    return ctx;
#else
    return nullptr;
#endif
}

void PerfDebugTracyHandler::PreCreateContexts(
    [[maybe_unused]] uint32_t chip_id, [[maybe_unused]] const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0) {
#if defined(TRACY_ENABLE)
    for (const auto& [cx, cy] : worker_noc0) {
        GetOrCreateContext(chip_id, cx, cy, fmt::format("Device: {} Physical ({},{})", chip_id, cx, cy));
    }
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] Device {}: pre-created {} per-core Tracy contexts (off the drain hot path)",
        chip_id,
        worker_noc0.size());
#endif
}

void PerfDebugTracyHandler::HandleWorkerZone([[maybe_unused]] const perf_debug::WorkerZonePacket& zone) {
#if defined(TRACY_ENABLE)
    if (!tracy::GetProfiler().IsConnected()) {
        return;
    }
    TracyTTCtx ctx = GetOrCreateContext(
        zone.chip_id,
        zone.core_noc0_x,
        zone.core_noc0_y,
        fmt::format("Device: {} Physical ({},{})", zone.chip_id, zone.core_noc0_x, zone.core_noc0_y));
    if (!ctx) {
        return;
    }

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

    // Mirror Tracy's per-lane GPU zone stack depth; drop an unmatched ZONE_END (would pop an empty
    // stack -> SEGV in tracy-capture). A never-opened lane's first END is a benign capture-start
    // straddle (its START predates the drain); an extra END after balanced traffic is a pairing bug.
    const uint64_t lane_key = (ContextKey(zone.chip_id, zone.core_noc0_x, zone.core_noc0_y) << 3) | (zone.risc & 0x7);
    // lane_depth_ / orphan_* are SHARED across the (multiple) socket-drain threads that call this. Guard the
    // read-modify-write: without the lock, concurrent inserts from two drain threads rehash the map and
    // corrupt an UNRELATED lane's depth, which then spuriously trips the orphan-END drop below and loses a
    // burst of real ZONE_ENDs -> a deep unclosed-zone staircase on a random single lane (rare, intermittent).
    // Release before the Tracy push so pushes stay concurrent (a lane is single-threaded, so push order is
    // preserved regardless; Tracy's serial queue is itself thread-safe).
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (zone.is_start) {
            lane_depth_[lane_key]++;
        } else {
            auto it = lane_depth_.find(lane_key);
            const int32_t depth = (it == lane_depth_.end()) ? 0 : it->second;
            if (depth <= 0) {
                ++orphan_end_count_;
                orphan_lanes_.insert(lane_key);
                return;  // orphan END -> drop (lock_guard releases on return)
            }
            --it->second;
        }
    }

    if (zone.is_start) {
        TracyTTPushStartMarker(ctx, marker);
    } else {
        TracyTTPushEndMarker(ctx, marker);
    }
#endif
}

}  // namespace tt::tt_metal
