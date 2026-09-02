// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "streaming_profiler_tracy_handler.hpp"

#include <cstring>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <string>
#include <utility>

#include <hostdevcommon/profiler_common.h>

#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>
#endif

namespace tt::tt_metal {

StreamingProfilerTracyHandler::StreamingProfilerTracyHandler() = default;

StreamingProfilerTracyHandler::~StreamingProfilerTracyHandler() {
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(TRACY_ENABLE)
    for (auto& entry : tracy_contexts_) {
        TracyTTDestroy(entry.second);
    }
    tracy_contexts_.clear();
#endif
}

void StreamingProfilerTracyHandler::AddDevice(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] int64_t host_start,
    [[maybe_unused]] double first_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    chip_anchors_[chip_id] = ChipAnchor{host_start, first_timestamp, frequency};
#endif
}

void StreamingProfilerTracyHandler::AddCore(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t noc0_x,
    [[maybe_unused]] uint32_t noc0_y,
    [[maybe_unused]] int64_t host_start,
    [[maybe_unused]] double first_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    core_anchors_[ContextKey(chip_id, noc0_x, noc0_y)] = ChipAnchor{host_start, first_timestamp, frequency};
#endif
}

bool StreamingProfilerTracyHandler::LookupAnchorLocked(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t core_x,
    [[maybe_unused]] uint32_t core_y,
    [[maybe_unused]] ChipAnchor& out) {
#if defined(TRACY_ENABLE)
    // Per-core first: an entry exists only for a core whose clock origin differs from the chip's.
    if (auto cit = core_anchors_.find(ContextKey(chip_id, core_x, core_y)); cit != core_anchors_.end()) {
        out = cit->second;
        return true;
    }
    if (auto ait = chip_anchors_.find(chip_id); ait != chip_anchors_.end()) {
        out = ait->second;
        return true;
    }
#endif
    return false;
}

bool StreamingProfilerTracyHandler::LookupAnchor(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    return LookupAnchorLocked(chip_id, core_x, core_y, out);
}

TracyTTCtx StreamingProfilerTracyHandler::GetOrCreateContext(
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
    ChipAnchor a{};
    if (!LookupAnchorLocked(chip_id, core_x, core_y, a)) {
        return nullptr;  // device was never AddDevice'd
    }
    ZoneScopedNC("ctx-create", 0xD35400);
    TracyTTCtx ctx = TracyTTContext();
    // Calibrated: timestamps are host-rebased, so the GUI's per-context drift correction must stay off.
    TracyTTContextPopulateCalibrated(ctx, a.host_start, a.first_timestamp, a.frequency);
    TracyTTContextName(ctx, name.c_str(), name.size());
    tracy_contexts_[key] = ctx;
    return ctx;
#else
    return nullptr;
#endif
}

void StreamingProfilerTracyHandler::PreCreateContexts(
    [[maybe_unused]] uint32_t chip_id, [[maybe_unused]] const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0) {
#if defined(TRACY_ENABLE)
    for (const auto& [cx, cy] : worker_noc0) {
        GetOrCreateContext(chip_id, cx, cy, fmt::format("Device: {} Physical ({},{})", chip_id, cx, cy));
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] Device {}: pre-created {} per-core Tracy contexts (off the drain hot path)",
        chip_id,
        worker_noc0.size());
#endif
}

void StreamingProfilerTracyHandler::HandleWorkerZone(
    [[maybe_unused]] const streaming_profiler::WorkerZonePacket& zone) {
#if defined(TRACY_ENABLE)
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

    // Same thread-id packing as the marker path (TTDeviceMarker::get_thread_id), so zones and markers share
    // the per-RISC row.
    tracy::TTDeviceMarker tm;
    tm.chip_id = zone.chip_id;
    tm.core_x = zone.core_noc0_x;
    tm.core_y = zone.core_noc0_y;
    tm.risc = kRisc[zone.risc % 5];
    const uint32_t thread = tm.get_thread_id();

    // Matches getMarkerColor (TracyTTDevice.hpp): explicit colour, then Tomato3 for PROFILER-keyword names,
    // then the per-RISC palette; colour 0 makes the GUI fall back to its own.
    uint32_t color = zone.color;
    if (color == 0) {
        static constexpr uint32_t kRiscColor[5] = {0xEE9A00u, 0x43CD80u, 0x6CA6CDu, 0x00E5EEu, 0x98F5FFu};
        color = zone.name.find("PROFILER") != std::string_view::npos ? 0xCD4F39u : kRiscColor[zone.risc % 5];
    }

    // The server dereferences this pointer by querying the process later, so the srcloc and its name are
    // allocated once per (id, colour) and never freed.
    const tracy::SourceLocationData* srcloc = nullptr;
    {
        const uint64_t key = (static_cast<uint64_t>(zone.timer_id) << 32) | color;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = zone_srclocs_.find(key);
        if (it == zone_srclocs_.end()) {
            const std::string nm = zone.name.empty() ? fmt::format("Zone_{}", zone.timer_id) : std::string(zone.name);
            char* nm_copy = new char[nm.size() + 1];
            std::memcpy(nm_copy, nm.c_str(), nm.size() + 1);
            auto* sl = new tracy::SourceLocationData{nm_copy, "kernel_profiler", "kernel_profiler", 0, color};
            it = zone_srclocs_.emplace(key, sl).first;
        }
        srcloc = static_cast<const tracy::SourceLocationData*>(it->second);
    }

    // Serial, deliberately: the client drains the lock-free queues before the serial one, so a lock-free zone
    // could overtake the GpuNewContext it references (an intermittent tracy-capture segfault).
    TracyTTPushZoneSerial(ctx, srcloc, thread, zone.start, zone.end);
#endif
}

void StreamingProfilerTracyHandler::HandleWorkerEvent(
    [[maybe_unused]] const streaming_profiler::WorkerEventPacket& event) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetOrCreateContext(
        event.chip_id,
        event.core_noc0_x,
        event.core_noc0_y,
        fmt::format("Device: {} Physical ({},{})", event.chip_id, event.core_noc0_x, event.core_noc0_y));
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
    marker.chip_id = event.chip_id;
    marker.core_x = event.core_noc0_x;
    marker.core_y = event.core_noc0_y;
    marker.risc = kRisc[event.risc % 5];
    marker.timestamp = event.timestamp;
    marker.runtime_host_id = event.runtime_host_id;
    marker.marker_type = (event.num_values != 0) ? tracy::TTDeviceMarkerType::DATA : tracy::TTDeviceMarkerType::FLAG;
    // Every id resolves from its kernel's ELF; an empty name is the bug the teardown summary counts.
    marker.marker_name = event.name.empty() ? fmt::format("Event_{}", event.id) : std::string(event.name);
    marker.file = "kernel_profiler";
    marker.line = 0;

    // The first two uint64s ride the marker's dedicated fields; the rest go into the metadata map.
    if (event.num_values > 0) {
        marker.data = event.values[0];
    }
    if (event.num_values > 1) {
        marker.data_high = event.values[1];
    }
#ifdef TRACY_TT_HAS_FULL_DEPS
    for (uint32_t i = 2; i < event.num_values; i++) {
        marker.meta_data[fmt::format("value{}", i)] = event.values[i];
    }
#endif

    TracyTTPushMarker(ctx, marker);
#endif
}

}  // namespace tt::tt_metal
