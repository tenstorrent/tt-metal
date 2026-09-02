// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "perf_debug_profiler_tracy_handler.hpp"

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

#if defined(TRACY_ENABLE)
namespace {}  // namespace
#endif

PerfDebugTracyHandler::PerfDebugTracyHandler() = default;

PerfDebugTracyHandler::~PerfDebugTracyHandler() {
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(TRACY_ENABLE)
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

void PerfDebugTracyHandler::AddCore(
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

bool PerfDebugTracyHandler::LookupAnchorLocked(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t core_x,
    [[maybe_unused]] uint32_t core_y,
    [[maybe_unused]] ChipAnchor& out) {
#if defined(TRACY_ENABLE)
    // Per-core FIRST: a DRAM core's entry exists precisely because its clock origin differs from the chip's.
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

bool PerfDebugTracyHandler::LookupAnchor(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    return LookupAnchorLocked(chip_id, core_x, core_y, out);
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
    ChipAnchor a{};
    if (!LookupAnchorLocked(chip_id, core_x, core_y, a)) {
        return nullptr;  // device was never AddDevice'd
    }
    ZoneScopedNC("ctx-create", 0xD35400);  // orange: creating a Tracy GPU context (GpuNewContext+Populate+name)
                                           // -- one per core, all lazily on the first batch -> startup spike
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

    // Same thread-id packing the marker path uses (TTDeviceMarker::get_thread_id), so zones and point
    // markers land on the same per-RISC row of the core's context.
    tracy::TTDeviceMarker tm;
    tm.chip_id = zone.chip_id;
    tm.core_x = zone.core_noc0_x;
    tm.core_y = zone.core_noc0_y;
    tm.risc = kRisc[zone.risc % 5];
    const uint32_t thread = tm.get_thread_id();

    // Colour resolution, matching the legacy wire's getMarkerColor exactly (TracyTTDevice.hpp): an
    // explicit colour wins (the DRISC role tables), then PROFILER-keyword names go Tomato3, then the
    // per-RISC palette -- BRISC Orange2, NCRISC SeaGreen3, TRISC_0/1/2 SkyBlue3/Turquoise2/CadetBlue1.
    // Without this every worker zone shipped colour 0 and the GUI fell back to its own palette -- the
    // "colors are all wrong" regression of the zones-at-arrival rework.
    uint32_t color = zone.color;
    if (color == 0) {
        static constexpr uint32_t kRiscColor[5] = {0xEE9A00u, 0x43CD80u, 0x6CA6CDu, 0x00E5EEu, 0x98F5FFu};
        color = zone.name.find("PROFILER") != std::string_view::npos ? 0xCD4F39u : kRiscColor[zone.risc % 5];
    }

    // Intern the srcloc: QueueGpuZone ships a raw pointer that the SERVER dereferences by querying this
    // process later, so the SourceLocationData and its name string must outlive the capture -- allocated
    // once per (zone id, colour), never freed. Bounded by distinct zone names x RISCs, not zone count
    // (the same name on two RISCs carries two colours, hence two entries -- the colour is in the key).
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

    // One complete zone, one SERIAL queue item. No begin/end split and no per-lane depth mirror (there
    // is nothing to orphan: the server never pops a stack). Serial rather than lock-free, deliberately:
    // context creation and point markers ride the serial queue, and the client drains the lock-free
    // queues BEFORE the serial one each pass, so a lock-free zone could overtake the GpuNewContext it
    // references -- an intermittent tracy-capture segfault, reproduced 2026-08-26. All-serial is totally
    // ordered by construction, and one serial item per zone is still cheaper than the legacy pair (two
    // serial items plus an alloc'd srcloc each).
    TracyTTPushZoneSerial(ctx, srcloc, thread, zone.start, zone.end);
#endif
}

void PerfDebugTracyHandler::HandleWorkerEvent([[maybe_unused]] const perf_debug::WorkerEventPacket& event) {
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
    // Two kinds, by payload -- every id on this wire is a compile-time structural id, so there is no
    // "runtime id" kind (a runtime value rides PP_DATA payload):
    //   DATA: compile-time tag + payload (DeviceData)
    //   FLAG: compile-time tag, no payload (DeviceFlag)
    marker.marker_type = (event.num_values != 0) ? tracy::TTDeviceMarkerType::DATA : tracy::TTDeviceMarkerType::FLAG;
    // Every id resolves from its kernel's ELF; an empty name here is the bug the teardown summary counts.
    marker.marker_name = event.name.empty() ? fmt::format("Event_{}", event.id) : std::string(event.name);
    marker.file = "kernel_profiler";
    marker.line = 0;

    // The first two uint64s ride the marker's dedicated fields (the Tracy tooltip prints them as
    // Data / Data high); any beyond that go into the metadata map so nothing is silently dropped.
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
