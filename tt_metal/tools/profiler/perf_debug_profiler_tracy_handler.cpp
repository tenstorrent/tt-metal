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
    ZoneScopedNC("ctx-create", 0xD35400);  // orange: creating a Tracy GPU context (GpuNewContext+Populate+name)
                                           // -- one per core, all lazily on the first batch -> startup spike
    TracyTTCtx ctx = TracyTTContext();
    // Calibrated variant: marks the context calibrated (calibrationMod=1.0, no calibration events) so the
    // Tracy GUI does NOT show a per-context "Drift (ns/s)/Auto" control under every core. Timestamps are
    // host-rebased, so the anchor mapping is exact and no drift correction is wanted.
    // LOCK-FREE variants: the context announcement must ride the SAME queue as the zones pushed right
    // after it (the profiler worker drains the lock-free queue before the serial one, so a serially-
    // announced context can reach the server AFTER its first lock-free zone and crash it).
    TracyTTContextPopulateCalibratedLockfree(ctx, a.host_start, a.first_timestamp, a.frequency);
    TracyTTContextNameLockfree(ctx, name.c_str(), name.size());
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

PerfDebugTracyHandler::ZoneSink PerfDebugTracyHandler::GetZoneSink(
    [[maybe_unused]] uint32_t chip_id, [[maybe_unused]] uint32_t core_noc0_x, [[maybe_unused]] uint32_t core_noc0_y) {
#if defined(TRACY_ENABLE)
    ZoneSink sink;
    sink.ctx = GetOrCreateContext(
        chip_id,
        core_noc0_x,
        core_noc0_y,
        fmt::format("Device: {} Physical ({},{})", chip_id, core_noc0_x, core_noc0_y));
    sink.thread_base = static_cast<uint32_t>(
        (core_noc0_x << tracy::TTDeviceMarker::CORE_X_BIT_SHIFT) |
        (core_noc0_y << tracy::TTDeviceMarker::CORE_Y_BIT_SHIFT) | (chip_id << tracy::TTDeviceMarker::CHIP_BIT_SHIFT));
    return sink;
#else
    return {};
#endif
}

const void* PerfDebugTracyHandler::InternZoneSrcloc(
    [[maybe_unused]] uint32_t hash, [[maybe_unused]] uint32_t risc, [[maybe_unused]] std::string_view name) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t key = (static_cast<uint64_t>(hash) << 3) | (risc & 0x7u);
    if (auto it = srclocs_.find(key); it != srclocs_.end()) {
        return it->second;
    }
    static constexpr tracy::Color::ColorType kRiscColor[5] = {
        tracy::Color::Orange2,      // BRISC
        tracy::Color::SeaGreen3,    // NCRISC
        tracy::Color::SkyBlue3,     // TRISC_0
        tracy::Color::Turquoise2,   // TRISC_1
        tracy::Color::CadetBlue1};  // TRISC_2
    // Leaked on purpose: the server reads the struct and its strings from client memory whenever it
    // first sees the pointer, which can be after this handler is destroyed.
    auto* name_str = new std::string(name.empty() ? fmt::format("Zone_{}", hash) : std::string(name));
    auto* srcloc = new tracy::SourceLocationData{
        name_str->c_str(), "", "kernel_profiler", 0, static_cast<uint32_t>(kRiscColor[risc % 5])};
    srclocs_[key] = srcloc;
    return srcloc;
#else
    return nullptr;
#endif
}

void PerfDebugTracyHandler::PushWorkerZone(
    [[maybe_unused]] const ZoneSink& sink,
    [[maybe_unused]] const void* srcloc,
    [[maybe_unused]] uint32_t risc,
    [[maybe_unused]] uint64_t start,
    [[maybe_unused]] uint64_t end) {
#if defined(TRACY_ENABLE)
    static constexpr tracy::RiscType kRisc[5] = {
        tracy::RiscType::BRISC,
        tracy::RiscType::NCRISC,
        tracy::RiscType::TRISC_0,
        tracy::RiscType::TRISC_1,
        tracy::RiscType::TRISC_2};
    const uint32_t thread = sink.thread_base | static_cast<uint32_t>(static_cast<uint8_t>(kRisc[risc % 5]));
    TracyTTPushZone(sink.ctx, static_cast<const tracy::SourceLocationData*>(srcloc), thread, start, end);
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
    // Three distinct kinds, by ID provenance and payload -- not one type with a size of zero:
    //   RUNTIME_EVENT: runtime id, no name exists (DeviceRuntimeEvent)
    //   DATA:          compile-time tag + payload (DeviceData)
    //   FLAG:          compile-time tag, no payload (DeviceFlag)
    if (event.runtime_id) {
        marker.marker_type = tracy::TTDeviceMarkerType::RUNTIME_EVENT;
    } else if (event.num_values != 0) {
        marker.marker_type = tracy::TTDeviceMarkerType::DATA;
    } else {
        marker.marker_type = tracy::TTDeviceMarkerType::FLAG;
    }
    // A runtime event has no source location and so no harvested name; show the id rather than a blank.
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
