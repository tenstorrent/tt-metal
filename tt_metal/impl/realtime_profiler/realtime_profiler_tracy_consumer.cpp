// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TRACY_ENABLE)

#include "realtime_profiler_tracy_consumer.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <string_view>

#include <client/TracyProfiler.hpp>
#include <common/TracyTTDeviceData.hpp>

#include "context/metal_context.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include "tt_metal/impl/profiler/profiler_state_manager.hpp"

namespace tt::tt_metal {

namespace {

tracy::TTDeviceMarker make_marker(
    const experimental::ProgramRealtimeRecord& record,
    uint64_t timestamp,
    tracy::TTDeviceMarkerType marker_type,
    const std::string& file,
    const std::string& name) {
    tracy::TTDeviceMarker marker;
    marker.runtime_host_id = record.runtime_id;
    marker.chip_id = record.chip_id;
    marker.core_x = 0;
    marker.core_y = 0;
    marker.risc = tracy::RiscType::BRISC;
    marker.timestamp = timestamp;
    // TracyTTContext drops runtime_host_id for zones not named BRISC-FW/ERISC-FW, so encode it in the name instead.
    marker.marker_name = name;
    marker.marker_type = marker_type;
    marker.file = file;
    marker.line = 0;
    marker.color = 0xee9a00;  // Orange2, matching the previous RT-profiler zone color
    return marker;
}

}  // namespace

void RealtimeProfilerTracyConsumer::on_records(const experimental::ProgramRealtimeRecordBatch& batch) {
    // Past its connect-timeout window, Tracy drops its backlog and refuses new connections.
    if (tracy::GetProfiler().IsEmitSuppressed()) {
        experimental::UnregisterProgramRealtimeProfilerCallback(handle_);
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

void RealtimeProfilerTracyConsumer::CalibrateFromRecord(const experimental::ProgramRealtimeRecord& record) {
    if (record.chip_id >= chips_.size()) {
        chips_.resize(record.chip_id + 1);
    }
    PerChip& s = chips_[record.chip_id];

    const bool first = s.ctx == nullptr;
    if (!first && s.last_seen_offset == record.clock_sync.device_cycle_offset) {
        return;
    }

    s.last_seen_offset = record.clock_sync.device_cycle_offset;

    // Anchor on the present instant, not record.host_start(): a record arrives ~100us after its program ran, and
    // back-dating the anchor that far made zones slide ~7us early. Projecting the device clock to now through the
    // record's mapping also keeps HostTimeToTracyCpuTicks's delta near zero, where its conversion error is smallest.
    const auto now = std::chrono::steady_clock::now();
    const int64_t host_anchor = HostTimeToTracyCpuTicks(now);
    const uint64_t device_anchor = record.device_timestamp_at(now);
    const double frequency = record.frequency;

    if (first) {
        s.ctx = AddDevice(record.chip_id, host_anchor, static_cast<double>(device_anchor), frequency);
    } else {
        CalibrateDevice(record.chip_id, host_anchor, device_anchor, frequency);
    }
    PublishDeviceProfilerSyncAnchor(record.chip_id, host_anchor, device_anchor, frequency);
}

RealtimeProfilerTracyConsumer::~RealtimeProfilerTracyConsumer() {
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
    // clock_sync is CLOCK_MONOTONIC; HostTimeToTracyCpuTicks bridges it into Tracy's rdtsc domain.
    if (!(TracyGetTimerMul() > 0.0)) {
        log_error(
            tt::LogMetal,
            "[Real-time profiler] Tracy CPU timer unavailable (TimerMul <= 0); disabling Tracy real-time calibration");
        return false;
    }
    return true;
}

int64_t RealtimeProfilerTracyConsumer::HostTimeToTracyCpuTicks(std::chrono::steady_clock::time_point host_time) {
    const double ns_per_tick = TracyGetTimerMul();
    if (!(ns_per_tick > 0.0)) {
        return TracyGetCpuTime();
    }
    const int64_t host_mono_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(host_time.time_since_epoch()).count();
    // CLOCK_MONOTONIC and rdtsc share the same TSC oscillator, so a side-by-side read pins their offset. An
    // interrupt between the two mono reads stretches the bracket and skews the midpoint, so keep the tightest gap
    // across several attempts.
    const auto mono_ns = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    constexpr int kCorrelationAttempts = 8;
    int64_t best_gap = std::numeric_limits<int64_t>::max();
    int64_t mono_now = 0;
    int64_t tracy_now = 0;
    for (int i = 0; i < kCorrelationAttempts; ++i) {
        const int64_t mono_before = mono_ns();
        const int64_t tracy = TracyGetCpuTime();
        const int64_t mono_after = mono_ns();
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
    profiler_state_manager->set_realtime_sync_anchor(
        static_cast<ChipId>(chip_id),
        DeviceProfiler::RealtimeSyncLine{
            static_cast<double>(host_anchor), static_cast<double>(device_anchor), frequency});
    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device-profiler clock anchor for device {}: "
        "host_anchor={}, device_anchor={}, freq={:.6f} GHz",
        chip_id,
        host_anchor,
        device_anchor,
        frequency);
}

void RealtimeProfilerTracyConsumer::HandleRecord(const experimental::ProgramRealtimeRecord& record) {
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

    const std::string name = fmt::format("Program op_id={}", record.runtime_id);
    TracyTTPushStartMarker(
        ctx, make_marker(record, record.start_timestamp, tracy::TTDeviceMarkerType::ZONE_START, file, name));
    TracyTTPushEndMarker(
        ctx, make_marker(record, record.end_timestamp, tracy::TTDeviceMarkerType::ZONE_END, file, name));
}

void RealtimeProfilerTracyConsumer::CalibrateDevice(
    uint32_t chip_id, int64_t host_anchor, uint64_t device_anchor, double frequency) {
    if (chip_id < chips_.size() && chips_[chip_id].ctx != nullptr) {
        TracyTTContextCalibrate(chips_[chip_id].ctx, host_anchor, static_cast<double>(device_anchor), frequency);
    }
}

}  // namespace tt::tt_metal

#endif  // TRACY_ENABLE
