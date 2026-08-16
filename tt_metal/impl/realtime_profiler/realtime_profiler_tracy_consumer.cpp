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

constexpr auto kHostClockCalibrationInterval = std::chrono::seconds(1);
constexpr double kHostClockSlopeClampFraction = 0.001;

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
    marker.marker_name = name;
    marker.marker_type = marker_type;
    marker.file = file;
    marker.line = 0;
    marker.color = 0xee9a00;
    return marker;
}

}  // namespace

void RealtimeProfilerTracyConsumer::on_records(const experimental::ProgramRealtimeRecordBatch& batch) {
    // Self-unregistration goes through the public API like any consumer's would; until the
    // handle from registration is visible, retirement just waits for a later batch.
    const auto unregister_self = [this] {
        const auto handle = handle_.load(std::memory_order_acquire);
        if (handle != kHandleNotSet) {
            experimental::UnregisterProgramRealtimeProfilerCallback(handle);
        }
    };

    // Past its connect-timeout window, Tracy drops its backlog and refuses new connections.
    if (tracy::GetProfiler().IsEmitSuppressed()) {
        unregister_self();
        return;
    }

    if (!host_clock_checked_) {
        if (!validate_host_clock_domain()) {
            unregister_self();
            return;
        }
        host_clock_checked_ = true;
    }
    const auto now = std::chrono::steady_clock::now();
    for (const auto& record : batch.records) {
        handle_record(refresh_tracy_calibration(record, now), record);
    }
}

TracyTTCtx RealtimeProfilerTracyConsumer::refresh_tracy_calibration(
    const experimental::ProgramRealtimeRecord& record, std::chrono::steady_clock::time_point now) {
    if (record.chip_id >= chips_.size()) {
        chips_.resize(record.chip_id + 1);
    }
    PerChip& s = chips_[record.chip_id];

    const bool first = s.ctx == nullptr;
    if (!first && now - s.last_calibrated < kHostClockCalibrationInterval) {
        return s.ctx;
    }
    s.last_calibrated = now;

    refresh_host_clock_mapping();
    const int64_t host_anchor = host_time_to_tracy_cpu_ticks(now);
    // Tracy's context period is 1ns per tick, so handing it host nanoseconds and a rate of 1 makes its second timeline
    // the host timeline outright.
    const double host_anchor_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    if (first) {
        s.ctx = add_device(record.chip_id, host_anchor, host_anchor_ns, 1.0);
    } else {
        TracyTTContextCalibrate(s.ctx, host_anchor, host_anchor_ns, 1.0);
    }
    // The post-run device profiler correlates raw device timestamps, so its anchor stays in the device domain.
    publish_device_profiler_sync_anchor(
        record.chip_id, host_time_to_tracy_cpu_ticks(record.host_end()), record.end_timestamp, record.frequency);
    return s.ctx;
}

RealtimeProfilerTracyConsumer::~RealtimeProfilerTracyConsumer() {
    for (auto& c : chips_) {
        if (c.ctx != nullptr) {
            TracyTTDestroy(c.ctx);
        }
    }
}

TracyTTCtx RealtimeProfilerTracyConsumer::add_device(
    uint32_t chip_id, int64_t host_anchor, double device_anchor, double frequency) {
    TracyTTCtx ctx = TracyTTContext();
    TracyTTContextPopulate(ctx, host_anchor, device_anchor, frequency);
    const std::string name = fmt::format("Device {}:", chip_id);
    TracyTTContextName(ctx, name.c_str(), static_cast<uint16_t>(name.size()));
    return ctx;
}

bool RealtimeProfilerTracyConsumer::validate_host_clock_domain() {
    // clock_sync is CLOCK_MONOTONIC; host_time_to_tracy_cpu_ticks bridges it into Tracy's rdtsc domain.
    const double ns_per_tick = TracyGetTimerMul();
    if (!(ns_per_tick > 0.0)) {
        log_error(
            tt::LogMetal,
            "[Real-time profiler] Tracy CPU timer unavailable (TimerMul <= 0); disabling Tracy real-time calibration");
        return false;
    }
    host_clock_ticks_per_ns_ = 1.0 / ns_per_tick;
    return true;
}

std::pair<int64_t, int64_t> RealtimeProfilerTracyConsumer::correlate_host_clocks() {
    const auto mono_ns = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    constexpr int kCorrelationAttempts = 8;
    int64_t best_gap = std::numeric_limits<int64_t>::max();
    int64_t mono = 0;
    int64_t tracy = 0;
    for (int i = 0; i < kCorrelationAttempts; ++i) {
        const int64_t mono_before = mono_ns();
        const int64_t sample = TracyGetCpuTime();
        const int64_t mono_after = mono_ns();
        if (const int64_t gap = mono_after - mono_before; gap < best_gap) {
            best_gap = gap;
            mono = mono_before + gap / 2;
            tracy = sample;
        }
    }
    return {mono, tracy};
}

void RealtimeProfilerTracyConsumer::refresh_host_clock_mapping() {
    const auto [mono, tracy] = correlate_host_clocks();
    // The first call measures against zeroed references and the clamp rejects it, leaving the
    // validated nominal slope; every later call refines within the clamp.
    const double span_ns = static_cast<double>(mono - host_clock_ref_mono_ns_);
    const double measured = static_cast<double>(tracy - host_clock_ref_tracy_) / span_ns;
    const double nominal = 1.0 / TracyGetTimerMul();
    if (std::abs(measured - nominal) <= nominal * kHostClockSlopeClampFraction) {
        host_clock_ticks_per_ns_ = measured;
    }
    host_clock_ref_mono_ns_ = mono;
    host_clock_ref_tracy_ = tracy;
}

int64_t RealtimeProfilerTracyConsumer::host_time_to_tracy_cpu_ticks(std::chrono::steady_clock::time_point host_time) {
    const int64_t host_mono_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(host_time.time_since_epoch()).count();
    return host_clock_ref_tracy_ +
           std::llround(static_cast<double>(host_mono_ns - host_clock_ref_mono_ns_) * host_clock_ticks_per_ns_);
}

void RealtimeProfilerTracyConsumer::publish_device_profiler_sync_anchor(
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

void RealtimeProfilerTracyConsumer::handle_record(TracyTTCtx ctx, const experimental::ProgramRealtimeRecord& record) {
    file_scratch_.clear();
    for (size_t i = 0; i < record.kernel_sources.size(); i++) {
        if (i > 0) {
            file_scratch_ += ",\n";
        }
        file_scratch_.append(record.kernel_sources[i].data(), record.kernel_sources[i].size());
    }
    if (file_scratch_.empty()) {
        file_scratch_ = "realtime_profiler";
    }

    name_scratch_.clear();
    fmt::format_to(std::back_inserter(name_scratch_), "Program op_id={}", record.runtime_id);
    const auto host_ns = [](std::chrono::steady_clock::time_point t) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
    };
    TracyTTPushStartMarker(
        ctx,
        make_marker(
            record, host_ns(record.host_start()), tracy::TTDeviceMarkerType::ZONE_START, file_scratch_, name_scratch_));
    TracyTTPushEndMarker(
        ctx,
        make_marker(
            record, host_ns(record.host_end()), tracy::TTDeviceMarkerType::ZONE_END, file_scratch_, name_scratch_));
}

}  // namespace tt::tt_metal

#endif  // TRACY_ENABLE
