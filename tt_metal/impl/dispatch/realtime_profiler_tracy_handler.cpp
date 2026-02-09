// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_tracy_handler.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

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
    marker.risc = tracy::RiscType::NONE;
    marker.timestamp = timestamp;
    marker.runtime_host_id = record.program_id;
    marker.marker_name = fmt::format("Program_{}", record.program_id);
    marker.marker_type = type;
    marker.file = file_str;
    marker.line = 0;
    return marker;
}
#endif

}  // namespace

RealtimeProfilerTracyHandler::RealtimeProfilerTracyHandler() {
    callback_handle_ =
        tt::RegisterProgramRealtimeCallback([this](const tt::ProgramRealtimeRecord& record) { HandleRecord(record); });
}

RealtimeProfilerTracyHandler::~RealtimeProfilerTracyHandler() {
    tt::UnregisterProgramRealtimeCallback(callback_handle_);

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [chip_id, ctx] : tracy_contexts_) {
        TracyTTDestroy(ctx);
    }
    tracy_contexts_.clear();
}

void RealtimeProfilerTracyHandler::AddDevice(
    uint32_t chip_id, int64_t host_start, double first_timestamp, double frequency) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracy_contexts_.count(chip_id)) {
        log_warning(tt::LogMetal, "RealtimeProfilerTracyHandler: device {} already added, skipping", chip_id);
        return;
    }

    TracyTTCtx ctx = TracyTTContext();
    TracyTTContextPopulate(ctx, host_start, first_timestamp, frequency);
    std::string name = fmt::format("Device {}:", chip_id);
    TracyTTContextName(ctx, name.c_str(), name.size());

    tracy_contexts_[chip_id] = ctx;
}

void RealtimeProfilerTracyHandler::RemoveDevice(uint32_t chip_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tracy_contexts_.find(chip_id);
    if (it == tracy_contexts_.end()) {
        return;
    }
    TracyTTDestroy(it->second);
    tracy_contexts_.erase(it);
}

TracyTTCtx RealtimeProfilerTracyHandler::GetContext(uint32_t chip_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tracy_contexts_.find(chip_id);
    return it != tracy_contexts_.end() ? it->second : nullptr;
}

void RealtimeProfilerTracyHandler::HandleRecord(const tt::ProgramRealtimeRecord& record) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetContext(record.chip_id);
    if (!ctx) {
        return;
    }

    std::string file_str = record.program_id > 0 ? tt::GetKernelSourcesForRuntimeId(record.program_id) : "";
    if (file_str.empty()) {
        file_str = "realtime_profiler";
    }

    auto start = make_marker(record, record.start_timestamp, tracy::TTDeviceMarkerType::ZONE_START, file_str);
    auto end = make_marker(record, record.end_timestamp, tracy::TTDeviceMarkerType::ZONE_END, file_str);

    TracyTTPushStartMarker(ctx, start);
    TracyTTPushEndMarker(ctx, end);
#endif
}

}  // namespace tt::tt_metal
