// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <tracy/TracyTTDevice.hpp>
#include "data_collection.hpp"

namespace tt::tt_metal {

// Manages Tracy contexts for real-time profiler.
// Registers itself as a ProgramRealtimeCallback and routes records to the correct
// Tracy context based on chip_id. Thread-safe for AddDevice/RemoveDevice.
class RealtimeProfilerTracyHandler {
public:
    RealtimeProfilerTracyHandler();
    ~RealtimeProfilerTracyHandler();

    RealtimeProfilerTracyHandler(const RealtimeProfilerTracyHandler&) = delete;
    RealtimeProfilerTracyHandler& operator=(const RealtimeProfilerTracyHandler&) = delete;

    // Create and calibrate a Tracy context for the given device.
    void AddDevice(uint32_t chip_id, int64_t host_start, double first_timestamp, double frequency);

    // Remove and destroy the Tracy context for the given device.
    void RemoveDevice(uint32_t chip_id);

    // Callback handler invoked by the real-time profiler system for each program record.
    void HandleRecord(const tt::ProgramRealtimeRecord& record);

private:
    TracyTTCtx GetContext(uint32_t chip_id);

    std::mutex mutex_;
    std::unordered_map<uint32_t, TracyTTCtx> tracy_contexts_;
    tt::ProgramRealtimeCallbackHandle callback_handle_;
};

}  // namespace tt::tt_metal
