// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental {

// Record passed to registered callbacks when real-time program performance
// telemetry data arrives from a device.
struct ProgramPerfRecord {
    uint32_t program_id;                      // Runtime program ID
    uint64_t start_timestamp;                 // Device start timestamp (raw ticks)
    uint64_t end_timestamp;                   // Device end timestamp (raw ticks)
    double frequency;                         // Device clock frequency (cycles per ns)
    uint32_t chip_id;                         // Device chip ID
    std::vector<std::string> kernel_sources;  // Kernel source paths for this program
};

// Callback type for program perf telemetry data.
using ProgramPerfCallback = std::function<void(const ProgramPerfRecord& record)>;

// Opaque handle returned by RegisterProgramPerfCallback, used to unregister.
using ProgramPerfCallbackHandle = uint64_t;

// clang-format off
/**
 * Register a callback to be invoked when real-time program performance telemetry
 * data arrives from a device. Multiple callbacks can be registered; they are called
 * in order of registration from the telemetry receiver thread.
 *
 * Return value: ProgramPerfCallbackHandle - handle that can be passed to
 *               UnregisterProgramPerfCallback to remove the callback.
 */
// clang-format on
ProgramPerfCallbackHandle RegisterProgramPerfCallback(ProgramPerfCallback callback);

/**
 * Unregister a previously registered callback by its handle.
 */
void UnregisterProgramPerfCallback(ProgramPerfCallbackHandle handle);

}  // namespace tt::tt_metal::experimental
