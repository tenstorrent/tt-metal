// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental {

// Record passed to registered callbacks when real-time profiler data arrives from a device.
struct ProgramRealtimeRecord {
    uint32_t program_id;                      // Runtime program ID
    uint64_t start_timestamp;                 // Device start timestamp (raw ticks)
    uint64_t end_timestamp;                   // Device end timestamp (raw ticks)
    double frequency;                         // Device clock frequency (cycles per ns)
    uint32_t chip_id;                         // Device chip ID
    std::vector<std::string> kernel_sources;  // Kernel source paths for this program
};

// Callback type for real-time profiler data.
using ProgramRealtimeProfilerCallback = std::function<void(const ProgramRealtimeRecord& record)>;

// Opaque handle returned by RegisterProgramRealtimeProfilerCallback, used to unregister.
using ProgramRealtimeProfilerCallbackHandle = uint64_t;

// clang-format off
/**
 * Register a callback to be invoked when real-time profiler data arrives from a device.
 * Multiple callbacks can be registered; they are called in order of registration from the
 * real-time profiler receiver thread.
 *
 * Return value: ProgramRealtimeProfilerCallbackHandle - handle that can be passed to
 *               UnregisterProgramRealtimeProfilerCallback to remove the callback.
 */
// clang-format on
ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallback callback);

/**
 * Unregister a previously registered callback by its handle.
 */
void UnregisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallbackHandle handle);

/**
 * Returns true if the real-time profiler is currently running on at least one chip.
 *
 * The real-time profiler is gated on host-accessible dispatch resources: it needs a
 * dedicated tensix core reserved from the dispatch pool and an MMIO-connected device
 * for the D2H socket. On configurations where those are not available (e.g. ETH
 * dispatch, remote chips on multi-host meshes), the profiler bows out silently and
 * no records will ever be delivered to registered callbacks.
 *
 * Callers that want to distinguish "profiler is on but has not produced records yet"
 * from "profiler is disabled by the current dispatch config" should query this before
 * asserting on collected record counts — the canonical use case is for tests that
 * want to gracefully skip when RT profiler is not supported.
 *
 * This is safe to call at any time after device construction. It becomes true after
 * the init-sync handshake for the first device completes, and returns to false when
 * every RT-profiler-enabled device has been closed.
 */
bool IsProgramRealtimeProfilerActive();

}  // namespace tt::tt_metal::experimental
