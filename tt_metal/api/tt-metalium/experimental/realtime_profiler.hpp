// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>

namespace tt::tt_metal::experimental {

struct ProgramRealtimeClockSync {
    int64_t device_cycle_offset;          // A device timestamp maps to std::chrono::steady_clock host time as
                                          // host_ns = (timestamp - device_cycle_offset) / frequency
    std::chrono::nanoseconds sync_error;  // Estimated error in a host time derived from this mapping, based on the most
                                          // recent measurement against the device clock
};

struct ProgramRealtimeRecord {
    uint32_t runtime_id;                               // Runtime ID. Currently truncated to 16 bits;
                                                       // widening tracked in #46103.
    uint32_t chip_id;                                  // Device chip ID
    uint64_t start_timestamp;                          // Device start timestamp (raw ticks)
    uint64_t end_timestamp;                            // Device end timestamp (raw ticks)
    double frequency;                                  // Device clock frequency (cycles per ns)
    ProgramRealtimeClockSync clock_sync;               // Device-to-host clock mapping for this record
    std::span<const std::string_view> kernel_sources;  // Kernel source paths; valid for the
                                                       // lifetime of the process.

    /**
     * @brief Program execution duration.
     *
     * Assumes the device clock frequency is stable.
     */
    [[nodiscard]] constexpr std::chrono::duration<double, std::nano> duration() const {
        return std::chrono::duration<double, std::nano>{
            static_cast<double>(end_timestamp - start_timestamp) / frequency};
    }

    /**
     * @brief When the program began executing, on the std::chrono::steady_clock timeline.
     *
     * Error estimated by clock_sync.sync_error.
     */
    [[nodiscard]] constexpr std::chrono::steady_clock::time_point host_start() const {
        const std::chrono::duration<double, std::nano> host_ns{
            (static_cast<double>(start_timestamp) - static_cast<double>(clock_sync.device_cycle_offset)) / frequency};
        return std::chrono::steady_clock::time_point{std::chrono::round<std::chrono::steady_clock::duration>(host_ns)};
    }

    /**
     * @brief When the program finished executing, on the std::chrono::steady_clock timeline.
     *
     * Error estimated by clock_sync.sync_error.
     */
    [[nodiscard]] constexpr std::chrono::steady_clock::time_point host_end() const {
        const std::chrono::duration<double, std::nano> host_ns{
            (static_cast<double>(end_timestamp) - static_cast<double>(clock_sync.device_cycle_offset)) / frequency};
        return std::chrono::steady_clock::time_point{std::chrono::round<std::chrono::steady_clock::duration>(host_ns)};
    }

    /**
     * @brief The device timestamp (raw ticks) a host time maps to.
     *
     * Accurate only near this record; a host time far from this record's window carries more error than
     * clock_sync.sync_error reports.
     */
    [[nodiscard]] constexpr uint64_t device_timestamp_at(std::chrono::steady_clock::time_point host_time) const {
        const double host_ns = std::chrono::duration<double, std::nano>{host_time.time_since_epoch()}.count();
        return static_cast<uint64_t>(host_ns * frequency + static_cast<double>(clock_sync.device_cycle_offset));
    }
};

struct ProgramRealtimeRecordBatch {
    std::span<const ProgramRealtimeRecord> records;  // Non-empty, oldest first; valid
                                                     // until the callback returns.
    uint64_t dropped;                                // Records lost since this callback last ran; nonzero if the
                                                     // callback could not keep up with incoming profiler data.
};

// Callback type for real-time profiler data. Invoked with a batch so a callback can
// amortize fixed costs (a lock, a file flush, a network/DB round-trip, etc.) across many records.
using ProgramRealtimeProfilerCallback = std::function<void(const ProgramRealtimeRecordBatch& batch)>;

// Opaque handle returned by RegisterProgramRealtimeProfilerCallback, used to unregister.
using ProgramRealtimeProfilerCallbackHandle = uint64_t;

// clang-format off
/**
 * Register a callback to be invoked when real-time profiler data arrives from a device.
 * Multiple callbacks can be registered; they are invoked concurrently, each on its own thread. If a
 * callback shares a resource with other callbacks, access it in a thread-safe way (e.g. with a lock).
 * Callbacks that are too slow to keep up with incoming profiler data may miss records; this
 * is reported by ProgramRealtimeRecordBatch::dropped.
 *
 * Return value: ProgramRealtimeProfilerCallbackHandle - handle that can be passed to
 *               UnregisterProgramRealtimeProfilerCallback to remove the callback.
 */
// clang-format on
ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallback callback);

/**
 * Unregister a previously registered callback by its handle.
 *
 * This call blocks until any in-flight invocation of that callback has completed. A callback may also unregister
 * itself; such a call returns immediately, and the callback is not invoked again after its current invocation ends.
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
