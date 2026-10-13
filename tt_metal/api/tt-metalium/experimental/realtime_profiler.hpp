// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string_view>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

struct ProgramRealtimeRecord {
    uint32_t runtime_id;                               // Runtime ID. Currently truncated to 16 bits;
                                                       // widening tracked in #46103.
    uint32_t chip_id;                                  // Device chip ID
    uint64_t start_timestamp;                          // Device start timestamp (raw ticks)
    uint64_t end_timestamp;                            // Device end timestamp (raw ticks)
    double frequency;                                  // Device clock frequency (cycles per ns)
    std::span<const std::string_view> kernel_sources;  // Kernel source paths; valid until
                                                       // MetalContext teardown or reinitialization.
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
 * Register a callback to be invoked with real-time profiler records for programs that ran on `mesh_device`'s local
 * chips. Multiple callbacks can be registered; they are invoked concurrently. If a callback shares a resource with
 * other callbacks or across multiple MeshDevices, access it in a thread-safe way (e.g. with a lock). Callbacks that
 * are too slow to keep up with incoming profiler data may miss records; this is reported by
 * ProgramRealtimeRecordBatch::dropped.
 *
 * The callback stays registered until it is unregistered or `mesh_device` is closed; closing first delivers the
 * records still in flight. A callback must not register or unregister callbacks.
 *
 * Return value: ProgramRealtimeProfilerCallbackHandle - handle that can be passed to
 *               UnregisterProgramRealtimeProfilerCallback to remove the callback.
 */
// clang-format on
ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
    distributed::MeshDevice& mesh_device, ProgramRealtimeProfilerCallback callback);

/**
 * Unregister a callback previously registered on `mesh_device` by its handle.
 *
 * This call blocks until any in-flight invocation of that callback has completed. It is a no-op if `mesh_device`
 * has already been closed.
 */
void UnregisterProgramRealtimeProfilerCallback(
    distributed::MeshDevice& mesh_device, ProgramRealtimeProfilerCallbackHandle handle);

/**
 * Returns true if the real-time profiler is running on at least one of `mesh_device`'s local chips.
 *
 * The real-time profiler is gated on host-accessible dispatch resources: it needs a
 * dedicated tensix core reserved from the dispatch pool and an MMIO-connected device
 * for the D2H socket. On configurations where those are not available (e.g. ETH
 * dispatch, remote chips on multi-host meshes), the profiler bows out silently and
 * no records will ever be delivered to registered callbacks.
 *
 * Tests that want to collect records should query this before asserting on collected record counts, and skip when
 * it is false. Returns false once `mesh_device` has been closed.
 */
bool IsProgramRealtimeProfilerActive(const distributed::MeshDevice& mesh_device);

}  // namespace tt::tt_metal::experimental
