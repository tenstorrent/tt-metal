// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <chrono>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::distributed {
class MeshCommandQueue;
}

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
    uint32_t command_queue_id = 0;                     // Fixed at 0 by the supported single-command-queue protocol.
    uint32_t dispatch_stream = 0;
    uint32_t sequence = 0;
    uint32_t schema_version = 0;
    uint32_t record_type = 0;
};

struct ProgramRealtimeProfilerDeviceCollection {
    uint32_t chip_id = 0;
    uint32_t expected_stream_mask = 0;
    uint32_t observed_stream_mask = 0;
    uint64_t record_count = 0;
    uint64_t descriptor_dropped = 0;
    uint64_t observer_dropped = 0;
    uint64_t record_dropped = 0;
    uint64_t source_dropped = 0;
    uint64_t transport_dropped = 0;
};

struct ProgramRealtimeProfilerCollectionResult {
    uint32_t requested_watermark = 0;
    uint32_t observed_watermark = 0;
    uint64_t record_count = 0;
    uint64_t descriptor_dropped = 0;
    uint64_t observer_dropped = 0;
    uint64_t record_dropped = 0;
    uint64_t source_dropped = 0;
    uint64_t transport_dropped = 0;
    bool timed_out = false;
    bool profiler_inactive = false;
    bool protocol_error = false;
    std::vector<ProgramRealtimeProfilerDeviceCollection> devices;

    // complete() proves the exact device watermark was observed for every
    // expected stream; it does not imply lossless collection. Check lossy().
    bool complete() const { return requested_watermark != 0 && observed_watermark == requested_watermark; }
    bool lossy() const {
        return timed_out || profiler_inactive || source_dropped != 0 || transport_dropped != 0 || protocol_error;
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
 * Multiple callbacks can be registered; they are invoked concurrently. If a callback shares a resource
 * with other callbacks or across multiple MeshDevices, access it in a thread-safe way (e.g. with a lock).
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
 * This call blocks until any in-flight invocation of that callback has completed.
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

/**
 * Finish the selected streams and wait for their exact device-produced
 * profiler watermark. The timeout controls only the host collection wait;
 * operation durations in the returned records remain raw device ticks. An
 * ineligible or disabled profiler is reported as profiler_inactive, separately
 * from a malformed device protocol response. The caller must keep the command
 * queue and its MeshDevice alive until this function returns.
 */
ProgramRealtimeProfilerCollectionResult FinishAndCollectProgramRealtimeProfiler(
    distributed::MeshCommandQueue& command_queue,
    std::chrono::milliseconds timeout,
    ttsl::Span<const SubDeviceId> sub_device_ids = {});

}  // namespace tt::tt_metal::experimental
