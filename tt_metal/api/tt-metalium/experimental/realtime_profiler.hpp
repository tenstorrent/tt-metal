// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace tt::tt_metal::experimental {

struct ProgramRealtimeRecord {
    uint32_t runtime_id;                               // Runtime ID. Currently truncated to 16 bits;
                                                       // widening tracked in #46103.
    uint32_t chip_id;                                  // Device chip ID
    uint64_t start_timestamp;                          // Device start timestamp (raw ticks)
    uint64_t end_timestamp;                            // Device end timestamp (raw ticks), or 0 when this schema does
                                                       // not provide a correlated endpoint.
    double frequency;                                  // Device clock frequency (cycles per ns)
    std::span<const std::string_view> kernel_sources;  // Kernel source paths; valid until
                                                       // MetalContext teardown or reinitialization.
    uint32_t command_queue_id = 0;
    uint32_t dispatch_stream = 0;
    uint32_t generation = 0;  // Zero-extended low 16 bits from the device record.
    uint32_t sequence = 0;
    uint32_t schema_version = 0;
    uint32_t record_type = 0;
    uint64_t cumulative_source_dropped = 0;
};

struct ProgramRealtimeProfilerDeviceLossSnapshot {
    uint32_t chip_id = 0;
    uint64_t cumulative_source_dropped = 0;
};

struct ProgramRealtimeRecordBatch {
    std::span<const ProgramRealtimeRecord> records;  // Non-empty, oldest first; valid
                                                     // until the callback returns.
    uint64_t dropped;                                // Records lost since this callback last ran; nonzero if the
                                                     // callback could not keep up with incoming profiler data.
    std::span<const ProgramRealtimeProfilerDeviceLossSnapshot> device_loss;
};

enum class ProgramRealtimeProfilerInactiveReason : uint8_t {
    None,
    NotInitialized,
    DisabledByEnvironment,
    UnsupportedArchitecture,
    MultipleHardwareCommandQueues,
    NonMmioDevice,
    IommuUnavailable,
    NonWorkerDispatch,
    DistributedDispatcher,
    VirtualizedUnicast,
    NoReservedProfilerCore,
    InsufficientL1,
    KernelsNullified,
    SocketInitializationFailed,
    ProtocolInitializationFailed,
};

struct ProgramRealtimeProfilerLossCounts {
    uint64_t descriptor_full = 0;
    uint64_t unsupported_launch = 0;
    uint64_t reset_descriptor = 0;
    uint64_t observer_coalesced = 0;
    uint64_t stuck_head = 0;
    uint64_t completed_record = 0;
    uint64_t terminal_descriptor = 0;
    uint64_t terminal_record = 0;
    uint64_t observer_stop_timeout = 0;
    uint64_t device_ring = 0;
};

struct ProgramRealtimeProfilerDeviceCapability {
    uint32_t chip_id = 0;
    bool active = false;
    ProgramRealtimeProfilerInactiveReason inactive_reason = ProgramRealtimeProfilerInactiveReason::NotInitialized;
    // Refreshed from device L1 only when the explicit capability query is
    // called; never used as a timing endpoint.
    ProgramRealtimeProfilerLossCounts loss;
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
 * The real-time profiler currently runs only on supported single-CQ Blackhole worker-dispatch
 * configurations with a dedicated tensix core and an MMIO-connected device for the D2H socket.
 * GetProgramRealtimeProfilerDeviceCapabilities() reports why any evaluated device is inactive.
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
 * Returns the newest capability snapshot for every evaluated chip. A closed
 * device remains present with active=false so shutdown-only loss counters can
 * be audited; the next evaluation of that chip replaces the snapshot.
 */
std::vector<ProgramRealtimeProfilerDeviceCapability> GetProgramRealtimeProfilerDeviceCapabilities();

}  // namespace tt::tt_metal::experimental
