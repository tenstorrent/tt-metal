// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <tracy/TracyTTDevice.hpp>
#include <tt-metalium/experimental/realtime_profiler_packets.hpp>
#include "data_collection.hpp"

namespace tt::tt_metal {

// Manages Tracy contexts for real-time profiler.
// Registers itself as a ProgramRealtimeProfilerCallback and routes records to the correct
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

    // Push a sync-check marker on the device GPU timeline at the given device timestamp.
    // Used to verify host-device clock sync accuracy in the Tracy GUI.
    void PushSyncCheckMarker(uint32_t chip_id, uint64_t device_timestamp, double frequency);

    // Subscriber for X280-drained worker-core kernel zones. The enriched packet is already fully
    // resolved by the host (NOC0 coords, deciphered name, is_start), so this just pushes a
    // ZONE_START or ZONE_END marker on the originating core's Tracy lane. Zones arrive in ring order
    // (== emission order == correct nest order per lane), so pushing in arrival order nests correctly.
    void HandleWorkerZone(const tt::tt_metal::experimental::WorkerZonePacket& zone);

    // Send a GpuCalibration event to Tracy, updating the host-device clock mapping.
    void CalibrateDevice(uint32_t chip_id, int64_t host_time, uint64_t device_timestamp, double frequency);

    // Eagerly create all per-core Tracy contexts up front (at init, before the workload runs) from the
    // given worker-core NOC0 coords, plus the program/sync (100,100) context. Creating a Tracy context
    // is expensive (~ms); doing it lazily on the receiver thread during draining blocks the socket
    // drain and back-pressures the X280. Pre-creating here (rings empty) keeps it off the hot path.
    void PreCreateContexts(uint32_t chip_id, const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0);

private:
    // Tracy renders one device ROW per context, so contexts are keyed by (chip, core) — like the
    // standard DeviceProfiler's device_tracy_contexts[{chip,core}] — otherwise every core's RISCs
    // collapse into a single row. Contexts are created lazily on the first marker for a core and
    // Populated ONCE with the chip's current anchor. (We deliberately do NOT Populate-then-Calibrate
    // a fresh context: two anchors at birth gave Tracy conflicting mappings.)
    static uint64_t ContextKey(uint32_t chip_id, uint32_t core_x, uint32_t core_y) {
        return (static_cast<uint64_t>(chip_id) << 40) | (static_cast<uint64_t>(core_x) << 20) |
               (static_cast<uint64_t>(core_y) & 0xFFFFF);
    }
    TracyTTCtx GetOrCreateContext(uint32_t chip_id, uint32_t core_x, uint32_t core_y, const std::string& name);

    void RecordSkippedZoneWithEndBeforeStart(const tt::ProgramRealtimeRecord& record, int64_t delta);
    void MaybeEmitSkippedZoneSummaryLocked();

    struct SkippedEndBeforeStartStats {
        uint64_t total_skipped = 0;
        uint64_t suppressed_since_last_summary = 0;
        bool logged_first_detail = false;
        std::unordered_map<uint32_t, uint64_t> count_by_runtime_id;
        std::unordered_map<uint32_t, uint64_t> count_by_chip_id;
        std::chrono::steady_clock::time_point last_summary_time{};
        static constexpr std::chrono::seconds kSummaryInterval{30};
    };

    // Current device<->host anchor per chip (set at AddDevice, refined by CalibrateDevice). Each
    // per-core context is Populated with this when lazily created, so all of a chip's contexts share
    // one consistent anchor.
    struct ChipAnchor {
        int64_t host_start = 0;
        double first_timestamp = 0.0;
        double frequency = 0.0;
    };

    std::mutex mutex_;
    std::unordered_map<uint32_t, ChipAnchor> chip_anchors_;    // chip_id -> current anchor
    std::unordered_map<uint64_t, TracyTTCtx> tracy_contexts_;  // ContextKey(chip,core) -> ctx
    SkippedEndBeforeStartStats skipped_end_before_start_stats_;
    tt::ProgramRealtimeProfilerCallbackHandle callback_handle_;
    tt::tt_metal::experimental::ProfilerPacketCallbackHandle packet_callback_handle_;
};

}  // namespace tt::tt_metal
