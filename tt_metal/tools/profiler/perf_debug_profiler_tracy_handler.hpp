// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tracy sink for the perf-debug (drainer) profiler. One Tracy context (device ROW) per worker core, keyed
// by NOC0 coord; the 5 RISCs are lanes within.
//
// Zones arrive from the decode pipeline as COMPLETE {start, end} pairs (paired upstream in the decode
// workers), so the hot path is a single lock-free Tracy queue item per zone: no serial-queue mutex, no
// per-zone source-location allocation and no begin/end balance bookkeeping. The handler's mutex guards
// only the cold memoization paths (context creation, srcloc interning); callers are expected to cache
// the returned ZoneSink / srcloc pointers and hit PushWorkerZone lock-free.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tracy/TracyTTDevice.hpp>

#include "perf_debug_profiler_packets.hpp"

namespace tt::tt_metal {

class PerfDebugTracyHandler {
public:
    PerfDebugTracyHandler();
    ~PerfDebugTracyHandler();

    PerfDebugTracyHandler(const PerfDebugTracyHandler&) = delete;
    PerfDebugTracyHandler& operator=(const PerfDebugTracyHandler&) = delete;

    // Record a device's host<->device anchor. Per-core contexts are Populated with it when created.
    void AddDevice(uint32_t chip_id, int64_t host_start, double first_timestamp, double frequency);

    // Eagerly create the per-core contexts up front (context creation is ~ms; keep it off the drain
    // hot path). worker_noc0 = the chip's worker-core NOC0 coords.
    void PreCreateContexts(uint32_t chip_id, const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0);

    // Everything a zone push needs that is per-(chip, core): the core's Tracy context and the
    // thread-id bits above the RISC field. ctx == nullptr means the device was never AddDevice'd
    // (or Tracy is compiled out) -- skip the push.
    struct ZoneSink {
        TracyTTCtx ctx = nullptr;
        uint32_t thread_base = 0;
    };

    // Resolve (create on first sight) the core's context. Cold path (mutex); cache the result per lane.
    // MUST be called on the same thread that will push this sink's zones: the context announcement rides
    // the same lock-free queue as the zones, which is what guarantees the server sees it first.
    ZoneSink GetZoneSink(uint32_t chip_id, uint32_t core_noc0_x, uint32_t core_noc0_y);

    // Intern one zone identity (name resolved from the srcloc hash, per-RISC fallback color) into a
    // process-lifetime Tracy source location. Cold path (mutex); cache the result per (hash, risc).
    const void* InternZoneSrcloc(uint32_t hash, uint32_t risc, std::string_view name);

    // Push one COMPLETE zone: one lock-free Tracy queue item, no locks taken. Per lane, calls must
    // arrive in zone-completion order (sorted by `end`); the decode pairing produces exactly that.
    static void PushWorkerZone(const ZoneSink& sink, const void* srcloc, uint32_t risc, uint64_t start, uint64_t end);

    // Push one point-in-time event onto its core's Tracy lane. Rare path; rides the serial queue.
    void HandleWorkerEvent(const perf_debug::WorkerEventPacket& event);

private:
    static uint64_t ContextKey(uint32_t chip_id, uint32_t core_x, uint32_t core_y) {
        return (static_cast<uint64_t>(chip_id) << 40) | (static_cast<uint64_t>(core_x) << 20) |
               (static_cast<uint64_t>(core_y) & 0xFFFFF);
    }
    TracyTTCtx GetOrCreateContext(uint32_t chip_id, uint32_t core_x, uint32_t core_y, const std::string& name);

    struct ChipAnchor {
        int64_t host_start = 0;
        double first_timestamp = 0.0;
        double frequency = 0.0;
    };

    std::mutex mutex_;
    std::unordered_map<uint32_t, ChipAnchor> chip_anchors_;
    std::unordered_map<uint64_t, TracyTTCtx> tracy_contexts_;
    // Interned zone identities: (hash, risc) -> SourceLocationData. The pointed-to structs and their
    // name strings are LEAKED deliberately: Tracy's worker reads them from client memory when the
    // SERVER asks, which can be long after this handler is gone.
    std::unordered_map<uint64_t, const void*> srclocs_;
};

}  // namespace tt::tt_metal
