// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tracy sink for the perf-debug (drainer) profiler. One Tracy context (device ROW) per worker core, keyed
// by NOC0 coord; the 5 RISCs are lanes within. Lifted from realtime_profiler's Tracy handler, trimmed to
// the worker-zone path only (no program records, no host<->device sync, no callback registry -- the
// PerfDebugProfiler drain threads call HandleWorkerZone directly).
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

    // Push one fully-resolved zone (ZONE_START or ZONE_END) onto its core's Tracy lane. Zones arrive in
    // emission order per lane, so pushing in arrival order nests correctly.
    void HandleWorkerZone(const perf_debug::WorkerZonePacket& zone);

    // Label a drainer core with its ROLE, so a plot row reads "DRISC 9-9 FILLER" rather than coordinates
    // nobody can map to a job. `role` MUST be a string literal or otherwise immortal: it ends up inside an
    // interned plot name, and a plot name pointer is dereferenced by the SERVER querying the client, so a
    // dangling one is read after free. Called once per drainer at boot, before any sample arrives.
    void SetDriscRole(uint32_t chip_id, uint32_t noc0_x, uint32_t noc0_y, const char* role);

    // Push one point-in-time event onto its core's Tracy lane. Unlike a zone this has no START/END pair,
    // so it bypasses the lane_depth_ bookkeeping entirely -- it cannot orphan or unbalance a stack.
    void HandleWorkerEvent(const perf_debug::WorkerEventPacket& event);

private:
    static uint64_t ContextKey(uint32_t chip_id, uint32_t core_x, uint32_t core_y) {
        return (static_cast<uint64_t>(chip_id) << 40) | (static_cast<uint64_t>(core_x) << 20) |
               (static_cast<uint64_t>(core_y) & 0xFFFFF);
    }
    TracyTTCtx GetOrCreateContext(uint32_t chip_id, uint32_t core_x, uint32_t core_y, const std::string& name);

    // ContextKey -> immortal role string ("FILLER"/"MOVER"). Empty for any core never registered, which is
    // why the label falls back to bare coordinates rather than asserting.
    std::unordered_map<uint64_t, const char*> drisc_roles_;

    struct ChipAnchor {
        int64_t host_start = 0;
        double first_timestamp = 0.0;
        double frequency = 0.0;
    };

    std::mutex mutex_;
    std::unordered_map<uint32_t, ChipAnchor> chip_anchors_;
    std::unordered_map<uint64_t, TracyTTCtx> tracy_contexts_;
    // Shadow of Tracy's per-(context,risc) GPU zone stack depth, so an unmatched ZONE_END (which would
    // pop an empty stack and SEGV tracy-capture) is dropped here. Key: (ContextKey<<3)|risc.
    std::unordered_map<uint64_t, int32_t> lane_depth_;
    uint64_t orphan_end_count_ = 0;
    std::unordered_set<uint64_t> orphan_lanes_;
    // Per-drainer latest per-sweep NoC KB, for the derived aggregate plot. Keyed by ContextKey.
    // Per-drainer state for the NoC-footprint RATE plots: the previous sample's device-ns instant, and the
    // latest computed rate. A rate needs two samples, so the first one per drainer only seeds the timestamp.
    std::unordered_map<uint64_t, double> nocfp_last_ns_;
    std::unordered_map<uint64_t, double> nocfp_last_gbps_;
};

}  // namespace tt::tt_metal
