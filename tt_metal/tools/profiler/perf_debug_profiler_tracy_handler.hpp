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

    // Record a PER-CORE anchor, overriding the chip's for that one core. Exists because the chip anchor is
    // measured on a TENSIX WORKER while these rows come off a DRAM tile, and on this part the two wall clocks
    // have banked very different totals: they share a zero point (chip reset) and neither is re-zeroed at device
    // open, but the Tensix domain is clocked only while out of reset and the DRAM one keeps running at a reduced
    // clock while idle -- ~11x duty ratio, so the gap grows with card activity. Sharing one anchor therefore
    // shifted every DRAM-core (DRISC drainer) row right by that whole gap -- 18 min on a card up half an hour,
    // 42 min an hour later -- while zone DURATIONS stayed correct to 0.15%, which is why it read as a mysterious
    // constant skew rather than a broken clock. Board-dependent: on parts whose two counters track each other
    // the offset is microseconds and this call changes nothing. See FINDINGS N+46.
    // MUST be called before PreCreateContexts/the first zone for that core: a context bakes its anchor in at
    // creation.
    void AddCore(
        uint32_t chip_id,
        uint32_t noc0_x,
        uint32_t noc0_y,
        int64_t host_start,
        double first_timestamp,
        double frequency);

    // Eagerly create the per-core contexts up front (context creation is ~ms; keep it off the drain
    // hot path). worker_noc0 = the chip's worker-core NOC0 coords.
    void PreCreateContexts(uint32_t chip_id, const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0);

    // Push one COMPLETE zone (both endpoints) onto its core's Tracy lane, as a single lock-free
    // QueueGpuZone item (TracyTTPushZone). Contract inherited from Tracy: per (context, thread),
    // calls must arrive in zone-completion order (non-decreasing end) -- which is exactly the paired
    // stream's per-lane arrival order, so the caller forwards in arrival order and nothing buffers.
    void HandleWorkerZone(const perf_debug::WorkerZonePacket& zone);

    // Push one point-in-time event onto its core's Tracy lane. Unlike a zone this has no START/END pair,
    // so it bypasses the lane_depth_ bookkeeping entirely -- it cannot orphan or unbalance a stack.
    void HandleWorkerEvent(const perf_debug::WorkerEventPacket& event);

private:
    static uint64_t ContextKey(uint32_t chip_id, uint32_t core_x, uint32_t core_y) {
        return (static_cast<uint64_t>(chip_id) << 40) | (static_cast<uint64_t>(core_x) << 20) |
               (static_cast<uint64_t>(core_y) & 0xFFFFF);
    }
    // Everything this handler emits -- context creation, zones (PushZoneSerial), markers -- rides the
    // SERIAL queue, so the stream is totally ordered by construction and a zone or marker can never
    // reach the server before the GpuNewContext it references (the server hard-asserts on an unknown
    // context). The lock-free PushZone/PopulateLockfree variants exist in the fork but are NOT used
    // here: the client drains the lock-free queues BEFORE the serial one each pass, so mixing queues
    // makes creation/use ordering racy -- measured as an intermittent tracy-capture segfault.
    TracyTTCtx GetOrCreateContext(uint32_t chip_id, uint32_t core_x, uint32_t core_y, const std::string& name);

    struct ChipAnchor {
        int64_t host_start = 0;
        double first_timestamp = 0.0;
        double frequency = 0.0;
    };

    // Resolve the anchor for one core: its own per-core entry if it has one, else its chip's. Returns false
    // when neither exists (the device was never AddDevice'd). Two flavours because GetOrCreateContext already
    // holds mutex_ across its whole body and mutex_ is not recursive -- taking it again would self-deadlock.
    bool LookupAnchorLocked(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out);  // lock HELD
    bool LookupAnchor(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out);        // takes lock

    std::mutex mutex_;
    std::unordered_map<uint32_t, ChipAnchor> chip_anchors_;
    // ContextKey -> anchor, for cores whose clock does NOT share the chip anchor's origin (DRAM tiles). Empty
    // on parts where the two counters agree, in which case every lookup falls through to chip_anchors_.
    std::unordered_map<uint64_t, ChipAnchor> core_anchors_;
    std::unordered_map<uint64_t, TracyTTCtx> tracy_contexts_;
    // Zone-id/colour -> immortal Tracy SourceLocationData. QueueGpuZone carries a POINTER the server
    // dereferences by querying this client whenever it likes, so both the struct and the name string
    // it points at must live for the rest of the process -- interned once, never freed (a few hundred
    // distinct zone names per capture). Key: id << 32 | color (the same name can carry per-role
    // colours, and colour lives inside the srcloc). Stored as void* so this header stays valid in
    // non-TRACY builds.
    std::unordered_map<uint64_t, const void*> zone_srclocs_;
};

}  // namespace tt::tt_metal
