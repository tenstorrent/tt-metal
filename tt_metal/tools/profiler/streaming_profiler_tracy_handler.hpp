// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tracy sink for the streaming profiler. One Tracy context (device row) per worker core, keyed by NOC0
// coord; the 5 RISCs are lanes within it. Worker zones and markers only: the drain threads call
// HandleWorkerZone / HandleWorkerEvent directly, with no callback registry in between.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tracy/TracyTTDevice.hpp>

#include "streaming_profiler_packets.hpp"

namespace tt::tt_metal {

class StreamingProfilerTracyHandler {
public:
    StreamingProfilerTracyHandler();
    ~StreamingProfilerTracyHandler();

    StreamingProfilerTracyHandler(const StreamingProfilerTracyHandler&) = delete;
    StreamingProfilerTracyHandler& operator=(const StreamingProfilerTracyHandler&) = delete;

    // Record a device's host<->device anchor. Per-core contexts are populated with it when created.
    void AddDevice(uint32_t chip_id, int64_t host_start, double first_timestamp, double frequency);

    // Record a per-core anchor, overriding the chip's for that one core. The chip anchor is measured on a
    // tensix worker while DRAM-tile rows come off a different clock: the two share a zero point (chip reset)
    // and neither is re-zeroed at device open, but the tensix domain is clocked only while out of reset while
    // the DRAM one keeps running at a reduced clock when idle, so the gap grows with card activity. One shared
    // anchor therefore shifts every DRAM-core row right by that whole gap while durations stay correct, which
    // reads as a constant skew rather than a broken clock. Board-dependent: where the two counters track each
    // other the offset is microseconds and this call changes nothing.
    // Must be called before PreCreateContexts / the first zone for that core: a context bakes its anchor in at
    // creation.
    void AddCore(
        uint32_t chip_id,
        uint32_t noc0_x,
        uint32_t noc0_y,
        int64_t host_start,
        double first_timestamp,
        double frequency);

    // Eagerly create the per-core contexts up front; creation is ~ms and belongs off the drain hot path.
    // worker_noc0 = the chip's worker-core NOC0 coords.
    void PreCreateContexts(uint32_t chip_id, const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0);

    // Push one complete zone (both endpoints) onto its core's Tracy lane as a single queue item.
    // Contract inherited from Tracy: per (context, thread), calls must arrive in zone-completion order
    // (non-decreasing end), which is exactly the paired stream's per-lane arrival order.
    void HandleWorkerZone(const streaming_profiler::WorkerZonePacket& zone);

    // Push one point-in-time event onto its core's Tracy lane.
    void HandleWorkerEvent(const streaming_profiler::WorkerEventPacket& event);

private:
    static uint64_t ContextKey(uint32_t chip_id, uint32_t core_x, uint32_t core_y) {
        return (static_cast<uint64_t>(chip_id) << 40) | (static_cast<uint64_t>(core_x) << 20) |
               (static_cast<uint64_t>(core_y) & 0xFFFFF);
    }
    // Everything this handler emits -- context creation, zones, markers -- rides the serial queue, so a
    // zone or marker can never reach the server before the GpuNewContext it references (the server
    // hard-asserts on an unknown context). The lock-free variants are deliberately unused: the client
    // drains the lock-free queues before the serial one each pass, so mixing them makes creation/use
    // ordering racy.
    TracyTTCtx GetOrCreateContext(uint32_t chip_id, uint32_t core_x, uint32_t core_y, const std::string& name);

    struct ChipAnchor {
        int64_t host_start = 0;
        double first_timestamp = 0.0;
        double frequency = 0.0;
    };

    // Resolve the anchor for one core: its own per-core entry if it has one, else its chip's. Returns false
    // when neither exists (the device was never AddDevice'd). Two flavours because GetOrCreateContext holds
    // mutex_ across its whole body and mutex_ is not recursive.
    bool LookupAnchorLocked(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out);  // lock held
    bool LookupAnchor(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out);        // takes lock

    std::mutex mutex_;
    std::unordered_map<uint32_t, ChipAnchor> chip_anchors_;
    // ContextKey -> anchor, for cores whose clock does not share the chip anchor's origin (DRAM tiles).
    // Empty on parts where the two counters agree, and every lookup then falls through to chip_anchors_.
    std::unordered_map<uint64_t, ChipAnchor> core_anchors_;
    std::unordered_map<uint64_t, TracyTTCtx> tracy_contexts_;
    // Zone-id/colour -> immortal Tracy SourceLocationData. QueueGpuZone carries a pointer the server
    // dereferences by querying this client whenever it likes, so both the struct and the name string it
    // points at must live for the rest of the process: interned once, never freed. Key: id << 32 | color,
    // because the same name can carry per-role colours and the colour lives inside the srcloc. Stored as
    // void* so this header stays valid in non-TRACY builds.
    std::unordered_map<uint64_t, const void*> zone_srclocs_;
};

}  // namespace tt::tt_metal
