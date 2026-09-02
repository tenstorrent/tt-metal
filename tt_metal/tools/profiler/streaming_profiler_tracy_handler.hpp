// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tracy sink: one context (device row) per worker core keyed by NOC0 coord, the 5 RISCs as lanes within it.
// The drain threads call HandleWorkerZone / HandleWorkerEvent directly.
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

    // Per-core anchor overriding the chip's: both counters zero at chip reset, but the Tensix domain is clocked
    // only while out of reset while the DRAM one keeps running, so one shared anchor shifts every DRAM-core row
    // right by a growing, board-dependent gap. Call before PreCreateContexts / the core's first zone: a context
    // bakes its anchor in at creation.
    void AddCore(
        uint32_t chip_id,
        uint32_t noc0_x,
        uint32_t noc0_y,
        int64_t host_start,
        double first_timestamp,
        double frequency);

    // Context creation is ~ms and belongs off the drain hot path.
    void PreCreateContexts(uint32_t chip_id, const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0);

    // Push one complete zone as a single queue item. Tracy requires per (context, thread) calls in
    // zone-completion order, which is the paired stream's per-lane arrival order.
    void HandleWorkerZone(const streaming_profiler::WorkerZonePacket& zone);

    void HandleWorkerEvent(const streaming_profiler::WorkerEventPacket& event);

private:
    static uint64_t ContextKey(uint32_t chip_id, uint32_t core_x, uint32_t core_y) {
        return (static_cast<uint64_t>(chip_id) << 40) | (static_cast<uint64_t>(core_x) << 20) |
               (static_cast<uint64_t>(core_y) & 0xFFFFF);
    }
    // Everything rides the serial queue so no zone or marker reaches the server before the GpuNewContext it
    // references (the server hard-asserts). The client drains the lock-free queues before the serial one, so
    // mixing them makes creation/use ordering racy.
    TracyTTCtx GetOrCreateContext(uint32_t chip_id, uint32_t core_x, uint32_t core_y, const std::string& name);

    struct ChipAnchor {
        int64_t host_start = 0;
        double first_timestamp = 0.0;
        double frequency = 0.0;
    };

    // Two flavours because GetOrCreateContext holds mutex_ across its body and mutex_ is not recursive.
    bool LookupAnchorLocked(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out);  // lock held
    bool LookupAnchor(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out);        // takes lock

    std::mutex mutex_;
    std::unordered_map<uint32_t, ChipAnchor> chip_anchors_;
    // Cores whose clock does not share the chip anchor's origin (DRAM tiles); lookups fall through to
    // chip_anchors_.
    std::unordered_map<uint64_t, ChipAnchor> core_anchors_;
    std::unordered_map<uint64_t, TracyTTCtx> tracy_contexts_;
    // Zone-id/colour -> immortal SourceLocationData: the server dereferences the pointer whenever it likes, so
    // the struct and its name string are interned once and never freed. Key id << 32 | color because the colour
    // lives inside the srcloc. void* keeps the header valid in non-TRACY builds.
    std::unordered_map<uint64_t, const void*> zone_srclocs_;
};

}  // namespace tt::tt_metal
