// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tracy sink on the PUBLIC paired-Zone contract -- an ordinary consumer callback, registered
// through add_consumer exactly like the ops CSV. Tracy takes device zones WHOLE now (QueueGpuZone:
// both timestamps in one lock-free item, TracyTTPushZone), and the paired stream already delivers
// zones whole in per-lane COMPLETION order -- exactly the order the Tracy server rebuilds nesting
// from. So every record forwards to Tracy as it arrives and NOTHING is buffered: no per-lane zone
// vectors, no teardown flush, no bracket reconstruction, no cross-lane timestamp merge. (All of
// those existed only because the old wire-level begin/end push encoded nesting in push order; see
// git history at 48b1e5c36f9 and before.) Point markers (Data/Event) forward as they arrive, as
// always.
#pragma once

#include <cstdint>
#include <vector>

#include "tools/profiler/perf_debug_consumer.hpp"

namespace tt::tt_metal {

class PerfDebugTracyHandler;

namespace perf_debug {

class PerfDebugTracyConsumer {
public:
    explicit PerfDebugTracyConsumer(PerfDebugTracyHandler* handler);
    ~PerfDebugTracyConsumer();

    void operator()(const PerfDebugRecordBatch& batch);

private:
    static constexpr uint32_t kMaxEventValues = 64;  // ceil(127 payload words / 2)

    struct PendingEvent {
        bool active = false;
        uint32_t dev = 0;
        uint32_t lane = 0;
        uint64_t ts = 0;
        uint32_t id = 0;
        uint32_t prog = 0;
        uint32_t want = 0;
        uint32_t got = 0;
        uint64_t vals[kMaxEventValues] = {};
    };

    void flush_event(const PerfDebugCaptureContext& ctx);
    void note_ts(uint32_t dev, uint64_t ts);

    PerfDebugTracyHandler* handler_;
    PendingEvent pend_;
    // Per device: rebase origin when the device clock is unsynced. A running MIN over zone starts
    // and marker timestamps. Everything forwards live now, so an unsynced device uses the min SO
    // FAR; the base only ever DECREASES, which shifts later pushes right and so cannot break the
    // per-lane end-order Tracy depends on. The synced path never uses this.
    std::vector<uint64_t> ts_base_;
    std::vector<uint8_t> clock_synced_;
    // id -> name, mirrored per-ELF from llrt::ZoneMetaRegistry. Member (not shared): this consumer runs
    // on its own delivery thread, so lookups take no lock.
    ZoneNameMirror names_;
};

}  // namespace perf_debug
}  // namespace tt::tt_metal
