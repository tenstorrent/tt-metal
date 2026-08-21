// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tracy sink on the PUBLIC paired-Zone contract -- an ordinary consumer callback, registered
// through add_consumer exactly like the ops CSV. Tracy's timeline encodes nesting through the
// ORDER of begin/end pushes per lane, but the paired stream emits a zone only when it CLOSES
// (child before parent, start not monotonic), so no push-in-arrival-order can render it. The
// consumer therefore buffers every zone and reconstructs the push order at teardown: per lane,
// sort to pre-order (start asc, end desc; ties broken toward the later-arrived = outer zone),
// then a single stack walk emits the begin/end bracket sequence. Point markers (Data/Event)
// carry no nesting and are forwarded as they arrive.
//
// The deferred flush is a real cost of zones-at-close, on purpose: nothing reaches Tracy until
// the capture ends, and the whole capture's zones are held in memory (24 B each). Lane identity
// is COPIED out of the capture context on first sight because the context dies with the
// receiver, before this consumer's destructor runs the flush.
#pragma once

#include <cstdint>
#include <string_view>
#include <unordered_map>
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

    // One completed zone, buffered until the teardown flush.
    struct BufZone {
        uint64_t start;
        uint64_t end;
        uint32_t id;
        uint32_t prog;
    };
    struct Lane {
        PerfDebugLaneInfo info;  // copied: the capture context is gone by flush time
        uint32_t dev = 0;
        std::vector<BufZone> zones;  // arrival order = per-lane END order
    };

    void flush_event(const PerfDebugCaptureContext& ctx);
    void note_ts(uint32_t dev, uint64_t ts);
    // Teardown: reconstruct each lane's begin/end push order and push everything to Tracy.
    void flush_zones();

    PerfDebugTracyHandler* handler_;
    PendingEvent pend_;
    // Per device: rebase origin when the device clock is unsynced. A running MIN over zone starts
    // and marker timestamps: with end-ordered zones the earliest start tends to arrive LAST (the
    // outermost zone closes last), so a first-record base would clamp it. Zones flush at teardown
    // and get the final min; a point marker forwarded live uses the min so far, so on an unsynced
    // device it can sit slightly right of zones -- accepted, the synced path never uses this.
    std::vector<uint64_t> ts_base_;
    std::vector<uint8_t> clock_synced_;
    std::unordered_map<uint32_t, Lane> lanes_;  // key: dev << 10 | lane
    uint64_t zones_buffered_ = 0;
    // id -> name, mirrored per-ELF from llrt::ZoneMetaRegistry. Member (not shared): this consumer runs
    // on its own delivery thread, so lookups take no lock.
    ZoneNameMirror names_;
    // Zone NAME -> explicit Tracy zone colour, for the drainer zones. KEYED BY NAME, never by id: a
    // structural zone id legitimately moves whenever a source line does, so an id-keyed table would
    // silently stop matching after any edit to the drain kernel. Filled in the constructor with
    // string literals, so string_view keys never dangle.
    std::unordered_map<std::string_view, uint32_t> zone_colors_;
    std::unordered_map<std::string_view, uint32_t> zone_colors_mover_;
};

}  // namespace perf_debug
}  // namespace tt::tt_metal
