// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tracy sink on the public paired-Zone contract: an ordinary consumer callback, registered through
// add_consumer exactly like the ops CSV. Tracy takes a device zone whole (both timestamps in one queue
// item) and the paired stream already delivers zones whole in per-lane completion order, which is the
// order the Tracy server rebuilds nesting from, so every record forwards as it arrives and nothing is
// buffered.
#pragma once

#include <cstdint>
#include <vector>

#include "tools/profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal {

class StreamingProfilerTracyHandler;

namespace streaming_profiler {

class StreamingProfilerTracyConsumer {
public:
    explicit StreamingProfilerTracyConsumer(StreamingProfilerTracyHandler* handler);
    ~StreamingProfilerTracyConsumer();

    void operator()(const StreamingProfilerRecordBatch& batch);

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

    void flush_event(const StreamingProfilerCaptureContext& ctx);
    void note_ts(uint32_t dev, uint64_t ts);

    StreamingProfilerTracyHandler* handler_;
    PendingEvent pend_;
    // Per device: rebase origin for an unsynced device clock, a running min over zone starts and
    // marker timestamps. Records forward live, so it is the min seen so far; the base only ever
    // decreases, which shifts later pushes right and so cannot break the per-lane end order Tracy
    // depends on. The synced path never uses this.
    std::vector<uint64_t> ts_base_;
    std::vector<uint8_t> clock_synced_;
    // id -> name, mirrored per-ELF from llrt::ZoneMetaRegistry.
    ZoneNameMirror names_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
