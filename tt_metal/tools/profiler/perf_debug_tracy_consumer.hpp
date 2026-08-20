// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RAW-stream consumer that drives PerfDebugTracyHandler: resolves lane identity through
// the capture context's lane table, reassembles Data/Event payloads from Ext/Cont records,
// and pushes zone starts/ends in arrival order. It consumes the receiver's INTERNAL raw
// stream (add_raw_consumer), not the public Zone-record contract: Tracy's timeline encodes
// nesting through the interleaving of begin/end pushes, and the paired stream emits a zone
// only when it CLOSES -- child before parent -- which no push order over whole zones can
// render as a properly nested timeline.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tools/profiler/perf_debug_receiver.hpp"

namespace tt::tt_metal {

class PerfDebugTracyHandler;

namespace perf_debug {

class PerfDebugTracyConsumer {
public:
    explicit PerfDebugTracyConsumer(PerfDebugTracyHandler* handler);

    void operator()(const PerfDebugRawRecordBatch& batch);

private:
    static constexpr uint32_t kMaxEventValues = 64;  // ceil(127 payload words / 2)

    struct PendingEvent {
        bool active = false;
        uint32_t dev = 0;
        uint32_t lane = 0;
        uint64_t ts = 0;
        uint32_t id = 0;
        bool runtime_id = false;
        uint32_t prog = 0;
        uint32_t want = 0;
        uint32_t got = 0;
        uint64_t vals[kMaxEventValues] = {};
    };

    void flush_event(const PerfDebugCaptureContext& ctx);

    PerfDebugTracyHandler* handler_;
    PendingEvent pend_;
    std::vector<uint64_t> ts_base_;  // per device: rebase origin when the device clock is unsynced
    std::unordered_map<uint16_t, uint32_t> zone_colors_;
    std::unordered_map<uint16_t, uint32_t> zone_colors_mover_;
};

}  // namespace perf_debug
}  // namespace tt::tt_metal
