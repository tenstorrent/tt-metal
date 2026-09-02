// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stall-timeline CSV consumer, enabled by TT_METAL_PERF_DEBUG_STALL_CSV=<path>: registers itself through
// register_consumer() at load and writes at exit, exactly like the ops CSV. It keeps every PRODUCER-STALL
// zone (a producer RISC blocked on its full ring) and, when DRISC self-profiling is on, every DRISC-* zone
// (SWEEP / PACE / the per-batch phases) -- one row each with identity, start timestamp and duration in raw
// device cycles. That is the timeline that says WHEN producers stall relative to what the drainer was doing,
// which no end-of-run counter can.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tools/profiler/perf_debug_consumer.hpp"

namespace tt::tt_metal::perf_debug {

class PerfDebugStallCsvConsumer {
public:
    void operator()(const PerfDebugRecordBatch& batch);
    void write_csv(const std::string& path) const;

private:
    struct Row {
        uint64_t start;
        uint64_t duration;
        uint32_t name_idx;
        uint32_t lane;
        uint32_t prog;
        uint8_t dev;
    };
    ZoneNameMirror names_;
    std::unordered_map<uint32_t, int32_t> keep_;  // id -> index into name_of_, or -1 = not ours
    std::vector<std::string> name_of_;
    std::vector<Row> rows_;
    std::vector<PerfDebugCaptureContext::Device> devctx_;
    uint64_t dropped_ = 0;
};

}  // namespace tt::tt_metal::perf_debug
