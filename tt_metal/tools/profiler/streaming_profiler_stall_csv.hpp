// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stall-timeline CSV consumer, enabled by TT_METAL_STREAMING_PROFILER_STALL_CSV=<path>: attaches when a
// capture starts and writes at exit, like the ops CSV. One row per PRODUCER-STALL zone (a producer RISC
// blocked on its full ring) and, when DRISC self-profiling is on, per DRISC-* zone: identity, start
// timestamp and duration in raw device cycles. That timeline says when producers stall relative to what
// the relay was doing, which no end-of-run counter can.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tools/profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

class StreamingProfilerStallCsvConsumer {
public:
    void operator()(const StreamingProfilerRecordBatch& batch);
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
    std::vector<StreamingProfilerCaptureContext::Device> devctx_;
    uint64_t dropped_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
