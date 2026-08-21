// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "perf_debug_consumer.hpp"

namespace tt::tt_metal::perf_debug {

// A streaming consumer that writes the CLASSIC per-zone device profiler CSV.
//
// Why this exists: enabling the streaming profiler makes the standard DRAM device
// profiler stand down (see external_ring_drainer_active() in impl/profiler/profiler.cpp
// -- its per-program control-buffer reset rewinds the ring tail and breaks the
// continuous drain), so `profile_log_device.csv` is never written. Every downstream
// analysis tool reads exactly that file, and the one built-in streaming sink
// (perf_debug_ops_csv) is aggregate: one row per op, no per-zone rows and no
// synchronization events at all.
//
// So this reproduces the classic file from the streamed records. The point is not to
// change the format -- it is to keep the format identical while gaining the property
// that motivated streaming, namely that the L1 marker ring is drained continuously
// instead of filling up and silently dropping the tail of a capture.
//
// Enable with: TT_METAL_PERF_DEBUG_ZONE_CSV=<path>
class PerfDebugZoneCsvConsumer {
public:
    void operator()(const PerfDebugRecordBatch& batch);
    void write_csv(const std::string& path) const;

private:
    // One emitted CSV row. Held rather than streamed to the file because the header
    // needs the chip frequency, which arrives with the first batch's context, and
    // because rows from different lanes interleave arbitrarily.
    struct Row {
        uint32_t chip = 0;
        uint16_t core_x = 0, core_y = 0;
        uint8_t risc = 0;
        uint32_t timer_id = 0;
        uint64_t timestamp = 0;
        uint64_t data = 0;
        // The op this lane was executing (PerfDebugRec::prog, 0 = none yet). Goes in
        // the CSV's `trace id` column, NOT `run host ID`: that one means "which host
        // run produced this row", and writing an op id there makes one run look like
        // several, which trips the reader's concatenated-capture warning.
        uint32_t prog = 0;
        std::string zone_name;
        const char* type = "";  // ZONE_START | ZONE_END | TS_DATA
    };

    // A TS_DATA event arrives split across records: Data (id + timestamp), then Ext
    // (payload word count), then Cont (the payload as one recombined uint64). They are
    // adjacent within a lane but lanes interleave, so the partial event is kept per
    // (device, lane) rather than in a single "last seen" slot.
    struct Pending {
        bool active = false;
        uint32_t id = 0;
        uint64_t ts = 0;
        uint32_t words_expected = 0;
        std::vector<uint64_t> payload;
    };

    void flush_pending(uint32_t dev, uint32_t lane, const PerfDebugCaptureContext& ctx);

    std::vector<Row> rows_;
    ZoneNameMirror names_;
    // key: (dev << 16) | lane
    std::unordered_map<uint32_t, Pending> pending_;
    double freq_mhz_ = 0.0;
    uint64_t dropped_ = 0;
    // Events whose payload never arrived (batch boundary at exit, or a dropped Cont).
    uint64_t incomplete_ = 0;
};

}  // namespace tt::tt_metal::perf_debug
