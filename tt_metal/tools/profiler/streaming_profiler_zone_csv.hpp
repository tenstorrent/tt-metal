// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

// A streaming consumer that writes the classic per-zone device profiler CSV.
//
// Enabling the streaming profiler makes the standard DRAM device profiler stand down (its per-program
// control-buffer reset rewinds the ring tail and breaks the continuous drain), so
// `profile_log_device.csv` is never written -- and every downstream analysis tool reads exactly that
// file. This reproduces it from the streamed records, format unchanged.
//
// Sync events reach this consumer in two shapes, and they land in the CSV differently:
//
//   * A blocking WAIT is a zone (synchronization_event_profiler.hpp), so it arrives as a complete Zone
//     record and is written out as an ordinary ZONE_START/ZONE_END pair, keyed by the `zone name`
//     column with its structural id as timer_id. There is no legacy id for a wait.
//   * A SIGNAL, and the "<name>-KEY" marker carried inside each wait zone, is a point marker with a
//     payload, and the classic reader's `data` column needs a numeric timer_id, so the name is mapped
//     back to a legacy id by the table in the .cpp.
//
// So a reader of this file keys waits by name and signals by id. That split is forced: the classic
// format cannot express an interval other than a zone, nor attach a payload to one.
//
// Enable with: TT_METAL_STREAMING_PROFILER_ZONE_CSV=<path>
class StreamingProfilerZoneCsvConsumer {
public:
    void operator()(const StreamingProfilerRecordBatch& batch);
    void write_csv(const std::string& path) const;

private:
    // One emitted CSV row. Buffered rather than streamed to the file: the header needs the chip
    // frequency, which only arrives with the first batch's context.
    struct Row {
        uint32_t chip = 0;
        uint16_t core_x = 0, core_y = 0;
        uint8_t risc = 0;
        uint32_t timer_id = 0;
        uint64_t timestamp = 0;
        uint64_t data = 0;
        // The op this lane was executing (StreamingProfilerRec::prog, 0 = none yet). Goes in the CSV's
        // `trace id` column, not `run host ID`: that one means "which host run produced this row", so an
        // op id there makes one run look like several and trips the reader's concatenated-capture warning.
        uint32_t prog = 0;
        std::string zone_name;
        const char* type = "";  // ZONE_START | ZONE_END | TS_DATA
    };

    // A sync point event arrives split across records: Data (id + timestamp), then Ext (payload word
    // count), then Cont (the payload as one recombined uint64). They are adjacent within a lane but
    // lanes interleave, so the partial event is kept per (device, lane).
    struct Pending {
        bool active = false;
        uint32_t legacy_id = 0;  // the classic numeric value the reader keys on; see kSyncNames
        uint64_t ts = 0;
        uint32_t words_expected = 0;
        std::vector<uint64_t> payload;
    };

    // "SYNC-CB-PUSH" -> 1000 etc.; 0 = not a sync event. Keyed by name because a wire id is a per-TU
    // structural id: the same event has a different id in every kernel.
    uint32_t sync_legacy_id(uint32_t wire_id);

    void flush_pending(uint32_t dev, uint32_t lane, const StreamingProfilerCaptureContext& ctx);

    std::vector<Row> rows_;
    ZoneNameMirror names_;
    // wire id -> legacy sync id (0 = not sync). Ids are 27-bit so a map, not an array.
    std::unordered_map<uint32_t, uint32_t> sync_id_cache_;
    // key: (dev << 16) | lane
    std::unordered_map<uint32_t, Pending> pending_;
    double freq_mhz_ = 0.0;
    uint64_t dropped_ = 0;
    // Events whose payload never arrived (batch boundary at exit, or a dropped Cont).
    uint64_t incomplete_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
