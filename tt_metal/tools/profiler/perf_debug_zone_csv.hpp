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
// profiler stand down (its per-program control-buffer reset rewinds the ring tail and
// breaks the continuous drain), so `profile_log_device.csv` is never written. Every
// downstream analysis tool -- the critical-path tool included -- reads exactly that
// file, and the one aggregate built-in sink (perf_debug_ops_csv) has no per-zone rows
// and no synchronization events at all.
//
// So this reproduces the classic file from the streamed records. The point is not to
// change the format -- it is to keep the format identical while gaining the property
// that motivated streaming, namely that the L1 marker ring is drained continuously
// instead of filling up and silently dropping the tail of a capture.
//
// Sync events reach this consumer in TWO shapes, and they land in the CSV differently:
//
//   * A blocking WAIT is a ZONE (see synchronization_event_profiler.hpp), so it arrives as a
//     complete Zone record and is written out as an ordinary ZONE_START/ZONE_END pair, keyed
//     by the `zone name` column like every other zone. Its timer_id is the structural id, not
//     a legacy value -- there is no legacy id for a wait, because the pre-port tool expressed
//     waits as two point markers (1001/1002, 1005/1006) and this one does not.
//   * A SIGNAL, and the "<name>-KEY" marker carried inside each wait zone, is a point marker
//     with a payload. Those still need a numeric timer_id for the classic reader's `data`
//     column, so the name is mapped back to a legacy id in the table in the .cpp.
//
// So a reader of this file keys waits BY NAME and signals BY ID. That split is not elegant,
// but it is forced: the classic format has no way to express an interval other than a zone,
// and no way to attach a payload to one.
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

    // A sync POINT event arrives split across records: Data (id + timestamp), then Ext
    // (payload word count), then Cont (the payload as one recombined uint64). They are
    // adjacent within a lane but lanes interleave, so the partial event is kept per
    // (device, lane) rather than in a single "last seen" slot.
    struct Pending {
        bool active = false;
        uint32_t legacy_id = 0;  // the classic numeric value the reader keys on; see kSyncNames
        uint64_t ts = 0;
        uint32_t words_expected = 0;
        std::vector<uint64_t> payload;
    };

    // "SYNC-CB-PUSH" -> 1000 etc.; 0 = not a sync event. By NAME because the wire ids
    // are per-TU structural ids now -- the same event has a different id in every
    // kernel, and only the name is stable.
    uint32_t sync_legacy_id(uint32_t wire_id);

    void flush_pending(uint32_t dev, uint32_t lane, const PerfDebugCaptureContext& ctx);

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

}  // namespace tt::tt_metal::perf_debug
