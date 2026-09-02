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

// Writes the classic per-zone device profiler CSV (profile_log_device.csv) from the streamed records, format
// unchanged: the DRAM device profiler stands down under the streaming profiler and every downstream tool
// reads exactly that file.
// Sync events land in two shapes. A blocking WAIT is a zone and is written as a ZONE_START/ZONE_END pair
// keyed by the `zone name` column with its structural id as timer_id (no legacy id exists for a wait). A
// SIGNAL, and the "-KEY" marker inside each wait zone, is a point marker whose `data` column needs a numeric
// timer_id, so its name maps to a legacy id via the table in the .cpp. The classic format can express no
// interval other than a zone and cannot attach a payload to one.
// Enable with TT_METAL_STREAMING_PROFILER_ZONE_CSV=<path>.
class StreamingProfilerZoneCsvConsumer {
public:
    void operator()(const StreamingProfilerRecordBatch& batch);
    void write_csv(const std::string& path) const;

private:
    // Buffered: the header needs the chip frequency, which arrives with the first batch's context.
    struct Row {
        uint32_t chip = 0;
        uint16_t core_x = 0, core_y = 0;
        uint8_t risc = 0;
        uint32_t timer_id = 0;
        uint64_t timestamp = 0;
        uint64_t data = 0;
        // StreamingProfilerRec::prog goes in `trace id`, not `run host ID`: that column means which host run produced
        // the row, and an op id there trips the reader's concatenated-capture warning.
        uint32_t prog = 0;
        std::string zone_name;
        const char* type = "";  // ZONE_START | ZONE_END | TS_DATA
    };

    // A sync point event arrives as Data, Ext, Cont records; lanes interleave, so the partial event is kept per
    // (device, lane).
    struct Pending {
        bool active = false;
        uint32_t legacy_id = 0;  // the classic numeric value the reader keys on; see kSyncNames
        uint64_t ts = 0;
        uint32_t words_expected = 0;
        std::vector<uint64_t> payload;
    };

    // Keyed by name: a wire id is per-TU, so the same event has a different id in every kernel.
    uint32_t sync_legacy_id(uint32_t wire_id);

    void flush_pending(uint32_t dev, uint32_t lane, const StreamingProfilerCaptureContext& ctx);

    std::vector<Row> rows_;
    ZoneNameMirror names_;
    // wire id -> legacy sync id (0 = not sync). Ids are 27-bit so a map, not an array.
    std::unordered_map<uint32_t, uint32_t> sync_id_cache_;
    // key: lane_key(dev, lane)
    std::unordered_map<uint32_t, Pending> pending_;
    double freq_mhz_ = 0.0;
    uint64_t dropped_ = 0;
    // Events whose payload never arrived (batch boundary at exit, or a dropped Cont).
    uint64_t incomplete_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
