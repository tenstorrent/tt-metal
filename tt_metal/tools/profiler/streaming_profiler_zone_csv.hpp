// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tools/profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

// Writes the classic per-zone device profiler CSV (profile_log_device.csv) from the subscribed records, format
// unchanged: the DRAM device profiler stands down under the streaming profiler and every downstream tool reads
// exactly that file.
//
// Sync events land in two shapes. A blocking WAIT is a zone and is written as a ZONE_START/ZONE_END pair, keyed
// by the `zone name` column with a name hash as timer_id. A SIGNAL, and the "-KEY" marker inside each wait zone,
// is a point marker whose `data` column needs a numeric timer_id, so its name maps to a legacy id via the table
// in the .cpp. Enable with TT_METAL_STREAMING_PROFILER_ZONE_CSV=<path>.
class ZoneCsvConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<
        experimental::streaming_profiler::Channel::Zones | experimental::streaming_profiler::Channel::TimestampedData>;
    void operator()(const Batch& batch);
    void write_csv(const std::string& path) const;

private:
    // Buffered: the header needs the chip frequency, which arrives with the first batch.
    struct Row {
        uint32_t chip = 0;
        uint16_t core_x = 0, core_y = 0;
        uint8_t risc = 0;
        uint32_t timer_id = 0;
        uint64_t timestamp = 0;
        uint64_t data = 0;
        // runtime_id goes in `trace id`, not `run host ID`: that column means which host run produced the row, and
        // an op id there trips the reader's concatenated-capture warning.
        uint32_t prog = 0;
        std::string zone_name;
        const char* type = "";  // ZONE_START | ZONE_END | TS_DATA
    };

    static uint32_t sync_legacy_id(std::string_view name);
    static uint32_t name_hash(std::string_view name);

    std::vector<Row> rows_;
    double freq_mhz_ = 0.0;
    uint64_t dropped_ = 0;
    uint64_t empty_payloads_ = 0;  // sync events with no payload: counted, not emitted as data 0
};

}  // namespace tt::tt_metal::streaming_profiler
