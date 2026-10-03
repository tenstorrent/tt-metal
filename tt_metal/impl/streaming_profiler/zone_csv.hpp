// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

// Writes the classic per-zone device profiler CSV (profile_log_device.csv) in that file's exact format: the DRAM
// device profiler stands down under the streaming profiler and every downstream tool reads that file.
class ZoneCsvConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<
        experimental::streaming_profiler::RecordType::Zones |
        experimental::streaming_profiler::RecordType::TimestampedData>;
    explicit ZoneCsvConsumer(const std::string& path);
    void operator()(const Batch& batch);
    // Call only between captures.
    void write_csv();

private:
    struct Row {
        uint32_t chip = 0;
        uint16_t core_x = 0, core_y = 0;
        uint16_t logical_x = 0, logical_y = 0;
        uint8_t processor = 0;
        uint32_t timer_id = 0;
        uint64_t timestamp = 0;
        uint64_t data = 0;
        // runtime_id goes in `trace id`, not `run host ID`: that column means which host run produced the row, and
        // an op id there trips the reader's concatenated-capture warning.
        uint32_t runtime_id = 0;
        std::string_view zone_name = "";  // in the zone-name registry, alive for the process
        const char* type = "";
    };

    static Row row_for(const experimental::streaming_profiler::Core& core);

    std::string path_;
    std::unique_ptr<FILE, int (*)(FILE*)> f_;
    bool header_written_ = false;
    std::vector<Row> rows_;
    uint64_t zone_cycles_ = 0;
    int64_t zone_tsc_ = 0;
    uint64_t dropped_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
