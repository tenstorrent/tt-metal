// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/zone_csv.hpp"

#include <unistd.h>

#include <array>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

// The ids of the old wait halves stay reserved, so a reader still keyed on them can't pick up something else.
struct SyncName {
    const char* name;
    uint32_t legacy_id;
};
constexpr SyncName kSyncNames[] = {
    {"SYNC-CB-PUSH", 1000},
    // 1001, 1002 reserved: the SYNC-CB-WAIT halves are one zone
    {"SYNC-SEM-SET", 1003},
    {"SYNC-SEM-SET-REMOTE", 1004},
    // 1005, 1006 reserved: the SYNC-SEM-WAIT halves are one zone
    {"SYNC-SEM-WAIT-KEY", 1007},
    {"SYNC-CB-WAIT-KEY", 1008},
    {"SYNC-CB-RESERVE-KEY", 1009},
    {"SYNC-CB-POP", 1010},
};

uint32_t sync_legacy_id(std::string_view name) {
    for (const SyncName& s : kSyncNames) {
        if (name == s.name) {
            return s.legacy_id;
        }
    }
    return 0;
}

// The reader pairs START and END rows by order and type, so the id only has to be stable per name and never 0.
uint32_t name_hash(std::string_view name) {
    uint32_t h = 2166136261u;
    for (unsigned char ch : name) {
        h = (h ^ ch) * 16777619u;
    }
    return (h & 0x7FFFFFFu) | 0x8000000u;  // outside the legacy sync-id range
}

// Named the way the classic CSV prints tracy::RiscType, which has a single ERISC for every eth RISC.
constexpr std::array<const char*, kProcessorCount> kCsvRiscNames = {
    "BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC", "ERISC"};

}  // namespace

ZoneCsvConsumer::ZoneCsvConsumer(const std::string& path) :
    path_(path), f_(std::fopen(path.c_str(), "w"), &std::fclose) {
    TT_FATAL(f_ != nullptr, "streaming profiler: cannot open {} for the zone CSV", path);
}

ZoneCsvConsumer::Row ZoneCsvConsumer::row_for(const api::Core& core) {
    Row row;
    row.chip = core.chip_id;
    row.core_x = static_cast<uint16_t>(core.physical.x);
    row.core_y = static_cast<uint16_t>(core.physical.y);
    row.logical_x = static_cast<uint16_t>(core.logical.x);
    row.logical_y = static_cast<uint16_t>(core.logical.y);
    row.processor = static_cast<uint8_t>(core.processor);
    return row;
}

void ZoneCsvConsumer::operator()(const Batch& batch) {
    dropped_ += batch.dropped_bytes();
    // Only the header's clock frequency reads them.
    const bool header_pending = !header_written_;
    for (const api::Zone& zone : batch.zones()) {
        if (header_pending) {
            zone_cycles_ += zone.end_device_cycles() - zone.start_device_cycles();
            zone_tsc_ += zone.end_tsc() - zone.start_tsc();
        }
        const uint32_t id = name_hash(zone.site().name);
        for (int end = 0; end < 2; end++) {
            Row& row = rows_.emplace_back(row_for(zone.core()));
            row.timer_id = id;
            row.timestamp = end ? zone.end_device_cycles() : zone.start_device_cycles();
            row.runtime_id = zone.runtime_id();
            row.zone_name = zone.site().name;
            row.type = end ? "ZONE_END" : "ZONE_START";
        }
    }
    for (const api::TimestampedData& data : batch.timestamped_data()) {
        // Only for sync events, where the reader treats `data` as a CB id or semaphore address.
        const uint32_t legacy = sync_legacy_id(data.site().name);
        if (legacy == 0) {
            continue;
        }
        const std::span<const uint64_t> payload = data.payload();
        if (payload.empty()) {
            continue;  // a semaphore event at address 0 would invent a dependency
        }
        Row& row = rows_.emplace_back(row_for(data.core()));
        row.timer_id = legacy;
        row.timestamp = data.device_cycles();
        row.data = payload.front();
        row.type = "TS_DATA";
    }
}

void ZoneCsvConsumer::write_csv() {
    // The header's CHIP_FREQ needs zones; until a capture brings some, the rows wait for that capture's write.
    if (zone_tsc_ <= 0) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] zone CSV: no zones yet for the header's clock frequency; {} rows wait for a capture "
            "with zones",
            rows_.size());
        return;
    }
    FILE* const out = f_.get();
    if (!header_written_) {
        const double freq_mhz =
            1000.0 * static_cast<double>(zone_cycles_) / (static_cast<double>(zone_tsc_) * api::NsPerTscTick());
        std::fprintf(out, "ARCH: blackhole, CHIP_FREQ[MHz]: %.0f, Max Compute Cores: 0\n", freq_mhz);
        std::fprintf(
            out,
            "PCIe slot, core_x, core_y, RISC processor type, timer_id, "
            "time[cycles since reset], data, run host ID, trace id, trace id counter, "
            "zone name, type, source line, source file, meta data, logical_x, logical_y\n");
        header_written_ = true;
    }
    // Use the PID rather than a constant, so two hand-concatenated captures get different ids and the reader's
    // multi-run warning still fires.
    const uint32_t run_id = static_cast<uint32_t>(::getpid());
    for (const Row& row : rows_) {
        std::fprintf(
            out,
            "%u, %u, %u, %s, %u, %llu, %llu, %u, %u, 0, %.*s, %s, 0, streaming, , %u, %u\n",
            row.chip,
            row.core_x,
            row.core_y,
            kCsvRiscNames[row.processor],
            row.timer_id,
            static_cast<unsigned long long>(row.timestamp),
            static_cast<unsigned long long>(row.data),
            run_id,
            row.runtime_id,
            static_cast<int>(row.zone_name.size()),
            row.zone_name.data(),
            row.type,
            row.logical_x,
            row.logical_y);
    }
    std::fflush(out);
    log_info(
        tt::LogMetal,
        "[streaming profiler] zone CSV: wrote {} rows to {} ({} dropped bytes)",
        rows_.size(),
        path_,
        dropped_);
    rows_.clear();
    dropped_ = 0;
}

}  // namespace tt::tt_metal::streaming_profiler
