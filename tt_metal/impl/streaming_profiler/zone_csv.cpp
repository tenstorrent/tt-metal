// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/zone_csv.hpp"

#include <unistd.h>

#include <cstdio>
#include <string>
#include <string_view>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

// Sync events by name with the legacy numeric id the classic reader keys on. Only payload-carrying events are
// listed; the wait-half ids stay reserved so a stale reader keying on them cannot pick up something else.
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

// Returns the name's stable CSV id, which is never 0 and is above every kSyncNames id.
uint32_t name_hash(std::string_view name) {
    uint32_t h = 2166136261u;
    for (unsigned char ch : name) {
        h = (h ^ ch) * 16777619u;
    }
    return (h & 0x7FFFFFFu) | 0x8000000u;
}

}  // namespace

ZoneCsvConsumer::ZoneCsvConsumer(const std::string& path) : path_(path), file_(std::fopen(path.c_str(), "w")) {
    TT_FATAL(file_ != nullptr, "streaming profiler: cannot open {} for the zone CSV", path);
}

ZoneCsvConsumer::Row ZoneCsvConsumer::row_for(const api::Core& core) {
    Row r;
    r.chip = core.chip_id;
    r.core_x = static_cast<uint16_t>(core.physical.x);
    r.core_y = static_cast<uint16_t>(core.physical.y);
    r.logical_x = static_cast<uint16_t>(core.logical.x);
    r.logical_y = static_cast<uint16_t>(core.logical.y);
    r.processor = static_cast<uint8_t>(core.processor);
    return r;
}

void ZoneCsvConsumer::operator()(const Batch& batch) {
    dropped_ += batch.dropped_bytes();
    for (const api::Zone& z : batch.records<api::Zone>()) {
        if (!header_written_) {
            zone_cycles_ += z.end_device_cycles() - z.start_device_cycles();
            zone_ns_ += z.duration().count();
        }
        const uint32_t id = name_hash(z.site().name);
        for (int end = 0; end < 2; end++) {
            Row& r = rows_.emplace_back(row_for(z.core()));
            r.timer_id = id;
            r.timestamp = end ? z.end_device_cycles() : z.start_device_cycles();
            r.runtime_id = z.runtime_id();
            r.zone_name = z.site().name;
            r.type = end ? "ZONE_END" : "ZONE_START";
        }
    }
    for (const api::TimestampedData& d : batch.records<api::TimestampedData>()) {
        const uint32_t legacy = sync_legacy_id(d.site().name);
        Row& r = rows_.emplace_back(row_for(d.core()));
        r.timer_id = legacy != 0 ? legacy : name_hash(d.site().name);
        r.timestamp = d.device_cycles();
        // The CSV has one data column, so a record with several values writes its first.
        r.data = d.payload().front();
        if (legacy == 0) {
            r.runtime_id = d.runtime_id();
            r.zone_name = d.site().name;
        }
        r.type = "TS_DATA";
    }
    for (const api::Event& e : batch.records<api::Event>()) {
        Row& r = rows_.emplace_back(row_for(e.core()));
        r.timer_id = name_hash(e.site().name);
        r.timestamp = e.device_cycles();
        r.runtime_id = e.runtime_id();
        r.zone_name = e.site().name;
        r.type = "TS_EVENT";
    }
}

void ZoneCsvConsumer::write_csv() {
    if (rows_.empty()) {
        return;
    }
    FILE* const f = file_.get();
    if (!header_written_) {
        const double freq_mhz =
            zone_ns_ != 0 ? 1000.0 * static_cast<double>(zone_cycles_) / static_cast<double>(zone_ns_) : 0.0;
        std::fprintf(f, "ARCH: blackhole, CHIP_FREQ[MHz]: %.0f, Max Compute Cores: 0\n", freq_mhz);
        std::fprintf(
            f,
            "PCIe slot, core_x, core_y, RISC processor type, timer_id, "
            "time[cycles since reset], data, run host ID, trace id, trace id counter, "
            "zone name, type, source line, source file, meta data, logical_x, logical_y\n");
        header_written_ = true;
    }
    // The PID, not a constant: two hand-concatenated captures then carry different ids and the reader's
    // multi-run warning still fires.
    const uint32_t run_id = static_cast<uint32_t>(::getpid());
    for (const Row& r : rows_) {
        std::fprintf(
            f,
            "%u, %u, %u, %s, %u, %llu, %llu, %u, %u, 0, %.*s, %s, 0, streaming, , %u, %u\n",
            r.chip,
            r.core_x,
            r.core_y,
            kProcessorNames[r.processor],
            r.timer_id,
            static_cast<unsigned long long>(r.timestamp),
            static_cast<unsigned long long>(r.data),
            run_id,
            r.runtime_id,
            static_cast<int>(r.zone_name.size()),
            r.zone_name.empty() ? "" : r.zone_name.data(),
            r.type,
            r.logical_x,
            r.logical_y);
    }
    std::fflush(f);
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
