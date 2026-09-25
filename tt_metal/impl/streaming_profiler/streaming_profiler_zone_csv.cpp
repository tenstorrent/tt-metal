// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_zone_csv.hpp"

#include <unistd.h>

#include <cstdio>
#include <span>
#include <string>
#include <string_view>

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

}  // namespace

uint32_t ZoneCsvConsumer::sync_legacy_id(std::string_view name) {
    for (const SyncName& s : kSyncNames) {
        if (name == s.name) {
            return s.legacy_id;
        }
    }
    return 0;
}

// The reader pairs START/END rows by order and type; the id only has to be stable per name and never 0.
uint32_t ZoneCsvConsumer::name_hash(std::string_view name) {
    uint32_t h = 2166136261u;
    for (unsigned char ch : name) {
        h = (h ^ ch) * 16777619u;
    }
    return (h & 0x7FFFFFFu) | 0x8000000u;  // outside the legacy sync-id range
}

ZoneCsvConsumer::ZoneCsvConsumer(const std::string& path) : path_(path), f_(std::fopen(path.c_str(), "w")) {
    TT_FATAL(f_ != nullptr, "streaming profiler: cannot open {} for the zone CSV", path);
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
    if (freq_mhz_ == 0.0 && !batch.zones().empty()) {
        freq_mhz_ = batch.zones().front().frequency_ghz() * 1000.0;
    }
    if (temp_global_ == nullptr) {  // TEMP: every record on the host timeline, tenths of a ns
        temp_global_ = std::fopen((path_ + ".global.csv").c_str(), "w");
        std::fprintf(temp_global_, "kind,name,chip,px,py,lx,ly,risc,prog,t0,t1,payload,tick0,tick1\n");
    }
    for (const api::Zone& z : batch.zones()) {
        const api::Core c = z.core();
        std::fprintf(
            temp_global_, "Z,%.*s,%u,%u,%u,%u,%u,%u,%u,%lld,%lld,,%llu,%llu\n", static_cast<int>(z.site().name.size()),
            z.site().name.data(), static_cast<unsigned>(c.chip_id), static_cast<unsigned>(c.physical.x),
            static_cast<unsigned>(c.physical.y), static_cast<unsigned>(c.logical.x), static_cast<unsigned>(c.logical.y),
            static_cast<unsigned>(c.risc), static_cast<unsigned>(z.runtime_id()),
            static_cast<long long>(z.start_time().time_since_epoch().count()),
            static_cast<long long>(z.end_time().time_since_epoch().count()),
            static_cast<unsigned long long>(z.start_timestamp()), static_cast<unsigned long long>(z.end_timestamp()));
    }
    for (const api::TimestampedData& d : batch.timestamped_data()) {
        const api::Core c = d.core();
        const std::span<const uint64_t> p = d.payload();
        std::fprintf(
            temp_global_, "D,%.*s,%u,%u,%u,%u,%u,%u,%u,%lld,,", static_cast<int>(d.site().name.size()),
            d.site().name.data(), static_cast<unsigned>(c.chip_id), static_cast<unsigned>(c.physical.x),
            static_cast<unsigned>(c.physical.y), static_cast<unsigned>(c.logical.x), static_cast<unsigned>(c.logical.y),
            static_cast<unsigned>(c.risc), static_cast<unsigned>(d.runtime_id()),
            static_cast<long long>(d.time().time_since_epoch().count()));
        for (size_t i = 0; i < p.size(); i++) {
            std::fprintf(temp_global_, i == 0 ? "%llu" : ";%llu", static_cast<unsigned long long>(p[i]));
        }
        std::fprintf(temp_global_, ",%llu,\n", static_cast<unsigned long long>(d.timestamp()));
    }
    for (const api::Zone& z : batch.zones()) {
        // Both rows emitted: the classic reader pairs ZONE_START with ZONE_END itself.
        const uint32_t id = name_hash(z.site().name);
        for (int end = 0; end < 2; end++) {
            Row& r = rows_.emplace_back(row_for(z.core()));
            r.timer_id = id;
            r.timestamp = end ? z.end_timestamp() : z.start_timestamp();
            r.prog = z.runtime_id();
            r.zone_name = z.site().name;
            r.type = end ? "ZONE_END" : "ZONE_START";
        }
    }
    for (const api::TimestampedData& d : batch.timestamped_data()) {
        // Sync events only: the reader interprets `data` as a CB id or semaphore address.
        const uint32_t legacy = sync_legacy_id(d.site().name);
        if (legacy == 0) {
            continue;
        }
        const std::span<const uint64_t> payload = d.payload();
        if (payload.empty()) {
            empty_payloads_++;  // a semaphore event at address 0 would invent a dependency
            continue;
        }
        Row& r = rows_.emplace_back(row_for(d.core()));
        r.timer_id = legacy;
        r.timestamp = d.timestamp();
        r.data = payload.front();
        r.type = "TS_DATA";
    }
}

void ZoneCsvConsumer::write_csv() {
    FILE* const f = f_;
    if (!header_written_) {
        // core_x/core_y are the NoC 0 coordinate, as in the device profiler log this file mirrors, so a reader of
        // that log needs no special case; the logical coordinate rides in two trailing columns.
        std::fprintf(f, "ARCH: blackhole, CHIP_FREQ[MHz]: %.0f, Max Compute Cores: 0\n", freq_mhz_);
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
            r.processor < kProcessorNames.size() ? kProcessorNames[r.processor] : "UNKNOWN",
            r.timer_id,
            static_cast<unsigned long long>(r.timestamp),
            static_cast<unsigned long long>(r.data),
            run_id,
            r.prog,
            static_cast<int>(r.zone_name.size()),
            r.zone_name.empty() ? "" : r.zone_name.data(),
            r.type,
            r.logical_x,
            r.logical_y);
    }
    std::fflush(f);
    if (temp_global_ != nullptr) {  // TEMP
        std::fflush(temp_global_);
    }
    std::fprintf(
        stderr,
        "[streaming profiler zone-csv] wrote %zu row(s) to %s (dropped records: %llu, events with no payload: %llu)\n",
        rows_.size(),
        path_.c_str(),
        static_cast<unsigned long long>(dropped_),
        static_cast<unsigned long long>(empty_payloads_));
    rows_.clear();
    dropped_ = 0;
    empty_payloads_ = 0;
}

}  // namespace tt::tt_metal::streaming_profiler
