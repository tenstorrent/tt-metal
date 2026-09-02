// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "streaming_profiler_zone_csv.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace tt::tt_metal::streaming_profiler {

namespace {

// risc index -> the name the classic CSV uses; the index order is tracy::RiscType
// (TracyTTDeviceData.hpp).
const char* risc_name(uint8_t risc) {
    switch (risc) {
        case 0: return "BRISC";
        case 1: return "NCRISC";
        case 2: return "TRISC_0";
        case 3: return "TRISC_1";
        case 4: return "TRISC_2";
        case 5: return "ERISC";
        default: return "UNKNOWN";
    }
}

// The synchronization point events by name (synchronization_event_profiler.hpp), each with the legacy
// numeric id the classic reader keys on: the streaming wire carries per-TU structural ids instead, so
// the mapping lives here.
//
// Only events that carry a payload are listed, because this table exists to fill the CSV's `data`
// column. A wait is a zone plus one "<name>-KEY" marker holding the join key, so the four wait-half ids
// below are never reused -- a stale reader keying on 1001 would otherwise silently pick up something
// else.
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

uint32_t StreamingProfilerZoneCsvConsumer::sync_legacy_id(uint32_t wire_id) {
    if (auto it = sync_id_cache_.find(wire_id); it != sync_id_cache_.end()) {
        return it->second;
    }
    uint32_t legacy = 0;
    const std::string_view name = names_.lookup(wire_id);
    for (const SyncName& s : kSyncNames) {
        if (name == s.name) {
            legacy = s.legacy_id;
            break;
        }
    }
    sync_id_cache_.emplace(wire_id, legacy);
    return legacy;
}

void StreamingProfilerZoneCsvConsumer::flush_pending(
    uint32_t dev, uint32_t lane, const StreamingProfilerCaptureContext& ctx) {
    const uint32_t key = (dev << 16) | lane;
    auto it = pending_.find(key);
    if (it == pending_.end() || !it->second.active) {
        return;
    }
    Pending& p = it->second;
    if (p.payload.empty()) {
        // A Data with no payload is counted, not emitted as zero: a semaphore event at address 0
        // would pair against a real waiter and invent a dependency.
        incomplete_++;
        p = Pending{};
        return;
    }
    const auto& li = ctx.devices[dev].lanes[lane];
    Row& r = rows_.emplace_back();
    r.chip = li.chip_id;
    r.core_x = li.noc0_x;
    r.core_y = li.noc0_y;
    r.risc = li.risc;
    r.timer_id = p.legacy_id;
    r.timestamp = p.ts;
    r.data = p.payload.front();
    r.type = "TS_DATA";
    p = Pending{};
}

void StreamingProfilerZoneCsvConsumer::operator()(const StreamingProfilerRecordBatch& batch) {
    names_.refresh();
    dropped_ += batch.dropped_delta;

    const StreamingProfilerCaptureContext& ctx = *batch.context;

    for (const auto& rec : batch.records) {
        const uint32_t dev = rec.meta.dev;
        const uint32_t lane = rec.meta.lane;
        if (dev >= ctx.devices.size() || lane >= ctx.devices[dev].lanes.size()) {
            continue;
        }
        const auto& li = ctx.devices[dev].lanes[lane];
        if (freq_mhz_ == 0.0 && ctx.devices[dev].frequency_ghz > 0.0) {
            freq_mhz_ = ctx.devices[dev].frequency_ghz * 1000.0;
        }

        switch (rec.meta.type) {
            case StreamingProfilerRecType::Zone: {
                // The record is a complete zone; both rows are emitted so the classic reader,
                // which pairs ZONE_START with ZONE_END itself, needs no change.
                const std::string name(names_.lookup(rec.id));
                for (int end = 0; end < 2; end++) {
                    Row& r = rows_.emplace_back();
                    r.chip = li.chip_id;
                    r.core_x = li.noc0_x;
                    r.core_y = li.noc0_y;
                    r.risc = li.risc;
                    r.timer_id = rec.id;
                    r.timestamp = end ? rec.data.zone.start + rec.data.zone.duration : rec.data.zone.start;
                    r.prog = rec.prog;
                    r.zone_name = name;
                    r.type = end ? "ZONE_END" : "ZONE_START";
                }
                break;
            }
            case StreamingProfilerRecType::Data: {
                // Only sync events belong here: the classic reader interprets the `data` column as
                // a CB id or semaphore address.
                const uint32_t legacy = sync_legacy_id(rec.id);
                if (legacy == 0) {
                    break;
                }
                flush_pending(dev, lane, ctx);  // a new Data ends any unfinished one
                Pending& p = pending_[(dev << 16) | lane];
                p = Pending{};
                p.active = true;
                p.legacy_id = legacy;
                p.ts = rec.data.ts;
                break;
            }
            case StreamingProfilerRecType::Ext: {
                Pending& p = pending_[(dev << 16) | lane];
                if (!p.active) {
                    break;
                }
                // Ext carries the payload word count in id and words 1-2 packed ((hi << 32) | lo) in
                // data.ext, and every sync macro passes exactly one datum, so the event completes
                // here; the Cont arm below only covers a >2-word payload.
                p.words_expected = rec.id;
                p.payload.push_back(rec.data.ext);
                if (p.words_expected <= 2) {
                    flush_pending(dev, lane, ctx);
                }
                break;
            }
            case StreamingProfilerRecType::Cont: {
                Pending& p = pending_[(dev << 16) | lane];
                if (!p.active) {
                    break;  // continuation of a marker we are not collecting
                }
                p.payload.push_back(rec.data.payload);
                if (p.payload.size() * 2 >= p.words_expected) {
                    flush_pending(dev, lane, ctx);
                }
                break;
            }
            default: break;  // Event carries nothing the classic reader consumes
        }
    }
}

void StreamingProfilerZoneCsvConsumer::write_csv(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "[streaming profiler zone-csv] cannot open %s\n", path.c_str());
        return;
    }
    // The classic preamble: only CHIP_FREQ is parsed downstream, but the shape is kept so an existing
    // reader does not have to special-case this file.
    std::fprintf(
        f, "ARCH: blackhole, CHIP_FREQ[MHz]: %.0f, Max Compute Cores: 0\n", freq_mhz_ > 0.0 ? freq_mhz_ : 1000.0);
    std::fprintf(
        f,
        "PCIe slot, core_x, core_y, RISC processor type, timer_id, "
        "time[cycles since reset], data, run host ID, trace id, trace id counter, "
        "zone name, type, source line, source file, meta data\n");

    // The PID rather than a constant: two captures concatenated by hand then carry different ids, so
    // the reader's multi-run warning still fires instead of going quiet.
    const uint32_t run_id = static_cast<uint32_t>(::getpid());

    for (const Row& r : rows_) {
        std::fprintf(
            f,
            "%u, %u, %u, %s, %u, %llu, %llu, %u, %u, 0, %s, %s, 0, streaming, \n",
            r.chip,
            r.core_x,
            r.core_y,
            risc_name(r.risc),
            r.timer_id,
            static_cast<unsigned long long>(r.timestamp),
            static_cast<unsigned long long>(r.data),
            run_id,
            r.prog,
            r.zone_name.c_str(),
            r.type);
    }
    std::fclose(f);

    std::fprintf(
        stderr,
        "[streaming profiler zone-csv] wrote %zu row(s) to %s (dropped batches: %llu, "
        "events with no payload: %llu)\n",
        rows_.size(),
        path.c_str(),
        static_cast<unsigned long long>(dropped_),
        static_cast<unsigned long long>(incomplete_));
}

namespace {

// TT_METAL_STREAMING_PROFILER_ZONE_CSV=<path>: register at load, write at exit. The atexit handler runs
// after receiver shutdown has delivered every buffered batch, and unregisters first so no batch races
// the write. State is leaked deliberately: an exit-time destructor would be ordered against other
// statics.
struct ZoneCsvState {
    std::string path;
    StreamingProfilerZoneCsvConsumer consumer;
    StreamingProfilerConsumerHandle handle = 0;
};
ZoneCsvState* g_zone_csv = nullptr;

const bool g_zone_csv_registered = [] {
    const char* p = std::getenv("TT_METAL_STREAMING_PROFILER_ZONE_CSV");
    if (p == nullptr || *p == '\0') {
        return false;
    }
    g_zone_csv = new ZoneCsvState{p, {}, 0};
    g_zone_csv->handle = register_consumer(
        "zone-csv", [](const StreamingProfilerRecordBatch& b) { g_zone_csv->consumer(b); });
    std::atexit([] {
        unregister_consumer(g_zone_csv->handle);
        g_zone_csv->consumer.write_csv(g_zone_csv->path);
    });
    return true;
}();

}  // namespace

}  // namespace tt::tt_metal::streaming_profiler
