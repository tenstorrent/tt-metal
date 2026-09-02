// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_stall_csv.hpp"

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace tt::tt_metal::perf_debug {

void PerfDebugStallCsvConsumer::operator()(const PerfDebugRecordBatch& batch) {
    if (devctx_.size() < batch.context->devices.size()) {
        devctx_ = batch.context->devices;
    }
    dropped_ += batch.dropped_delta;
    names_.refresh();
    for (const PerfDebugRec& rec : batch.records) {
        if (rec.meta.type != PerfDebugRecType::Zone) {
            continue;
        }
        int32_t ni;
        if (auto it = keep_.find(rec.id); it != keep_.end()) {
            ni = it->second;
        } else {
            const std::string_view name = names_.lookup(rec.id);
            ni = -1;
            if (name == "PRODUCER-STALL" || name.substr(0, 6) == "DRISC-") {
                ni = static_cast<int32_t>(name_of_.size());
                name_of_.emplace_back(name);
            }
            keep_.emplace(rec.id, ni);
        }
        if (ni < 0) {
            continue;
        }
        rows_.push_back(
            {rec.data.zone.start,
             rec.data.zone.duration,
             static_cast<uint32_t>(ni),
             rec.meta.lane,
             rec.prog,
             static_cast<uint8_t>(rec.meta.dev)});
    }
}

void PerfDebugStallCsvConsumer::write_csv(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        return;
    }
    for (size_t d = 0; d < devctx_.size(); d++) {
        std::fprintf(
            f,
            "# dev %zu chip %u freq_ghz %.6f clock_synced %d dropped %llu\n",
            d,
            devctx_[d].chip_id,
            devctx_[d].frequency_ghz,
            devctx_[d].clock_synced ? 1 : 0,
            static_cast<unsigned long long>(dropped_));
    }
    std::fputs("chip,x,y,risc,role,name,start_cycles,duration_cycles,prog\n", f);
    for (const Row& r : rows_) {
        uint32_t chip = r.dev, x = 0, y = 0, risc = r.lane % 5, role = 0;
        if (r.dev < devctx_.size()) {
            const auto& dev = devctx_[r.dev];
            chip = dev.chip_id;
            if (r.lane < dev.lanes.size()) {
                const auto& l = dev.lanes[r.lane];
                x = l.noc0_x;
                y = l.noc0_y;
                risc = l.risc;
                role = static_cast<uint32_t>(l.role);
            }
        }
        std::fprintf(
            f,
            "%u,%u,%u,%u,%u,%s,%llu,%llu,%u\n",
            chip,
            x,
            y,
            risc,
            role,
            name_of_[r.name_idx].c_str(),
            static_cast<unsigned long long>(r.start),
            static_cast<unsigned long long>(r.duration),
            r.prog);
    }
    std::fclose(f);
}

namespace {

// TT_METAL_PERF_DEBUG_STALL_CSV=<path>: same lifecycle as the ops CSV -- register at load, write at exit,
// state leaked on purpose (an exit-time destructor would be ordered against other statics).
struct StallCsvState {
    std::string path;
    PerfDebugStallCsvConsumer consumer;
    PerfDebugConsumerHandle handle = 0;
};
StallCsvState* g_stall_csv = nullptr;

const bool g_stall_csv_registered = [] {
    const char* p = std::getenv("TT_METAL_PERF_DEBUG_STALL_CSV");
    if (p == nullptr || *p == '\0') {
        return false;
    }
    g_stall_csv = new StallCsvState{p, {}, 0};
    g_stall_csv->handle =
        register_consumer("stall-csv", [](const PerfDebugRecordBatch& b) { g_stall_csv->consumer(b); });
    std::atexit([] {
        unregister_consumer(g_stall_csv->handle);
        g_stall_csv->consumer.write_csv(g_stall_csv->path);
    });
    return true;
}();

}  // namespace

}  // namespace tt::tt_metal::perf_debug
