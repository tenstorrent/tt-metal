// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/streaming_profiler_ops_csv.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <string>
#include <string_view>

#include "context/metal_context.hpp"

namespace tt::tt_metal::streaming_profiler {

void StreamingProfilerOpsCsvConsumer::operator()(const StreamingProfilerRecordBatch& batch) {
    const StreamingProfilerCaptureContext* ctx = batch.context;
    if (devices_.size() < ctx->devices.size()) {
        devices_.clear();
        for (const auto& d : ctx->devices) {
            devices_.push_back({d.chip_id, d.frequency_ghz});
        }
    }
    names_.refresh();
    for (const StreamingProfilerRec& rec : batch.records) {
        if (rec.meta.type != StreamingProfilerRecType::Zone || rec.prog == 0) {
            continue;
        }
        ZoneClass cls = ZoneClass::Unseen;
        if (auto it = class_of_id_.find(rec.id); it != class_of_id_.end()) {
            cls = it->second;
        } else if (const std::string_view name = names_.lookup(rec.id); !name.empty()) {
            cls = name.ends_with("-KERNEL") ? ZoneClass::Kernel : ZoneClass::Other;
            class_of_id_.emplace(rec.id, cls);
        }
        if (cls != ZoneClass::Kernel) {
            continue;
        }
        const auto& lanes = ctx->devices[rec.meta.dev].lanes;
        if (rec.meta.lane >= lanes.size() || lanes[rec.meta.lane].role != StreamingProfilerLaneRole::Worker) {
            continue;
        }
        constexpr uint32_t kLaneShift = 32;
        constexpr uint32_t kDevShift = kLaneShift + std::bit_width(kStreamingProfilerMaxLanes - 1u);
        // The wrapper zone never self-nests, so the k-th Zone on a lane for a prog is execution k.
        uint32_t& completed = pair_count_
            [(static_cast<uint64_t>(rec.meta.dev) << kDevShift) | (static_cast<uint64_t>(rec.meta.lane) << kLaneShift) |
             rec.prog];
        OpAgg& op = ops_[{rec.meta.dev, rec.prog, completed}];
        completed++;
        const uint64_t start = rec.data.zone.start;
        const uint64_t end = start + rec.data.zone.duration;
        const uint32_t risc = rec.meta.lane % kNumRisc;
        auto& core = op.cores[rec.meta.lane / kNumRisc];
        op.k_start = std::min(op.k_start, start);
        op.k_start_last = std::max(op.k_start_last, start);
        if (risc <= 1) {
            op.dm_start = std::min(op.dm_start, start);
        }
        op.risc_start[risc] = std::min(op.risc_start[risc], start);
        core.first = core.first == 0 ? start : std::min(core.first, start);
        op.k_end = std::max(op.k_end, end);
        op.risc_end[risc] = std::max(op.risc_end[risc], end);
        core.second = std::max(core.second, end);
    }
}

void StreamingProfilerOpsCsvConsumer::write_csv(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        return;
    }
    std::fputs(
        "DEVICE ID,GLOBAL CALL COUNT,EXECUTION,CORE COUNT,DEVICE KERNEL START CYCLE,DEVICE KERNEL END CYCLE,"
        "DEVICE KERNEL DURATION [ns],DEVICE KERNEL DURATION DM START [ns],"
        "DEVICE KERNEL DURATION PER CORE MIN [ns],DEVICE KERNEL DURATION PER CORE MAX [ns],"
        "DEVICE KERNEL DURATION PER CORE AVG [ns],DEVICE KERNEL FIRST TO LAST START [ns],"
        "DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],"
        "DEVICE TRISC0 KERNEL DURATION [ns],DEVICE TRISC1 KERNEL DURATION [ns],"
        "DEVICE TRISC2 KERNEL DURATION [ns]\n",
        f);
    for (const auto& [key, op] : ops_) {
        const auto& [dev, prog, exec] = key;
        const DeviceMeta meta = dev < devices_.size() ? devices_[dev] : DeviceMeta{};
        const double freq = meta.frequency_ghz;
        auto ns = [&](uint64_t start, uint64_t end) {
            return (freq > 0.0 && end > start && start != UINT64_MAX) ? (end - start) / freq : 0.0;
        };
        uint64_t core_min = UINT64_MAX, core_max = 0, core_sum = 0;
        uint32_t core_n = 0;
        for (const auto& [c, se] : op.cores) {
            if (se.first == 0 || se.second <= se.first) {
                continue;
            }
            const uint64_t d = se.second - se.first;
            core_min = std::min(core_min, d);
            core_max = std::max(core_max, d);
            core_sum += d;
            core_n++;
        }
        auto cyc_ns = [&](uint64_t cyc) { return freq > 0.0 ? cyc / freq : 0.0; };
        std::fprintf(
            f,
            "%u,%u,%u,%u,%llu,%llu,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
            meta.chip_id,
            prog,
            exec,
            core_n,
            static_cast<unsigned long long>(op.k_start == UINT64_MAX ? 0 : op.k_start),
            static_cast<unsigned long long>(op.k_end),
            ns(op.k_start, op.k_end),
            ns(op.dm_start, op.k_end),
            core_n != 0 ? cyc_ns(core_min) : 0.0,
            core_n != 0 ? cyc_ns(core_max) : 0.0,
            core_n != 0 ? cyc_ns(core_sum) / core_n : 0.0,
            ns(op.k_start, op.k_start_last),
            ns(op.risc_start[0], op.risc_end[0]),
            ns(op.risc_start[1], op.risc_end[1]),
            ns(op.risc_start[2], op.risc_end[2]),
            ns(op.risc_start[3], op.risc_end[3]),
            ns(op.risc_start[4], op.risc_end[4]));
    }
    std::fclose(f);
}

namespace {

const bool g_ops_csv_declared = [] {
    register_file_consumer<StreamingProfilerOpsCsvConsumer>("ops-csv", []() -> std::string {
        return MetalContext::instance().rtoptions().get_streaming_profiler_ops_csv_path();
    });
    return true;
}();

}  // namespace

}  // namespace tt::tt_metal::streaming_profiler
