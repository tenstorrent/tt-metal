// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_ops_csv.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

OpsCsvConsumer::OpsCsvConsumer(const std::string& path) : f_(std::fopen(path.c_str(), "w")) {
    TT_FATAL(f_ != nullptr, "streaming profiler: cannot open {} for the ops CSV", path);
}

void OpsCsvConsumer::operator()(const Batch& batch) {
    for (const auto& z : batch.zones()) {
        if (z.runtime_id() == 0 || !z.site().name.ends_with("-KERNEL")) {
            continue;
        }
        const api::Core c = z.core();
        const uint32_t risc = static_cast<uint32_t>(c.risc);
        const uint32_t core_key = (static_cast<uint32_t>(c.logical.y) << 16) | static_cast<uint32_t>(c.logical.x);
        // The wrapper zone never self-nests, so the k-th one on a lane for a prog is execution k.
        uint32_t& completed = pair_count_[{c.chip_id, core_key, risc, z.runtime_id()}];
        OpAgg& op = ops_[{c.chip_id, z.runtime_id(), completed}];
        completed++;
        op.k_start = std::min(op.k_start, z.start_timestamp());
        op.k_end = std::max(op.k_end, z.end_timestamp());
        const auto [hs, he] = z.host_span<api::tsc_clock>();
        const int64_t start = hs.time_since_epoch().count(), end = he.time_since_epoch().count();
        if (start == 0 || end == 0) {
            continue;  // released before the sync covered it: no host span
        }
        auto& core = op.cores[core_key];
        op.h_start = std::min(op.h_start, start);
        op.h_start_last = std::max(op.h_start_last, start);
        if (risc <= 1) {
            op.h_dm_start = std::min(op.h_dm_start, start);
        }
        op.h_risc_start[risc] = std::min(op.h_risc_start[risc], start);
        core.first = core.first == 0 ? start : std::min(core.first, start);
        op.h_end = std::max(op.h_end, end);
        op.h_risc_end[risc] = std::max(op.h_risc_end[risc], end);
        core.second = std::max(core.second, end);
    }
}

void OpsCsvConsumer::write_csv() {
    FILE* const f = f_;
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
        const auto& [chip, prog, exec] = key;
        auto ns = [](int64_t start, int64_t end) {
            return end > start && start != INT64_MAX
                       ? static_cast<double>(api::tsc_clock::to_ns(api::tsc_clock::duration(end - start)).count())
                       : 0.0;
        };
        double core_min = 0.0, core_max = 0.0, core_sum = 0.0;
        uint32_t core_n = 0;
        for (const auto& [c, se] : op.cores) {
            if (se.first == 0 || se.second <= se.first) {
                continue;
            }
            const double d = ns(se.first, se.second);
            core_min = core_n == 0 ? d : std::min(core_min, d);
            core_max = std::max(core_max, d);
            core_sum += d;
            core_n++;
        }
        std::fprintf(
            f,
            "%u,%u,%u,%u,%llu,%llu,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
            chip,
            prog,
            exec,
            core_n,
            static_cast<unsigned long long>(op.k_start == UINT64_MAX ? 0 : op.k_start),
            static_cast<unsigned long long>(op.k_end),
            ns(op.h_start, op.h_end),
            ns(op.h_dm_start, op.h_end),
            core_min,
            core_max,
            core_n != 0 ? core_sum / core_n : 0.0,
            ns(op.h_start, op.h_start_last),
            ns(op.h_risc_start[0], op.h_risc_end[0]),
            ns(op.h_risc_start[1], op.h_risc_end[1]),
            ns(op.h_risc_start[2], op.h_risc_end[2]),
            ns(op.h_risc_start[3], op.h_risc_end[3]),
            ns(op.h_risc_start[4], op.h_risc_end[4]));
    }
    std::fclose(f);
}

}  // namespace tt::tt_metal::streaming_profiler
