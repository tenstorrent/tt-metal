// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_ops_csv.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>

namespace tt::tt_metal::streaming_profiler {

void OpsCsvConsumer::operator()(const Batch& batch) {
    for (const auto& clk : batch.clocks) {
        devices_[clk.chip_id] = DeviceMeta{clk.chip_id, clk.frequency_ghz};
    }
    for (const auto& z : batch.zones) {
        if (z.runtime_id == 0 || !z.site.name.ends_with("-KERNEL")) {
            continue;
        }
        const uint32_t risc = static_cast<uint32_t>(z.core.risc);
        const uint32_t core_key = (static_cast<uint32_t>(z.core.logical.y) << 16) | static_cast<uint32_t>(z.core.logical.x);
        // The wrapper zone never self-nests, so the k-th one on a lane for a prog is execution k.
        uint32_t& completed = pair_count_
            [(static_cast<uint64_t>(z.core.chip_id) << 56) | (static_cast<uint64_t>(core_key) << 24) |
             (static_cast<uint64_t>(risc) << 20) | (z.runtime_id & 0xFFFFFu)];
        OpAgg& op = ops_[{z.core.chip_id, z.runtime_id, completed}];
        completed++;
        const uint64_t start = z.start_timestamp;
        const uint64_t end = z.end_timestamp;
        auto& core = op.cores[core_key];
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

void OpsCsvConsumer::write_csv(const std::string& path) const {
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
        const auto& [chip, prog, exec] = key;
        const auto mit = devices_.find(chip);
        const DeviceMeta meta = mit != devices_.end() ? mit->second : DeviceMeta{chip, 0.0};
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

}  // namespace tt::tt_metal::streaming_profiler
