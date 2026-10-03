// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/ops_csv.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

OpsCsvConsumer::OpsCsvConsumer(const std::string& path) : f_(std::fopen(path.c_str(), "w"), &std::fclose) {
    TT_FATAL(f_ != nullptr, "streaming profiler: cannot open {} for the ops CSV", path);
}

void OpsCsvConsumer::operator()(const Batch& batch) {
    for (const auto& zone : batch.zones()) {
        if (zone.runtime_id() == 0 || !zone.site().name.ends_with("-KERNEL")) {
            continue;
        }
        const api::Core core = zone.core();
        const bool eth = core.processor >= api::Processor::ERISC0;
        const uint32_t column = static_cast<uint32_t>(eth ? api::Processor::ERISC0 : core.processor);
        const CoreKey core_key{
            .eth = eth, .y = static_cast<uint32_t>(core.logical.y), .x = static_cast<uint32_t>(core.logical.x)};
        // The wrapper zone never nests, so the k-th one on a lane for a program is its k-th execution.
        uint32_t& completed = executions_seen_[{
            .chip = static_cast<uint32_t>(core.chip_id),
            .core = core_key,
            .processor = static_cast<uint32_t>(core.processor),
            .program = zone.runtime_id()}];
        OpAgg& op =
            ops_[{.chip = static_cast<uint32_t>(core.chip_id), .program = zone.runtime_id(), .execution = completed}];
        completed++;
        if (!eth) {
            op.start_cycles = std::min(op.start_cycles, zone.start_device_cycles());
            op.end_cycles = std::max(op.end_cycles, zone.end_device_cycles());
        }
        const int64_t start = zone.start_tsc();
        const int64_t end = zone.end_tsc();
        Span& core_span = op.cores[core_key];
        op.last_kernel_start_tsc = std::max(op.last_kernel_start_tsc, start);
        op.risc_start_tsc[column] = std::min(op.risc_start_tsc[column], start);
        core_span.start = std::min(core_span.start, start);
        op.risc_end_tsc[column] = std::max(op.risc_end_tsc[column], end);
        core_span.end = std::max(core_span.end, end);
    }
}

void OpsCsvConsumer::write_csv() {
    FILE* const out = f_.get();
    if (!header_written_) {
        header_written_ = true;
        std::fputs(
            "DEVICE ID,GLOBAL CALL COUNT,EXECUTION,CORE COUNT,DEVICE KERNEL START CYCLE,DEVICE KERNEL END CYCLE,"
            "DEVICE KERNEL DURATION [ns],DEVICE KERNEL DURATION DM START [ns],"
            "DEVICE KERNEL DURATION PER CORE MIN [ns],DEVICE KERNEL DURATION PER CORE MAX [ns],"
            "DEVICE KERNEL DURATION PER CORE AVG [ns],DEVICE KERNEL FIRST TO LAST START [ns],"
            "DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],"
            "DEVICE TRISC0 KERNEL DURATION [ns],DEVICE TRISC1 KERNEL DURATION [ns],"
            "DEVICE TRISC2 KERNEL DURATION [ns],DEVICE ERISC KERNEL DURATION [ns]\n",
            out);
    }
    const auto elapsed_ns = [](int64_t start, int64_t end) {
        return end > start ? static_cast<double>(end - start) * api::NsPerTscTick() : 0.0;
    };
    for (const auto& [key, op] : ops_) {
        const auto risc_start = [&](api::Processor processor) {
            return op.risc_start_tsc[static_cast<uint32_t>(processor)];
        };
        const auto risc_ns = [&](api::Processor processor) {
            return elapsed_ns(risc_start(processor), op.risc_end_tsc[static_cast<uint32_t>(processor)]);
        };
        const int64_t start_tsc = std::ranges::min(op.risc_start_tsc);
        const int64_t end_tsc = std::ranges::max(op.risc_end_tsc);
        const int64_t dm_start_tsc = std::min(
            {risc_start(api::Processor::BRISC),
             risc_start(api::Processor::NCRISC),
             risc_start(api::Processor::ERISC0)});
        double core_min = 0.0, core_max = 0.0, core_sum = 0.0;
        uint32_t core_n = 0;
        for (const auto& [core, span] : op.cores) {
            const double core_ns = elapsed_ns(span.start, span.end);
            core_min = core_n == 0 ? core_ns : std::min(core_min, core_ns);
            core_max = std::max(core_max, core_ns);
            core_sum += core_ns;
            core_n++;
        }
        std::fprintf(out, "%u,%u,%u,%u,", key.chip, key.program, key.execution, core_n);
        // An op with no Tensix kernel zone leaves the CYCLE columns empty.
        if (op.start_cycles != UINT64_MAX) {
            std::fprintf(
                out,
                "%llu,%llu,",
                static_cast<unsigned long long>(op.start_cycles),
                static_cast<unsigned long long>(op.end_cycles));
        } else {
            std::fputs(",,", out);
        }
        std::fprintf(
            out,
            "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,",
            elapsed_ns(start_tsc, end_tsc),
            elapsed_ns(dm_start_tsc, end_tsc),
            core_min,
            core_max,
            core_n != 0 ? core_sum / core_n : 0.0,
            elapsed_ns(start_tsc, op.last_kernel_start_tsc),
            risc_ns(api::Processor::BRISC),
            risc_ns(api::Processor::NCRISC),
            risc_ns(api::Processor::TRISC0),
            risc_ns(api::Processor::TRISC1),
            risc_ns(api::Processor::TRISC2));
        // The classic report leaves the column empty for an op with no eth kernel zone.
        if (op.risc_end_tsc[static_cast<uint32_t>(api::Processor::ERISC0)] != INT64_MIN) {
            std::fprintf(out, "%.0f", risc_ns(api::Processor::ERISC0));
        }
        std::fputc('\n', out);
    }
    std::fflush(out);
    ops_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
