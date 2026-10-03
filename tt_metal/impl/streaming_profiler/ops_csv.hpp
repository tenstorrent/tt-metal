// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>

#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

// Rows join against a classic ops_perf_results CSV on GLOBAL CALL COUNT. Trace replays reuse a host id, so an op's
// executions are split by ordinal.
class OpsCsvConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::RecordType::Zones>;
    explicit OpsCsvConsumer(const std::string& path);
    void operator()(const Batch& batch);
    // Call only between captures.
    void write_csv();

private:
    // The Tensix RISCs' columns, then ERISC0's, which holds both ERISCs.
    static constexpr uint32_t kRiscColumns =
        static_cast<uint32_t>(experimental::streaming_profiler::Processor::ERISC0) + 1;

    // An eth core's logical coordinate can equal a Tensix core's.
    struct CoreKey {
        bool eth;
        uint32_t y, x;
        auto operator<=>(const CoreKey&) const = default;
    };
    struct OpKey {
        uint32_t chip, program, execution;
        auto operator<=>(const OpKey&) const = default;
    };
    struct LaneKey {
        uint32_t chip;
        CoreKey core;
        uint32_t processor, program;
        auto operator<=>(const LaneKey&) const = default;
    };
    struct Span {
        int64_t start = INT64_MAX, end = INT64_MIN;
    };
    // Kernel start and end in device cycles for the CSV's CYCLE columns and on the host TSC for every [ns] column,
    // since under DVFS a difference of cycles has no single rate to convert with. The cycles are the Tensix zones'
    // only: the classic report subtracts them across a chip's ops, and an eth tile's counter runs up to seconds from
    // the Tensix tiles' (tile_sync.hpp).
    struct OpAgg {
        uint64_t start_cycles = UINT64_MAX, end_cycles = 0;
        int64_t last_kernel_start_tsc = INT64_MIN;
        std::array<int64_t, kRiscColumns> risc_start_tsc{};
        std::array<int64_t, kRiscColumns> risc_end_tsc{};
        std::map<CoreKey, Span> cores;
        OpAgg() {
            risc_start_tsc.fill(INT64_MAX);
            risc_end_tsc.fill(INT64_MIN);
        }
    };

    std::unique_ptr<FILE, int (*)(FILE*)> f_;
    bool header_written_ = false;
    std::map<OpKey, OpAgg> ops_;
    // Kept across captures, since every capture's rows go to the one file.
    std::map<LaneKey, uint32_t> executions_seen_;
};

}  // namespace tt::tt_metal::streaming_profiler
