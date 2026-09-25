// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Op-perf CSV consumer: one row per program launch (keyed by runtime host-id) with the classic report's
// kernel columns (first-start to last-end unions over the "<RISC>-KERNEL" wrapper zones, per-core and
// per-RISC splits), joinable against a classic ops_perf_results CSV on GLOBAL CALL COUNT. The classic FW
// columns have no counterpart: this producer's FW wrapper emits no markers. Trace replays reuse a host-id,
// so an op's executions are split by ordinal: per (lane, prog) the k-th wrapper pair is execution k.
// Enabled by TT_METAL_STREAMING_PROFILER_OPS_CSV=<path>; the file is opened on construction and each capture's ops are
// appended when it detaches.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

class OpsCsvConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::RecordType::Zones>;
    explicit OpsCsvConsumer(const std::string& path);
    void operator()(const Batch& batch);
    // Appends the ops received since the previous call, after the header on the first. Call only between captures.
    void write_csv();

private:
    static constexpr uint32_t kNumRisc = 5;

    // Kernel start and end in device cycles (the CSV's CYCLE columns) and on the host TSC (every [ns] column): a
    // difference of cycles has no single rate to convert with under DVFS, the host span has.
    struct OpAgg {
        uint64_t k_start = UINT64_MAX, k_end = 0;
        int64_t h_start = INT64_MAX, h_start_last = INT64_MIN, h_end = INT64_MIN;
        int64_t h_dm_start = INT64_MAX;  // earliest BRISC/NCRISC kernel start
        std::array<int64_t, kNumRisc> h_risc_start{};
        std::array<int64_t, kNumRisc> h_risc_end{};
        std::map<uint32_t, std::pair<int64_t, int64_t>> cores;  // core -> (kernel start, end), host TSC
        OpAgg() {
            h_risc_start.fill(INT64_MAX);
            h_risc_end.fill(INT64_MIN);
        }
    };

    FILE* f_ = nullptr;
    bool header_written_ = false;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, OpAgg> ops_;  // (chip, runtime host-id, execution)
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> pair_count_;  // (chip, core, risc, prog)
};

}  // namespace tt::tt_metal::streaming_profiler
