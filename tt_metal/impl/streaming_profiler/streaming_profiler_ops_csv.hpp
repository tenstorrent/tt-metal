// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Op-perf CSV consumer: one row per program launch (keyed by runtime host-id) with the classic report's
// kernel columns (first-start to last-end unions over the "<RISC>-KERNEL" wrapper zones, per-core and
// per-RISC splits), joinable against a classic ops_perf_results CSV on GLOBAL CALL COUNT. The classic FW
// columns have no counterpart: this producer's FW wrapper emits no markers. Trace replays reuse a host-id,
// so an op's executions are split by ordinal: per (lane, prog) the k-th wrapper pair is execution k.
// Enabled by TT_METAL_STREAMING_PROFILER_OPS_CSV=<path>; the file is opened on construction and written when the last
// capture detaches.
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
    // Call only after the consumer can no longer receive batches.
    void write_csv();

private:
    static constexpr uint32_t kNumRisc = 5;

    struct OpAgg {
        uint64_t k_start = UINT64_MAX, k_start_last = 0, k_end = 0;
        uint64_t dm_start = UINT64_MAX;  // earliest BRISC/NCRISC kernel start
        std::array<uint64_t, kNumRisc> risc_start{};
        std::array<uint64_t, kNumRisc> risc_end{};
        std::map<uint32_t, std::pair<uint64_t, uint64_t>> cores;  // core -> (kernel start, end)
        OpAgg() { risc_start.fill(UINT64_MAX); }
    };

    struct DeviceMeta {
        uint32_t chip_id = 0;
        double frequency_ghz = 0.0;
    };

    FILE* f_ = nullptr;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, OpAgg> ops_;  // (chip, runtime host-id, execution)
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> pair_count_;  // (chip, core, risc, prog)
    std::unordered_map<uint32_t, DeviceMeta> devices_;  // from the records: one frequency per chip
};

}  // namespace tt::tt_metal::streaming_profiler
