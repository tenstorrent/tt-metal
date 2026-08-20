// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Op-perf CSV consumer: aggregates the perf-debug record stream into one row per program launch
// (keyed by runtime host-id) with the classic device-profiler report's kernel columns -- first-start
// -> last-end unions over the "<RISC>-KERNEL" wrapper zones, per-core and per-RISC splits. Column
// names and semantics match tools/tracy/process_ops_logs.py so rows join against a classic
// ops_perf_results CSV on GLOBAL CALL COUNT. The classic FW columns have no counterpart: this
// producer's FW wrapper deliberately emits no markers (kernel_profiler.hpp, profileScopeLifecycle).
//
// rec.prog is exact per lane (every RISC emits its own STICKY_PROG at launch, ordered within its
// ring), so aggregation keys on it directly. Trace replays reuse the same runtime host-id, so an op's
// executions are split by ordinal: per (lane, prog), the k-th kernel wrapper pair is execution k --
// exact, because a given program's lane participation is fixed, so counts align across lanes.
//
// Enabled by TT_METAL_PERF_DEBUG_OPS_CSV=<path>: registers itself through register_consumer() at
// load and writes the CSV at process exit, after the capture has drained. It is deliberately a plain
// registry consumer with no profiler hooks -- it doubles as the reference for writing one.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "tools/profiler/perf_debug_consumer.hpp"

namespace tt::tt_metal::perf_debug {

class PerfDebugOpsCsvConsumer {
public:
    void operator()(const PerfDebugRecordBatch& batch);
    // Writes the CSV. Call only after the consumer can no longer receive batches (unregistered, or
    // the capture is fully shut down).
    void write_csv(const std::string& path) const;

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

    enum class ZoneClass : uint8_t { Unseen = 0, Other, Kernel };

    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, OpAgg> ops_;  // (dev, runtime host-id, execution)
    std::unordered_map<uint64_t, uint32_t> pair_count_;              // (dev, lane, prog) -> completed pairs
    // Classification cache over the 27-bit structural id space (a flat array would be 2^27 entries).
    // An id is cached only once its name resolves; names arrive per-ELF strictly before the binary can
    // emit, so a miss is transient at worst and the retry is one map probe.
    std::unordered_map<uint32_t, ZoneClass> class_of_id_;
    ZoneNameMirror names_;  // id -> name, mirrored per-ELF from llrt::ZoneMetaRegistry
    // Snapshot of the capture context's device table: the context lives on the receiver and is gone
    // by the time the exit-path write_csv runs.
    std::vector<DeviceMeta> devices_;
};

}  // namespace tt::tt_metal::perf_debug
