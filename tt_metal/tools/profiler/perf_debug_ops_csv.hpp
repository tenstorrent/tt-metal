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
// Attribution: only a core's BRISC emits STICKY_PROG, so rec.prog on the other lanes is exact only
// to drainer-sweep granularity -- a frame straddling an op boundary stamps neighbouring zones with
// the wrong op and back-to-back launches then union to ~2x (measured). BRISC's own records ARE exact
// (the sticky is ordered within its ring), so BRISC kernel windows define each op's time span per
// core and the other lanes' kernel pairs are assigned by timestamp: BRISC opens its wrapper before
// it launches NCRISC/TRISCs, so "latest BRISC start at or before the zone's start" is the owner.
//
// Enabled by TT_METAL_PERF_DEBUG_OPS_CSV=<path>: registers itself through register_consumer() at
// load and writes the CSV at process exit, after the capture has drained. It is deliberately a plain
// registry consumer with no profiler hooks -- it doubles as the reference for writing one.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "tools/profiler/perf_debug_consumer.hpp"

namespace tt::tt_metal::perf_debug {

class PerfDebugOpsCsvConsumer {
public:
    void operator()(const PerfDebugRecordBatch& batch);
    // Assigns lane pairs to ops and writes the CSV. Call only after the consumer can no longer
    // receive batches (unregistered, or the capture is fully shut down).
    void write_csv(const std::string& path);

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

    struct BriscWindow {
        uint64_t start = 0, end = 0;
        uint32_t prog = 0;
    };
    struct LanePair {
        uint64_t start = 0, end = 0;
        uint8_t risc = 0;
    };
    struct CoreState {
        std::vector<BriscWindow> windows;             // in stream order == time order
        std::vector<LanePair> pairs;                  // non-BRISC kernel pairs, assigned at write
        std::array<uint64_t, kNumRisc> open_start{};  // per-lane kernel start awaiting its end
    };

    struct DeviceMeta {
        uint32_t chip_id = 0;
        double frequency_ghz = 0.0;
    };

    enum class ZoneClass : uint8_t { Unseen = 0, Other, Kernel };

    void fold(uint32_t dev, uint32_t prog, uint32_t core, uint8_t risc, uint64_t start, uint64_t end);

    std::map<std::pair<uint32_t, uint32_t>, CoreState> cores_;  // (dev, core)
    std::map<std::pair<uint32_t, uint32_t>, OpAgg> ops_;        // (dev, runtime host-id)
    uint64_t unassigned_pairs_ = 0;
    std::array<ZoneClass, 1 << 16> class_of_hash_{};
    // Snapshot of the capture context's device table: the context lives on the receiver and is gone
    // by the time the exit-path write_csv runs.
    std::vector<DeviceMeta> devices_;
};

}  // namespace tt::tt_metal::perf_debug
