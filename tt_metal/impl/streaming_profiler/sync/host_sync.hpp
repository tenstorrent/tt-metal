// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>

#include "impl/streaming_profiler/sync/clock_map.hpp"

namespace tt {
class Cluster;
namespace umd {
class IoWindow;
}
}  // namespace tt

namespace tt::tt_metal::streaming_profiler {

// The first call measures the rate, sleeping 100 ms.
double tsc_ticks_per_ns();

// Maps the root chip's refclk onto the host TSC from bursts of reads of the PCIe tile's count-from-reset timer, which
// counts the same distributed clock as the eth tiles, one NoC hop from the PCIe entry, and which no other reader
// latches. It must be destroyed before its device is torn down.
class HostSync {
public:
    HostSync(tt::Cluster& cluster, uint32_t chip_id, SteadyClock& steady);
    ~HostSync();

    const ClockBases& bases() const { return bases_; }
    void burst_if_due(ClockMap& map);
    void finish(ClockMap& map);

private:
    static constexpr uint32_t kBurstReads = 1000;
    struct Read {
        int64_t mid, rtt;
        uint64_t refclk;
    };
    struct BurstPoint {
        double tsc, refclk;
    };
    void burst(ClockMap& map);

    const int numa_node_;
    SteadyClock& steady_;
    std::unique_ptr<tt::umd::IoWindow> window_;
    ClockBases bases_;
    std::deque<BurstPoint> points_;
    std::optional<HostNode> node_;
    std::chrono::steady_clock::time_point next_due_{};
};

}  // namespace tt::tt_metal::streaming_profiler
