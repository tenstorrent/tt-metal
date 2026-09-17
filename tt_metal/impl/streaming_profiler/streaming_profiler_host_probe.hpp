// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_placement_map.hpp"

namespace tt {
class Cluster;
}
namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal::streaming_profiler {

// The host TSC, fenced, in ticks; and its rate, measured once per process against CLOCK_MONOTONIC_RAW.
int64_t tsc_now() noexcept;
double tsc_ticks_per_ns();

// The root chip's refclk on the host TSC: tsc = a + b * refclk, fitted over the recent probe bursts.
struct HostLine {
    double a = 0.0, b = 0.0;
    double sigma_ns = 0.0;  // rms residual of the burst points about the line, in ns
    uint32_t bursts = 0;
    bool ok = false;
    double tsc_of(double refclk) const { return a + b * refclk; }
};

// Ties ONE chip's refclk to the host: bursts of back-to-back reads of that chip's PCIe-tile count-from-reset timer
// (the same distributed ordinary clock the eth tiles count, one NoC hop from the PCIe entry, nobody else's latch),
// each read bracketed by fenced TSC reads through a static TLB window (720 ns round trip), the tightest kept, a line
// fitted across bursts. Every other chip reaches this one through the eth link sync, so this is the fleet's only
// host relation; each burst's line becomes a node of the sync engine's host series. The same thread pairs TSC with
// CLOCK_MONOTONIC for the steady_clock view.
class HostProbe {
public:
    // Writes the host series of `map` while it runs.
    // `csv_path` non-empty: the bursts are written to <csv_path>.probe.csv when the probe stops.
    HostProbe(tt::Cluster& cluster, uint32_t chip_id, PlacementMap& map, std::string csv_path);
    ~HostProbe();
    HostProbe(const HostProbe&) = delete;
    HostProbe& operator=(const HostProbe&) = delete;

    uint32_t chip_id() const { return chip_id_; }
    HostLine line() const;
    SteadySegment steady() const;
    // Ends the reads; the last line and segment stay readable. Must precede the device's teardown.
    void stop();

private:
    struct BurstPoint {
        double tsc, refclk;  // means of the kept reads
        uint32_t kept;
        int64_t rtt_min_ticks, rtt_p50_ticks;
    };
    void run();
    uint32_t read_cfr_lo();
    bool burst(BurstPoint& out);
    void refit();
    void steady_pair();

    tt::Cluster& cluster_;
    const uint32_t chip_id_;
    PlacementMap& map_;
    const std::string csv_path_;
    uint32_t pcie_x_ = 0, pcie_y_ = 0;  // translated
    tt::umd::TlbWindow* window_ = nullptr;
    uint32_t cfr_hi_ = 0, cfr_lo_last_ = 0;
    double ticks_per_ns_ = 0.0;
    std::deque<BurstPoint> points_;
    std::deque<std::pair<int64_t, int64_t>> pairs_;  // (tsc, mono)
    mutable std::mutex mu_;
    HostLine line_;
    SteadySegment steady_;
    uint64_t bursts_ = 0, reads_ = 0, kept_ = 0;
    // Each burst against the line the previous bursts predicted for it: the host placement's error 100 ms ahead.
    uint64_t predicted_ = 0;
    double pred_ss_ns_ = 0.0, pred_worst_ns_ = 0.0;
    double rtt_lo_ns_ = 1e9, rtt_hi_ns_ = 0.0;  // the bursts' tightest round trips: one mode when the thread stays put
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

}  // namespace tt::tt_metal::streaming_profiler
