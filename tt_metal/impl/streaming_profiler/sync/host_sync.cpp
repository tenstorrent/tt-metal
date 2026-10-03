// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/host_sync.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <limits>
#include <thread>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include <numa.h>
#include <tt_stl/assert.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/types/io_window_config.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/sync/least_squares.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

// The PCIe tile's SII register block in its own address space, and the count-from-reset timer inside it, per tables 4
// and 12 of the Blackhole PCIE_SS spec.
constexpr uint64_t kSiiBase = 0xFFFFFFFFF0000000ull;
constexpr uint32_t kCfrLo = 0xA8, kCfrHi = 0xAC;
// On an 8-chip LoudBox, a line fitted over 1 s of bursts and extrapolated 10 ms ahead misses the next burst's line by
// 0.07 ns rms (0.6 ns max).
constexpr auto kBurstPeriod = std::chrono::milliseconds(10);
constexpr size_t kWindowBursts = 100;
// One host node per burst period, on a fixed grid of root refclk ticks.
constexpr double kNodeTicks = kernel_profiler::kEthRefclkHz * std::chrono::duration<double>(kBurstPeriod).count();
constexpr size_t kMinLineBursts = 3;
constexpr int kSteadyBrackets = 16;
constexpr auto kRateSpan = std::chrono::milliseconds(100);

constexpr uint64_t join(uint32_t hi, uint32_t lo) { return (uint64_t{hi} << 32) | lo; }

int64_t clock_ns(clockid_t id) {
    timespec now{};
    clock_gettime(id, &now);
    return static_cast<int64_t>(now.tv_sec) * 1'000'000'000 + now.tv_nsec;
}

int64_t tsc_now() noexcept {
#if defined(__x86_64__)
    _mm_lfence();
    const int64_t tsc = static_cast<int64_t>(__rdtsc());
    _mm_lfence();
    return tsc;
#else
    return clock_ns(CLOCK_MONOTONIC_RAW);
#endif
}

}  // namespace

double tsc_ticks_per_ns() {
    static const double rate = [] {
        const int64_t tsc_start = tsc_now(), raw_start = clock_ns(CLOCK_MONOTONIC_RAW);
        std::this_thread::sleep_for(kRateSpan);
        const int64_t raw_end = clock_ns(CLOCK_MONOTONIC_RAW), tsc_end = tsc_now();
        return static_cast<double>(tsc_end - tsc_start) / static_cast<double>(raw_end - raw_start);
    }();
    return rate;
}

HostSync::HostSync(tt::Cluster& cluster, uint32_t chip_id, SteadyClock& steady) :
    numa_node_(static_cast<int>(cluster.get_numa_node_for_device(chip_id))), steady_(steady) {
    const auto pcie =
        cluster.get_driver()->get_soc_descriptor(chip_id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    TT_FATAL(!pcie.empty(), "streaming profiler: host sync chip {} has no PCIe tile in its descriptor", chip_id);
    // Write-combined like other UMD windows. Reads are uncached either way, and tsc_now()'s fences order them.
    window_ = cluster.get_driver()->create_io_window(
        chip_id,
        pcie.front(),
        kSiiBase,
        tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::WC, .size = kCfrHi + sizeof(uint32_t)});
    // Reading LO latches HI, and this is the only reader, so the pair is consistent.
    const uint32_t lo = window_->read32(kCfrLo);
    const uint32_t hi = window_->read32(kCfrHi);
    bases_ = {.root_refclk = static_cast<int64_t>(join(hi, lo)), .tsc = tsc_now()};
    // The first call sleeps 100 ms, so take it here rather than on the sync thread.
    static_cast<void>(tsc_ticks_per_ns());
}

HostSync::~HostSync() = default;

void HostSync::burst(ClockMap& map) {
    // Reads from the other CPU socket take ~90 ns longer on one leg, which shifts a burst's midpoint by tens of ns.
    if (numa_node_ >= 0 && numa_available() != -1) {
        numa_run_on_node(numa_node_);
    }
    // LO wraps every 86 s and bursts can be further apart, so HI is re-read each burst.
    uint32_t hi = 0, lo_last = 0;
    std::array<Read, kBurstReads> reads;
    for (uint32_t i = 0; i < kBurstReads; i++) {
        const int64_t before = tsc_now();
        const uint32_t lo = window_->read32(kCfrLo);
        const int64_t after = tsc_now();
        if (i == 0) {
            hi = window_->read32(kCfrHi);
        } else if (lo < lo_last) {
            hi++;
        }
        lo_last = lo;
        reads[i] = Read{.mid = before + (after - before) / 2, .rtt = after - before, .refclk = join(hi, lo)};
    }
    std::array<int64_t, kBurstReads> rtts;
    std::ranges::transform(reads, rtts.begin(), &Read::rtt);
    std::ranges::nth_element(rtts, rtts.begin() + rtts.size() / 2);
    const int64_t median_rtt = rtts[rtts.size() / 2];
    const Read& origin = reads.front();
    double sum_tsc = 0.0, sum_refclk = 0.0;
    uint32_t kept = 0;
    for (const Read& read : reads) {
        if (read.rtt <= median_rtt) {
            sum_tsc += static_cast<double>(read.mid - origin.mid);
            sum_refclk += static_cast<double>(read.refclk - origin.refclk);
            kept++;
        }
    }
    points_.push_back(BurstPoint{
        .tsc = static_cast<double>(origin.mid - bases_.tsc) + sum_tsc / kept,
        .refclk = static_cast<double>(static_cast<int64_t>(origin.refclk) - bases_.root_refclk) + sum_refclk / kept});
    if (points_.size() > kWindowBursts) {
        points_.pop_front();
    }
    if (points_.size() >= kMinLineBursts) {
        const LineFit line = fit_line(points_, &BurstPoint::refclk, &BurstPoint::tsc);
        // Each node starts where the previous node's tangent reaches and aims at this line one grid step on, so host
        // times never step back at a node.
        const double grid_at = std::ceil(points_.back().refclk / kNodeTicks) * kNodeTicks;
        const double node_at = node_ ? std::max(grid_at, node_->at + kNodeTicks) : grid_at;
        const double value = node_ ? node_->value + node_->tangent * (node_at - node_->at) : line.at(node_at);
        node_ =
            HostNode{.at = node_at, .value = value, .tangent = (line.at(node_at + kNodeTicks) - value) / kNodeTicks};
        map.append_host(*node_, node_at + kNodeTicks);
    }
    // CLOCK_MONOTONIC is slewed but never stepped, so the steady series is just these pairs, joined by straight lines.
    int64_t best_gap = std::numeric_limits<int64_t>::max(), best_tsc = 0, best_mono = 0;
    for (int i = 0; i < kSteadyBrackets; i++) {
        const int64_t before = tsc_now();
        const int64_t mono = clock_ns(CLOCK_MONOTONIC);
        const int64_t after = tsc_now();
        if (after - before < best_gap) {
            best_gap = after - before;
            best_tsc = before + (after - before) / 2;
            best_mono = mono;
        }
    }
    steady_.append(best_tsc, best_mono);
}

void HostSync::burst_if_due(ClockMap& map) {
    const auto now = std::chrono::steady_clock::now();
    if (now < next_due_) {
        return;
    }
    next_due_ = now + kBurstPeriod;
    burst(map);
}

// The closing burst carries the host series' cover past the capture's last records.
void HostSync::finish(ClockMap& map) {
    do {
        burst(map);
    } while (points_.size() < kMinLineBursts);
}

}  // namespace tt::tt_metal::streaming_profiler
