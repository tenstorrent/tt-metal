// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_host_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <x86intrin.h>

#include <numa.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

// The PCIe tile's SII register block in its own address space, and the count-from-reset timer inside it
// (Blackhole PCIE_SS spec, tables 4 and 12).
constexpr uint64_t kSiiBase = 0xFFFFFFFFF0000000ull;
constexpr uint32_t kCfrLo = 0xA8, kCfrHi = 0xAC;
constexpr size_t kTlbBytes = 2 * 1024 * 1024;
constexpr uint32_t kBurstReads = 1000;
// Reads within this much of the burst's tightest round trip carry the least queueing on either leg.
constexpr double kRttSlackNs = 50.0;
constexpr size_t kWindowBursts = 10;
constexpr size_t kWindowPairs = 20;
// A steady pair off the segment by more than the pair bracket can explain is a slew step.
constexpr double kSteadyKinkNs = 40.0;

}  // namespace

int64_t clock_ns(clockid_t id) {
    timespec ts{};
    clock_gettime(id, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

int64_t tsc_now() noexcept {
    _mm_lfence();
    const int64_t t = static_cast<int64_t>(__rdtsc());
    _mm_lfence();
    return t;
}

double tsc_ticks_per_ns() {
    static const double rate = [] {
        const int64_t t0 = tsc_now(), r0 = clock_ns(CLOCK_MONOTONIC_RAW);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int64_t r1 = clock_ns(CLOCK_MONOTONIC_RAW), t1 = tsc_now();
        return static_cast<double>(t1 - t0) / static_cast<double>(r1 - r0);
    }();
    return rate;
}

double units_per_tsc() {
    static const double u = 10.0 / tsc_ticks_per_ns();
    return u;
}

namespace {
// Double-buffered under a generation, so a reader that sees the new generation sees the whole segment.
struct SteadySlots {
    SteadySegment seg[2];
    std::atomic<uint32_t> gen{0};
};
SteadySlots g_steady;
}  // namespace

void SteadyView::set(const SteadySegment& segment) noexcept {
    const uint32_t g = g_steady.gen.load(std::memory_order_relaxed);
    g_steady.seg[(g + 1) & 1] = segment;
    g_steady.gen.store(g + 1, std::memory_order_release);
}

int64_t SteadyView::mono_ns(int64_t tsc) noexcept {
    thread_local uint32_t gen = ~0u;
    thread_local SteadySegment seg;
    const uint32_t g = g_steady.gen.load(std::memory_order_acquire);
    if (g != gen) {
        seg = g_steady.seg[g & 1];
        gen = g;
    }
    if (!seg.ok) {
        // The stand-in pair is taken once per thread and generation.
        seg = SteadySegment{static_cast<int64_t>(__rdtsc()), clock_ns(CLOCK_MONOTONIC), 1.0 / tsc_ticks_per_ns(), true};
    }
    return seg.mono_of(tsc);
}

HostProbe::HostProbe(tt::Cluster& cluster, uint32_t chip_id, PlacementMap& map) :
    cluster_(cluster), chip_id_(chip_id), map_(map) {
    ticks_per_ns_ = tsc_ticks_per_ns();
    const auto pcie =
        cluster_.get_driver()->get_soc_descriptor(chip_id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    if (pcie.empty()) {
        log_warning(
            tt::LogMetal, "[streaming profiler] host probe: chip {} has no PCIe tile in its descriptor", chip_id);
        return;
    }
    pcie_x_ = static_cast<uint32_t>(pcie.front().x);
    pcie_y_ = static_cast<uint32_t>(pcie.front().y);
    auto* tlb_manager = cluster_.get_driver()->get_chip(chip_id)->get_tlb_manager();
    const tt_xy_pair xy(pcie_x_, pcie_y_);
    try {
        if (!tlb_manager->is_tlb_mapped(xy)) {
            tlb_manager->configure_tlb(xy, kTlbBytes, kSiiBase, tt::umd::tlb_data::Strict);
        }
        if (tlb_manager->is_tlb_mapped(xy, kSiiBase + kCfrLo, 4)) {
            window_ = tlb_manager->get_tlb_window(xy);
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[streaming profiler] host probe: no static TLB for the PCIe tile: {}", e.what());
    }
    // The LO read latches HI; the host is this instance's only reader, so the pair is consistent.
    cfr_lo_last_ = read_cfr_lo();
    uint32_t hi = 0;
    cluster_.read_reg(&hi, tt_cxy_pair(chip_id_, CoreCoord(pcie_x_, pcie_y_)), kSiiBase + kCfrHi);
    cfr_hi_ = hi;
    thread_ = std::thread(&HostProbe::run, this);
}

HostProbe::~HostProbe() { stop(); }

void HostProbe::stop() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

HostLine HostProbe::line() const {
    std::lock_guard<std::mutex> g(mu_);
    return line_;
}

SteadySegment HostProbe::steady() const {
    std::lock_guard<std::mutex> g(mu_);
    return steady_;
}

uint32_t HostProbe::read_cfr_lo() {
    if (window_ != nullptr) {
        return window_->read32(kCfrLo);
    }
    uint32_t v = 0;
    cluster_.read_reg(&v, tt_cxy_pair(chip_id_, CoreCoord(pcie_x_, pcie_y_)), kSiiBase + kCfrLo);
    return v;
}

bool HostProbe::burst(BurstPoint& out) {
    struct Read {
        int64_t mid, rtt;
        uint64_t refclk;
    };
    std::vector<Read> reads;
    reads.reserve(kBurstReads);
    for (uint32_t i = 0; i < kBurstReads; i++) {
        const int64_t t0 = tsc_now();
        const uint32_t lo = read_cfr_lo();
        const int64_t t1 = tsc_now();
        if (lo < cfr_lo_last_) {
            cfr_hi_++;
        }
        cfr_lo_last_ = lo;
        reads.push_back(Read{t0 + (t1 - t0) / 2, t1 - t0, (static_cast<uint64_t>(cfr_hi_) << 32) | lo});
    }
    int64_t rtt_min = reads.front().rtt;
    for (const Read& r : reads) {
        rtt_min = std::min(rtt_min, r.rtt);
    }
    const int64_t cut = rtt_min + static_cast<int64_t>(kRttSlackNs * ticks_per_ns_);
    const int64_t tsc_ref = reads.front().mid;
    const uint64_t ref_ref = reads.front().refclk;
    double st = 0.0, sr = 0.0;
    uint32_t n = 0;
    for (const Read& r : reads) {
        if (r.rtt <= cut) {
            st += static_cast<double>(r.mid - tsc_ref);
            sr += static_cast<double>(r.refclk - ref_ref);
            n++;
        }
    }
    reads_ += reads.size();
    kept_ += n;
    if (n == 0) {
        return false;
    }
    out = BurstPoint{static_cast<double>(tsc_ref) + st / n, static_cast<double>(ref_ref) + sr / n, n, rtt_min};
    return true;
}

// Least squares of the burst points, centred on their means so the ~1e12-tick magnitudes cancel.
void HostProbe::refit() {
    const size_t n = points_.size();
    HostLine l;
    l.bursts = static_cast<uint32_t>(n);
    if (n >= 2) {
        double mt = 0.0, mr = 0.0;
        for (const BurstPoint& p : points_) {
            mt += p.tsc;
            mr += p.refclk;
        }
        mt /= n;
        mr /= n;
        double srr = 0.0, srt = 0.0;
        for (const BurstPoint& p : points_) {
            srr += (p.refclk - mr) * (p.refclk - mr);
            srt += (p.refclk - mr) * (p.tsc - mt);
        }
        if (srr > 0.0) {
            l.b = srt / srr;
            l.a = mt - l.b * mr;
            double rss = 0.0;
            for (const BurstPoint& p : points_) {
                const double e = p.tsc - l.tsc_of(p.refclk);
                rss += e * e;
            }
            l.sigma_ns = n > 2 ? std::sqrt(rss / (n - 2)) / ticks_per_ns_ : 0.0;
            l.ok = true;
        }
    }
    std::lock_guard<std::mutex> g(mu_);
    line_ = l;
}

void HostProbe::steady_pair() {
    int64_t best_gap = INT64_MAX, best_tsc = 0, best_mono = 0;
    for (int i = 0; i < 16; i++) {
        const int64_t t0 = tsc_now();
        const int64_t m = clock_ns(CLOCK_MONOTONIC);
        const int64_t t1 = tsc_now();
        if (t1 - t0 < best_gap) {
            best_gap = t1 - t0;
            best_tsc = t0 + (t1 - t0) / 2;
            best_mono = m;
        }
    }
    SteadySegment cur = steady();
    if (cur.ok) {
        const double resid = static_cast<double>(best_mono - cur.mono_of(best_tsc));
        if (std::abs(resid) > kSteadyKinkNs) {
            log_debug(tt::LogMetal, "[streaming profiler] host probe: steady_clock slew step of {:+.0f} ns", resid);
            pairs_.clear();
        }
    }
    pairs_.emplace_back(best_tsc, best_mono);
    while (pairs_.size() > kWindowPairs) {
        pairs_.pop_front();
    }
    SteadySegment s;
    const size_t n = pairs_.size();
    if (n >= 2) {
        double mt = 0.0, mm = 0.0;
        for (const auto& [t, m] : pairs_) {
            mt += static_cast<double>(t - pairs_.front().first);
            mm += static_cast<double>(m - pairs_.front().second);
        }
        mt /= n;
        mm /= n;
        double stt = 0.0, stm = 0.0;
        for (const auto& [t, m] : pairs_) {
            const double dt = static_cast<double>(t - pairs_.front().first) - mt;
            stt += dt * dt;
            stm += dt * (static_cast<double>(m - pairs_.front().second) - mm);
        }
        if (stt > 0.0) {
            s.ns_per_tick = stm / stt;
            s.tsc0 = pairs_.front().first + static_cast<int64_t>(mt);
            s.mono0 = pairs_.front().second + static_cast<int64_t>(mm);
            s.ok = true;
        }
    } else {
        s.tsc0 = best_tsc;
        s.mono0 = best_mono;
        s.ns_per_tick = 1.0 / ticks_per_ns_;
        s.ok = true;
    }
    {
        std::lock_guard<std::mutex> g(mu_);
        steady_ = s;
    }
    SteadyView::set(s);
}

void HostProbe::run() {
    set_os_thread_name("sp-hostprobe");
    // The read's round trip is 90 ns longer from the other socket, and the extra sits on one leg, so a thread that
    // wanders between sockets shifts the bracket's midpoint by tens of ns burst to burst.
    if (numa_available() != -1) {
        numa_run_on_node(static_cast<int>(cluster_.get_numa_node_for_device(chip_id_)));
    }
    // The first bursts come quickly so a line exists before records need it; then one every 100 ms, a cadence the
    // refclk period's ramp on the TSC (~0.05 ppm/s) keeps the line's prediction within tens of ns of.
    static constexpr int64_t kSchedule[] = {0, 100, 200, 300};
    const auto t_start = std::chrono::steady_clock::now();
    size_t k = 0;
    int64_t next_pair_ms = 0;
    const auto elapsed_ms = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start)
            .count();
    };
    const auto take = [&] {
        HostLine before = line();
        BurstPoint p{};
        if (!burst(p)) {
            return;
        }
        const double pred_err_ns = before.ok ? (p.tsc - before.tsc_of(p.refclk)) / ticks_per_ns_ : 0.0;
        points_.push_back(p);
        while (points_.size() > kWindowBursts) {
            points_.pop_front();
        }
        refit();
        bursts_++;
        const HostLine l = line();
        rtt_lo_ns_ = std::min(rtt_lo_ns_, static_cast<double>(p.rtt_min_ticks) / ticks_per_ns_);
        rtt_hi_ns_ = std::max(rtt_hi_ns_, static_cast<double>(p.rtt_min_ticks) / ticks_per_ns_);
        if (before.ok && before.bursts >= 3) {
            predicted_++;
            pred_ss_ns_ += pred_err_ns * pred_err_ns;
            pred_worst_ns_ = std::max(pred_worst_ns_, std::abs(pred_err_ns));
        }
        // A node per burst: the line as it stands, at the burst's own instant. Records placed between two
        // bursts run on the newer node's tangent; those placed later interpolate between the nodes.
        if (l.ok && l.bursts >= 3) {
            map_.append_host(HostNode{.at = p.refclk, .value = l.tsc_of(p.refclk), .tangent = l.b});
        }
    };
    while (!stop_.load(std::memory_order_acquire)) {
        const int64_t now_ms = elapsed_ms();
        const int64_t due = k < std::size(kSchedule) ? kSchedule[k]
                                                     : kSchedule[std::size(kSchedule) - 1] +
                                                           100 * static_cast<int64_t>(k + 1 - std::size(kSchedule));
        if (now_ms >= due) {
            take();
            k++;
        }
        if (now_ms >= next_pair_ms) {
            steady_pair();
            next_pair_ms = now_ms + 100;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // A capture shorter than the schedule's third burst would end without a line: the bursts still owed come now,
    // and the last spans the capture with the first.
    for (int tries = 0; tries < 8; tries++) {
        if (const HostLine l = line(); l.ok && l.bursts >= 3) {
            break;
        }
        take();
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] host probe chip {}: {} bursts, {} of {} reads kept, tightest round trips {:.0f}-{:.0f} "
        "ns; "
        "final refclk period {:.6f} ns, residual {:.1f} ns; each burst against the line predicted for it: rms {:.1f} "
        "ns, "
        "worst {:.1f} ns over {}; steady_clock {:.9f} ns per tick",
        chip_id_,
        bursts_,
        kept_,
        reads_,
        rtt_lo_ns_,
        rtt_hi_ns_,
        line().ok ? line().b / ticks_per_ns_ : 0.0,
        line().sigma_ns,
        predicted_ != 0 ? std::sqrt(pred_ss_ns_ / static_cast<double>(predicted_)) : 0.0,
        pred_worst_ns_,
        predicted_,
        steady().ns_per_tick);
}

}  // namespace tt::tt_metal::streaming_profiler
