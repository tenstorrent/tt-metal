// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <map>
#include <utility>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"

namespace tt::tt_metal::streaming_profiler {

// The chip's AICLK wall clock against its refclk, as its idle-eth pusher models it (eth_clock_pusher.cpp): one
// segment per PLL multiple, each a line the pusher sends POINTS of -- (refclk r, the line's wall at r) with the
// multiple k8 and the count of samples behind the line in the record's round word. The newest point of the open
// segment stands for it; a CLOSE point ends a segment at its last on-line sample; a new multiple, or a point after
// a close, opens the next; and consecutive segments meet where their lines cross. The raw samples the pusher sends
// around a transition are counted here and kept by the consumer for its CSV dump only.
class LocalClockModel {
public:
    static constexpr double kRefclkHz = 50e6;
    // A line behind fewer samples than this has its intercept fixed too loosely (>0.25 tick) to freeze a node on.
    static constexpr uint32_t kSettledCount = 16;
    // How far from a segment's close the crossing with the next line may lie: the pusher confirms a step within
    // ~25 us of samples and the close is its last on-line sample, so the crossing precedes the close by at most that.
    static constexpr double kKnotSlackTicks = 4000.0;

    struct Run {
        double k8 = 0.0;                     // wall ticks per refclk tick, in eighths
        double ax = 0.0, ay = 0.0;           // the newest point: refclk, wall
        double r_first = 0.0, r_last = 0.0;  // the first and the newest point's refclk
        double w_first = 0.0;                // the first point's wall, the segment's key in wall order
        uint32_t n = 0;                      // samples behind the line
        bool closed = false;
        double slope() const { return k8 / 8.0; }
        double ratio() const { return slope(); }
        bool settled() const { return n >= kSettledCount; }
        double wall_of_refclk(double r) const { return ay + slope() * (r - ax); }
        double refclk_of_wall(double w) const { return ax + (w - ay) / slope(); }
        // A sample is the wall tick of one refclk update, caught to within a cycle; the intercept is their mean.
        double residual_ticks() const { return 1.0; }
        double se_ticks(double) const { return n > 0 ? residual_ticks() / std::sqrt(static_cast<double>(n)) : 0.0; }
    };

    std::vector<Run> runs;  // in time order, disjoint in refclk
    uint64_t n_total = 0;   // clock records received: points, closes and raw samples
    uint64_t points = 0;
    uint64_t raws = 0;
    uint64_t transitions = 0;  // segments after the first

    // A point of the open segment's line, or the segment's close. An older point than the newest is superseded.
    void add_point(uint64_t refclk, uint64_t wall, uint32_t k8, uint32_t n, bool close) {
        n_total++;
        points++;
        const double r = static_cast<double>(refclk), w = static_cast<double>(wall);
        if (runs.empty() || runs.back().closed || runs.back().k8 != static_cast<double>(k8)) {
            transitions += runs.empty() ? 0 : 1;
            runs.emplace_back();
            runs.back().k8 = k8;
            runs.back().r_first = r;
            runs.back().w_first = w;
        }
        Run& cur = runs.back();
        if (r < cur.r_last) {
            return;
        }
        cur.ax = r;
        cur.ay = w;
        cur.r_last = r;
        cur.n = n;
        cur.closed = close;
    }
    void add_raw() {
        n_total++;
        raws++;
    }

    // The segment holding refclk r: the last one starting at or before it (the first, for anything earlier).
    const Run& run_at(double r) const {
        auto it = std::upper_bound(runs.begin(), runs.end(), r, [](double x, const Run& a) { return x < a.r_first; });
        return it == runs.begin() ? runs.front() : *(it - 1);
    }
    // Where consecutive segments hand over: their lines' intersection, when it lies within kKnotSlackTicks of the
    // seam. No value (parallel lines, or a crossing far from the seam): the seam is bridged from a.r_last to b.r_first.
    static std::optional<double> knot(const Run& a, const Run& b) {
        const double ds = a.slope() - b.slope();
        if (!(std::abs(ds) > 1e-9)) {
            return std::nullopt;
        }
        const double r_x = (b.ay - b.slope() * b.ax - a.ay + a.slope() * a.ax) / ds;
        if (std::isfinite(r_x) && r_x >= a.r_last - kKnotSlackTicks && r_x <= b.r_first + kKnotSlackTicks) {
            return r_x;
        }
        return std::nullopt;
    }
    // The wall instant of refclk tick r exactly as the published correction places a record there: each segment's
    // line up to its knot with the next, a straight bridge across a seam without one. 0 when no line holds r yet.
    double wall_at(double r) const {
        if (runs.empty()) {
            return 0.0;
        }
        const Run* run = &run_at(r);
        const size_t idx = static_cast<size_t>(run - runs.data());
        const auto usable = [](const Run& b) { return b.n > 0 && b.slope() > 0.0; };
        if (idx + 1 < runs.size() && usable(runs[idx + 1])) {
            const Run& next = runs[idx + 1];
            if (const auto k = knot(*run, next)) {
                if (r >= *k) {
                    run = &next;
                }
            } else if (r > run->r_last && r < next.r_first && usable(*run)) {
                const double w0 = run->wall_of_refclk(run->r_last), w1 = next.wall_of_refclk(next.r_first);
                return w0 + (w1 - w0) * (r - run->r_last) / (next.r_first - run->r_last);
            }
        }
        if (run == &runs[idx] && idx > 0 && usable(runs[idx - 1])) {
            if (const auto k = knot(runs[idx - 1], *run); k && r < *k) {
                run = &runs[idx - 1];
            }
        }
        return usable(*run) ? run->wall_of_refclk(r) : 0.0;
    }
    // The segment holding wall tick w, by the segments' wall order (monotone with their refclk order).
    const Run& run_at_wall(double w) const {
        auto it = std::upper_bound(runs.begin(), runs.end(), w, [](double x, const Run& a) { return x < a.w_first; });
        return it == runs.begin() ? runs.front() : *(it - 1);
    }
};

// Device<->device sync from the PP_CLOCK samples the idle-eth pushers carry, and the correction it publishes.
//
// LOCAL points feed one LocalClockModel per device. LINK samples (the boot-time eth sync rounds: sender round start
// and end, receiver arrival) are paired by round and solved refclk against refclk, so DVFS on either wall clock
// cannot enter the link solve. From those the consumer publishes, per chip, a time-indexed correction to the baked
// host anchor every Record carries (SyncCorrections; Record::host_time composes it):
//
//   root chip r:      host(T) = H_r + (R_r(T) - R_r(A_r)) * P_r         (static host anchor o applied-AICLK term)
//   non-root chip c:  host(T) = H_r + (link(R_c(T)) - R_r(A_r)) * P_r   (the same, on the root's timeline)
//
// where R_x(T) inverts the constant-rate run holding wall tick T, A_x/H_x are the chip's boot anchor (tick, host ns), P_x
// its refclk period taken as k_mean/hz so it is consistent with that anchor, and link() maps c's refclk onto r's by
// the solved offset and rate about the burst midpoint. Published incrementally for live sinks, finally at capture end.
// Runs entirely on its consumer's thread.
class D2dSyncConsumer {
public:
    void on_attach(const CaptureContext& ctx);
    void on_clock(const ClockSample& s);
    void on_capture_end(const CaptureContext& ctx);

private:
    struct LocalState {
        LocalClockModel model;
        std::vector<std::pair<uint64_t, uint64_t>>
            samples;  // the raw transition samples (refclk, wall), for the CSV dump
    };
    // One end's stamp of a round: the reading (refclk ticks for software stamps, ns for hardware ones) and the eth
    // core's wall clock when it was recorded.
    struct Stamp {
        uint64_t value = 0, wall = 0;
        bool have = false;
    };
    // A round under the number the sender gave it, with both ends' stamps: the sender's frame egress and echo
    // ingress, the receiver's frame ingress and echo egress, so each end has a midpoint.
    struct Round {
        uint32_t id = 0;
        Stamp t0, t1, t1b, t2;
        bool complete() const { return t0.have && t2.have && t1.have && t1b.have; }
    };
    // A link's rounds of one stamp kind: the complete ones in the order they completed, the rest waiting for their
    // other end. A round whose other end never reports (no ring room there, a lapped consumer) is evicted once
    // kPendingMax newer rounds are waiting; nothing behind it shifts.
    struct LinkRounds {
        std::vector<Round> rounds;
        std::map<uint32_t, Round> pending;
    };
    struct LinkStreams {
        LinkRounds sw, hw;
        bool have_hw() const { return !hw.rounds.empty(); }
        const LinkRounds& primary() const { return have_hw() ? hw : sw; }
    };
    // A solved link: receiver refclk = sender refclk + offset + rate * (sender refclk - mid).
    struct LinkSolution {
        bool ok = false;
        bool hw = false;
        double solved_at = 0.0;     // the sender chip's refclk at the newest round of the last solve
        double precision_ns = 0.0;  // residual_rms_ns / sqrt(kept): the offset estimate's own precision
        uint32_t dev_snd = 0, dev_rcv = 0;
        double offset_ticks = 0.0, rate = 0.0, mid = 0.0;
        double offset_ns = 0.0, rate_ppm = 0.0, residual_rms_ns = 0.0;
        size_t rounds = 0, kept = 0, path_dropped = 0;
    };

    int64_t core_index(uint32_t dev, const CoreCoord& eth) const;
    // A round in the refclk domain: each end's midpoint, and for hardware rounds the sender's round trip, the
    // receiver's turnaround and the one-way delay inside the stamps, in ns.
    static double mid_a(const Round& r, bool hw);
    static double mid_b(const Round& r, bool hw);
    static double rtt_ns(const Round& r) {
        return (static_cast<double>(r.t2.value) - static_cast<double>(r.t0.value)) * kHwUnitTicks * 20.0;
    }
    static double turn_ns(const Round& r) {
        return (static_cast<double>(r.t1b.value) - static_cast<double>(r.t1.value)) * kHwUnitTicks * 20.0;
    }
    static double path_ns(const Round& r) { return 0.5 * (rtt_ns(r) - turn_ns(r)); }
    static double path_median(const std::vector<Round>& rounds, size_t begin, size_t n);
    // The fleet timeline's root: the chip the host probe reads, fixed for the capture.
    uint32_t root_dev() const { return ctx_.root_dev; }
    void try_solve_links(bool final);
    // One round in the refclk domain: the sender's midpoint, the receiver's stamp minus it, and the round trip in
    // wall ticks (0 for hardware stamps, which need no trip-time filter).
    struct RoundPoint {
        double mid, off, rtt;
    };
    // Whether the solution was accepted into `out`.
    bool solve_link(const CaptureContext::Link& L, std::vector<RoundPoint> pts, bool hw, LinkSolution& out) const;
    // A device's refclk onto the root's: root_refclk = scale * dev_refclk + shift; prec_ns the precision of the
    // solutions composed along the way.
    struct RootXf {
        double scale = 1.0, shift = 0.0, prec_ns = 0.0;
        bool ok = false;
    };
    // The largest disagreement of a solved link with the tree's composition around its loop: the fleet's path
    // asymmetry as far as its loops reveal it.
    double max_closure_ns() const;
    std::map<uint32_t, RootXf> root_transforms(uint32_t root, std::vector<bool>* used) const;
    void publish_all();
    void log_summary() const;
    void dump_csv() const;
    void publish_rate_plots();
    void publish_error_plots() const;
    // The receiver's stamp and the sender's round midpoint placed on the root's refclk as the sink places records
    // from each chip's eth core, and their difference in ns; tsc_a is the sender's host placement, the plots'
    // abscissa. False when a chip has no fitted run or no node to place a stamp with.
    struct RoundTerms {
        double wall_a = 0, wall_b = 0, root_a = 0, root_b = 0;
    };
    bool round_error(
        const CaptureContext::Link& L,
        const Round& r,
        bool hw,
        int64_t& tsc_a,
        double& err,
        RoundTerms* terms = nullptr) const;
    // Per link, the round errors computed with the corrections as they stood when the round's last stamp arrived:
    // what a sink converting records on arrival actually applied, against the final map publish_error_plots uses.
    std::vector<std::vector<SyncPlotPoint>> live_err_;
    std::vector<size_t> live_done_;

    // A placement node: at eth wall tick H the chip sits at root refclk tick `root`; r is the chip's own refclk it
    // was placed at, tangent the run's rate on the root (root refclk ticks per wall tick), the map past the newest
    // node, and sigma the standard deviation of `root` in ns (the run's line at r and the link solutions the chip
    // reaches the root through).
    struct Node {
        double H, root, r, tangent, sigma;
    };
    // One published series of a chip. Its nodes are frozen (consumers have placed records against them), so a
    // publish only appends beyond them; `cover_H` is how far the newest node's tangent has been confirmed by the
    // fit, `knots` how many run boundaries have their nodes, `last_r` the refclk of the newest node or cover.
    struct Series {
        std::vector<Node> nodes;
        size_t knots = 0;
        double last_r = -1.0;
        double cover_H = -1.0;
        double cover_r = -1.0;
        size_t dropped = 0;   // nodes refused: behind the frozen series, or not a correction below one ns per ns
        size_t extended = 0;  // frontier samples that only advanced the cover
    };
    struct Published {
        Series linked;
    };
    std::map<uint32_t, Published> published_;
    // The composed root transforms as of the newest accepted link solution.
    std::map<uint32_t, RootXf> to_root_;
    uint64_t solve_gen_ = 0;
    uint64_t to_root_gen_ = ~0ull;
    // The nodes a chip's fit yields beyond a series: those at run boundaries, and the open run's frontier.
    struct Fresh {
        std::vector<Node> knots;
        std::optional<Node> frontier;
        size_t knots_after = 0;  // run boundaries consumed once the knots are placed
    };
    Fresh fresh_nodes(const Series& s, const LocalClockModel& fit, const RootXf& xf) const;
    // Publishes one chip's series from its fit as it stands; true when the chip's cover moved.
    bool publish_dev(uint32_t dev);
    // Appends the knots and, if the frontier left the newest tangent by more than kFreezeNs, freezes the tangent where
    // it stood and appends the frontier; otherwise advances the cover. True when the cover moved.
    bool advance(Series& s, uint32_t chip, Fresh fresh);
    void freeze_append(Series& s, uint32_t chip, const Node& n);
    void push_node(Series& s, uint32_t chip, const Node& n);
    CaptureContext ctx_;
    std::map<uint32_t, LocalState> local_;  // device index -> local fit
    const char* const csv_path_ = std::getenv("TT_METAL_STREAMING_PROFILER_D2D_CSV");
    std::vector<LinkStreams> links_;  // per ctx_.links index
    // (device index, decoder core index) -> the link the core stamps for, and whether as its sender.
    std::map<std::pair<uint32_t, uint32_t>, std::pair<size_t, bool>> side_of_;
    std::vector<LinkSolution> solved_;                         // per ctx_.links index
    uint64_t dropped_kind_ = 0;
    // One hardware-stamp payload unit in 20 ns refclk ticks: the kernels report round averages in quarter-ns units
    // (eth_ptp_link.hpp kHwUnitsPerNs); the two must agree.
    static constexpr double kHwUnitTicks = 1.0 / 80.0;
    // A hardware round whose one-way delay inside the stamps sits this far from the window's median had a frame
    // delayed on one leg, and its offset is off by that same amount; the delay itself holds to 0.5 ns.
    static constexpr double kPathDevNs = 2.0;
    // A frontier within this much of the newest tangent extends the cover instead of freezing a node; the published
    // map then sits within it of the fit's own estimate. A mature run's estimate moves ~0.02 ns per keepalive sample,
    // so it adds a node every few hundred ms; a young run adds one per burst sample for its first ms.
    static constexpr double kFreezeNs = 0.25;
    // The refclk span the tangent is measured over along a run's exact line; any span gives the same slope.
    static constexpr double kTangentTicks = 50000.0;
    // A knot (where two runs' exact lines meet) must land within this much refclk (50 us) of the samples that
    // bracket the transition; a split is detected up to ~10 us after the transition it follows.
    // The link solve's window in the sender chip's refclk, re-solved every half window. Two chips' crystals hold a
    // line to ~0.4 ns over 250 ms and their rate moves a few ppb from one such window to the next (measured on the
    // 8-chip runs), so a fit extrapolated half a window past its end stays within ~0.4 ns rms and doubles that at
    // 500 ms. The rounds inside the window only average the fit's own noise, ~0.1 ns at 100 Hz.
    static constexpr double kLinkWindowTicks = 12'500'000.0;  // 250 ms
    static constexpr double kFirstSolveTicks = 10'000'000.0;  // 200 ms of rounds before the first live solution
    static constexpr size_t kMinSolveRounds = 8;
    static constexpr size_t kPendingMax = 4096;
};

}  // namespace tt::tt_metal::streaming_profiler
