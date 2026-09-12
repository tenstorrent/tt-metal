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

// The local half of the device<->device sync: one chip's AICLK wall clock against its eth tile's free-running
// 50 MHz refclk, from the idle-eth tracker's PP_CLOCK(LOCAL) samples.
//
// AICLK is a PLL multiple of the crystal the refclk counts, so between DVFS transitions the wall clock is EXACTLY
// linear in the refclk with a slope that is a multiple of 1/8 wall ticks per refclk tick (27 at 1.35 GHz, 26.875 at
// 1.34375, 20.375 at 1.01875), and a transition is a ~3 us glide at most once per ms (the ARC firmware's DVFS timer).
// The fit is therefore one straight line per RUN of constant rate, each relative to its own first sample: a sample
// joins the open run while it lies on that run's exact line and opens a new run when it does not, so the run
// boundaries ARE the transitions, and between two of them the map is one fitted line: its slope resolves the ratio
// to 0.03 ppm within 10 ms and its intercept averages every sample of the run (~1 ns after a few dozen) -- where a
// chord through a fixed window bends for the whole window when a transition falls inside it. The PLL multiple is
// only nominal at the ppm level (a run of 10 s at the snapped 27.000 drifted 0.5 ppm off the samples), so the
// multiple serves to name the rate, never to place a record.
struct LocalClockFit {
    static constexpr double kRefclkHz = 50e6;
    // A sample this far (wall ticks) off the open run's line opens a new run: the refclk quantisation puts a sample
    // at most ~14 ticks off the line, a single 1/8 step in the ratio walks 19 ticks per 3 us stride.
    static constexpr double kSplitWallTicks = 64.0;
    // A run is cut after this much refclk (0.5 s) regardless: the ratio wanders with temperature at the 0.01 ppm/s
    // level, which one line per half second follows to ~ns, and the relative sums stay small.
    static constexpr double kMaxRunTicks = 25e6;
    // Fewer samples than this fix the slope too loosely (>100 ppm) to test a newcomer against.
    static constexpr uint64_t kSettledSamples = 16;

    struct Run {
        bool anchored = false;
        double ax = 0.0, ay = 0.0;  // this run's own origin: (refclk, wall)
        double r_first = 0.0, r_last = 0.0;
        double last_x = 0.0, last_y = 0.0;  // the newest sample, so it can be handed to the next run
        double prev_x = 0.0;                // the sample before it, for r_last once the newest is removed
        long double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        uint64_t n = 0;
        // Wall ticks per refclk tick over this run: the applied AICLK / 50 MHz.
        double fitted_slope() const {
            if (n < 2) {
                return 0.0;
            }
            const long double nn = static_cast<long double>(n);
            const long double den = nn * sxx - sx * sx;
            return den > 0 ? static_cast<double>((nn * sxy - sx * sy) / den) : 0.0;
        }
        // The line's slope is the DVFS step's exact multiple whenever the fit lies within a step's width of one. A
        // young run's fitted slope (16 samples over 48 us, refclk quantised to 20 ns) is off by up to ~100 ppm,
        // which is 20 ns of placement 200 us in; runs longer than 50 ms all land on their multiple to <0.005 ppm.
        double slope() const {
            const double r = ratio();
            return r != 0.0 ? r : fitted_slope();
        }
        // The DVFS step this run sits on: the nearest multiple of 1/8 (0 when the fit is not within 1000 ppm of one,
        // a run cut across a glide).
        double ratio() const {
            const double s = fitted_slope();
            const double sn = std::round(s * 8.0) / 8.0;
            return (sn > 0.0 && std::abs(s - sn) <= 1000e-6 * sn) ? sn : 0.0;
        }
        // Whether the line is fixed well enough (slope to ~100 ppm, a stride's extrapolation to a fraction of a
        // tick) to test a newcomer against it.
        bool settled() const { return n >= kSettledSamples; }
        double intercept() const {
            if (n < 2) {
                return 0.0;
            }
            return static_cast<double>((sy - static_cast<long double>(slope()) * sx) / static_cast<long double>(n));
        }
        // The fitted line, both ways, through this run's own anchor so a conversion never leaves the interval.
        double wall_of_refclk(double r) const { return ay + intercept() + slope() * (r - ax); }
        double refclk_of_wall(double w) const { return ax + (w - ay - intercept()) / slope(); }
        void add(double x, double y) {
            if (!anchored) {
                ax = x;
                ay = y;
                r_first = x;
                anchored = true;
            }
            r_last = x;
            prev_x = last_x;
            last_x = x;
            last_y = y;
            const long double dx = x - ax, dy = y - ay;
            sx += dx;
            sy += dy;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
            n++;
        }
        // The scatter of the run's samples about its line, in wall ticks: the refclk's 20 ns quantisation spread
        // over the ~27 wall ticks it spans, ~8 ticks for any run long enough to fit.
        double residual_ticks() const {
            if (n < 3) {
                return 0.0;
            }
            const long double b = slope();
            const long double a = intercept();
            const long double rss =
                syy - 2 * a * sy - 2 * b * sxy + static_cast<long double>(n) * a * a + 2 * a * b * sx + b * b * sxx;
            return rss > 0 ? std::sqrt(static_cast<double>(rss / static_cast<long double>(n - 2))) : 0.0;
        }
        // Standard error of the line's wall value at refclk r: the intercept's alone when the slope is the exact PLL
        // multiple, with the slope's contribution when it had to be fitted.
        double se_ticks(double r) const {
            if (n < 3) {
                return 0.0;
            }
            const double nn = static_cast<double>(n);
            const double sigma = residual_ticks();
            if (ratio() != 0.0) {
                return sigma / std::sqrt(nn);
            }
            const double xbar = static_cast<double>(sx) / nn;
            const double sxx_c = static_cast<double>(sxx) - static_cast<double>(sx) * xbar;
            const double dx = (r - ax) - xbar;
            return sigma * std::sqrt(1.0 / nn + (sxx_c > 0.0 ? dx * dx / sxx_c : 0.0));
        }
        // A sample older than the run's own (handed over from the run before): the sums and the start move, the
        // newest-sample bookkeeping does not.
        void add_front(double x, double y) {
            r_first = std::min(r_first, x);
            const long double dx = x - ax, dy = y - ay;
            sx += dx;
            sy += dy;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
            n++;
        }
        // Takes the newest sample back out; the sums are exact, so this is exact.
        void remove_last() {
            const long double dx = last_x - ax, dy = last_y - ay;
            sx -= dx;
            sy -= dy;
            sxx -= dx * dx;
            sxy -= dx * dy;
            syy -= dy * dy;
            n--;
            last_x = prev_x;
            r_last = prev_x;
        }
    };

    std::vector<Run> runs;  // in time order, disjoint in refclk
    uint64_t n_total = 0;
    uint64_t transitions = 0;  // runs opened by a sample off the line or by a burst (the kMaxRunTicks cuts are not counted)
    uint64_t handed_over = 0;  // samples moved from a run's tail to the run after it
    double prev_gap = 0.0;     // refclk between the last two samples
    bool pending_handover = false;

    // The tracker emits a burst of consecutive 3 us samples when it detects a rate change, a sample per 100 us for
    // the run's first ms, then one per ms: a sample following the previous one by less than kBurstGapTicks after a
    // gap of at least kSparseGapTicks is the first of a burst, i.e. the tracker's own verdict that a new rate
    // began, and opens a run whether or not it has yet left the old line by the threshold. Testing the same sample
    // against a fitted line with the same threshold let it join the old run one sample too often, and that one
    // sample pulled a 1 ms run's slope 40 ppm off, which a node frozen on it carried as 20-45 ns.
    static constexpr double kBurstGapTicks = 300.0;    // 6 us
    static constexpr double kSparseGapTicks = 1500.0;  // 30 us

    void add(uint64_t refclk_ticks, uint64_t wall_ticks) {
        const double r = static_cast<double>(refclk_ticks), w = static_cast<double>(wall_ticks);
        n_total++;
        if (!runs.empty()) {
            Run& cur = runs.back();
            const double gap = r - cur.r_last;
            const bool burst_start = cur.n >= 2 && gap < kBurstGapTicks && prev_gap >= kSparseGapTicks;
            const bool off = cur.settled() && std::abs(w - cur.wall_of_refclk(r)) > kSplitWallTicks;
            if (!off && !burst_start && r - cur.r_first < kMaxRunTicks) {
                cur.add(r, w);
                prev_gap = gap;
                settle_handover();
                return;
            }
            transitions += (off || burst_start) ? 1 : 0;
            // The old run's newest sample may already sit on the new rate (a sparse sample landing in the ~12 us
            // between a transition and the tracker's detection of it); judged once the new run's line is settled.
            pending_handover = burst_start && cur.n >= 3;
            prev_gap = gap;
        }
        runs.emplace_back();
        runs.back().add(r, w);
    }

private:
    void settle_handover() {
        if (!pending_handover || runs.size() < 2 || !runs.back().settled()) {
            return;
        }
        pending_handover = false;
        Run& prev = runs[runs.size() - 2];
        Run& cur = runs.back();
        const double x = prev.last_x, y = prev.last_y;
        const double off_prev = std::abs(y - prev.wall_of_refclk(x));
        const double off_cur = std::abs(y - cur.wall_of_refclk(x));
        if (off_cur < off_prev) {
            prev.remove_last();
            cur.add_front(x, y);
            handed_over++;
        }
    }

public:
    // The run holding refclk r: the last one starting at or before it (the first, for anything earlier).
    const Run& run_at(double r) const {
        auto it = std::upper_bound(runs.begin(), runs.end(), r, [](double x, const Run& a) { return x < a.r_first; });
        return it == runs.begin() ? runs.front() : *(it - 1);
    }
    // Where consecutive runs hand over: their lines' intersection, when it falls within kKnotSlackTicks of the seam
    // (a small DVFS step stays inside the tracker's detection band for ~10 us, so the old run's last samples lie
    // past the true transition). No value: the seam is bridged straight from a.r_last to b.r_first.
    static constexpr double kKnotSlackTicks = 2500.0;
    static std::optional<double> knot(const Run& a, const Run& b) {
        const double ds = a.slope() - b.slope();
        if (!(std::abs(ds) > 1e-6)) {
            return std::nullopt;
        }
        const double r_x = (b.ay + b.intercept() - b.slope() * b.ax - a.ay - a.intercept() + a.slope() * a.ax) / ds;
        if (std::isfinite(r_x) && r_x >= a.r_last - kKnotSlackTicks && r_x <= b.r_first + kKnotSlackTicks) {
            return r_x;
        }
        return std::nullopt;
    }
    // The wall instant of refclk tick r exactly as the published correction places a record there: each run's line
    // up to its knot with the next, a straight bridge across a seam without one. 0 when no line holds r yet.
    double wall_at(double r) const {
        if (runs.empty()) {
            return 0.0;
        }
        const Run* run = &run_at(r);
        const size_t idx = static_cast<size_t>(run - runs.data());
        const auto usable = [](const Run& b) { return b.n >= 2 && b.slope() > 0.0; };
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
    // The run holding wall tick w, by the runs' wall order (monotone with their refclk order).
    const Run& run_at_wall(double w) const {
        auto it = std::upper_bound(runs.begin(), runs.end(), w, [](double x, const Run& a) { return x < a.ay; });
        return it == runs.begin() ? runs.front() : *(it - 1);
    }
};

// Device<->device sync from the PP_CLOCK samples the idle-eth pushers carry, and the correction it publishes.
//
// LOCAL samples feed one LocalClockFit per device. LINK samples (the boot-time eth sync rounds: sender round start
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
        LocalClockFit fit;
        std::vector<std::pair<uint64_t, uint64_t>> samples;  // (refclk, wall) as received, kept for the CSV dump only
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
    // A chip's refclk frame: its anchor tick's refclk and its refclk period, once its buckets allow.
    struct Frame {
        bool ok = false;
        double refclk_at_anchor = 0.0;
        double period_ns = 0.0;
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
    // The fleet timeline's root: the lowest device index with an eth clock anchor, fixed for the capture. Taking the
    // lowest index seen so far instead let whichever tracker decoded first be root for its first publish, and its
    // identity nodes froze.
    uint32_t root_dev() const;
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
    Frame frame_of(uint32_t dev) const;
    // frame_of(dev), computed once the chip's first bucket is complete and kept: the anchor precedes every bucket,
    // so the frame never changes afterwards, and a walk over every bucket per publish is what it would cost.
    const Frame& frame_cached(uint32_t dev);
    void publish_all();
    void log_summary() const;
    void dump_csv() const;
    void publish_rate_plots();
    void publish_error_plots() const;
    // The receiver's stamp and the sender's round midpoint placed on the host timeline as the sink places records
    // from each chip's eth core (baked eth anchor, then the published correction at that instant), and their
    // difference in ns. False when a chip has no fitted run to place a stamp with.
    // The conversion chain of one round, for the error CSV: each chip's wall instant, its host time before the
    // correction, and the correction applied.
    struct RoundTerms {
        double wall_a = 0, wall_b = 0, baked_a = 0, baked_b = 0, corr_a = 0, corr_b = 0;
    };
    bool round_error(
        const CaptureContext::Link& L,
        const Round& r,
        bool hw,
        double& host_a,
        double& err,
        RoundTerms* terms = nullptr) const;
    // Per link, the round errors computed with the corrections as they stood when the round's last stamp arrived:
    // what a sink converting records on arrival actually applied, against the final map publish_error_plots uses.
    std::vector<std::vector<SyncPlotPoint>> live_err_;
    std::vector<size_t> live_done_;

    // A correction node: at host time H the chip's record time moves by d (ns); r is the refclk it was placed at,
    // tangent the correction's slope along the run it sits on (ns per ns), the map past the newest node, and sigma
    // the standard deviation of d (the run's line at r, and the link solutions the chip reaches the root through).
    struct Node {
        double H, d, r, tangent, sigma;
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
        Series linked, local;
    };
    std::map<uint32_t, Published> published_;
    std::map<uint32_t, Frame> frames_;
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
    template <typename Corr>
    Fresh fresh_nodes(
        const Series& s, const LocalClockFit& fit, const DeviceClock& eclk, double prec_ns, const Corr& corr) const;
    // Publishes one chip's series from its fit as it stands; true when the chip's linked cover moved.
    bool publish_dev(uint32_t dev);
    // Appends the knots and, if the frontier left the newest tangent by more than kFreezeNs, freezes the tangent where
    // it stood and appends the frontier; otherwise advances the cover. True when the cover moved.
    bool advance(Series& s, SyncSeries kind, uint32_t chip, Fresh fresh);
    void freeze_append(Series& s, SyncSeries kind, uint32_t chip, const Node& n);
    void push_node(Series& s, SyncSeries kind, uint32_t chip, const Node& n);
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
    static constexpr double kProvisionalTicks = 1500.0;  // the open run's newest 30 us: no node is frozen there
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
