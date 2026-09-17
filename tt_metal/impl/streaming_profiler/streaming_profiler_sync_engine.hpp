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

#include <string>
#include <unordered_set>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_placement_map.hpp"

namespace tt::tt_metal::streaming_profiler {


// The chip's AICLK wall clock against its refclk, as its idle-eth pusher models it (eth_clock_pusher.cpp): one
// segment per PLL multiple, each a line the pusher sends POINTS of -- (refclk r, the line's wall at r) with the
// multiple k8 and the count of samples behind the line in the record's round word. The newest point of the open
// segment stands for it; a CLOSE point ends a segment at its last on-line sample; a new multiple, or a point after
// a close, opens the next; and consecutive segments meet where their lines cross. The raw samples the pusher sends
// around a transition are only counted.
class LocalClockModel {
public:
    static constexpr double kRefclkHz = kernel_profiler::kEthRefclkHz;
    // A line behind fewer samples than this has its intercept fixed too loosely (>0.25 tick) to freeze a node on.
    static constexpr uint32_t kSettledCount = 16;
    // How far from a segment's close the crossing with the next line may lie: the pusher confirms a step within
    // ~25 us of samples and the close is its last on-line sample, so the crossing precedes the close by at most that.
    static constexpr double kKnotSlackTicks = 4000.0;

    struct Run {
        double k8 = 0.0;                     // wall ticks per refclk tick, in eighths
        double r_first = 0.0, r_last = 0.0;  // the first and the newest point's refclk
        double w_first = 0.0, w_last = 0.0;  // their walls; w_first is the segment's key in wall order
        uint32_t n = 0;                      // samples behind the line
        bool closed = false;
        double slope() const { return k8 / 8.0; }
        bool settled() const { return n >= kSettledCount; }
        double wall_of_refclk(double r) const { return w_last + slope() * (r - r_last); }
        double refclk_of_wall(double w) const { return r_last + (w - w_last) / slope(); }
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
        cur.r_last = r;
        cur.w_last = w;
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
        const double r_x = (b.w_last - b.slope() * b.r_last - a.w_last + a.slope() * a.r_last) / ds;
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

constexpr double kNsPerRefclk = 1e9 / kernel_profiler::kEthRefclkHz;

// A device's refclk onto the root chip's: root_refclk = scale * dev_refclk + shift.
struct RootXf {
    double scale = 1.0, shift = 0.0;
    bool ok = false;
};

// The eth links' rounds and what they solve to. Each link's two ends stamp their frames with the 1588 hardware and
// report a round's stamp averages as PP_CLOCK link samples; the solver pairs the two ends' samples by round number,
// fits each link's offset and rate over a window of rounds in the refclk domain (so DVFS on either wall clock cannot
// enter), combines a chip pair's parallel links, and composes every chip onto the root along the solved tree.
class LinkSolver {
public:
    // One end's stamp of a round: the reading, in the link's stamp units of the refclk domain.
    struct Stamp {
        uint64_t units = 0;
        bool have = false;
    };
    // A round under the number the sender gave it, with both ends' stamps: the sender's frame egress and echo
    // ingress, the receiver's frame ingress and echo egress, so each end has a midpoint.
    struct Round {
        uint32_t id = 0;
        Stamp t0, t1, t1b, t2;
        bool complete() const { return t0.have && t2.have && t1.have && t1b.have; }
    };
    // A solved link: receiver refclk = sender refclk + offset_refclk + rate * (sender refclk - mid_refclk).
    struct LinkSolution {
        bool ok = false;
        double solved_at_refclk = 0.0;  // the sender chip's refclk at the newest round of the last solve
        double precision_ns = 0.0;      // residual_rms_ns / sqrt(kept): the offset estimate's own precision
        uint32_t dev_snd = 0, dev_rcv = 0;
        double offset_refclk = 0.0, rate = 0.0, mid_refclk = 0.0;
        double residual_rms_ns = 0.0;
        size_t rounds = 0, kept = 0, path_dropped = 0;
    };
    // A hardware round whose one-way delay inside the stamps sits this far from the window's median had a frame
    // delayed on one leg, and its offset is off by that same amount; the delay itself holds to 0.5 ns.
    static constexpr double kPathDevNs = 2.0;

    // Starts over on a capture's links; `ctx` must outlive the solver's use of it.
    void reset(const CaptureContext& ctx);
    // A link stamp: paired into its round, and once the round is complete the link is re-solved if its window is due.
    void on_stamp(const ClockSample& s);
    // Every link solved over whatever its window holds.
    void solve_final();
    const std::vector<LinkSolution>& solutions() const { return solved_; }
    // The complete rounds of link li in the order they completed, and how many wait for their other end.
    const std::vector<Round>& rounds(size_t li) const { return rounds_[li].rounds; }
    size_t pending(size_t li) const { return rounds_[li].pending.size(); }
    // Moves whenever a solution is accepted.
    uint64_t generation() const { return gen_; }
    // Samples ignored: an unknown kind, a core on no link, or a role that end does not stamp.
    uint64_t dropped() const { return dropped_; }
    // How many solved links share link li's chip pair.
    size_t pair_size(size_t li) const;
    // The tree over the pair solutions: every chip reachable from `root` onto its refclk. `used` marks the links
    // whose pair the tree took and that are that pair's only member: a loop through such a link closes to zero by
    // construction.
    std::map<uint32_t, RootXf> root_transforms(uint32_t root, std::vector<bool>* used) const;

    // A round in the refclk domain: each end's midpoint; the sender's round trip, the receiver's turnaround and the
    // one-way delay inside the stamps, in ns.
    static double mid_a_refclk(const Round& r);
    static double mid_b_refclk(const Round& r);
    static double rtt_ns(const Round& r) {
        return (static_cast<double>(r.t2.units) - static_cast<double>(r.t0.units)) * kRefclkPerStampUnit * kNsPerRefclk;
    }
    static double turn_ns(const Round& r) {
        return (static_cast<double>(r.t1b.units) - static_cast<double>(r.t1.units)) * kRefclkPerStampUnit *
               kNsPerRefclk;
    }
    static double path_ns(const Round& r) { return 0.5 * (rtt_ns(r) - turn_ns(r)); }
    static double path_median(const std::vector<Round>& rounds, size_t begin, size_t n);

private:
    // A link's rounds: the complete ones in the order they completed, the rest waiting for their other end. A round
    // whose other end never reports (no ring room there, a lapped consumer) is evicted once kPendingMax newer rounds
    // are waiting; nothing behind it shifts.
    struct LinkRounds {
        std::vector<Round> rounds;
        std::map<uint32_t, Round> pending;
    };
    // One round in the refclk domain: the sender's midpoint and the receiver's minus it.
    struct RoundPoint {
        double mid_refclk, off_refclk;
    };
    void try_solve_links(bool final);
    // Whether the solution was accepted into `out`.
    bool solve_link(const CaptureContext::Link& L, std::vector<RoundPoint> pts, LinkSolution& out) const;
    // One solution per chip pair: a pair's solved links combined by precision-weighted means of their rates and of
    // their offsets at a common midpoint, so parallel links average their path asymmetries. `members` gets the
    // links behind each.
    std::vector<LinkSolution> pair_solutions(std::vector<std::vector<size_t>>* members) const;

    const CaptureContext* ctx_ = nullptr;
    std::vector<LinkRounds> rounds_;  // per ctx_->links index
    // (device index, decoder core index) -> the link the core stamps for, and whether as its sender.
    std::map<std::pair<uint32_t, uint32_t>, std::pair<size_t, bool>> side_of_;
    std::vector<LinkSolution> solved_;  // per ctx_->links index
    uint64_t gen_ = 0;
    uint64_t dropped_ = 0;
    // Refclk ticks per stamp unit: the kernels report a round's stamp averages in kLinkSyncStampUnitsPerNs per ns.
    static constexpr double kRefclkPerStampUnit = 1.0 / (kernel_profiler::kLinkSyncStampUnitsPerNs * kNsPerRefclk);
    // The link solve's window in the sender chip's refclk, re-solved every half window. Two chips' crystals hold a
    // line to ~0.4 ns over 250 ms and their rate moves a few ppb from one such window to the next (measured on the
    // 8-chip runs), so a fit extrapolated half a window past its end stays within ~0.4 ns rms and doubles that at
    // 500 ms. The rounds inside the window only average the fit's own noise, ~0.1 ns at 100 Hz.
    static constexpr double kLinkWindowTicks = 12'500'000.0;  // 250 ms
    static constexpr double kFirstSolveTicks = 10'000'000.0;  // 200 ms of rounds before the first live solution
    static constexpr size_t kMinSolveRounds = 8;
    static constexpr size_t kPendingMax = 4096;
};

// Each chip's published placement series: its clock model's segments and its root transform turned into the
// PlacementMap's nodes. Nodes are frozen once published (consumers have placed records against them), so a publish
// only appends beyond them: a node where the map bends (a segment boundary, the first sample) and the open
// segment's frontier when it has left the frozen tangent.
class SeriesPublisher {
public:
    // A placement node: at eth wall tick H the chip sits at root refclk tick `root`; r is the chip's own refclk it
    // was placed at, tangent the run's rate on the root (root refclk ticks per wall tick), the map past the newest
    // node.
    struct Node {
        double H, root, r, tangent;
    };
    // One chip's series. `cover_H` is how far the newest node's tangent has been confirmed by the fit, `knots` how
    // many run boundaries have their nodes, `last_r` the refclk of the newest node or cover.
    struct Series {
        std::vector<Node> nodes;
        size_t knots = 0;
        double last_r = -1.0;
        double cover_H = -1.0;
        double cover_r = -1.0;
        size_t dropped = 0;   // nodes refused: behind the frozen series, or not a correction below one ns per ns
        size_t extended = 0;  // frontier samples that only advanced the cover
    };

    explicit SeriesPublisher(PlacementMap& map) : map_(map) {}
    void reset() { series_.clear(); }
    // Publishes one chip's series from its fit and root transform as they stand; true when the chip's cover moved.
    bool publish(uint32_t dev, uint32_t chip, const LocalClockModel& fit, const RootXf& xf);
    // A chip's series, null before its first publish.
    const Series* series(uint32_t dev) const {
        const auto it = series_.find(dev);
        return it == series_.end() ? nullptr : &it->second;
    }
    bool has_nodes(uint32_t dev) const {
        const Series* s = series(dev);
        return s != nullptr && !s->nodes.empty();
    }

private:
    // The nodes a chip's fit yields beyond a series: those at run boundaries, and the open run's frontier.
    struct Fresh {
        std::vector<Node> knots;
        std::optional<Node> frontier;
        size_t knots_after = 0;  // run boundaries consumed once the knots are placed
    };
    Fresh fresh_nodes(const Series& s, const LocalClockModel& fit, const RootXf& xf) const;
    // Appends the knots and, if the frontier left the newest tangent by more than kFreezeNs, freezes the tangent where
    // it stood and appends the frontier; otherwise advances the cover. True when the cover moved.
    bool advance(Series& s, uint32_t chip, Fresh fresh);
    void freeze_append(Series& s, uint32_t chip, const Node& n);
    void push_node(Series& s, uint32_t chip, const Node& n);

    PlacementMap& map_;
    std::map<uint32_t, Series> series_;  // device index -> series
    // A frontier within this much of the newest tangent extends the cover instead of freezing a node; the published
    // map then sits within it of the fit's own estimate. A mature run's estimate moves ~0.02 ns per keepalive sample,
    // so it adds a node every few hundred ms; a young run adds one per burst sample for its first ms.
    static constexpr double kFreezeNs = 0.25;
};

// Device<->device sync from the PP_CLOCK samples the idle-eth pushers carry, and the correction it publishes.
//
// LOCAL points feed one LocalClockModel per device. LINK samples feed the LinkSolver, refclk against refclk, so DVFS
// on either wall clock cannot enter the link solve. From those the SeriesPublisher publishes, per chip, a placement
// series in the PlacementMap every record is placed through: the chip's eth wall tick onto the root chip's refclk,
// which the host probe's series takes onto the host. Published incrementally for live sinks, finally at capture end.
// Driven from the Service's sync thread, which decodes the eth pushers' streams and hands it every clock sample in
// order; a capture is on_attach, the samples, on_capture_end. The unit test drives it the same way.
class SyncEngine {
public:
    SyncEngine() : series_(map_) {}
    void on_attach(const CaptureContext& ctx);
    void on_clock(const ClockSample& s);
    void on_capture_end(const CaptureContext& ctx);
    // The placement map the service places records with: this engine writes its chip series, the host probe its
    // host series.
    PlacementMap& map() { return map_; }
    const PlacementMap& map() const { return map_; }

private:
    using Round = LinkSolver::Round;
    // The fleet timeline's root: the chip the host probe reads, fixed for the capture.
    uint32_t root_dev() const { return ctx_.root_dev; }
    // Publishes one chip's series from its fit as it stands; true when the chip's cover moved.
    bool publish_dev(uint32_t dev);
    void publish_all();
    // The capture-end report: each chip's clock model, every link's solution, the loop closures; the sync error per
    // round and each chip's AICLK as plots.
    void log_summary() const;
    void log_clock_models() const;
    void log_link_solutions() const;
    void log_loop_closures() const;
    void publish_error_plots();
    void publish_clock_plots();
    // One link's rounds placed through the final map: per round the error (as a plot point), the placement's terms,
    // the raw offset, the stamps' own residual and the path figures.
    struct LinkErrors;
    LinkErrors link_errors(size_t li) const;
    static void stamp_residuals(LinkErrors& e);
    void log_link_stats(const CaptureContext::Link& L, const LinkErrors& e, size_t rounds) const;
    void log_worst_rounds(const CaptureContext::Link& L, const LinkErrors& e) const;
    void write_err_csv(const CaptureContext::Link& L, const LinkErrors& e) const;
    // The receiver's stamp and the sender's round midpoint placed on the root's refclk as the sink places records
    // from each chip's eth core, and their difference in ns; tsc_a is the sender's host placement, the plots'
    // abscissa. False when a chip has no fitted run or no node to place a stamp with.
    struct RoundTerms {
        double wall_a = 0, wall_b = 0, root_a = 0, root_b = 0;
    };
    bool round_error(
        const CaptureContext::Link& L, const Round& r, int64_t& tsc_a, double& err, RoundTerms* terms = nullptr) const;
    // A (host TSC tick, value) series as a Tracy plot (each chip's AICLK, the sync error per link); the TSC is the
    // timer stamp Tracy places it by. Emitted whenever Tracy is compiled in: the record sink is what an env var
    // turns on, the plots ride with the profiler.
    struct PlotPoint {
        int64_t tsc = 0;
        double value = 0.0;
    };
    void plot(const std::string& name, const std::vector<PlotPoint>& points);

    PlacementMap map_;
    CaptureContext ctx_;
    std::map<uint32_t, LocalClockModel> local_;  // device index -> local fit
    LinkSolver links_;
    SeriesPublisher series_;
    // The composed root transforms as of the newest accepted link solution.
    std::map<uint32_t, RootXf> to_root_;
    uint64_t to_root_gen_ = ~0ull;
    std::unordered_set<std::string> plot_names_;  // Tracy keys a plot by its name's address, for the process
};

}  // namespace tt::tt_metal::streaming_profiler
