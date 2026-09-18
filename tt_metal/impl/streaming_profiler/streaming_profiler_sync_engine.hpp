// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <string>
#include <unordered_set>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"

namespace tt::tt_metal::streaming_profiler {

// One record of the sync stream (hostdev/streaming_profiler_common.h, kSyncRecordWords) as the engine takes it: the
// device and roster core that wrote it, its kind and role, the round a link stamp names, the reading, and the wall
// clock at it.
struct ClockSample {
    uint32_t dev;
    uint32_t core;
    uint32_t kind;
    uint32_t round;
    uint32_t role;
    uint64_t value;
    uint64_t ts;
    uint64_t ref = 0;  // link records: the refclk read with ts
    uint32_t spins = 0;  // link records: the bracketed read's spins to the caught refclk update, plus one
};

// The chip's AICLK wall clock against its refclk, as its idle-eth pusher models it (eth_clock_pusher.cpp): one
// segment per PLL multiple, each a line the pusher sends POINTS of -- (refclk r, the line's wall at r) with the
// multiple k8 and the count of samples behind the line in the record's round word. The newest point of the open
// segment stands for it; a CLOSE point ends a segment at its last on-line sample; a new multiple, or a point after
// a close, opens the next; and consecutive segments meet where their lines cross.
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
    // The pusher's own check of each line against its samples: the largest residual any point reported, in wall
    // ticks, and how many points reported one beyond kResidWarnTicks.
    static constexpr double kResidWarnTicks = 8.0;
    double max_resid_ticks = 0.0;
    uint64_t resid_warn_points = 0;
    // Each line point's residual as (refclk, wall ticks): the model's error bound over the samples between the
    // previous point and it, in refclk order.
    std::vector<std::pair<double, double>> resid;
    // The bound in force at refclk r: the first point at or after it, the last point's past the end.
    double resid_at(double r) const {
        if (resid.empty()) {
            return 0.0;
        }
        const auto it = std::lower_bound(
            resid.begin(), resid.end(), r, [](const std::pair<double, double>& p, double x) { return p.first < x; });
        return it == resid.end() ? resid.back().second : it->second;
    }
    // Instants of the wall clock on no line (k8 0): the pusher's own samples while it acquires a slope across a
    // glide, in refclk order. They bend the map through the seam they fall in.
    std::vector<std::pair<double, double>> raw;  // (refclk, wall)
    uint64_t points = 0;    // points and closes received
    uint64_t transitions = 0;  // segments after the first

    // A point of the open segment's line, or the segment's close. An older point than the newest is superseded.
    void add_point(uint64_t refclk, uint64_t wall, uint32_t k8, uint32_t n, bool close, uint32_t resid8 = 0) {
        points++;
        const double r = static_cast<double>(refclk), w = static_cast<double>(wall);
        const double resid = resid8 / 8.0;
        max_resid_ticks = std::max(max_resid_ticks, resid);
        resid_warn_points += resid > kResidWarnTicks;
        if (k8 != 0 && (this->resid.empty() || r > this->resid.back().first)) {
            this->resid.emplace_back(r, resid);
        }
        if (k8 == 0) {
            if (raw.empty() || r > raw.back().first) {
                raw.emplace_back(r, w);
            }
            return;
        }
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
    // The raw instants inside a seam (r_lo, r_hi), in order.
    std::vector<std::pair<double, double>> raw_between(double r_lo, double r_hi) const {
        std::vector<std::pair<double, double>> out;
        for (const auto& p : raw) {
            if (p.first > r_lo && p.first < r_hi) {
                out.push_back(p);
            }
        }
        return out;
    }
    // The newest instant the samples reach when the model ends inside a transition (the last segment closed, or raw
    // instants past it): nothing later is measured. No value while the last segment is open: its line stands on.
    std::optional<double> modelled_until() const {
        if (runs.empty()) {
            return 0.0;
        }
        const double raw_r = raw.empty() ? 0.0 : raw.back().first;
        if (!runs.back().closed && runs.back().r_last >= raw_r) {
            return std::nullopt;
        }
        return std::max(runs.back().r_last, raw_r);
    }
    // The segment holding refclk r: the last one starting at or before it (the first, for anything earlier).
    const Run& run_at(double r) const {
        auto it = std::upper_bound(runs.begin(), runs.end(), r, [](double x, const Run& a) { return x < a.r_first; });
        return it == runs.begin() ? runs.front() : *(it - 1);
    }
    // A raw instant this far off the line a step would put it on says the seam is a ramp. The pusher's raw instants
    // hold a chord within three ticks of every sample across the seam (its cone), so a step's two lines must do the
    // same or the instants themselves place the seam.
    static constexpr double kStepFitTicks = 3.0;
    // Where consecutive segments hand over: their lines' intersection, when it lies within kKnotSlackTicks of the
    // seam and the two lines account for every raw instant recorded across it, i.e. the seam is a step. No value (a
    // ramp, parallel lines, or a crossing far from the seam): the seam is bridged from a.r_last to b.r_first through
    // its raw instants.
    std::optional<double> knot(const Run& a, const Run& b) const {
        const double ds = a.slope() - b.slope();
        if (!(std::abs(ds) > 1e-9)) {
            return std::nullopt;
        }
        const double r_x = (b.w_last - b.slope() * b.r_last - a.w_last + a.slope() * a.r_last) / ds;
        if (!(std::isfinite(r_x) && r_x >= a.r_last - kKnotSlackTicks && r_x <= b.r_first + kKnotSlackTicks)) {
            return std::nullopt;
        }
        for (const auto& [rr, ww] : raw_between(a.r_last, b.r_first)) {
            const Run& on = rr < r_x ? a : b;
            if (std::abs(ww - on.wall_of_refclk(rr)) > kStepFitTicks) {
                return std::nullopt;
            }
        }
        return r_x;
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
                // A bridged seam: straight between the raw instants inside it, the two lines' ends included.
                double r0 = run->r_last, w0 = run->wall_of_refclk(run->r_last);
                for (const auto& [rr, ww] : raw_between(run->r_last, next.r_first)) {
                    if (r < rr) {
                        return w0 + (ww - w0) * (r - r0) / (rr - r0);
                    }
                    r0 = rr;
                    w0 = ww;
                }
                const double w1 = next.wall_of_refclk(next.r_first);
                return w0 + (w1 - w0) * (r - r0) / (next.r_first - r0);
            }
        } else if (
            idx + 1 == runs.size() && run->closed && r > run->r_last && usable(*run) && !raw.empty() &&
            raw.back().first > run->r_last) {
            // The seam still open: through the raw instants, then along the last two.
            double r0 = run->r_last, w0 = run->wall_of_refclk(run->r_last);
            for (const auto& [rr, ww] : raw_between(run->r_last, std::numeric_limits<double>::infinity())) {
                if (r < rr) {
                    return w0 + (ww - w0) * (r - r0) / (rr - r0);
                }
                if (rr == raw.back().first) {
                    return ww + (ww - w0) * (r - rr) / (rr - r0);
                }
                r0 = rr;
                w0 = ww;
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
        uint64_t wall = 0, ref = 0;  // the end's AICLK wall clock and refclk read together when it recorded this
        uint32_t spins = 0;
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

// One frozen node of a placement series: at `at` the placement is `value`, linear to the next node, and past the
// newest node along `tangent` (d value / d at) as far as the series' cover reaches.
template <typename Key>
struct ClockNode {
    Key at{};
    double value = 0.0;
    double tangent = 0.0;
};
// A chip's series: its eth wall tick -> the root chip's refclk tick, from the link solutions and the local fits alone,
// so two chips' records at one instant differ by nothing the host contributes. Worker lanes reach the eth wall
// domain through their chip's constant tile offset, so one series places every record of a chip.
using SyncNode = ClockNode<int64_t>;
// The fleet's one host series: the root's refclk tick -> host TSC tick, from the host probe.
using HostNode = ClockNode<double>;

// The sync engine's clock map: one series per chip (its eth wall tick -> the root chip's refclk tick) and the
// host series (the root's refclk tick -> host TSC tick). The sync engine writes the chip series and the host probe
// the host series; the service's consumer threads read them to place records. Each series is append-only within a
// capture, strictly increasing in its key, and keeps its newest kSeriesNodes: a record before the oldest kept node
// converts on that node's tangent. Before a chip's first node, or the host's, its records have no place on the host
// timeline.
//
// Reads never lock and never block the writer (IndexedRing). A reader keeps a thread-local cursor on the segment it
// last converted in and converts without touching shared state until a record leaves the segment. A series' cover
// is the key up to which the newest node's tangent has been confirmed: a record at or before it converts against
// frozen data on both sides.
class ClockMap {
public:
    static constexpr uint32_t kMaxChips = 256;
    // Nodes a series keeps (32 MB at most); the oldest go as newer ones arrive. Nodes come per local clock step and
    // per host burst, so this spans hours of a capture and any consumer's lag behind the sync.
    static constexpr uint32_t kSeriesNodes = 1u << 20;

    ClockMap();
    ~ClockMap();
    ClockMap(const ClockMap&) = delete;
    ClockMap& operator=(const ClockMap&) = delete;

    // Appends a node past every earlier one (a node at the last node's key is dropped) and moves the cover to it.
    void append(uint32_t chip_id, SyncNode node);
    // The newest node's tangent holds up to cover_ticks; the cover never moves back.
    void extend(uint32_t chip_id, int64_t cover_ticks);
    // The series is complete for the capture: every later instant converts on the newest tangent.
    void finish(uint32_t chip_id);
    // Empties a chip's series for a new capture.
    void clear(uint32_t chip_id);
    void append_host(HostNode node);

    // The root refclk tick of a chip's eth wall tick; 0 before the chip's first node.
    double lookup_root(uint32_t chip_id, int64_t wall) const noexcept;
    // Wall tick `wall` of chip `chip_id` on host_clock (tenths of a ns of the TSC): the chip series and the host
    // series composed into one line per segment pair, one multiply-add per record while a batch stays inside it. 0
    // when nothing places the tick yet.
    int64_t place_host(uint32_t chip_id, int64_t wall) const noexcept;
    // The host TSC tick of a root refclk tick; 0 before the host's first node.
    double host_tsc(double root) const noexcept;
    size_t host_published() const noexcept;
    // The wall tick the chip's series covers: INT64_MIN before its first node, INT64_MAX once finished.
    int64_t cover_ticks(uint32_t chip_id) const noexcept;
    // Moves whenever any chip's cover does, so a consumer holding batches re-reads covers only then.
    uint64_t cover_generation() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Each chip's published placement series: its clock model's segments and its root transform turned into the
// ClockMap's nodes. Nodes are frozen once published (consumers have placed records against them), so a publish
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

    explicit SeriesPublisher(ClockMap& map) : map_(map) {}
    void reset() { series_.clear(); }
    // Publishes one chip's series from its fit and root transform as they stand; true when the chip's cover moved.
    bool publish(uint32_t dev, uint32_t chip, const LocalClockModel& fit, const RootXf& xf, bool final = false);
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
    Fresh fresh_nodes(const Series& s, const LocalClockModel& fit, const RootXf& xf, bool final) const;
    // Appends the knots and, if the frontier left the newest tangent by more than kFreezeNs, freezes the tangent where
    // it stood and appends the frontier; otherwise advances the cover. True when the cover moved.
    bool advance(Series& s, uint32_t chip, Fresh fresh);
    void freeze_append(Series& s, uint32_t chip, const Node& n);
    void push_node(Series& s, uint32_t chip, const Node& n);

    ClockMap& map_;
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
// series in the ClockMap every record is placed through: the chip's eth wall tick onto the root chip's refclk,
// which the host probe's series takes onto the host. Published incrementally for live sinks, finally at capture end.
// Driven from the Service's sync thread, which decodes the eth pushers' streams and hands it every clock sample in
// order; a capture is on_attach, the samples, on_capture_end. The unit test drives it the same way.
class SyncEngine {
public:
    SyncEngine() : series_(map_) {}
    void on_attach(const CaptureContext& ctx);
    void on_clock(const ClockSample& s);
    void on_capture_end(const CaptureContext& ctx);
    // The clock map the service places records with: this engine writes its chip series, the host probe its
    // host series.
    ClockMap& map() { return map_; }
    const ClockMap& map() const { return map_; }

private:
    using Round = LinkSolver::Round;
    // The fleet timeline's root: the chip the host probe reads, fixed for the capture.
    uint32_t root_dev() const { return ctx_.root_dev; }
    // Publishes one chip's series from its fit as it stands; true when the chip's cover moved.
    bool publish_dev(uint32_t dev, bool final = false);
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
    LinkErrors link_errors(size_t li, bool anchored) const;
    static void stamp_residuals(LinkErrors& e);
    void log_link_stats(const CaptureContext::Link& L, const LinkErrors& e, size_t rounds) const;
    void log_worst_rounds(const CaptureContext::Link& L, const LinkErrors& e) const;
    void write_err_csv(const CaptureContext::Link& L, const LinkErrors& e) const;
    // The receiver's stamp and the sender's round midpoint placed on the root's refclk as the sink places records
    // from each chip's eth core, and their difference in ns; tsc_a is the sender's host placement, the plots'
    // abscissa. False when a chip has no fitted run or no node to place a stamp with.
    struct RoundTerms {
        double wall_a = 0, wall_b = 0, root_a = 0, root_b = 0;
        double res_a = 0, res_b = 0;  // each end's recorded pair against its chip's model, ns
        uint32_t spins_a = 0, spins_b = 0;
    };
    // `anchored`: each end's wall from the (wall, refclk) pair its record carries, the wall clock at one refclk
    // update; otherwise from the local model, which then cancels and the error is the links' and the map's.
    bool round_error(
        const CaptureContext::Link& L,
        const Round& r,
        bool anchored,
        int64_t& tsc_a,
        double& err,
        RoundTerms* terms = nullptr) const;
    // A (host TSC tick, value) series as a Tracy plot (each chip's AICLK, the sync error per link); the TSC is the
    // timer stamp Tracy places it by. Emitted whenever Tracy is compiled in: the record sink is what an env var
    // turns on, the plots ride with the profiler.
    struct PlotPoint {
        int64_t tsc = 0;
        double value = 0.0;
    };
    void plot(const std::string& name, const std::vector<PlotPoint>& points);

    ClockMap map_;
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
