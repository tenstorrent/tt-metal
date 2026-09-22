// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
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
    uint64_t ref = 0;    // link records: the refclk read with ts
    uint32_t spins = 0;  // link records: the bracketed read's spins to the caught refclk update, plus one
};

// A chip's clock as the pusher describes it: instants (its refclk tick, its eth wall tick) in refclk order, the wall
// linear between consecutive ones. On a steady line the pusher sends a point every kEthPointUs with the line's exact
// k/8 slope and the residual bound of the samples behind it; inside a transition it sends the samples that keep the
// chord through them within kRawEpsTicks of every sample (k8 0), so the map bends through a glide as the clock did.
// Records are placed only up to the newest instant, on the chord to it.
class LocalClockModel {
public:
    static constexpr double kRefclkHz = kernel_profiler::kEthRefclkHz;
    struct Instant {
        double r = 0.0, w = 0.0;
        uint32_t k8 = 0;  // the line's slope in eighths of a wall tick per refclk tick at a point; 0 at a raw instant
        bool close = false;  // a line's last instant
    };
    std::vector<Instant> pts;  // in refclk order
    uint64_t points = 0;       // instants received, those behind the frontier included
    uint64_t transitions = 0;  // closes
    // The pusher's own check of each line against its samples: the largest residual any point reported, in wall
    // ticks, and how many points reported one beyond kResidWarnTicks.
    static constexpr double kResidWarnTicks = 8.0;
    double max_resid_ticks = 0.0;
    uint64_t resid_warn_points = 0;
    // Each point's residual as (refclk, wall ticks): the line's error bound over the samples between the previous
    // point and it, in refclk order.
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

    void add_point(uint64_t refclk, uint64_t wall8, uint32_t k8, uint32_t, bool close, uint32_t resid8 = 0) {
        points++;
        const double r = static_cast<double>(refclk), w = static_cast<double>(wall8) / 8.0;
        const double res = resid8 / 8.0;
        max_resid_ticks = std::max(max_resid_ticks, res);
        resid_warn_points += res > kResidWarnTicks;
        if (k8 != 0 && (resid.empty() || r > resid.back().first)) {
            resid.emplace_back(r, res);
        }
        // A new line's first points sit at its lock, behind the seam's last instants: nothing behind the frontier
        // is placed again.
        if (!pts.empty() && (r <= pts.back().r || w <= pts.back().w)) {
            return;
        }
        transitions += close;
        pts.push_back(Instant{r, w, k8, close});
    }
    // The newest instant's refclk: nothing later is known.
    double frontier() const { return pts.empty() ? 0.0 : pts.back().r; }
    // Between the raw instants pts[i-1] and pts[i] the wall clock's rate runs linearly from the rate at the first
    // to the rate at the second (each from its neighbours) when the pair is close: a glide's instants are ~1 us
    // apart and a chord across them misses the bend by (rate change per tick) x gap^2 / 8, up to 5 cycles. Across a
    // longer gap -- a straight stretch the cone held through, or a hole -- the same construction scales rate noise
    // by gap^2 (a 64 ms pair bowed 55 cycles on a 0.001 cycle/tick difference) and the chord stands.
    static constexpr double kBendMaxTicks = 250.0;  // 5 us
    bool curved(size_t i) const {
        return i >= 2 && i + 1 < pts.size() && pts[i - 1].k8 == 0 && pts[i].k8 == 0 &&
               pts[i].r - pts[i - 1].r <= kBendMaxTicks;
    }
    // The wall tick at refclk r: on the curve between the instants around it (see curved), along the last two
    // past the newest, and 0 before the first, where nothing places.
    double wall_at(double r) const {
        if (pts.empty() || r < pts.front().r) {
            return 0.0;
        }
        if (pts.size() == 1) {
            return pts[0].w + (r - pts[0].r) * pts[0].k8 / 8.0;
        }
        auto hi = std::lower_bound(pts.begin(), pts.end(), r, [](const Instant& p, double x) { return p.r < x; });
        if (hi == pts.end()) {
            hi = pts.end() - 1;
        }
        if (hi == pts.begin()) {
            ++hi;
        }
        const size_t i = static_cast<size_t>(hi - pts.begin());
        const Instant& a = pts[i - 1];
        const Instant& b = pts[i];
        const double dr = b.r - a.r, x = r - a.r;
        const double chord = (b.w - a.w) / dr;
        if (!curved(i)) {
            return a.w + chord * x;
        }
        const Instant& p = pts[i - 2];
        const Instant& n = pts[i + 1];
        const double rate_a = (b.w - p.w) / (b.r - p.r);
        const double rate_b = (n.w - a.w) / (n.r - a.r);
        const double kappa = (rate_b - rate_a) / (2.0 * dr);
        return a.w + chord * x + kappa * x * (x - dr);
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
    // A link stamp: paired into its round, and once the round is complete the link is re-solved over its window.
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
    // Every chip the root reaches over solved links, onto the root's refclk, from all the links at once (see the
    // definition). `weights`, per link, gets the robust weight each ended with: 1 on the mesh, 0 left out.
    std::map<uint32_t, RootXf> root_transforms(uint32_t root, std::vector<double>* weights) const;

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

    const CaptureContext* ctx_ = nullptr;
    std::vector<LinkRounds> rounds_;  // per ctx_->links index
    // (device index, decoder core index) -> the link the core stamps for, and whether as its sender.
    std::map<std::pair<uint32_t, uint32_t>, std::pair<size_t, bool>> side_of_;
    std::vector<LinkSolution> solved_;  // per ctx_->links index
    uint64_t gen_ = 0;
    uint64_t dropped_ = 0;
    // Refclk ticks per stamp unit: the kernels report a round's stamp averages in kLinkSyncStampUnitsPerNs per ns.
    static constexpr double kRefclkPerStampUnit = 1.0 / (kernel_profiler::kLinkSyncStampUnitsPerNs * kNsPerRefclk);
    // The link solve's window in the sender chip's refclk, re-solved as each round completes so the live solution
    // ends at the newest round. Two chips' crystals hold a line to ~0.4 ns over this long (measured on the 8-chip
    // runs); the ~25 rounds inside average the stamps' 0.3 ns to under 0.1 ns, and a longer window measured no better.
    static constexpr double kLinkWindowTicks = 12'500'000.0;  // 250 ms
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
    // Nodes a series keeps before the oldest go: 2 GB at most per chip, allocated as nodes arrive (32 B each). Under
    // heavy DVFS a chip takes ~300 nodes a second, so a series holds two and a half days of it; the sync report
    // counts any round that fell off the front.
    static constexpr uint32_t kSeriesNodes = 1u << 26;

    explicit ClockMap(uint32_t series_nodes = kSeriesNodes);
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
    // The wall tick of the chip's oldest kept node, INT64_MIN while it has none: an earlier instant converts on that
    // node's tangent, which is not a placement.
    int64_t oldest_at(uint32_t chip_id) const noexcept;
    // Nodes each series keeps, whole chunks of the ring.
    uint32_t series_nodes() const noexcept;
    // Moves whenever any chip's cover does, so a consumer holding batches re-reads covers only then.
    uint64_t cover_generation() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Each chip's published placement series: a node per instant of its clock model, the instant's eth wall tick and
// the root's refclk tick there through the chip's refclk and the solved links, the chord to the next instant as its
// tangent. Nodes are frozen once published (consumers have placed records against them), so a publish only appends
// beyond them; records are placed only up to the newest node, on the chord to it.
class SeriesPublisher {
public:
    // A placement node: at eth wall tick H the chip sits at root refclk tick `root`; r is the chip's own refclk it
    // was placed at, tangent the rate on the root (root refclk ticks per wall tick) past it.
    struct Node {
        double H, root, r, tangent;
    };
    // One chip's series; `last_r` is the refclk of the newest instant published.
    struct Series {
        std::vector<Node> nodes;
        double last_r = -1.0;
        size_t dropped = 0;  // nodes refused: behind the frozen series, or not a rate
    };

    explicit SeriesPublisher(ClockMap& map) : map_(map) {}
    void reset() { series_.clear(); }
    // Publishes one chip's instants beyond its series' end, from its model and root transform as they stand; true
    // when a node was added. A raw instant waits for the one after it (the curve across the gap before it needs
    // the rate at both ends); `final` publishes the newest regardless.
    bool publish(uint32_t dev, uint32_t chip, const LocalClockModel& fit, const RootXf& xf, bool final);
    // Where the model curves between two instants (LocalClockModel::curved), kBendNodes nodes sampled from its
    // wall_at go between them, so consumers interpolate linearly as ever and follow the curve to a sixteenth of
    // the chord's error.
    static constexpr uint32_t kBendNodes = 3;
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
    void append_node(Series& s, uint32_t chip, const Node& n);
    void push_node(Series& s, uint32_t chip, const Node& n);

    ClockMap& map_;
    std::map<uint32_t, Series> series_;  // device index -> series
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
    explicit SyncEngine(uint32_t series_nodes = ClockMap::kSeriesNodes) : map_(series_nodes), series_(map_) {}
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
    void write_model_csv() const;
    // The receiver's stamp and the sender's round midpoint placed on the root's refclk as the sink places records
    // from each chip's eth core, and their difference in ns; tsc_a is the sender's host placement, the plots'
    // abscissa. False when a chip has no fitted run or no node to place a stamp with.
    struct RoundTerms {
        double wall_a = 0, wall_b = 0, root_a = 0, root_b = 0;
        double res_a = 0, res_b = 0;  // each end's recorded pair against its chip's model, ns
        double ref_a = 0, ref_b = 0, wraw_a = 0, wraw_b = 0;  // the pairs themselves
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
