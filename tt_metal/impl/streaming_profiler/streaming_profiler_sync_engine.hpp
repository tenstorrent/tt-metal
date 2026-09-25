// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
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

// A LINK, ANCHOR or ANCHOR_HIST record of the sync stream (hostdev/streaming_profiler_common.h, kSyncRecordWords)
// split into its fields: the device and roster core that wrote it, its kind and role, the round a link stamp names,
// the reading, and the wall clock at it.
struct ClockSample {
    uint32_t dev;
    uint32_t core;
    uint32_t kind;
    uint32_t round;
    uint32_t role;
    uint64_t value;
    uint64_t ts;
    uint64_t ref = 0;  // anchor records: the refclk read with ts
};

// A chip's clock as the pusher describes it: instants (its refclk tick, its eth wall tick) in refclk order, the wall
// linear between consecutive ones. An instant is the centroid of a window of samples at one FBDIV (k8 its slope) or,
// where FBDIV changed, a single sample (k8 0). Records are placed only up to the newest instant, on the chord to it.
class LocalClockModel {
public:
    static constexpr double kRefclkHz = kernel_profiler::kEthRefclkHz;
    struct Instant {
        int64_t r = 0;
        int64_t w8 = 0;   // the wall tick in eighths
        uint32_t k8 = 0;  // the line's slope in eighths of a wall tick per refclk tick at a point; 0 at a raw instant
        double wall() const { return static_cast<double>(w8) / 8.0; }
        int64_t wall_tick() const { return (w8 + 4) >> 3; }
    };
    std::deque<Instant> pts;  // in refclk order; a deque, so growth never copies on the sync thread
    void add_point(uint64_t refclk, uint64_t wall8, uint32_t k8) {
        const auto r = static_cast<int64_t>(refclk), w8 = static_cast<int64_t>(wall8);
        if (!pts.empty() && (r <= pts.back().r || w8 <= pts.back().w8)) {
            return;
        }
        pts.push_back(Instant{r, w8, k8});
    }
    // Whether instant i is the last of a line whose slope the next instant does not share: an FBDIV change.
    bool ends_line(size_t i) const { return pts[i].k8 != 0 && i + 1 < pts.size() && pts[i + 1].k8 != pts[i].k8; }
    // The newest instant's refclk: nothing later is known.
    int64_t frontier() const { return pts.empty() ? 0 : pts.back().r; }
    // The wall tick at refclk r: on the chord between the instants around it, along the last two past the newest,
    // and 0 before the first, where nothing places.
    double wall_at(double r) const {
        if (pts.empty() || r < static_cast<double>(pts.front().r)) {
            return 0.0;
        }
        if (pts.size() == 1) {
            return pts[0].wall() + (r - static_cast<double>(pts[0].r)) * pts[0].k8 / 8.0;
        }
        const auto key = static_cast<int64_t>(std::ceil(r));
        auto hi = std::lower_bound(pts.begin(), pts.end(), key, [](const Instant& p, int64_t x) { return p.r < x; });
        if (hi == pts.end()) {
            hi = pts.end() - 1;
        }
        if (hi == pts.begin()) {
            ++hi;
        }
        const size_t i = static_cast<size_t>(hi - pts.begin());
        const Instant& a = pts[i - 1];
        const Instant& b = pts[i];
        const double dr = static_cast<double>(b.r - a.r), x = r - static_cast<double>(a.r);
        return a.wall() + static_cast<double>(b.w8 - a.w8) / 8.0 / dr * x;
    }
};

constexpr double kNsPerRefclk = 1e9 / kernel_profiler::kEthRefclkHz;

// A signed error distribution in ns, in kBinsPerNs bins over +-kRangeNs; beyond the range only counted and the worst
// kept.
struct ErrorHistogram {
    static constexpr int kBinsPerNs = 16;
    static constexpr int kRangeNs = 256;
    static constexpr int kBins = 2 * kRangeNs * kBinsPerNs;
    std::vector<double> bins = std::vector<double>(kBins, 0.0);
    double n = 0.0, sum = 0.0, sumsq = 0.0, worst = 0.0, beyond = 0.0;

    static int bin_of(double ns) { return static_cast<int>(std::floor(ns * kBinsPerNs)) + kRangeNs * kBinsPerNs; }
    static double ns_of(int b) { return (b - kRangeNs * kBinsPerNs + 0.5) / kBinsPerNs; }
    void add(double ns, double weight = 1.0) {
        n += weight;
        sum += weight * ns;
        sumsq += weight * ns * ns;
        worst = std::max(worst, std::abs(ns));
        const int b = bin_of(ns);
        if (b < 0 || b >= kBins) {
            beyond += weight;
            return;
        }
        bins[b] += weight;
    }
    double mean() const { return n > 0.0 ? sum / n : 0.0; }
    double rms() const { return n > 0.0 ? std::sqrt(sumsq / n) : 0.0; }
    // The q-quantile of |error|, those beyond the range counted at the top.
    double abs_quantile(double q) const;
    // The distribution of x + y (sign -1: x - y) for independent x ~ *this and y ~ other, normalised to one.
    ErrorHistogram convolve(const ErrorHistogram& other, int sign) const;
};

// A chip's model audited against anchors: the chip's eth wall clock read at a refclk update by a core that builds no
// model (the drainer), each held until the model is final past it, then its distance from the model in ns. The drainer
// checks most of them itself and sends its histogram at stop (add_drainer_audit); those it sends are checked here. The
// refclk reaches the eth cores at different times (up to ~1 ns earlier at the die centre than at its edges), so the
// audit carries the drainer's and the pusher's difference.
struct AnchorAudit {
    struct Pending {
        int64_t r, w;
    };
    std::deque<Pending> pending;
    ErrorHistogram err;
    double worst_r = 0.0, worst_ns = 0.0;
    uint64_t before_model = 0;  // anchors before the model's first instant: nothing places them
    uint64_t past_model = 0;    // anchors after its last, at capture end

    // Checks every pending anchor the model is final past; `final` checks the rest up to its frontier.
    void settle(const LocalClockModel& m, bool final);
    // One record of the drainer's own audit of the anchors it did not send (hostdev/streaming_profiler_common.h,
    // ANCHOR_HIST).
    void add_drainer_audit(const ClockSample& s);
    uint64_t drainer_unbracketed = 0;  // anchors the drainer dropped: their read found no refclk update
};

// A device's refclk onto the root chip's: root_refclk = scale * dev_refclk + shift.
struct RootXf {
    double scale = 1.0, shift = 0.0;
    bool ok = false;
};

// The eth links' rounds and what they solve to. Each link's two ends stamp their frames with the 1588 hardware and
// report a round's stamp averages as LINK sync records; the solver pairs the two ends' records by round number,
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
        double precision_ns = 0.0;      // residual_rms_ns / sqrt(rounds): the offset estimate's own precision
        uint32_t dev_snd = 0, dev_rcv = 0;
        double offset_refclk = 0.0, rate = 0.0, mid_refclk = 0.0;
        double residual_rms_ns = 0.0;
        size_t rounds = 0;
    };

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
    // Per device index, the chip onto the root's refclk from all the links at once (see the definition); not ok for a
    // chip the root does not reach over solved links.
    std::vector<RootXf> root_transforms(uint32_t root) const;

    // A round in the refclk domain: each end's midpoint; the sender's round trip, the receiver's turnaround and the
    // one-way delay inside the stamps, in ns. Each end averages the frames it could pair, which need not be the same
    // frames at both ends, so the round trip can span the round: the one-way delay carries it into the receiver's
    // refclk at the link's `rate` (LinkSolution::rate). The midpoints need no such term, the link's relation being a
    // line.
    static double mid_a_refclk(const Round& r);
    static double mid_b_refclk(const Round& r);
    static double rtt_ns(const Round& r) {
        return (static_cast<double>(r.t2.units) - static_cast<double>(r.t0.units)) * kRefclkPerStampUnit * kNsPerRefclk;
    }
    static double turn_ns(const Round& r) {
        return (static_cast<double>(r.t1b.units) - static_cast<double>(r.t1.units)) * kRefclkPerStampUnit *
               kNsPerRefclk;
    }
    static double path_ns(const Round& r, double rate) { return 0.5 * (rtt_ns(r) * (1.0 + rate) - turn_ns(r)); }

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
    static void solve_link(const CaptureContext::Link& L, const std::vector<RoundPoint>& pts, LinkSolution& out);

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
    // runs); the ~25 rounds inside average a round's ~0.6 ns of stamp noise to ~0.12 ns, and a longer window measured
    // no better.
    static constexpr double kLinkWindowTicks = 12'500'000.0;  // 250 ms
    static constexpr size_t kMinSolveRounds = 8;
    static constexpr size_t kPendingMax = 4096;
};

// One frozen node of a placement series: at `at` the placement is `value`, linear to the next node. Before the oldest
// kept node, and past the newest once the series is finished, it carries on along `tangent` (d value / d at).
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
// is its newest node's key, and every key once the series is finished: a record at or before it converts against
// frozen data on both sides.
class ClockMap {
public:
    static constexpr uint32_t kMaxChips = 256;
    // Nodes a series keeps before the oldest go: 1.5 GB at most per chip, allocated as nodes arrive (24 B each). A
    // chip takes a node at least every millisecond (kEthPointUs) and more under DVFS, so a series holds ~18.6 hours
    // of a steady clock; the sync report counts any round that fell off the front.
    static constexpr uint32_t kSeriesNodes = 1u << 26;

    explicit ClockMap(uint32_t series_nodes = kSeriesNodes);
    ~ClockMap();
    ClockMap(const ClockMap&) = delete;
    ClockMap& operator=(const ClockMap&) = delete;

    // Appends a node past every earlier one (a node at the last node's key is dropped) and moves the cover to it.
    void append(uint32_t chip_id, SyncNode node);
    // The series is complete for the capture: every later instant converts on the newest tangent.
    void finish(uint32_t chip_id);
    // Empties a chip's series for a new capture.
    void clear(uint32_t chip_id);
    void append_host(HostNode node);
    // The host's steady series: host TSC tick -> CLOCK_MONOTONIC ns, a node per host probe pair, linear between them
    // and on the newest node's tangent past it. Never cleared: steady_clock outlives any capture.
    void append_steady(ClockNode<int64_t> node);
    // CLOCK_MONOTONIC ns at host TSC tick `tsc`; false before the series' first node.
    bool steady_ns(int64_t tsc, double& ns) const noexcept;

    // The root refclk tick of a chip's eth wall tick, which may be fractional; 0 before the chip's first node.
    double lookup_root(uint32_t chip_id, double wall) const noexcept;
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

// Each chip's published placement series: a node per instant of its clock model, at the instant's eth wall tick the
// root's refclk tick through the chip's refclk and the solved links. A node's tangent is its line's k8 slope, or at a
// raw instant the chord to the instant before it (for the first instant, the one after it). Nodes are frozen once
// published (consumers have placed records against them), so a publish only appends beyond them; records are placed
// only up to the newest node, on the chord to it.
class SeriesPublisher {
public:
    explicit SeriesPublisher(ClockMap& map) : map_(map) {}
    void reset(size_t devices) { series_.assign(devices, Series{}); }
    // Publishes one chip's instants beyond its series' end, from its model and root transform as they stand; true
    // when a node was added.
    bool publish(uint32_t dev, uint32_t chip, const LocalClockModel& fit, const RootXf& xf);
    size_t nodes(uint32_t dev) const { return series_[dev].count; }

private:
    // One chip's series: how many nodes it has (the map holds them; a vector of them here doubled into a 45 ms copy
    // on the sync thread at two million nodes, and every drainer's ring overflowed meanwhile) and its model's first
    // instant not yet published.
    struct Series {
        size_t count = 0;
        size_t next = 0;
    };

    ClockMap& map_;
    std::vector<Series> series_;  // per device index
};

// Device<->device sync from the sync records the idle-eth pushers carry, and the correction it publishes.
//
// LOCAL points feed one LocalClockModel per device. LINK samples feed the LinkSolver, refclk against refclk, so DVFS
// on either wall clock cannot enter the link solve. From those the SeriesPublisher publishes, per chip, a placement
// series in the ClockMap every record is placed through: the chip's eth wall tick onto the root chip's refclk,
// which the host probe's series takes onto the host. Published incrementally for live sinks, finally at capture end.
// Driven from the Service's sync thread, which walks the eth pushers' streams and hands it every sync record in
// order; a capture is on_attach, the records, on_capture_end. The unit test drives it the same way.
class SyncEngine {
public:
    explicit SyncEngine(uint32_t series_nodes = ClockMap::kSeriesNodes) : map_(series_nodes), series_(map_) {}
    void on_attach(const CaptureContext& ctx);
    // One sync record (kSyncRecordWords words) from roster core `core` of device index `dev`.
    void on_clock(uint32_t dev, uint32_t core, const uint32_t* rec);
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
    // the raw offset and the path figures.
    struct LinkErrors;
    LinkErrors link_errors(size_t li) const;
    void log_link_stats(const CaptureContext::Link& L, const LinkErrors& e, size_t rounds) const;
    void write_err_csv(const CaptureContext::Link& L, const LinkErrors& e) const;
    void write_model_csv() const;
    // The receiver's round midpoint and the sender's, each through its chip's model to its wall clock and placed on
    // the root's refclk as the sink places records from that chip's eth core, and their difference in ns: the links'
    // and the map's error, the models cancelling. tsc_a is the sender's host placement, the plots' abscissa. False
    // when a chip has no fitted run or no node to place a stamp with.
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

    ClockMap map_;
    CaptureContext ctx_;
    std::vector<LocalClockModel> local_;  // per device index
    std::vector<AnchorAudit> audit_;      // per device index, its model's audit
    void log_audit() const;
    LinkSolver links_;
    SeriesPublisher series_;
    // The composed root transforms as of the newest accepted link solution.
    std::vector<RootXf> to_root_;
    uint64_t to_root_gen_ = ~0ull;
    std::unordered_set<std::string> plot_names_;  // Tracy keys a plot by its name's address, for the process
};

}  // namespace tt::tt_metal::streaming_profiler
