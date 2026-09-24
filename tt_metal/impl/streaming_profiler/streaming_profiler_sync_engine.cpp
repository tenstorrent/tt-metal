// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>

#include <tt_stl/assert.hpp>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>

#include "impl/streaming_profiler/spsc_packet.h"
#include "tt_metal/common/indexed_ring.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

template <typename Key>
constexpr Key key_min() {
    if constexpr (std::is_same_v<Key, double>) {
        return -std::numeric_limits<double>::infinity();
    } else {
        return std::numeric_limits<Key>::min();
    }
}
template <typename Key>
constexpr Key key_max() {
    if constexpr (std::is_same_v<Key, double>) {
        return std::numeric_limits<double>::infinity();
    } else {
        return std::numeric_limits<Key>::max();
    }
}

constexpr uint32_t kHostSeries = std::numeric_limits<uint32_t>::max();

template <typename Key>
struct Log {
    using Node = ClockNode<Key>;
    explicit Log(uint32_t series_nodes) : nodes(series_nodes) {}
    IndexedRing<Node> nodes;
    Key last_at = key_min<Key>();  // the writer's own copy
    alignas(64) std::atomic<Key> cover{key_min<Key>()};
    alignas(64) std::atomic<uint32_t> gen{0};
    bool full_warned = false;

    void append(uint32_t chip_id, const Node& node) {
        if (nodes.count() != nodes.first() && node.at <= last_at) {
            return;
        }
        if (!full_warned && nodes.count() - nodes.first() == nodes.capacity()) {
            full_warned = true;
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync: {} has {} placement nodes, the series' capacity; records before its "
                "oldest kept node convert on that node's tangent",
                chip_id == kHostSeries ? std::string("the host") : "chip " + std::to_string(chip_id),
                nodes.capacity());
        }
        nodes.push(node);
        last_at = node.at;
        extend(node.at);
    }
    void extend(Key c) {
        if (c > cover.load(std::memory_order_relaxed)) {
            cover.store(c, std::memory_order_release);
        }
    }
    void clear() {
        gen.fetch_add(1, std::memory_order_release);
        cover.store(key_min<Key>(), std::memory_order_release);
        nodes.clear();
        last_at = key_min<Key>();
        full_warned = false;
    }
};

// A reader's place in one series: the segment [a, b] it last converted in and that segment's line through
// (origin, value). For the open segment b is the cover the reader last saw, so a record past that cover re-reads it.
template <typename Key>
struct Cursor {
    uint32_t gen = 0;
    Key a = key_max<Key>();
    Key b = key_min<Key>();
    Key origin{};
    double value = 0.0;
    double slope = 0.0;
};
template <typename Key>
inline double on_line(const Cursor<Key>& c, Key t) noexcept {
    return c.value + c.slope * static_cast<double>(t - c.origin);
}

// Puts the cursor on the segment holding t and places t; false when the series has no node. A read that fails (the
// writer retired that chunk meanwhile) starts over from the new oldest node.
template <typename Key>
bool refill(const Log<Key>& log, Cursor<Key>& c, Key t, double& value) noexcept {
    using Node = ClockNode<Key>;
    for (;;) {
        const uint32_t gen = log.gen.load(std::memory_order_acquire);
        const Key cover = log.cover.load(std::memory_order_acquire);
        const uint64_t f = log.nodes.first();
        const uint64_t n = log.nodes.count();
        if (n == f) {
            c = Cursor<Key>{.gen = gen};
            return false;
        }
        Node a{}, b{};
        uint64_t lo = f, hi = n;  // first node past t
        bool retired = false;
        while (lo < hi && !retired) {
            const uint64_t mid = lo + (hi - lo) / 2;
            retired = !log.nodes.read(mid, a);
            if (a.at <= t) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (retired || !log.nodes.read(lo == f ? f : lo - 1, a)) {
            continue;
        }
        if (lo == f) {
            c = Cursor<Key>{gen, key_min<Key>(), a.at, a.at, a.value, a.tangent};
        } else if (lo < n) {
            if (!log.nodes.read(lo, b)) {
                continue;
            }
            c = Cursor<Key>{gen, a.at, b.at, a.at, a.value, (b.value - a.value) / static_cast<double>(b.at - a.at)};
        } else if (t <= cover) {
            c = Cursor<Key>{gen, a.at, cover, a.at, a.value, a.tangent};
        } else {
            // Past the cover: a batch the service released before the sync covered it (counted and reported as a
            // fault). The newest tangent carries on; holding still here would collapse every such record onto one
            // instant. Not cached, the cover moves.
            c = Cursor<Key>{.gen = gen};
            value = a.value + a.tangent * static_cast<double>(t - a.at);
            return true;
        }
        value = on_line(c, t);
        return true;
    }
}

template <typename Key>
inline bool place(const Log<Key>& log, Cursor<Key>& c, Key t, double& value) noexcept {
    if (c.gen == log.gen.load(std::memory_order_relaxed) && t >= c.a && t <= c.b) {
        value = on_line(c, t);
        return true;
    }
    return refill(log, c, t, value);
}

// One chip's composed placement line: wall -> host_clock units, the chip segment (wall -> root refclk) multiplied by
// the host segment (root refclk -> TSC), valid over [a, b] in wall ticks where both hold. Anchored at the record that
// built it so a 1e15 value stays exact in the double.
struct Composed {
    uint32_t gen_c = 0, gen_h = 0;
    int64_t a = std::numeric_limits<int64_t>::max(), b = std::numeric_limits<int64_t>::min();
    int64_t origin = 0;
    double value = 0.0, slope = 0.0;
};

struct ThreadView {
    const void* owner = nullptr;
    std::array<Cursor<int64_t>, ClockMap::kMaxChips> chip{};
    Cursor<double> host{};
    std::array<Composed, ClockMap::kMaxChips> composed{};
};
constinit thread_local ThreadView t_view;
ThreadView& view_of(const void* owner) noexcept {
    if (t_view.owner != owner) {
        t_view = ThreadView{};
        t_view.owner = owner;
    }
    return t_view;
}

}  // namespace

struct ClockMap::Impl {
    template <size_t... I>
    explicit Impl(uint32_t series_nodes, std::index_sequence<I...>) :
        chips{{((void)I, Log<int64_t>(series_nodes))...}}, host(series_nodes) {}
    std::array<Log<int64_t>, kMaxChips> chips;
    Log<double> host;
    alignas(64) std::atomic<uint64_t> cover_generation{0};
};

ClockMap::ClockMap(uint32_t series_nodes) :
    impl_(std::make_unique<Impl>(series_nodes, std::make_index_sequence<kMaxChips>{})) {}

uint32_t ClockMap::series_nodes() const noexcept { return static_cast<uint32_t>(impl_->host.nodes.capacity()); }
ClockMap::~ClockMap() = default;

void ClockMap::append(uint32_t chip_id, SyncNode node) {
    if (chip_id >= kMaxChips) {
        return;
    }
    TT_FATAL(
        std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0 && node.tangent < 1.0,
        "streaming profiler: placement node for chip {} at wall {} is not a rate: root {} tangent {}",
        chip_id,
        node.at,
        node.value,
        node.tangent);
    impl_->chips[chip_id].append(chip_id, node);
    impl_->cover_generation.fetch_add(1, std::memory_order_release);
}

void ClockMap::extend(uint32_t chip_id, int64_t cover_ticks) {
    if (chip_id < kMaxChips) {
        impl_->chips[chip_id].extend(cover_ticks);
        impl_->cover_generation.fetch_add(1, std::memory_order_release);
    }
}

void ClockMap::finish(uint32_t chip_id) { extend(chip_id, std::numeric_limits<int64_t>::max()); }

void ClockMap::clear(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        impl_->chips[chip_id].clear();
    }
}

void ClockMap::append_host(HostNode node) {
    TT_FATAL(
        std::isfinite(node.at) && std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0 &&
            node.tangent < 1e4,
        "streaming profiler: host placement node at refclk {} is not a rate: tsc {} tangent {}",
        node.at,
        node.value,
        node.tangent);
    impl_->host.append(kHostSeries, node);
    impl_->cover_generation.fetch_add(1, std::memory_order_release);
}

// The chip's cover, held back to where its line reaches the host series' last node: a record placed beyond that
// node sits on its tangent while a later record at the same tick takes the chord to the next node, so two lanes of
// one chip would part by the line's move between bursts. A finished chip releases everything.
int64_t ClockMap::cover_ticks(uint32_t chip_id) const noexcept {
    if (chip_id >= kMaxChips) {
        return std::numeric_limits<int64_t>::max();
    }
    const Log<int64_t>& cl = impl_->chips[chip_id];
    const int64_t cover = cl.cover.load(std::memory_order_acquire);
    if (cover == std::numeric_limits<int64_t>::max()) {
        return cover;
    }
    const double host_cover = impl_->host.cover.load(std::memory_order_acquire);
    ClockNode<int64_t> last{};
    const uint64_t n = cl.nodes.count();
    if (n == cl.nodes.first() || !cl.nodes.read(n - 1, last) || !(last.tangent > 0.0)) {
        return cover;
    }
    const double wall = static_cast<double>(last.at) + (host_cover - last.value) / last.tangent;
    if (wall >= static_cast<double>(cover)) {
        return cover;
    }
    if (wall <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(std::floor(wall));
}

int64_t ClockMap::oldest_at(uint32_t chip_id) const noexcept {
    if (chip_id >= kMaxChips) {
        return std::numeric_limits<int64_t>::min();
    }
    const Log<int64_t>& log = impl_->chips[chip_id];
    ClockNode<int64_t> n{};
    const uint64_t f = log.nodes.first();
    if (log.nodes.count() == f || !log.nodes.read(f, n)) {
        return std::numeric_limits<int64_t>::min();
    }
    return n.at;
}

uint64_t ClockMap::cover_generation() const noexcept { return impl_->cover_generation.load(std::memory_order_acquire); }

size_t ClockMap::host_published() const noexcept { return impl_->host.nodes.count() - impl_->host.nodes.first(); }

double ClockMap::lookup_root(uint32_t chip_id, int64_t wall) const noexcept {
    double root = 0.0;
    if (chip_id >= kMaxChips || !place(impl_->chips[chip_id], view_of(impl_.get()).chip[chip_id], wall, root)) {
        return 0.0;
    }
    return root;
}

double ClockMap::host_tsc(double root) const noexcept {
    double tsc = 0.0;
    if (!place(impl_->host, view_of(impl_.get()).host, root, tsc)) {
        return 0.0;
    }
    return tsc;
}

int64_t ClockMap::place_host(uint32_t chip_id, int64_t wall) const noexcept {
    if (chip_id >= kMaxChips) {
        return 0;
    }
    ThreadView& v = view_of(impl_.get());
    const Log<int64_t>& cl = impl_->chips[chip_id];
    const Log<double>& hl = impl_->host;
    Composed& k = v.composed[chip_id];
    if (wall >= k.a && wall <= k.b && k.gen_c == cl.gen.load(std::memory_order_relaxed) &&
        k.gen_h == hl.gen.load(std::memory_order_relaxed)) {
        return std::llround(k.value + k.slope * static_cast<double>(wall - k.origin));
    }
    double root = 0.0, tsc = 0.0;
    Cursor<int64_t>& cc = v.chip[chip_id];
    Cursor<double>& hc = v.host;
    if (!place(cl, cc, wall, root) || !place(hl, hc, root, tsc)) {
        return 0;
    }
    const double u = units_per_tsc();
    k.gen_c = cc.gen;
    k.gen_h = hc.gen;
    k.origin = wall;
    k.value = tsc * u;
    k.slope = cc.slope * hc.slope * u;
    // The range both cursors cover; a cursor past its cover caches nothing and neither does the composition.
    k.a = std::numeric_limits<int64_t>::max();
    k.b = std::numeric_limits<int64_t>::min();
    if (cc.a <= cc.b && hc.a <= hc.b && cc.slope > 0.0) {
        const double wa = static_cast<double>(cc.origin) + (hc.a - cc.value) / cc.slope;
        const double wb = static_cast<double>(cc.origin) + (hc.b - cc.value) / cc.slope;
        k.a = std::max(cc.a, static_cast<int64_t>(std::ceil(wa)));
        k.b = std::min(cc.b, static_cast<int64_t>(std::floor(wb)));
    }
    return std::llround(k.value);
}

double ErrorHistogram::abs_quantile(double q) const {
    if (n <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<std::pair<double, double>> v;
    for (int b = 0; b < kBins; b++) {
        if (bins[b] != 0.0) {
            v.emplace_back(std::abs(ns_of(b)), bins[b]);
        }
    }
    std::sort(v.begin(), v.end());
    double acc = 0.0;
    for (const auto& [ns, w] : v) {
        acc += w;
        if (acc >= q * n) {
            return ns;
        }
    }
    return worst;
}

ErrorHistogram ErrorHistogram::convolve(const ErrorHistogram& other, int sign) const {
    ErrorHistogram out;
    if (n <= 0.0 || other.n <= 0.0) {
        return out;
    }
    std::vector<int> mine, theirs;
    for (int b = 0; b < kBins; b++) {
        if (bins[b] != 0.0) {
            mine.push_back(b);
        }
        if (other.bins[b] != 0.0) {
            theirs.push_back(b);
        }
    }
    const double norm = 1.0 / (n * other.n);
    for (const int i : mine) {
        for (const int j : theirs) {
            const int b = bin_of(ns_of(i) + sign * ns_of(j));
            if (b >= 0 && b < kBins) {
                out.bins[b] += bins[i] * other.bins[j] * norm;
            } else {
                out.beyond += bins[i] * other.bins[j] * norm;
            }
        }
    }
    out.beyond += (beyond * other.n + n * other.beyond - beyond * other.beyond) * norm;
    out.n = 1.0;
    out.sum = mean() + sign * other.mean();
    out.sumsq = sumsq / n + other.sumsq / other.n + 2.0 * sign * mean() * other.mean();
    out.worst = worst + other.worst;
    return out;
}

ErrorHistogram AnchorAudit::both() const {
    ErrorHistogram h = line;
    for (int b = 0; b < ErrorHistogram::kBins; b++) {
        h.bins[b] += glide.bins[b];
    }
    h.n += glide.n;
    h.sum += glide.sum;
    h.sumsq += glide.sumsq;
    h.beyond += glide.beyond;
    h.worst = std::max(h.worst, glide.worst);
    return h;
}

// An anchor is checked once the model's instants around it can no longer change: the curve between two instants
// needs the one after them.
void AnchorAudit::settle(const LocalClockModel& m, bool final) {
    const auto& pts = m.pts;
    if (pts.size() < 3 && !final) {
        return;
    }
    const double until = final ? m.frontier() : pts[pts.size() - 2].r;
    while (!pending.empty() && pending.front().r < until) {
        const Pending a = pending.front();
        pending.pop_front();
        if (a.r < pts.front().r) {
            before_model++;
            continue;
        }
        const double w = m.wall_at(a.r);
        const double ns = (a.w - w) * kNsPerRefclk / (m.wall_at(a.r + 1.0) - w);
        const auto hi = std::lower_bound(
            pts.begin(), pts.end(), a.r, [](const LocalClockModel::Instant& p, double x) { return p.r < x; });
        const bool in_glide = hi != pts.end() && hi != pts.begin() && hi->k8 == 0 && (hi - 1)->k8 == 0;
        (in_glide ? glide : line).add(ns);
        if (std::abs(ns) > std::abs(worst_ns)) {
            worst_ns = ns;
            worst_r = a.r;
        }
    }
    if (final) {
        past_model += pending.size();
        pending.clear();
    }
}

void SyncEngine::on_attach(const CaptureContext& ctx) {
    ctx_ = ctx;
    local_.clear();
    audit_.clear();
    links_.reset(ctx_);
    series_.reset();
    to_root_gen_ = ~0ull;
    // A chip the capture cannot place -- no eth tracker, or no link path to the root -- is finished at once, so no
    // consumer waits for it.
    std::vector<bool> reach(ctx.devices.size(), false);
    const uint32_t root = ctx.root_dev;
    if (root < ctx.devices.size() && ctx.devices[root].has_eth_tracker) {
        reach[root] = true;
        for (bool progress = true; progress;) {
            progress = false;
            for (const CaptureContext::Link& L : ctx.links) {
                if (L.dev_a < reach.size() && L.dev_b < reach.size() && reach[L.dev_a] != reach[L.dev_b]) {
                    reach[L.dev_a] = reach[L.dev_b] = true;
                    progress = true;
                }
            }
        }
    }
    for (size_t dev = 0; dev < ctx.devices.size(); dev++) {
        const CaptureContext::Device& d = ctx.devices[dev];
        map_.clear(d.chip_id);
        if (!reach[dev] || !d.has_eth_tracker) {
            map_.finish(d.chip_id);
        }
    }
}

void SyncEngine::on_clock(const ClockSample& s) {
    if (s.kind == kernel_profiler::kSyncKindAnchor) {
        const int64_t off = s.dev < ctx_.devices.size() ? ctx_.devices[s.dev].drainer_offset : 0;
        audit_[s.dev].pending.push_back(
            AnchorAudit::Pending{static_cast<double>(s.ref), static_cast<double>(s.ts) + static_cast<double>(off)});
        return;
    }
    if (s.kind != kernel_profiler::kSyncKindLocal) {
        links_.on_stamp(s);
        return;
    }
    local_[s.dev].add_point(
        s.value,
        s.ts,
        s.round & 0xFFu,
        s.round >> 8,
        s.role == kernel_profiler::kSyncLocalClose,
        static_cast<uint32_t>(s.ref));
    if (const auto a = audit_.find(s.dev); a != audit_.end()) {
        a->second.settle(local_[s.dev], /*final=*/false);
    }
    if (publish_dev(s.dev)) {
        service().wake_consumers();
    }
}

void LinkSolver::reset(const CaptureContext& ctx) {
    ctx_ = &ctx;
    rounds_.assign(ctx.links.size(), LinkRounds{});
    side_of_.clear();
    for (size_t li = 0; li < ctx.links.size(); li++) {
        const CaptureContext::Link& L = ctx.links[li];
        side_of_[{L.dev_a, L.core_a}] = {li, true};
        side_of_[{L.dev_b, L.core_b}] = {li, false};
    }
    solved_.assign(ctx.links.size(), LinkSolution{});
    gen_ = 0;
    dropped_ = 0;
}

void LinkSolver::on_stamp(const ClockSample& s) {
    namespace kp = kernel_profiler;
    const auto side = side_of_.find({s.dev, s.core});
    if (s.kind != kp::kSyncKindLink || side == side_of_.end()) {
        dropped_++;
        return;
    }
    const auto [li, sender] = side->second;
    // Each end records the peer's egress average (read from the frames it received) and its own ingress average:
    // the receiver T0 and T1, the sender T1B and T2.
    Stamp Round::* slot = nullptr;
    if (sender) {
        slot = s.role == kp::kSyncRoleT1B ? &Round::t1b : s.role == kp::kSyncRoleT2 ? &Round::t2 : nullptr;
    } else {
        slot = s.role == kp::kSyncRoleT0 ? &Round::t0 : s.role == kp::kSyncRoleT1 ? &Round::t1 : nullptr;
    }
    if (slot == nullptr) {
        dropped_++;
        return;
    }
    LinkRounds& lr = rounds_[li];
    Round& r = lr.pending[s.round];
    r.id = s.round;
    r.*slot = Stamp{.units = s.value, .wall = s.ts, .ref = s.ref, .spins = s.spins, .have = true};
    if (r.complete()) {
        lr.rounds.push_back(r);
        lr.pending.erase(s.round);
        try_solve_links(/*final=*/false);
    }
    while (lr.pending.size() > kPendingMax) {
        lr.pending.erase(lr.pending.begin());
    }
}

void LinkSolver::solve_final() { try_solve_links(/*final=*/true); }

void LinkSolver::try_solve_links(bool final) {
    for (size_t li = 0; li < ctx_->links.size(); li++) {
        LinkSolution& out = solved_[li];
        const CaptureContext::Link& L = ctx_->links[li];
        const std::vector<Round>& rounds = rounds_[li].rounds;
        const size_t n = rounds.size();
        if (final) {
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link {} (dev {} eth({},{}) core {}) -> (dev {} eth({},{}) core {}): "
                "{} rounds complete, {} pending",
                li,
                L.dev_a,
                L.eth_a.x,
                L.eth_a.y,
                L.core_a,
                L.dev_b,
                L.eth_b.x,
                L.eth_b.y,
                L.core_b,
                rounds.size(),
                rounds_[li].pending.size());
        }
        if (n == 0) {
            continue;
        }
        const auto pos = [](const Round& r) { return mid_a_refclk(r); };
        const double newest = pos(rounds[n - 1]);
        if (!final && newest <= out.solved_at_refclk) {
            continue;
        }
        size_t begin = n;
        while (begin > 0 && pos(rounds[begin - 1]) > newest - kLinkWindowTicks) {
            begin--;
        }
        const size_t w = n - begin;
        if (w < kMinSolveRounds) {
            continue;
        }
        out.solved_at_refclk = newest;
        std::vector<RoundPoint> pts;
        pts.reserve(w);
        const double path_med = path_median(rounds, begin, n);
        out.path_dropped = 0;
        for (size_t i = begin; i < n; i++) {
            const Round& r = rounds[i];
            if (std::abs(path_ns(r) - path_med) > kPathDevNs) {
                out.path_dropped++;
                continue;
            }
            const double mid = mid_a_refclk(r);
            pts.push_back(RoundPoint{mid, mid_b_refclk(r) - mid});
        }
        if (solve_link(L, std::move(pts), out)) {
            out.rounds = w;
            gen_++;
            if (!final) {
                continue;
            }
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} -> chip {}: solved at round {} over {} ({} kept): offset "
                "{:.1f} ns, rate {:.3f} ppm, residual {:.1f} ns{}",
                L.chip_a,
                L.chip_b,
                n,
                w,
                out.kept,
                out.offset_refclk * kNsPerRefclk,
                out.rate * 1e6,
                out.residual_rms_ns,
                out.path_dropped != 0 ? fmt::format(", {} rounds off the stamp path band", out.path_dropped) : "");
        }
    }
}

double LinkSolver::mid_a_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(r.t0.units) + static_cast<double>(r.t2.units)) * kRefclkPerStampUnit;
}

double LinkSolver::mid_b_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(r.t1.units) + static_cast<double>(r.t1b.units)) * kRefclkPerStampUnit;
}

double LinkSolver::path_median(const std::vector<Round>& rounds, size_t begin, size_t n) {
    if (n <= begin) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> v;
    v.reserve(n - begin);
    for (size_t i = begin; i < n; i++) {
        v.push_back(path_ns(rounds[i]));
    }
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

namespace {

// Gaussian elimination with partial pivoting on a dense system; n is the number of chips off the root, a few dozen
// at most. False when the system is singular (a chip with no weight left on any of its links).
bool solve_dense(std::vector<double> m, std::vector<double> rhs, std::vector<double>& x) {
    const size_t n = rhs.size();
    for (size_t k = 0; k < n; k++) {
        size_t piv = k;
        for (size_t i = k + 1; i < n; i++) {
            if (std::abs(m[i * n + k]) > std::abs(m[piv * n + k])) {
                piv = i;
            }
        }
        if (!(std::abs(m[piv * n + k]) > 1e-12)) {
            return false;
        }
        if (piv != k) {
            for (size_t j = 0; j < n; j++) {
                std::swap(m[k * n + j], m[piv * n + j]);
            }
            std::swap(rhs[k], rhs[piv]);
        }
        for (size_t i = k + 1; i < n; i++) {
            const double f = m[i * n + k] / m[k * n + k];
            for (size_t j = k; j < n; j++) {
                m[i * n + j] -= f * m[k * n + j];
            }
            rhs[i] -= f * rhs[k];
        }
    }
    x.assign(n, 0.0);
    for (size_t k = n; k-- > 0;) {
        double acc = rhs[k];
        for (size_t j = k + 1; j < n; j++) {
            acc -= m[k * n + j] * x[j];
        }
        x[k] = acc / m[k * n + k];
    }
    return true;
}

// Weighted least squares for a potential on a graph: each edge says node_a - node_b = value with weight w; the root
// is fixed at zero. `idx` maps a device to its unknown, -1 for the root.
bool solve_potential(
    const std::vector<std::array<int, 2>>& ends,
    const std::vector<double>& value,
    const std::vector<double>& w,
    size_t n,
    std::vector<double>& x) {
    std::vector<double> m(n * n, 0.0), rhs(n, 0.0);
    for (size_t i = 0; i < ends.size(); i++) {
        const int a = ends[i][0], b = ends[i][1];
        if (a >= 0) {
            m[a * n + a] += w[i];
            rhs[a] += w[i] * value[i];
        }
        if (b >= 0) {
            m[b * n + b] += w[i];
            rhs[b] -= w[i] * value[i];
        }
        if (a >= 0 && b >= 0) {
            m[a * n + b] -= w[i];
            m[b * n + a] -= w[i];
        }
    }
    return solve_dense(std::move(m), std::move(rhs), x);
}

}  // namespace

// The mesh solve: every chip's refclk onto the root's from all solved links at once, a link's two ends being the
// same instant. The log of a chip's rate against the root is a potential on the graph whose edges are the links'
// log rates, so rates compose exactly. A chip's offset is the potential whose edges equate the two placements of a
// link's midpoint through the solved rates: anchored there, a link's constraint holds exactly at its own midpoint
// whatever the mesh's rates differ from its own by (a ppb over the refclk count is a microsecond; over the distance
// from the midpoint, nothing). The offsets are reweighted by residual against half a stamp tick (Tukey's biweight):
// the cables' path asymmetries leave a link within 4 ns of the mesh, while a stamp reference that moved at one end
// (a hardware latency quantum, one launch in four, never under 9 ns and up to a tick) puts its link a good part of
// a tick off, and such a link loses its weight instead of averaging into its pair; the report names it (weights,
// per link, in `weights`).
std::map<uint32_t, RootXf> LinkSolver::root_transforms(uint32_t root, std::vector<double>* weights) const {
    std::map<uint32_t, RootXf> to_root;
    to_root[root] = RootXf{1.0, 0.0, true};
    if (weights != nullptr) {
        weights->assign(solved_.size(), 0.0);
    }
    // The chips the root reaches over solved links, each an unknown; the root is fixed.
    std::map<uint32_t, int> idx;
    idx[root] = -1;
    for (bool grew = true; grew;) {
        grew = false;
        for (const LinkSolution& s : solved_) {
            if (!s.ok) {
                continue;
            }
            const bool have_s = idx.count(s.dev_snd) != 0, have_r = idx.count(s.dev_rcv) != 0;
            if (have_s != have_r) {
                idx[have_s ? s.dev_rcv : s.dev_snd] = static_cast<int>(idx.size()) - 1;
                grew = true;
            }
        }
    }
    const size_t n = idx.size() - 1;
    if (n == 0) {
        return to_root;
    }
    std::vector<size_t> li_of;
    std::vector<std::array<int, 2>> ends;
    std::vector<double> log_rate, mid, offset, w0;
    for (size_t li = 0; li < solved_.size(); li++) {
        const LinkSolution& s = solved_[li];
        if (!s.ok || idx.count(s.dev_snd) == 0 || idx.count(s.dev_rcv) == 0) {
            continue;
        }
        li_of.push_back(li);
        ends.push_back({idx[s.dev_snd], idx[s.dev_rcv]});
        log_rate.push_back(std::log1p(s.rate));  // A_snd = A_rcv * (1 + rate)
        mid.push_back(s.mid_refclk);
        offset.push_back(s.offset_refclk);
        const double p = std::max(s.precision_ns, 0.01);
        w0.push_back(1.0 / (p * p));
    }
    std::vector<double> x;
    if (!solve_potential(ends, log_rate, std::vector<double>(ends.size(), 1.0), n, x)) {
        return to_root;
    }
    const auto A_of = [&](int i) { return i < 0 ? 1.0 : std::exp(x[i]); };
    // Offsets: at the link's midpoint the sender reads mid and the receiver mid + offset, one instant on the root:
    // A_snd * mid + B_snd = A_rcv * (mid + offset) + B_rcv. Four passes of reweighting on the residuals' robust scale.
    std::vector<double> value(ends.size()), w = w0, B, resid(ends.size(), 0.0);
    for (size_t i = 0; i < ends.size(); i++) {
        value[i] = A_of(ends[i][1]) * (mid[i] + offset[i]) - A_of(ends[i][0]) * mid[i];
    }
    std::vector<double> robust(ends.size(), 1.0);
    for (int pass = 0; pass < 4; pass++) {
        if (!solve_potential(ends, value, w, n, B)) {
            return to_root;
        }
        const auto B_of = [&](int i) { return i < 0 ? 0.0 : B[i]; };
        for (size_t i = 0; i < ends.size(); i++) {
            resid[i] = (B_of(ends[i][0]) - B_of(ends[i][1]) - value[i]) * kNsPerRefclk;
            const double u = resid[i] / (kNsPerRefclk / 2.0);
            robust[i] = std::abs(u) < 1.0 ? (1.0 - u * u) * (1.0 - u * u) : 0.0;
            w[i] = w0[i] * std::max(robust[i], 1e-6);
        }
    }
    for (const auto& [dev, i] : idx) {
        if (i >= 0) {
            to_root[dev] = RootXf{A_of(i), B[i], true};
        }
    }
    if (weights != nullptr) {
        for (size_t i = 0; i < ends.size(); i++) {
            (*weights)[li_of[i]] = robust[i];
        }
    }
    return to_root;
}

void SeriesPublisher::push_node(Series& s, uint32_t chip, const Node& n) {
    const int64_t wall = static_cast<int64_t>(std::llround(n.H));
    if (s.count != 0 && wall <= static_cast<int64_t>(std::llround(s.last.H))) {
        return;  // within the tick of the last node: the placement cannot differ measurably there
    }
    s.last = n;
    s.count++;
    s.last_r = std::max(s.last_r, n.r);
    map_.append(chip, SyncNode{.at = wall, .value = n.root, .tangent = n.tangent});
}

// Frozen nodes never move (consumers have placed records against them), so a publish can only add beyond them, at
// the newest estimate's values; a join carries whatever the estimate moved by since the last node (a few ns at a
// knot). Shifting fresh nodes to meet the frozen tail and fading that shift over a quarter second is worse: every
// discrepancy at a join becomes a level the map carries for 250 ms, 30-60 ns during DVFS dithering at 1 ms.
void SeriesPublisher::append_node(Series& s, uint32_t chip, const Node& n) {
    const double end_H = s.count == 0 ? -1.0 : s.last.H;
    if (n.H >= end_H && n.H < end_H + 1.0) {
        return;  // the series' end re-derived, or a knot within the tick of it: the same node
    }
    if (n.H < end_H) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync: placement node at wall {:.0f} lies {:.0f} wall ticks behind the frozen "
            "series' end; the frozen node stands",
            n.H,
            end_H - n.H);
        s.dropped++;
        return;
    }
    if (!std::isfinite(n.H) || !std::isfinite(n.root) || !std::isfinite(n.tangent) ||
        !(n.tangent > 0.0 && n.tangent < 1.0)) {
        s.dropped++;
        return;
    }
    push_node(s, chip, n);
}

bool SeriesPublisher::publish(uint32_t dev, uint32_t chip, const LocalClockModel& fit, const RootXf& xf, bool final) {
    Series& s = series_[dev];
    const auto& pts = fit.pts;
    const size_t before = s.count;
    const auto root_at = [&](const LocalClockModel::Instant& p) { return xf.scale * p.r + xf.shift; };
    auto it = std::upper_bound(
        pts.begin(), pts.end(), s.last_r, [](double x, const LocalClockModel::Instant& p) { return x < p.r; });
    for (; it != pts.end(); ++it) {
        const size_t i = static_cast<size_t>(it - pts.begin());
        const bool has_next = i + 1 < pts.size();
        if (it->k8 == 0 && !has_next && !final) {
            break;
        }
        const double root = root_at(*it);
        if (fit.curved(i)) {
            const LocalClockModel::Instant& a = pts[i - 1];
            for (uint32_t q = 1; q <= kBendNodes; q++) {
                const double r = a.r + (it->r - a.r) * q / (kBendNodes + 1);
                const double w = fit.wall_at(r);
                const double w2 = fit.wall_at(r + 1.0);
                append_node(s, chip, Node{w, xf.scale * r + xf.shift, r, xf.scale / (w2 - w)});
            }
        }
        double tangent = 0.0;
        if (has_next) {
            tangent = (root_at(*(it + 1)) - root) / ((it + 1)->w - it->w);
        } else if (it->k8 != 0) {
            tangent = xf.scale * 8.0 / it->k8;
        } else if (it != pts.begin()) {
            tangent = (root - root_at(*(it - 1))) / (it->w - (it - 1)->w);
        } else {
            break;  // a lone raw instant: no rate yet
        }
        append_node(s, chip, Node{it->w, root, it->r, tangent});
        s.last_r = it->r;
    }
    return s.count > before;
}

bool SyncEngine::publish_dev(uint32_t dev, bool final) {
    const auto st = local_.find(dev);
    if (st == local_.end() || st->second.pts.empty() || dev >= ctx_.devices.size()) {
        return false;
    }
    // Nothing is placed before the host series exists: a record converted then would land nowhere.
    if (map_.host_published() == 0) {
        return false;
    }
    if (to_root_gen_ != links_.generation()) {
        to_root_ = links_.root_transforms(root_dev(), nullptr);
        to_root_gen_ = links_.generation();
    }
    // The series starts only once the chip is on the root's tree (the root is there from the start): its first node
    // fixes the placement every later node joins.
    const auto xf = to_root_.find(dev);
    if (xf == to_root_.end() || !xf->second.ok) {
        return false;
    }
    return series_.publish(dev, ctx_.devices[dev].chip_id, st->second, xf->second, final);
}

void SyncEngine::publish_all() {
    for (const auto& kv : local_) {
        publish_dev(kv.first, /*final=*/true);
    }
    for (const CaptureContext::Device& d : ctx_.devices) {
        map_.finish(d.chip_id);
    }
}

void SyncEngine::log_summary() const {
    log_clock_models();
    log_link_solutions();
    log_loop_closures();
    if (links_.dropped() != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync: {} sync records ignored (unknown kind, a core on no link, or a role "
            "that end does not stamp)",
            links_.dropped());
    }
}

void SyncEngine::log_clock_models() const {
    for (const auto& [dev, l] : local_) {
        const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
        if (l.pts.empty()) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync chip {}: {} clock records but no instant of the local clock",
                chip,
                l.points);
            continue;
        }
        // The applied AICLK: each point's k/8 slope over the refclk until the next instant.
        long double ssum = 0.0, wsum = 0.0;
        double smin = std::numeric_limits<double>::max(), smax = 0.0;
        size_t raw = 0;
        for (size_t i = 0; i < l.pts.size(); i++) {
            const LocalClockModel::Instant& p = l.pts[i];
            if (p.k8 == 0) {
                raw++;
                continue;
            }
            const double s = p.k8 / 8.0;
            const double span = i + 1 < l.pts.size() ? l.pts[i + 1].r - p.r : 0.0;
            ssum += static_cast<long double>(s) * span;
            wsum += span;
            smin = std::min(smin, s);
            smax = std::max(smax, s);
        }
        const double to_ghz = LocalClockModel::kRefclkHz * 1e-9;
        const double anchor_ghz = dev < ctx_.devices.size() ? ctx_.devices[dev].frequency_ghz : 0.0;
        const double mean_ghz = wsum > 0.0 ? static_cast<double>(ssum / wsum) * to_ghz : 0.0;
        const double ns_ghz = mean_ghz > 0.0 ? mean_ghz : std::max(anchor_ghz, 0.1);
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync chip {}: local clock {} instants, {} of them inside its {} transitions; "
            "applied AICLK mean {:.5f} GHz (min {:.5f}, max {:.5f}; boot anchor {:.5f}); {} placement nodes; samples "
            "within {:.2f} ns of their line at worst, {} points over {:.1f} ns",
            chip,
            l.pts.size(),
            raw,
            l.transitions,
            mean_ghz,
            smax > 0.0 ? smin * to_ghz : 0.0,
            smax * to_ghz,
            anchor_ghz,
            series_.series(dev) != nullptr ? series_.series(dev)->count : 0,
            l.max_resid_ticks / ns_ghz,
            l.resid_warn_points,
            LocalClockModel::kResidWarnTicks / ns_ghz);
    }
}

void SyncEngine::log_link_solutions() const {
    const std::vector<LinkSolver::LinkSolution>& solved = links_.solutions();
    for (size_t li = 0; li < solved.size() && li < ctx_.links.size(); li++) {
        const LinkSolver::LinkSolution& s = solved[li];
        const CaptureContext::Link& L = ctx_.links[li];
        if (!s.ok) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): not solved (no or too "
                "few link stamps drained)",
                L.chip_a,
                L.eth_a.x,
                L.eth_a.y,
                L.chip_b,
                L.eth_b.x,
                L.eth_b.y);
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): {} rounds ({} kept); refclk "
            "domain offset {:.1f} ns, rate {:.3f} ppm, residual rms {:.1f} ns, estimate precision {:.2f} ns",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            s.rounds,
            s.kept,
            s.offset_refclk * kNsPerRefclk,
            s.rate * 1e6,
            s.residual_rms_ns,
            s.precision_ns);
    }
}

// Each link against the mesh solve: the difference between its own solution and the two chips' placements on the
// root. With every link on one consistent mesh the residuals are the cables' path asymmetries, a nanosecond or two;
// a link the reweighting dropped carries a stamp bias, and its rounds still place through the mesh.
void SyncEngine::log_loop_closures() const {
    if (local_.empty()) {
        return;
    }
    const std::vector<LinkSolver::LinkSolution>& solved = links_.solutions();
    std::vector<double> weights;
    const std::map<uint32_t, RootXf> to_root = links_.root_transforms(root_dev(), &weights);
    for (size_t li = 0; li < solved.size() && li < ctx_.links.size(); li++) {
        const LinkSolver::LinkSolution& s = solved[li];
        if (!s.ok) {
            continue;
        }
        const auto S = to_root.find(s.dev_snd);
        const auto R = to_root.find(s.dev_rcv);
        if (S == to_root.end() || R == to_root.end() || !S->second.ok || !R->second.ok) {
            continue;
        }
        const double direct = (1.0 + s.rate) * s.mid_refclk + (s.offset_refclk - s.rate * s.mid_refclk);
        const double via_mesh = (S->second.scale * s.mid_refclk + S->second.shift - R->second.shift) / R->second.scale;
        const double off_ns = (via_mesh - direct) * kNsPerRefclk;
        if (weights[li] < 0.5) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): its stamps sit {:+.1f} "
                "ns "
                "off the mesh (weight {:.2f}); a stamp reference at one end moved this launch, the link is left out "
                "of the placement",
                ctx_.links[li].chip_a,
                ctx_.links[li].eth_a.x,
                ctx_.links[li].eth_a.y,
                ctx_.links[li].chip_b,
                ctx_.links[li].eth_b.x,
                ctx_.links[li].eth_b.y,
                off_ns,
                weights[li]);
        } else {
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {}: {:+.1f} ns off the mesh (path "
                "asymmetry; weight {:.2f}, solution good to ~{:.1f} ns)",
                ctx_.links[li].chip_a,
                ctx_.links[li].eth_a.x,
                ctx_.links[li].eth_a.y,
                ctx_.links[li].chip_b,
                off_ns,
                weights[li],
                s.precision_ns);
        }
    }
}

// The link solve: receiver refclk = sender refclk + offset + rate * (sender refclk - mean midpoint), a straight line
// through the rounds' (midpoint, receiver minus midpoint) points with two passes of 3-sigma trimming. A solution
// that is not finite, beyond 100 ppm or a millisecond of residual is refused and the previous one stands.
bool LinkSolver::solve_link(const CaptureContext::Link& L, std::vector<RoundPoint> pts, LinkSolution& out) const {
    if (pts.size() < 4) {
        return false;
    }
    double mid0 = pts.front().mid_refclk;
    long double mid_acc = 0;
    for (const RoundPoint& p : pts) {
        mid0 = std::min(mid0, p.mid_refclk);
        mid_acc += p.mid_refclk;
    }
    std::vector<char> keep(pts.size(), 1);
    double inter = 0.0, slope = 0.0, rms = 0.0;
    size_t nk = 0;
    for (int pass = 0; pass < 3; pass++) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        nk = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (!keep[i]) {
                continue;
            }
            const double x = pts[i].mid_refclk - mid0, y = pts[i].off_refclk;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            nk++;
        }
        if (nk < 4) {
            return false;
        }
        const double nn = static_cast<double>(nk);
        const double den = nn * sxx - sx * sx;
        slope = std::abs(den) > 1e-9 ? (nn * sxy - sx * sy) / den : 0.0;
        inter = (sy - slope * sx) / nn;
        double ss = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (keep[i]) {
                const double res = pts[i].off_refclk - (inter + slope * (pts[i].mid_refclk - mid0));
                ss += res * res;
            }
        }
        rms = std::sqrt(ss / nn);
        if (pass == 2) {
            break;
        }
        const double cut = 3.0 * std::max(rms, 0.5);
        for (size_t i = 0; i < pts.size(); i++) {
            if (keep[i] && std::abs(pts[i].off_refclk - (inter + slope * (pts[i].mid_refclk - mid0))) > cut) {
                keep[i] = 0;
            }
        }
    }
    if (!std::isfinite(inter) || !std::isfinite(slope) || std::abs(slope) > 1e-4 || rms * kNsPerRefclk > 1e6) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} -> chip {}: link solution refused (offset {:.1f} ns, rate "
            "{:.3f} ppm, "
            "residual {:.1f} ns); keeping the previous one",
            L.chip_a,
            L.chip_b,
            inter * kNsPerRefclk,
            slope * 1e6,
            rms * kNsPerRefclk);
        return false;
    }
    out.ok = true;
    out.dev_snd = L.dev_a;
    out.dev_rcv = L.dev_b;
    out.rate = slope;
    out.mid_refclk = static_cast<double>(mid_acc / static_cast<long double>(pts.size()));
    // The fit's intercept sits at mid0, the window's first round; the solution is read as offset + rate * (mid -
    // out.mid_refclk), so move it to the mean or every consumer carries rate * half a window (0.45 ppm * 0.5 s = 225
    // ns).
    out.offset_refclk = inter + slope * (out.mid_refclk - mid0);
    out.residual_rms_ns = rms * kNsPerRefclk;
    out.precision_ns = rms * kNsPerRefclk / std::sqrt(static_cast<double>(nk));
    out.rounds = pts.size();
    out.kept = nk;
    return true;
}

bool SyncEngine::round_error(
    const CaptureContext::Link& L, const Round& r, bool anchored, int64_t& tsc_a, double& err, RoundTerms* terms)
    const {
    if (L.dev_a >= ctx_.devices.size() || L.dev_b >= ctx_.devices.size()) {
        return false;
    }
    const auto la = local_.find(L.dev_a);
    const auto lb = local_.find(L.dev_b);
    if (la == local_.cend() || lb == local_.cend()) {
        return false;
    }
    if (!series_.has_nodes(L.dev_a) || !series_.has_nodes(L.dev_b)) {
        return false;
    }
    // Anchored: each end's wall clock at the round's midpoint from the (wall, refclk) pair it read together when it
    // recorded the stamp, moved to the midpoint by the model's slope over that ~1 ms, a measured AICLK instant.
    // Otherwise the model's wall for the midpoint, which cancels the model out of the error.
    const double mid_a = LinkSolver::mid_a_refclk(r), mid_b = LinkSolver::mid_b_refclk(r);
    anchored = anchored && r.t2.ref != 0 && r.t1.ref != 0;
    const double wa = anchored ? static_cast<double>(r.t2.wall) + la->second.wall_at(mid_a) -
                                     la->second.wall_at(static_cast<double>(r.t2.ref))
                               : la->second.wall_at(mid_a);
    const double wb = anchored ? static_cast<double>(r.t1.wall) + lb->second.wall_at(mid_b) -
                                     lb->second.wall_at(static_cast<double>(r.t1.ref))
                               : lb->second.wall_at(mid_b);
    if (wa <= 0.0 || wb <= 0.0) {
        return false;
    }
    RoundTerms t;
    t.wall_a = wa;
    t.wall_b = wb;
    if (anchored) {
        const auto res_ns = [](const LocalClockModel& m, const LinkSolver::Stamp& st) {
            const double ref = static_cast<double>(st.ref);
            const double ghz = (m.wall_at(ref + 1.0) - m.wall_at(ref)) / kNsPerRefclk;
            return (static_cast<double>(st.wall) - m.wall_at(ref)) / std::max(ghz, 0.1);
        };
        t.res_a = res_ns(la->second, r.t2);
        t.res_b = res_ns(lb->second, r.t1);
        t.spins_a = r.t2.spins;
        t.spins_b = r.t1.spins;
        t.ref_a = static_cast<double>(r.t2.ref);
        t.wraw_a = static_cast<double>(r.t2.wall);
        t.ref_b = static_cast<double>(r.t1.ref);
        t.wraw_b = static_cast<double>(r.t1.wall);
    }
    t.root_a = map_.lookup_root(ctx_.devices[L.dev_a].chip_id, std::llround(wa));
    t.root_b = map_.lookup_root(ctx_.devices[L.dev_b].chip_id, std::llround(wb));
    tsc_a = std::llround(map_.host_tsc(t.root_a));
    err = (t.root_b - t.root_a) * kNsPerRefclk;
    if (terms != nullptr) {
        *terms = t;
    }
    return true;
}

struct SyncEngine::LinkErrors {
    std::vector<PlotPoint> pts;  // (sender's host placement, receiver minus sender on the root, ns)
    std::vector<RoundTerms> terms;
    std::vector<double> raw_x, raw_y;  // the round's sender midpoint and the receiver's offset from it, refclk
    std::vector<double> rtt, turn, path, resid;
    double path_med = 0.0, rtt_median = 0.0;
    size_t off_path = 0;
    size_t past_model = 0;  // rounds after an end's clock model stopped, in a transition the capture cut short
    size_t unbracketed = 0;  // rounds an end recorded with a plain (wall, refclk) pair: read_bracketed gave up
    size_t before_series = 0;  // rounds before a chip's oldest kept node: the series wrapped, nothing places them
};

// The rounds inside the path band, placed through the final map. Next to the placement error: the sender's round
// trip and the one way and turnaround inside the stamps.
SyncEngine::LinkErrors SyncEngine::link_errors(size_t li, bool anchored) const {
    const CaptureContext::Link& L = ctx_.links[li];
    const std::vector<Round>& rounds = links_.rounds(li);
    LinkErrors e;
    e.path_med = LinkSolver::path_median(rounds, 0, rounds.size());
    const auto until = [&](uint32_t dev) {
        const auto it = local_.find(dev);
        return it == local_.end() ? 0.0 : it->second.frontier();
    };
    const double until_a = until(L.dev_a), until_b = until(L.dev_b);
    const int64_t oldest_a = map_.oldest_at(L.chip_a), oldest_b = map_.oldest_at(L.chip_b);
    for (const Round& r : rounds) {
        if (std::abs(LinkSolver::path_ns(r) - e.path_med) > LinkSolver::kPathDevNs) {
            e.off_path++;
            continue;
        }
        if (LinkSolver::mid_a_refclk(r) > until_a || LinkSolver::mid_b_refclk(r) > until_b) {
            e.past_model++;
            continue;
        }
        if (static_cast<int64_t>(r.t2.wall) < oldest_a || static_cast<int64_t>(r.t1.wall) < oldest_b) {
            e.before_series++;
            continue;
        }
        int64_t H = 0;
        double err = 0.0;
        RoundTerms t;
        if (!round_error(L, r, anchored, H, err, &t)) {
            continue;
        }
        e.pts.push_back(PlotPoint{H, err});
        e.terms.push_back(t);
        e.unbracketed += anchored && (r.t2.spins == 0 || r.t1.spins == 0);
        e.raw_x.push_back(LinkSolver::mid_a_refclk(r));
        e.raw_y.push_back(LinkSolver::mid_b_refclk(r) - e.raw_x.back());
        e.rtt.push_back(LinkSolver::rtt_ns(r));
        e.path.push_back(LinkSolver::path_ns(r));
        e.turn.push_back(e.rtt.back() - 2.0 * e.path.back());
    }
    if (!e.rtt.empty()) {
        std::vector<double> tmp = e.rtt;
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        e.rtt_median = tmp[tmp.size() / 2];
    }
    return e;
}

// Each round's raw offset against a line through the neighbouring rounds': the ruler's own noise. A stamp glitch
// shows here, a map error does not; the link rate wanders ~0.1 ppm over a run, so a single run-wide line would not do.
void SyncEngine::stamp_residuals(LinkErrors& e) {
    e.resid.assign(e.pts.size(), 0.0);
    if (e.pts.size() < 8) {
        return;
    }
    constexpr size_t kHalf = 12;
    for (size_t i = 0; i < e.pts.size(); i++) {
        const size_t lo = i > kHalf ? i - kHalf : 0, hi = std::min(e.pts.size(), i + kHalf + 1);
        long double sx = 0, sy = 0, sxx = 0, sxy = 0;
        size_t m = 0;
        for (size_t j = lo; j < hi; j++) {
            if (j == i) {
                continue;
            }
            const long double x = e.raw_x[j] - e.raw_x[i], y = e.raw_y[j];
            sx += x, sy += y, sxx += x * x, sxy += x * y, m++;
        }
        const long double den = static_cast<long double>(m) * sxx - sx * sx;
        const long double b = den > 0 ? (static_cast<long double>(m) * sxy - sx * sy) / den : 0;
        const long double a = (sy - b * sx) / static_cast<long double>(m);
        e.resid[i] = static_cast<double>((e.raw_y[i] - a) * kNsPerRefclk);
    }
}

void SyncEngine::log_link_stats(const CaptureContext::Link& L, const LinkErrors& e, size_t rounds) const {
    const auto pct = [](std::vector<double> v, double q) {
        std::erase_if(v, [](double x) { return std::isnan(x); });
        if (v.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const size_t k = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    };
    log_info(
        tt::LogMetal,
        "[streaming profiler] d2d sync link chip {} -> chip {}: one way inside the stamps {:.1f} ns (p10 {:.1f}, p90 "
        "{:.1f}); receiver's stamped turnaround {:.1f} ns (p10 {:.1f}, p90 {:.1f}); sender's round trip {:.1f} ns "
        "(p10 {:.1f}, p90 {:.1f}); {} rounds, {} off the path band dropped, {} past a chip's clock model, {} read "
        "unbracketed, {} before a chip's oldest kept node",
        L.chip_a,
        L.chip_b,
        pct(e.path, 0.5),
        pct(e.path, 0.1),
        pct(e.path, 0.9),
        pct(e.turn, 0.5),
        pct(e.turn, 0.1),
        pct(e.turn, 0.9),
        pct(e.rtt, 0.5),
        pct(e.rtt, 0.1),
        pct(e.rtt, 0.9),
        rounds,
        e.off_path,
        e.past_model,
        e.unbracketed,
        e.before_series);
    long double se = 0, ss = 0, srr = 0;
    for (size_t i = 0; i < e.pts.size(); i++) {
        se += e.pts[i].value;
        ss += static_cast<long double>(e.pts[i].value) * e.pts[i].value;
        srr += static_cast<long double>(e.resid[i]) * e.resid[i];
    }
    const double nn = static_cast<double>(e.pts.size());
    log_info(
        tt::LogMetal,
        "[streaming profiler] d2d sync check chip {} vs chip {} (AICLK-anchored{}): {} rounds, mean {:+.1f} ns, rms "
        "{:.1f} ns{}",
        L.chip_b,
        L.chip_a,
        e.unbracketed != 0 ? fmt::format(", {} of them from plain reads, ~30 ns each", e.unbracketed) : "",
        e.pts.size(),
        static_cast<double>(se) / nn,
        std::sqrt(static_cast<double>(ss) / nn),
        e.pts.size() >= 8 ? fmt::format(
                                "; the stamps' own noise (residual to the neighbouring rounds): rms {:.1f} ns",
                                std::sqrt(static_cast<double>(srr) / nn))
                          : "");
}

// The five worst rounds and their two glitch indicators: a map error has a small stamp residual; a stamp glitch has a
// large one and often a shifted round trip.
void SyncEngine::log_worst_rounds(const CaptureContext::Link& L, const LinkErrors& e) const {
    std::vector<size_t> order(e.pts.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = i;
    }
    std::partial_sort(
        order.begin(), order.begin() + std::min<size_t>(5, order.size()), order.end(), [&](size_t a, size_t b) {
            return std::abs(e.pts[a].value) > std::abs(e.pts[b].value);
        });
    std::string worst;
    for (size_t k = 0; k < std::min<size_t>(5, order.size()); k++) {
        const size_t i = order[k];
        const PlotPoint& p = e.pts[i];
        worst += fmt::format(
            " {:+.0f} ns at {:.3f} s (stamp resid {:+.0f} ns, path {:+.1f} ns vs median);",
            p.value,
            static_cast<double>(SteadyView::mono_ns(p.tsc) - SteadyView::mono_ns(e.pts.front().tsc)) / 1e9,
            e.resid[i],
            e.path[i] - e.path_med);
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] d2d sync check chip {} vs chip {}: worst rounds{}",
        L.chip_b,
        L.chip_a,
        worst);
}

void SyncEngine::write_err_csv(const CaptureContext::Link& L, const LinkErrors& e) const {
    const std::string& csv = ctx_.d2d_csv_path;
    if (csv.empty()) {
        return;
    }
    std::FILE* ef = std::fopen(
        fmt::format("{}.err_{}_{}_eth{}_{}.csv", csv, L.chip_b, L.chip_a, L.eth_a.x, L.eth_a.y).c_str(), "w");
    if (ef == nullptr) {
        return;
    }
    std::fprintf(
        ef,
        "host_ns,err_ns,stamp_resid_ns,rtt_dev_ns,mid_a_refclk,r1_b_refclk,wall_a,wall_b,root_a,root_b,rtt_ns,"
        "turn_ns,path_ns,res_a_ns,res_b_ns,spins_a,spins_b,ref_a,wraw_a,ref_b,wraw_b\n");
    for (size_t i = 0; i < e.pts.size(); i++) {
        const RoundTerms& t = e.terms[i];
        std::fprintf(
            ef,
            "%lld,%.2f,%.2f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.2f,%.2f,%u,%u,%.0f,%.0f,%.0f,%.0f\n",
            static_cast<long long>(SteadyView::mono_ns(e.pts[i].tsc)),
            e.pts[i].value,
            e.resid[i],
            e.rtt[i] - e.rtt_median,
            e.raw_x[i],
            e.raw_x[i] + e.raw_y[i],
            t.wall_a,
            t.wall_b,
            t.root_a,
            t.root_b,
            e.rtt[i],
            e.turn[i],
            e.path[i],
            t.res_a,
            t.res_b,
            t.spins_a,
            t.spins_b,
            t.ref_a,
            t.wraw_a,
            t.ref_b,
            t.wraw_b);
    }
    std::fclose(ef);
}

void SyncEngine::write_model_csv() const {
    const std::string& csv = ctx_.d2d_csv_path;
    if (csv.empty()) {
        return;
    }
    for (const auto& [dev, l] : local_) {
        const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
        std::FILE* f = std::fopen(fmt::format("{}.model_{}.csv", csv, chip).c_str(), "w");
        if (f == nullptr) {
            continue;
        }
        std::fprintf(f, "kind,r,w,k8,resid\n");
        for (const LocalClockModel::Instant& p : l.pts) {
            std::fprintf(f, "%s,%.1f,%.3f,%u,0\n", p.k8 == 0 ? "raw" : p.close ? "close" : "point", p.r, p.w, p.k8);
        }
        for (const auto& [r, e] : l.resid) {
            std::fprintf(f, "resid,%.1f,0,0,%.3f\n", r, e);
        }
        std::fclose(f);
    }
}

// Per link and round: the receiver's stamp and the sender's round midpoint (one instant, under the symmetric path)
// placed on the host timeline exactly as the sink places a record from each chip's eth core, against the final map.
void SyncEngine::publish_error_plots() {
    write_model_csv();
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        const size_t rounds = links_.rounds(li).size();
        if (rounds == 0) {
            continue;
        }
        LinkErrors e = link_errors(li, /*anchored=*/true);
        if (e.pts.empty()) {
            continue;
        }
        stamp_residuals(e);
        log_link_stats(L, e, rounds);
        log_worst_rounds(L, e);
        write_err_csv(L, e);
        // The anchored check reads the wall clock to a cycle only where the link end brackets its reads; a router's
        // plain pair puts the refclk register's 80 ns step into every round, so it is not plotted as an error.
        if (e.unbracketed == 0) {
            plot(fmt::format("d2d sync check chip{} vs chip{}, AICLK-anchored (ns)", L.chip_b, L.chip_a), e.pts);
        }
        // The same rounds with the model cancelled: the links' and the map's own error, at the stamps' 0.3 ns. With
        // each chip's model bound added, the worst a record of either chip can be off the other's. That series is
        // drawn at every point of either chip's model, each carrying the worst residual over all the samples since
        // the previous point, with the cross-chip error taken from the nearest round: crystals and nodes move slowly,
        // the models do not. The signed cross-chip error stays in the log and the CSV.
        const LinkErrors em = link_errors(li, /*anchored=*/false);
        if (em.pts.empty()) {
            continue;
        }
        const auto la = local_.find(L.dev_a), lb = local_.find(L.dev_b);
        const auto xa = to_root_.find(L.dev_a), xb = to_root_.find(L.dev_b);
        if (la == local_.end() || lb == local_.end() || xa == to_root_.end() || xb == to_root_.end()) {
            continue;
        }
        const double ghz_a = std::max(ctx_.devices[L.dev_a].frequency_ghz, 0.1);
        const double ghz_b = std::max(ctx_.devices[L.dev_b].frequency_ghz, 0.1);
        double se = 0, ss = 0, worst = 0;
        for (const PlotPoint& p : em.pts) {
            se += p.value;
            ss += p.value * p.value;
            worst = std::max(worst, std::abs(p.value));
        }
        const double nn = static_cast<double>(em.pts.size());
        // Both chips' point instants on the root, in order; the round nearest each carries the cross-chip term.
        struct At {
            int64_t tsc;
            double root;
            bool a;
            double resid_ns;
        };
        std::vector<At> at;
        at.reserve(la->second.resid.size() + lb->second.resid.size());
        for (const auto& [r, ticks] : la->second.resid) {
            const double root = xa->second.scale * r + xa->second.shift;
            at.push_back(At{std::llround(map_.host_tsc(root)), root, true, ticks / ghz_a});
        }
        for (const auto& [r, ticks] : lb->second.resid) {
            const double root = xb->second.scale * r + xb->second.shift;
            at.push_back(At{std::llround(map_.host_tsc(root)), root, false, ticks / ghz_b});
        }
        std::sort(at.begin(), at.end(), [](const At& x, const At& y) { return x.tsc < y.tsc; });
        std::vector<PlotPoint> bound;
        bound.reserve(at.size());
        std::vector<double> bounds;
        bounds.reserve(at.size());
        size_t ri = 0;
        double other_a = 0.0, other_b = 0.0;  // each chip's newest residual, in force until its next point
        for (const At& x : at) {
            if (x.tsc < em.pts.front().tsc) {
                continue;  // before the first placed round: no cross-chip term measured there
            }
            while (ri + 1 < em.pts.size() && em.pts[ri + 1].tsc <= x.tsc) {
                ri++;
            }
            (x.a ? other_a : other_b) = x.resid_ns;
            bound.push_back(PlotPoint{x.tsc, std::abs(em.pts[ri].value) + other_a + other_b});
            bounds.push_back(bound.back().value);
        }
        if (bounds.empty()) {
            continue;
        }
        std::nth_element(bounds.begin(), bounds.begin() + bounds.size() / 2, bounds.end());
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync error chip {} vs chip {} (links and map, model cancelled): {} rounds, mean "
            "{:+.2f} ns, rms {:.2f} ns, worst {:.2f} ns; with both chips' model bounds at their {} points: median "
            "{:.2f} ns, worst {:.2f} ns",
            L.chip_b,
            L.chip_a,
            em.pts.size(),
            se / nn,
            std::sqrt(ss / nn),
            worst,
            bounds.size(),
            bounds[bounds.size() / 2],
            *std::max_element(bounds.begin(), bounds.end()));
        plot(fmt::format("d2d sync error chip{} vs chip{} (ns)", L.chip_b, L.chip_a), bound);
    }
}

// The drainer's anchors against each chip's model, and per link what a record of either chip is off the other's:
// the link and map term of each round with both models' errors, the chips' clocks being independent processes.
void SyncEngine::log_audit() const {
    const auto parts = [](const ErrorHistogram& h) {
        return fmt::format(
            "{:.0f} (mean {:+.2f}, rms {:.2f}, |err| p50 {:.2f}, p99 {:.2f}, p99.9 {:.2f}, worst {:.2f} ns)",
            h.n,
            h.mean(),
            h.rms(),
            h.abs_quantile(0.5),
            h.abs_quantile(0.99),
            h.abs_quantile(0.999),
            h.worst);
    };
    std::map<uint32_t, AnchorAudit> audit = audit_;
    for (auto& [dev, a] : audit) {
        const auto m = local_.find(dev);
        if (m != local_.end()) {
            a.settle(m->second, /*final=*/true);
        }
        const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync audit chip {}: the drainer's anchors against the clock model, on lines {}; "
            "inside transitions {}; worst at refclk {:.0f}; {} before the model, {} past it",
            chip,
            parts(a.line),
            parts(a.glide),
            a.worst_r,
            a.before_model,
            a.past_model);
    }
    ErrorHistogram pooled;
    double bound = 0.0;
    size_t links = 0;
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        const auto aa = audit.find(L.dev_a), ab = audit.find(L.dev_b);
        if (aa == audit.end() || ab == audit.end() || links_.rounds(li).empty()) {
            continue;
        }
        const LinkErrors em = link_errors(li, /*anchored=*/false);
        if (em.pts.empty()) {
            continue;
        }
        ErrorHistogram lh;
        for (const PlotPoint& p : em.pts) {
            lh.add(p.value);
        }
        const ErrorHistogram ea = aa->second.both(), eb = ab->second.both();
        if (ea.n <= 0.0 || eb.n <= 0.0) {
            continue;
        }
        const ErrorHistogram total = lh.convolve(eb, 1).convolve(ea, -1);
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync audit chip {} vs chip {} eth({},{}): a record of one against one of the other, "
            "|err| p50 {:.2f}, p99 {:.2f}, p99.9 {:.2f} ns, mean {:+.2f} ns; bound {:.2f} ns (link worst {:.2f} + "
            "models {:.2f} + {:.2f})",
            L.chip_b,
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            total.abs_quantile(0.5),
            total.abs_quantile(0.99),
            total.abs_quantile(0.999),
            total.mean(),
            total.worst,
            lh.worst,
            eb.worst,
            ea.worst);
        for (int b = 0; b < ErrorHistogram::kBins; b++) {
            pooled.bins[b] += total.bins[b];
        }
        pooled.n += 1.0;
        pooled.sum += total.sum;
        pooled.sumsq += total.sumsq;
        pooled.beyond += total.beyond;
        bound = std::max(bound, total.worst);
        links++;
    }
    if (links != 0) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync audit over {} links: a record against one on a linked chip, |err| p50 {:.2f}, "
            "p99 {:.2f}, p99.9 {:.2f} ns; bound {:.2f} ns",
            links,
            pooled.abs_quantile(0.5),
            pooled.abs_quantile(0.99),
            pooled.abs_quantile(0.999),
            bound);
    }
}

void SyncEngine::publish_clock_plots() {
    for (const auto& [dev, fit] : local_) {
        if (!series_.has_nodes(dev) || dev >= ctx_.devices.size()) {
            continue;
        }
        const uint32_t chip = ctx_.devices[dev].chip_id;
        std::vector<PlotPoint> pts;
        for (const LocalClockModel::Instant& p : fit.pts) {
            if (p.k8 == 0) {
                continue;
            }
            const double root = map_.lookup_root(chip, std::llround(p.w));
            pts.push_back(PlotPoint{std::llround(map_.host_tsc(root)), p.k8 / 8.0 * LocalClockModel::kRefclkHz * 1e-9});
        }
        if (!pts.empty()) {
            plot(fmt::format("AICLK chip{} (GHz)", chip), pts);
        }
    }
}

void SyncEngine::plot([[maybe_unused]] const std::string& name, [[maybe_unused]] const std::vector<PlotPoint>& points) {
#if defined(TRACY_ENABLE)
    const char* nm = plot_names_.insert(name).first->c_str();
    for (const PlotPoint& p : points) {
        if (p.tsc > 0) {
            tracy::Profiler::PlotDataAt(nm, p.value, p.tsc);
        }
    }
#endif
}

void SyncEngine::on_capture_end(const CaptureContext& ctx) {
    (void)ctx;
    links_.solve_final();
    publish_all();
    publish_error_plots();
    publish_clock_plots();
    log_summary();
    log_audit();
    // The published corrections stay for the sinks that write at process end; the next attach starts fresh.
    local_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
