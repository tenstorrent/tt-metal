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
constexpr uint32_t kSteadySeries = kHostSeries - 1;

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
                chip_id == kHostSeries     ? std::string("the host")
                : chip_id == kSteadySeries ? std::string("the host steady clock")
                                           : "chip " + std::to_string(chip_id),
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
            c = Cursor<Key>{.gen = gen, .origin = a.at, .value = a.value, .slope = a.tangent};
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
    Cursor<int64_t> steady{};
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
        chips{{((void)I, Log<int64_t>(series_nodes))...}}, host(series_nodes), steady(series_nodes) {}
    std::array<Log<int64_t>, kMaxChips> chips;
    Log<double> host;
    Log<int64_t> steady;
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

void ClockMap::finish(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        impl_->chips[chip_id].extend(std::numeric_limits<int64_t>::max());
        impl_->cover_generation.fetch_add(1, std::memory_order_release);
    }
}

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

double ClockMap::lookup_root(uint32_t chip_id, double wall) const noexcept {
    if (chip_id >= kMaxChips) {
        return 0.0;
    }
    Cursor<int64_t>& c = view_of(impl_.get()).chip[chip_id];
    const auto tick = static_cast<int64_t>(std::floor(wall));
    double root = 0.0;
    if (!place(impl_->chips[chip_id], c, tick, root)) {
        return 0.0;
    }
    return root + c.slope * (wall - static_cast<double>(tick));
}

void ClockMap::append_steady(ClockNode<int64_t> node) {
    TT_FATAL(
        std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0,
        "streaming profiler: steady clock node at tsc {} is not a rate: {} ns, tangent {}",
        node.at,
        node.value,
        node.tangent);
    impl_->steady.append(kSteadySeries, node);
}

bool ClockMap::steady_ns(int64_t tsc, double& ns) const noexcept {
    return place(impl_->steady, view_of(impl_.get()).steady, tsc, ns);
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

// An anchor is checked once an instant lies past it: the chord between the two around it no longer changes.
void AnchorAudit::settle(const LocalClockModel& m, bool final) {
    const auto& pts = m.pts;
    if (pts.size() < 2 && !final) {
        return;
    }
    const int64_t until = m.frontier();
    while (!pending.empty() && pending.front().r < until) {
        const Pending a = pending.front();
        pending.pop_front();
        if (a.r < pts.front().r) {
            before_model++;
            continue;
        }
        const double r = static_cast<double>(a.r);
        const double w = m.wall_at(r);
        const double ns = (static_cast<double>(a.w) - w) * kNsPerRefclk / (m.wall_at(r + 1.0) - w);
        err.add(ns);
        if (std::abs(ns) > std::abs(worst_ns)) {
            worst_ns = ns;
            worst_r = r;
        }
    }
    if (final) {
        past_model += pending.size();
        pending.clear();
    }
}

void AnchorAudit::add_drainer_audit(const ClockSample& s) {
    constexpr double kUnitsPerNs = 16.0;  // the drainer's bins and sums are in 1/16 ns
    if (s.round == kernel_profiler::kSyncAnchorHistSummary) {
        const double w = static_cast<int32_t>(static_cast<uint32_t>(s.ref >> 32)) / kUnitsPerNs;
        err.sum += static_cast<int64_t>(s.value) / kUnitsPerNs;
        err.sumsq += static_cast<double>(s.ts) / (kUnitsPerNs * kUnitsPerNs);
        if (std::abs(w) > std::abs(drainer_worst_ns)) {
            drainer_worst_ns = w;
        }
        err.worst = std::max(err.worst, std::abs(w));
        return;
    }
    if (s.round == kernel_profiler::kSyncAnchorHistWorstAt) {
        drainer_worst_r = static_cast<double>(s.value);
        drainer_unbracketed += s.ts;
        return;
    }
    const uint32_t counts[6] = {
        static_cast<uint32_t>(s.value),
        static_cast<uint32_t>(s.value >> 32),
        static_cast<uint32_t>(s.ts),
        static_cast<uint32_t>(s.ts >> 32),
        static_cast<uint32_t>(s.ref),
        static_cast<uint32_t>(s.ref >> 32)};
    for (uint32_t j = 0; j < 6; j++) {
        const int b = static_cast<int>(s.round + j) - static_cast<int>(kernel_profiler::kSyncAnchorHistBins / 2) +
                      ErrorHistogram::kRangeNs * ErrorHistogram::kBinsPerNs;
        err.bins[b] += counts[j];
        err.n += counts[j];
    }
}

void SyncEngine::on_attach(const CaptureContext& ctx) {
    const size_t n = ctx.devices.size();
    TT_FATAL(ctx.root_dev < n, "streaming profiler: the sync root is device {} of {}", ctx.root_dev, n);
    for (const CaptureContext::Link& L : ctx.links) {
        TT_FATAL(
            L.dev_a < n && L.dev_b < n,
            "streaming profiler: the sync link chip {} -> chip {} names a device outside the capture",
            L.chip_a,
            L.chip_b);
    }
    ctx_ = ctx;
    local_.assign(n, LocalClockModel{});
    audit_.assign(n, AnchorAudit{});
    links_.reset(ctx_);
    series_.reset(n);
    to_root_gen_ = ~0ull;
    // A chip the capture cannot place -- no eth tracker, or no link path to the root -- is finished at once, so no
    // consumer waits for it.
    std::vector<bool> reach(n, false);
    const uint32_t root = ctx.root_dev;
    if (ctx.devices[root].has_eth_tracker) {
        reach[root] = true;
        for (bool progress = true; progress;) {
            progress = false;
            for (const CaptureContext::Link& L : ctx.links) {
                if (reach[L.dev_a] != reach[L.dev_b]) {
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

void SyncEngine::on_clock(uint32_t dev, uint32_t core, const uint32_t* rec) {
    namespace kp = kernel_profiler;
    TT_FATAL(dev < local_.size(), "streaming profiler: a sync record names device {} of {}", dev, local_.size());
    const uint32_t kind = (rec[kp::SYNC_META] >> 8) & 0xFFu;
    if (kind == kp::kSyncKindLocal) {
        kp::SyncLocalPoint pts[kp::kSyncLocalPoints];
        const uint32_t n = kp::sync_local_unpack(rec, pts);
        LocalClockModel& m = local_[dev];
        for (uint32_t i = 0; i < n; i++) {
            m.add_point(pts[i].r, pts[i].w8, pts[i].k8);
        }
        audit_[dev].settle(m, /*final=*/false);
        if (publish_dev(dev)) {
            service().wake_consumers();
        }
        return;
    }
    const auto word64 = [&](uint32_t lo, uint32_t hi) { return (static_cast<uint64_t>(rec[hi]) << 32) | rec[lo]; };
    const ClockSample s{
        .dev = dev,
        .core = core,
        .kind = kind,
        .round = rec[kp::SYNC_ROUND],
        .role = rec[kp::SYNC_META] & 0xFFu,
        .value = word64(kp::SYNC_VALUE_LO, kp::SYNC_VALUE_HI),
        .ts = word64(kp::SYNC_WALL_LO, kp::SYNC_WALL_HI),
        .ref = word64(kp::SYNC_REF_LO, kp::SYNC_REF_HI)};
    if (kind == kp::kSyncKindAnchor) {
        audit_[dev].pending.push_back(AnchorAudit::Pending{
            static_cast<int64_t>(s.ref), static_cast<int64_t>(s.ts) + ctx_.devices[dev].drainer_offset});
    } else if (kind == kp::kSyncKindAnchorHist) {
        audit_[dev].add_drainer_audit(s);
    } else {
        links_.on_stamp(s);
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
    r.*slot = Stamp{.units = s.value, .have = true};
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
        for (size_t i = begin; i < n; i++) {
            const double mid = mid_a_refclk(rounds[i]);
            pts.push_back(RoundPoint{mid, mid_b_refclk(rounds[i]) - mid});
        }
        solve_link(L, pts, out);
        gen_++;
    }
}

double LinkSolver::mid_a_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(r.t0.units) + static_cast<double>(r.t2.units)) * kRefclkPerStampUnit;
}

double LinkSolver::mid_b_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(r.t1.units) + static_cast<double>(r.t1b.units)) * kRefclkPerStampUnit;
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

// Least squares for a potential on a graph: each edge says node_a - node_b = value; the root is fixed at zero. An
// edge's end is the device's unknown, -1 for the root.
bool solve_potential(
    const std::vector<std::array<int, 2>>& ends, const std::vector<double>& value, size_t n, std::vector<double>& x) {
    std::vector<double> m(n * n, 0.0), rhs(n, 0.0);
    for (size_t i = 0; i < ends.size(); i++) {
        const int a = ends[i][0], b = ends[i][1];
        if (a >= 0) {
            m[a * n + a] += 1.0;
            rhs[a] += value[i];
        }
        if (b >= 0) {
            m[b * n + b] += 1.0;
            rhs[b] -= value[i];
        }
        if (a >= 0 && b >= 0) {
            m[a * n + b] -= 1.0;
            m[b * n + a] -= 1.0;
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
// from the midpoint, nothing). Every link weighs the same: its error is the fixed term each link training draws
// (~0.5 ns rms), which its stamps' precision (~0.06 ns) says nothing about.
std::vector<RootXf> LinkSolver::root_transforms(uint32_t root) const {
    std::vector<RootXf> to_root(ctx_->devices.size());
    to_root[root] = RootXf{1.0, 0.0, true};
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
    std::vector<std::array<int, 2>> ends;
    std::vector<double> log_rate, mid, offset;
    for (size_t li = 0; li < solved_.size(); li++) {
        const LinkSolution& s = solved_[li];
        if (!s.ok || idx.count(s.dev_snd) == 0 || idx.count(s.dev_rcv) == 0) {
            continue;
        }
        ends.push_back({idx[s.dev_snd], idx[s.dev_rcv]});
        log_rate.push_back(std::log1p(s.rate));  // A_snd = A_rcv * (1 + rate)
        mid.push_back(s.mid_refclk);
        offset.push_back(s.offset_refclk);
    }
    std::vector<double> x;
    if (!solve_potential(ends, log_rate, n, x)) {
        return to_root;
    }
    const auto A_of = [&](int i) { return i < 0 ? 1.0 : std::exp(x[i]); };
    // Offsets: at the link's midpoint the sender reads mid and the receiver mid + offset, one instant on the root:
    // A_snd * mid + B_snd = A_rcv * (mid + offset) + B_rcv.
    std::vector<double> value(ends.size()), B;
    for (size_t i = 0; i < ends.size(); i++) {
        value[i] = A_of(ends[i][1]) * (mid[i] + offset[i]) - A_of(ends[i][0]) * mid[i];
    }
    if (!solve_potential(ends, value, n, B)) {
        return to_root;
    }
    for (const auto& [dev, i] : idx) {
        if (i >= 0) {
            to_root[dev] = RootXf{A_of(i), B[i], true};
        }
    }
    return to_root;
}

// Frozen nodes never move (consumers have placed records against them), so a publish can only add beyond them, at
// the newest estimate's values; a join carries whatever the estimate moved by since the last node (a few ns at a
// knot). Shifting fresh nodes to meet the frozen tail and fading that shift over a quarter second is worse: every
// discrepancy at a join becomes a level the map carries for 250 ms, 30-60 ns during DVFS dithering at 1 ms.
bool SeriesPublisher::publish(uint32_t dev, uint32_t chip, const LocalClockModel& fit, const RootXf& xf) {
    Series& s = series_[dev];
    const auto& pts = fit.pts;
    const size_t before = s.count;
    const auto root_at = [&](const LocalClockModel::Instant& p) {
        return xf.scale * static_cast<double>(p.r) + xf.shift;
    };
    for (; s.next < pts.size(); s.next++) {
        const LocalClockModel::Instant& p = pts[s.next];
        const double root = root_at(p);
        double tangent = 0.0;
        if (p.k8 != 0) {
            tangent = xf.scale * 8.0 / p.k8;
        } else if (pts.size() > 1) {
            const LocalClockModel::Instant& o = pts[s.next == 0 ? 1 : s.next - 1];
            tangent = (root_at(o) - root) / (o.wall() - p.wall());
        } else {
            break;
        }
        if (std::isfinite(root) && tangent > 0.0 && tangent < 1.0) {
            map_.append(chip, SyncNode{.at = p.wall_tick(), .value = root, .tangent = tangent});
            s.count++;
        }
    }
    return s.count > before;
}

bool SyncEngine::publish_dev(uint32_t dev) {
    // Nothing is placed before the host series exists: a record converted then would land nowhere.
    if (map_.host_published() == 0) {
        return false;
    }
    if (to_root_gen_ != links_.generation()) {
        to_root_ = links_.root_transforms(root_dev());
        to_root_gen_ = links_.generation();
    }
    // The series starts only once the chip is on the root's tree (the root is there from the start): its first node
    // fixes the placement every later node joins.
    if (!to_root_[dev].ok) {
        return false;
    }
    return series_.publish(dev, ctx_.devices[dev].chip_id, local_[dev], to_root_[dev]);
}

void SyncEngine::publish_all() {
    for (uint32_t dev = 0; dev < local_.size(); dev++) {
        publish_dev(dev);
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
    for (uint32_t dev = 0; dev < local_.size(); dev++) {
        const LocalClockModel& l = local_[dev];
        if (l.pts.empty()) {
            continue;
        }
        // The applied AICLK: each point's k/8 slope over the refclk until the next instant.
        long double ssum = 0.0, wsum = 0.0;
        double smin = std::numeric_limits<double>::max(), smax = 0.0;
        size_t raw = 0, changes = 0;
        for (size_t i = 0; i < l.pts.size(); i++) {
            const LocalClockModel::Instant& p = l.pts[i];
            changes += l.ends_line(i);
            if (p.k8 == 0) {
                raw++;
                continue;
            }
            const double s = p.k8 / 8.0;
            const double span = i + 1 < l.pts.size() ? static_cast<double>(l.pts[i + 1].r - p.r) : 0.0;
            ssum += static_cast<long double>(s) * span;
            wsum += span;
            smin = std::min(smin, s);
            smax = std::max(smax, s);
        }
        const double to_ghz = LocalClockModel::kRefclkHz * 1e-9;
        const double mean_ghz = wsum > 0.0 ? static_cast<double>(ssum / wsum) * to_ghz : 0.0;
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync chip {}: local clock {} instants ({} single samples) across {} FBDIV "
            "changes; applied AICLK mean {:.5f} GHz (min {:.5f}, max {:.5f}); {} placement nodes",
            ctx_.devices[dev].chip_id,
            l.pts.size(),
            raw,
            changes,
            mean_ghz,
            smax > 0.0 ? smin * to_ghz : 0.0,
            smax * to_ghz,
            series_.nodes(dev));
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
            "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): {} rounds; refclk domain "
            "offset {:.1f} ns, rate {:.3f} ppm, residual rms {:.1f} ns, estimate precision {:.2f} ns",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            s.rounds,
            s.offset_refclk * kNsPerRefclk,
            s.rate * 1e6,
            s.residual_rms_ns,
            s.precision_ns);
    }
}

// Each link against the mesh solve: the difference between its own solution and the two chips' placements on the
// root. With every link on one consistent mesh the residuals are the cables' path asymmetries, a nanosecond or two.
void SyncEngine::log_loop_closures() const {
    const std::vector<LinkSolver::LinkSolution>& solved = links_.solutions();
    const std::vector<RootXf> to_root = links_.root_transforms(root_dev());
    for (size_t li = 0; li < solved.size() && li < ctx_.links.size(); li++) {
        const LinkSolver::LinkSolution& s = solved[li];
        if (!s.ok) {
            continue;
        }
        const RootXf& S = to_root[s.dev_snd];
        const RootXf& R = to_root[s.dev_rcv];
        if (!S.ok || !R.ok) {
            continue;
        }
        const double direct = (1.0 + s.rate) * s.mid_refclk + (s.offset_refclk - s.rate * s.mid_refclk);
        const double via_mesh = (S.scale * s.mid_refclk + S.shift - R.shift) / R.scale;
        const double off_ns = (via_mesh - direct) * kNsPerRefclk;
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {}: {:+.1f} ns off the mesh (path "
            "asymmetry; solution good to ~{:.1f} ns)",
            ctx_.links[li].chip_a,
            ctx_.links[li].eth_a.x,
            ctx_.links[li].eth_a.y,
            ctx_.links[li].chip_b,
            off_ns,
            s.precision_ns);
    }
}

// The link solve: receiver refclk = sender refclk + offset + rate * (sender refclk - mean midpoint), the least-squares
// line through the rounds' (midpoint, receiver minus midpoint) points.
void LinkSolver::solve_link(const CaptureContext::Link& L, const std::vector<RoundPoint>& pts, LinkSolution& out) {
    double mid0 = pts.front().mid_refclk;
    long double mid_acc = 0;
    for (const RoundPoint& p : pts) {
        mid0 = std::min(mid0, p.mid_refclk);
        mid_acc += p.mid_refclk;
    }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const RoundPoint& p : pts) {
        const double x = p.mid_refclk - mid0, y = p.off_refclk;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double nn = static_cast<double>(pts.size());
    const double den = nn * sxx - sx * sx;
    const double slope = std::abs(den) > 1e-9 ? (nn * sxy - sx * sy) / den : 0.0;
    const double inter = (sy - slope * sx) / nn;
    double ss = 0;
    for (const RoundPoint& p : pts) {
        const double res = p.off_refclk - (inter + slope * (p.mid_refclk - mid0));
        ss += res * res;
    }
    const double rms = std::sqrt(ss / nn);
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
    out.precision_ns = rms * kNsPerRefclk / std::sqrt(nn);
    out.rounds = pts.size();
}

bool SyncEngine::round_error(
    const CaptureContext::Link& L, const Round& r, int64_t& tsc_a, double& err, RoundTerms* terms) const {
    if (series_.nodes(L.dev_a) == 0 || series_.nodes(L.dev_b) == 0) {
        return false;
    }
    RoundTerms t;
    t.wall_a = local_[L.dev_a].wall_at(LinkSolver::mid_a_refclk(r));
    t.wall_b = local_[L.dev_b].wall_at(LinkSolver::mid_b_refclk(r));
    if (t.wall_a <= 0.0 || t.wall_b <= 0.0) {
        return false;
    }
    t.root_a = map_.lookup_root(ctx_.devices[L.dev_a].chip_id, t.wall_a);
    t.root_b = map_.lookup_root(ctx_.devices[L.dev_b].chip_id, t.wall_b);
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
    std::vector<double> rtt, turn, path;
    size_t past_model = 0;     // rounds past an end's newest instant
    size_t before_series = 0;  // rounds before a chip's oldest kept node: the series wrapped, nothing places them
};

// The rounds, placed through the final map. Next to the placement error: the sender's round trip and the one way and
// turnaround inside the stamps.
SyncEngine::LinkErrors SyncEngine::link_errors(size_t li) const {
    const CaptureContext::Link& L = ctx_.links[li];
    const std::vector<Round>& rounds = links_.rounds(li);
    LinkErrors e;
    const double rate = links_.solutions()[li].rate;
    const double until_a = static_cast<double>(local_[L.dev_a].frontier());
    const double until_b = static_cast<double>(local_[L.dev_b].frontier());
    const auto oldest_a = static_cast<double>(map_.oldest_at(L.chip_a));
    const auto oldest_b = static_cast<double>(map_.oldest_at(L.chip_b));
    for (const Round& r : rounds) {
        if (LinkSolver::mid_a_refclk(r) > until_a || LinkSolver::mid_b_refclk(r) > until_b) {
            e.past_model++;
            continue;
        }
        int64_t H = 0;
        double err = 0.0;
        RoundTerms t;
        if (!round_error(L, r, H, err, &t)) {
            continue;
        }
        if (t.wall_a < oldest_a || t.wall_b < oldest_b) {
            e.before_series++;
            continue;
        }
        e.pts.push_back(PlotPoint{H, err});
        e.terms.push_back(t);
        e.raw_x.push_back(LinkSolver::mid_a_refclk(r));
        e.raw_y.push_back(LinkSolver::mid_b_refclk(r) - e.raw_x.back());
        e.rtt.push_back(LinkSolver::rtt_ns(r));
        e.path.push_back(LinkSolver::path_ns(r, rate));
        e.turn.push_back(e.rtt.back() - 2.0 * e.path.back());
    }
    return e;
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
        "(p10 {:.1f}, p90 {:.1f}); {} rounds, {} past a chip's clock model, {} before a chip's oldest kept node",
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
        e.past_model,
        e.before_series);
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
    std::fprintf(ef, "host_ns,err_ns,mid_a_refclk,r1_b_refclk,wall_a,wall_b,root_a,root_b,rtt_ns,turn_ns,path_ns\n");
    for (size_t i = 0; i < e.pts.size(); i++) {
        const RoundTerms& t = e.terms[i];
        std::fprintf(
            ef,
            "%lld,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f\n",
            static_cast<long long>(SteadyView::mono_ns(e.pts[i].tsc)),
            e.pts[i].value,
            e.raw_x[i],
            e.raw_x[i] + e.raw_y[i],
            t.wall_a,
            t.wall_b,
            t.root_a,
            t.root_b,
            e.rtt[i],
            e.turn[i],
            e.path[i]);
    }
    std::fclose(ef);
}

void SyncEngine::write_model_csv() const {
    const std::string& csv = ctx_.d2d_csv_path;
    if (csv.empty()) {
        return;
    }
    for (uint32_t dev = 0; dev < local_.size(); dev++) {
        const LocalClockModel& l = local_[dev];
        if (l.pts.empty()) {
            continue;
        }
        std::FILE* f = std::fopen(fmt::format("{}.model_{}.csv", csv, ctx_.devices[dev].chip_id).c_str(), "w");
        if (f == nullptr) {
            continue;
        }
        std::fprintf(f, "kind,r,w,k8\n");
        for (size_t i = 0; i < l.pts.size(); i++) {
            const LocalClockModel::Instant& p = l.pts[i];
            const char* kind = p.k8 == 0 ? "raw" : l.ends_line(i) ? "close" : "point";
            std::fprintf(f, "%s,%.1f,%.3f,%u\n", kind, static_cast<double>(p.r), p.wall(), p.k8);
        }
        std::fclose(f);
    }
}

// Per link and round: the two ends' round midpoints (one instant, under the symmetric path) placed on the host
// timeline exactly as the sink places a record from each chip's eth core, against the final map. The chips' models
// cancel, so this is the links' and the map's own error; the signed cross-chip error stays in the CSV.
void SyncEngine::publish_error_plots() {
    write_model_csv();
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        const size_t rounds = links_.rounds(li).size();
        if (rounds == 0) {
            continue;
        }
        const LinkErrors e = link_errors(li);
        if (e.pts.empty()) {
            continue;
        }
        log_link_stats(L, e, rounds);
        write_err_csv(L, e);
        double se = 0, ss = 0, worst = 0;
        for (const PlotPoint& p : e.pts) {
            se += p.value;
            ss += p.value * p.value;
            worst = std::max(worst, std::abs(p.value));
        }
        const double nn = static_cast<double>(e.pts.size());
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync error chip {} vs chip {} (links and map, model cancelled): {} rounds, mean "
            "{:+.2f} ns, rms {:.2f} ns, worst {:.2f} ns",
            L.chip_b,
            L.chip_a,
            e.pts.size(),
            se / nn,
            std::sqrt(ss / nn),
            worst);
        plot(fmt::format("d2d sync error chip{} vs chip{} (ns)", L.chip_b, L.chip_a), e.pts);
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
    for (uint32_t dev = 0; dev < audit_.size(); dev++) {
        const AnchorAudit& a = audit_[dev];
        if (a.err.n <= 0.0 && a.pending.empty() && a.before_model + a.past_model + a.drainer_unbracketed == 0) {
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync audit chip {}: the drainer's anchors against the clock model, {}; worst at "
            "refclk {:.0f}; {} before the model, {} past it, {} dropped for a read with no refclk update",
            ctx_.devices[dev].chip_id,
            parts(a.err),
            std::abs(a.drainer_worst_ns) > std::abs(a.worst_ns) ? a.drainer_worst_r : a.worst_r,
            a.before_model,
            a.past_model,
            a.drainer_unbracketed);
    }
    ErrorHistogram pooled;
    size_t links = 0;
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        const ErrorHistogram& ea = audit_[L.dev_a].err;
        const ErrorHistogram& eb = audit_[L.dev_b].err;
        if (ea.n <= 0.0 || eb.n <= 0.0 || links_.rounds(li).empty()) {
            continue;
        }
        const LinkErrors em = link_errors(li);
        if (em.pts.empty()) {
            continue;
        }
        ErrorHistogram lh;
        for (const PlotPoint& p : em.pts) {
            lh.add(p.value);
        }
        const ErrorHistogram total = lh.convolve(eb, 1).convolve(ea, -1);
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync audit chip {} vs chip {} eth({},{}): a record of one against one of the other, "
            "|err| p50 {:.2f}, p99 {:.2f}, p99.9 {:.2f} ns, mean {:+.2f} ns",
            L.chip_b,
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            total.abs_quantile(0.5),
            total.abs_quantile(0.99),
            total.abs_quantile(0.999),
            total.mean());
        for (int b = 0; b < ErrorHistogram::kBins; b++) {
            pooled.bins[b] += total.bins[b];
        }
        pooled.n += 1.0;
        pooled.sum += total.sum;
        pooled.sumsq += total.sumsq;
        pooled.beyond += total.beyond;
        links++;
    }
    if (links != 0) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync audit over {} links: a record against one on a linked chip, |err| p50 {:.2f}, "
            "p99 {:.2f}, p99.9 {:.2f} ns",
            links,
            pooled.abs_quantile(0.5),
            pooled.abs_quantile(0.99),
            pooled.abs_quantile(0.999));
    }
}

void SyncEngine::publish_clock_plots() {
    for (uint32_t dev = 0; dev < local_.size(); dev++) {
        if (series_.nodes(dev) == 0) {
            continue;
        }
        const uint32_t chip = ctx_.devices[dev].chip_id;
        std::vector<PlotPoint> pts;
        for (const LocalClockModel::Instant& p : local_[dev].pts) {
            if (p.k8 == 0) {
                continue;
            }
            const double root = map_.lookup_root(chip, static_cast<double>(p.wall_tick()));
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
    for (size_t dev = 0; dev < audit_.size(); dev++) {
        if (!local_[dev].pts.empty()) {
            audit_[dev].settle(local_[dev], /*final=*/true);
        }
    }
    log_audit();
    // The published corrections stay for the sinks that write at process end; the next attach starts fresh.
    local_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
