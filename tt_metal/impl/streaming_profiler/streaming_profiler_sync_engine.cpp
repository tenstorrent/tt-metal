// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>

#include <tt-logger/tt-logger.hpp>
#include <cstdio>
#include <string>
#include <utility>

#include <tt_stl/assert.hpp>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>

#include "tt_metal/common/indexed_ring.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

double aiclk_ghz(double wall_per_refclk) { return wall_per_refclk * kernel_profiler::kEthRefclkHz * 1e-9; }

constexpr uint32_t kHostSeries = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kSteadySeries = kHostSeries - 1;

template <typename Key>
struct Log {
    using Node = ClockNode<Key>;
    explicit Log(uint32_t series_nodes) : nodes(series_nodes) {}
    IndexedRing<Node> nodes;
    Key last_at = std::numeric_limits<Key>::lowest();  // the writer's own copy
    alignas(64) std::atomic<Key> cover{std::numeric_limits<Key>::lowest()};
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
        cover.store(std::numeric_limits<Key>::lowest(), std::memory_order_release);
        nodes.clear();
        last_at = std::numeric_limits<Key>::lowest();
        full_warned = false;
    }
};

// A reader's place in one series: the segment [a, b] it last converted in and that segment's line through
// (origin, value). For the open segment b is the cover the reader last saw, so a record past that cover re-reads it.
template <typename Key>
struct Cursor {
    uint32_t gen = 0;
    Key a = std::numeric_limits<Key>::max();
    Key b = std::numeric_limits<Key>::lowest();
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
            c = Cursor<Key>{gen, std::numeric_limits<Key>::lowest(), a.at, a.at, a.value, a.tangent};
        } else if (lo < n) {
            if (!log.nodes.read(lo, b)) {
                continue;
            }
            c = Cursor<Key>{gen, a.at, b.at, a.at, a.value, (b.value - a.value) / static_cast<double>(b.at - a.at)};
        } else if (t <= cover) {
            c = Cursor<Key>{gen, a.at, cover, a.at, a.value, a.tangent};
        } else {
            // Past the cover (the sync's own reads of a series still growing): the newest tangent carries on. Not
            // cached, the cover moves.
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
// the host segment (root refclk -> TSC), valid over [a, b] in wall ticks where both hold: base + value at the record
// that built it, and slope per wall tick from there.
struct Composed {
    uint32_t gen_c = 0, gen_h = 0;
    int64_t a = std::numeric_limits<int64_t>::max(), b = std::numeric_limits<int64_t>::min();
    int64_t origin = 0;
    int64_t base = 0;
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
    int64_t root_base = 0, tsc_base = 0;
    // tsc_base on host_clock: units_base + units_frac units, exactly as units_of_tsc() converts it.
    int64_t units_base = 0;
    double units_frac = 0.0;
    int64_t steady_base = 0;  // CLOCK_MONOTONIC ns the steady series' values are offsets from
    bool steady_based = false;
};

ClockMap::ClockMap(uint32_t series_nodes) :
    impl_(std::make_unique<Impl>(series_nodes, std::make_index_sequence<kMaxChips>{})) {}

uint32_t ClockMap::series_nodes() const noexcept { return static_cast<uint32_t>(impl_->host.nodes.capacity()); }
ClockMap::~ClockMap() = default;

void ClockMap::append(uint32_t chip_id, SyncNode node) {
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
    impl_->chips[chip_id].extend(std::numeric_limits<int64_t>::max());
    impl_->cover_generation.fetch_add(1, std::memory_order_release);
}

void ClockMap::clear(uint32_t chip_id) { impl_->chips[chip_id].clear(); }

void ClockMap::set_bases(int64_t root_refclk, int64_t tsc) {
    Impl& m = *impl_;
    m.root_base = root_refclk;
    m.tsc_base = tsc;
    const __int128 units = static_cast<__int128>(tsc) * units_per_tsc_q32();
    m.units_base = static_cast<int64_t>(units >> 32);
    m.units_frac = static_cast<double>(static_cast<uint32_t>(units)) / 4294967296.0;
    m.host.clear();
}

int64_t ClockMap::root_base() const noexcept { return impl_->root_base; }
int64_t ClockMap::tsc_base() const noexcept { return impl_->tsc_base; }

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
    const Log<int64_t>& cl = impl_->chips[chip_id];
    const int64_t cover = cl.cover.load(std::memory_order_acquire);
    if (cover == std::numeric_limits<int64_t>::max()) {
        return cover;
    }
    const double host_cover = impl_->host.cover.load(std::memory_order_acquire);
    ClockNode<int64_t> last{};
    const uint64_t n = cl.nodes.count();
    if (n == cl.nodes.first() || !cl.nodes.read(n - 1, last)) {
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

std::optional<double> ClockMap::lookup_root(uint32_t chip_id, int64_t wall, double frac) const noexcept {
    Cursor<int64_t>& c = view_of(impl_.get()).chip[chip_id];
    double root = 0.0;
    if (!place(impl_->chips[chip_id], c, wall, root)) {
        return std::nullopt;
    }
    return root + c.slope * frac;
}

void ClockMap::append_steady(int64_t tsc, int64_t mono_ns, double ns_per_tick) {
    TT_FATAL(ns_per_tick > 0.0, "streaming profiler: steady clock node at tsc {} is not a rate: {}", tsc, ns_per_tick);
    Impl& m = *impl_;
    if (!m.steady_based) {
        m.steady_base = mono_ns;
        m.steady_based = true;
    }
    m.steady.append(
        kSteadySeries,
        ClockNode<int64_t>{.at = tsc, .value = static_cast<double>(mono_ns - m.steady_base), .tangent = ns_per_tick});
}

bool ClockMap::steady_ns(int64_t tsc, int64_t& ns) const noexcept {
    double rel = 0.0;
    if (!place(impl_->steady, view_of(impl_.get()).steady, tsc, rel)) {
        return false;
    }
    ns = impl_->steady_base + std::llround(rel);
    return true;
}

std::optional<double> ClockMap::host_tsc(double root) const noexcept {
    double tsc = 0.0;
    if (!place(impl_->host, view_of(impl_.get()).host, root, tsc)) {
        return std::nullopt;
    }
    return tsc;
}

int64_t ClockMap::place_host(uint32_t chip_id, int64_t wall) const noexcept {
    ThreadView& v = view_of(impl_.get());
    const Log<int64_t>& cl = impl_->chips[chip_id];
    const Log<double>& hl = impl_->host;
    Composed& k = v.composed[chip_id];
    if (wall >= k.a && wall <= k.b && k.gen_c == cl.gen.load(std::memory_order_relaxed) &&
        k.gen_h == hl.gen.load(std::memory_order_relaxed)) {
        return k.base + std::llround(k.value + k.slope * static_cast<double>(wall - k.origin));
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
    k.base = impl_->units_base;
    k.value = impl_->units_frac + tsc * u;
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
    return k.base + std::llround(k.value);
}

double ErrorHistogram::abs_quantile(double q) const {
    if (n <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Bins kCentre + i and kCentre - 1 - i hold the same |error|.
    constexpr int kCentre = kBins / 2;
    double acc = 0.0;
    for (int i = 0; i < kCentre; i++) {
        acc += bins[kCentre + i] + bins[kCentre - 1 - i];
        if (acc >= q * n) {
            return ns_of(kCentre + i);
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
            // Two bin centres combine onto a bin edge; half the weight goes to each bin beside it.
            const int b = i + sign * j + (sign > 0 ? 1 - kBins / 2 : kBins / 2);
            const double w = 0.5 * bins[i] * other.bins[j] * norm;
            for (const int e : {b - 1, b}) {
                if (e >= 0 && e < kBins) {
                    out.bins[e] += w;
                } else {
                    out.beyond += w;
                }
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
        const double w = *m.wall_at(r);
        const double ns = (static_cast<double>(a.w) - w) * kNsPerRefclk / (*m.wall_at(r + 1.0) - w);
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

void AnchorAudit::add_drainer_audit(const ClockSample& s, int64_t refclk_base) {
    if (s.round == kernel_profiler::kSyncAnchorHistWorst) {
        const double w =
            static_cast<int32_t>(static_cast<uint32_t>(s.ref)) / static_cast<double>(ErrorHistogram::kBinsPerNs);
        if (std::abs(w) > std::abs(worst_ns)) {
            worst_ns = w;
            worst_r = static_cast<double>(static_cast<int64_t>(s.value) - refclk_base);
        }
        err.worst = std::max(err.worst, std::abs(w));
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
        const double ns = ErrorHistogram::ns_of(b), c = counts[j];
        err.bins[b] += c;
        err.n += c;
        err.sum += c * ns;
        err.sumsq += c * ns * ns;
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
    for (const CaptureContext::Device& d : ctx.devices) {
        TT_FATAL(
            d.chip_id < ClockMap::kMaxChips,
            "streaming profiler: chip {} is past the clock map's {} chips",
            d.chip_id,
            ClockMap::kMaxChips);
    }
    ctx_ = ctx;
    bases_.reset(n, ctx.root_dev, map_.root_base());
    chips_.assign(n, Chip{});
    links_.reset(ctx_, bases_);
    to_root_gen_ = ~0ull;
    // A chip with no link path to the root is finished at once, so no consumer waits for it.
    std::vector<bool> reach(n, false);
    reach[ctx.root_dev] = true;
    for (bool progress = true; progress;) {
        progress = false;
        for (const CaptureContext::Link& L : ctx.links) {
            if (reach[L.dev_a] != reach[L.dev_b]) {
                reach[L.dev_a] = reach[L.dev_b] = true;
                progress = true;
            }
        }
    }
    for (size_t dev = 0; dev < ctx.devices.size(); dev++) {
        const CaptureContext::Device& d = ctx.devices[dev];
        map_.clear(d.chip_id);
        if (!reach[dev]) {
            map_.finish(d.chip_id);
        }
    }
}

void SyncEngine::on_clock(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& rec) {
    namespace kp = kernel_profiler;
    TT_FATAL(dev < chips_.size(), "streaming profiler: a sync record names device {} of {}", dev, chips_.size());
    const auto meta = kp::word_as<kp::SyncMeta>(rec.meta);
    const uint32_t kind = meta.kind;
    if (kind == kp::kSyncKindLocal) {
        kp::SyncLocalPoint pts[kp::kSyncLocalPoints];
        const uint32_t n = kp::sync_local_unpack(rec, pts);
        Chip& c = chips_[dev];
        for (uint32_t i = 0; i < n; i++) {
            const auto r = static_cast<int64_t>(pts[i].r), w8 = static_cast<int64_t>(pts[i].w8);
            c.model.add_point(r - bases_.refclk(dev, r), w8 - 8 * bases_.wall(dev, w8 >> 3), pts[i].k8);
        }
        c.audit.settle(c.model, /*final=*/false);
        if (publish_dev(dev)) {
            service().wake_consumers();
        }
        return;
    }
    const auto word64 = [](uint32_t lo, uint32_t hi) { return (static_cast<uint64_t>(hi) << 32) | lo; };
    const ClockSample s{
        .dev = dev,
        .core = core,
        .round = rec.round,
        .role = meta.role,
        .value = word64(rec.value_lo, rec.value_hi),
        .ts = word64(rec.wall_lo, rec.wall_hi),
        .ref = word64(rec.ref[0], rec.ref[1])};
    if (kind == kp::kSyncKindAnchor) {
        const auto r = static_cast<int64_t>(s.ref), w = static_cast<int64_t>(s.ts) + ctx_.devices[dev].drainer_offset;
        chips_[dev].audit.pending.push_back(AnchorAudit::Pending{r - bases_.refclk(dev, r), w - bases_.wall(dev, w)});
    } else if (kind == kp::kSyncKindAnchorHist) {
        const bool worst = s.round == kp::kSyncAnchorHistWorst;
        chips_[dev].audit.add_drainer_audit(s, worst ? bases_.refclk(dev, static_cast<int64_t>(s.value)) : 0);
    } else {
        TT_FATAL(kind == kp::kSyncKindLink, "streaming profiler: device {} sent a sync record of kind {}", dev, kind);
        links_.on_stamp(s);
    }
}

void LinkSolver::reset(const CaptureContext& ctx, ChipBases& bases) {
    ctx_ = &ctx;
    bases_ = &bases;
    rounds_.assign(ctx.links.size(), LinkRounds{});
    side_of_.clear();
    for (size_t li = 0; li < ctx.links.size(); li++) {
        const CaptureContext::Link& L = ctx.links[li];
        side_of_[{L.dev_a, L.core_a}] = {li, true};
        side_of_[{L.dev_b, L.core_b}] = {li, false};
    }
    solved_.assign(ctx.links.size(), LinkSolution{});
    gen_ = 0;
}

void LinkSolver::on_stamp(const ClockSample& s) {
    namespace kp = kernel_profiler;
    const auto side = side_of_.find({s.dev, s.core});
    TT_FATAL(
        side != side_of_.end(), "streaming profiler: device {} core {} sent a link stamp for no link", s.dev, s.core);
    const auto [li, sender] = side->second;
    // Each end records the peer's egress average (read from the frames it received) and its own ingress average:
    // the receiver T0 and T1, the sender T1B and T2.
    std::optional<int64_t> Round::* slot = nullptr;
    if (sender) {
        slot = s.role == kp::kSyncRoleT1B ? &Round::t1b : s.role == kp::kSyncRoleT2 ? &Round::t2 : nullptr;
    } else {
        slot = s.role == kp::kSyncRoleT0 ? &Round::t0 : s.role == kp::kSyncRoleT1 ? &Round::t1 : nullptr;
    }
    TT_FATAL(
        slot != nullptr,
        "streaming profiler: device {} core {} sent link stamp role {}, which its end does not stamp",
        s.dev,
        s.core,
        s.role);
    // T0 and T2 are in the sender's refclk domain, T1 and T1B in the receiver's, whichever end recorded them.
    const CaptureContext::Link& L = ctx_->links[li];
    const uint32_t domain = slot == &Round::t0 || slot == &Round::t2 ? L.dev_a : L.dev_b;
    const auto units = static_cast<int64_t>(s.value);
    LinkRounds& lr = rounds_[li];
    Round& r = lr.pending[s.round];
    r.*slot = units - kUnitsPerRefclk * bases_->refclk(domain, units / kUnitsPerRefclk);
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
        const double newest = mid_a_refclk(rounds[n - 1]);
        if (!final && newest <= out.solved_at_refclk) {
            continue;
        }
        size_t begin = n;
        while (begin > 0 && mid_a_refclk(rounds[begin - 1]) > newest - kLinkWindowTicks) {
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
        solve_link(pts, out);
        gen_++;
    }
}

double LinkSolver::mid_a_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(*r.t0) + static_cast<double>(*r.t2)) * kRefclkPerStampUnit;
}

double LinkSolver::mid_b_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(*r.t1) + static_cast<double>(*r.t1b)) * kRefclkPerStampUnit;
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
        for (size_t li = 0; li < solved_.size(); li++) {
            if (!solved_[li].ok) {
                continue;
            }
            const CaptureContext::Link& L = ctx_->links[li];
            const bool have_s = idx.count(L.dev_a) != 0, have_r = idx.count(L.dev_b) != 0;
            if (have_s != have_r) {
                idx[have_s ? L.dev_b : L.dev_a] = static_cast<int>(idx.size()) - 1;
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
        const CaptureContext::Link& L = ctx_->links[li];
        if (!s.ok || idx.count(L.dev_a) == 0 || idx.count(L.dev_b) == 0) {
            continue;
        }
        ends.push_back({idx[L.dev_a], idx[L.dev_b]});
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
bool SyncEngine::publish_dev(uint32_t dev) {
    // Nothing is placed before the host series exists: a record converted then would land nowhere.
    if (map_.host_published() == 0) {
        return false;
    }
    if (to_root_gen_ != links_.generation()) {
        to_root_ = links_.root_transforms(ctx_.root_dev);
        to_root_gen_ = links_.generation();
    }
    // The series starts only once the chip is on the root's tree (the root is there from the start): its first node
    // fixes the placement every later node joins.
    const RootXf& xf = to_root_[dev];
    if (!xf.ok) {
        return false;
    }
    Chip& c = chips_[dev];
    const auto& pts = c.model.pts;
    const uint32_t chip = ctx_.devices[dev].chip_id;
    const int64_t wall_base = bases_.wall(dev);
    const size_t before = c.published;
    const auto root_at = [&](const LocalClockModel::Instant& p) {
        return xf.scale * static_cast<double>(p.r) + xf.shift;
    };
    for (; c.next < pts.size(); c.next++) {
        const LocalClockModel::Instant& p = pts[c.next];
        const double root = root_at(p);
        double tangent = 0.0;
        if (p.k8 != 0) {
            tangent = xf.scale * 8.0 / p.k8;
        } else if (pts.size() > 1) {
            const LocalClockModel::Instant& o = pts[c.next == 0 ? 1 : c.next - 1];
            tangent = (root_at(o) - root) / (o.wall() - p.wall());
        } else {
            break;
        }
        map_.append(chip, SyncNode{.at = wall_base + p.wall_tick(), .value = root, .tangent = tangent});
        c.published++;
    }
    return c.published > before;
}

void SyncEngine::publish_all() {
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
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
}

void SyncEngine::log_clock_models() const {
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        const LocalClockModel& l = chips_[dev].model;
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
        const double mean_ghz = wsum > 0.0 ? aiclk_ghz(static_cast<double>(ssum / wsum)) : 0.0;
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync chip {}: local clock {} instants ({} single samples) across {} FBDIV "
            "changes; applied AICLK mean {:.5f} GHz (min {:.5f}, max {:.5f}); {} placement nodes",
            ctx_.devices[dev].chip_id,
            l.pts.size(),
            raw,
            changes,
            mean_ghz,
            smax > 0.0 ? aiclk_ghz(smin) : 0.0,
            aiclk_ghz(smax),
            chips_[dev].published);
    }
}

void SyncEngine::log_link_solutions() const {
    const std::vector<LinkSolver::LinkSolution>& solved = links_.solutions();
    for (size_t li = 0; li < solved.size(); li++) {
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
            (s.offset_refclk + static_cast<double>(bases_.refclk(L.dev_b) - bases_.refclk(L.dev_a))) * kNsPerRefclk,
            s.rate * 1e6,
            s.residual_rms_ns,
            s.precision_ns);
    }
}

// Each link against the mesh solve: the difference between its own solution and the two chips' placements on the
// root. With every link on one consistent mesh the residuals are the cables' path asymmetries, a nanosecond or two.
void SyncEngine::log_loop_closures() const {
    const std::vector<LinkSolver::LinkSolution>& solved = links_.solutions();
    const std::vector<RootXf> to_root = links_.root_transforms(ctx_.root_dev);
    for (size_t li = 0; li < solved.size(); li++) {
        const LinkSolver::LinkSolution& s = solved[li];
        if (!s.ok) {
            continue;
        }
        const CaptureContext::Link& L = ctx_.links[li];
        const RootXf& S = to_root[L.dev_a];
        const RootXf& R = to_root[L.dev_b];
        if (!S.ok || !R.ok) {
            continue;
        }
        const double direct = s.mid_refclk + s.offset_refclk;
        const double via_mesh = (S.scale * s.mid_refclk + S.shift - R.shift) / R.scale;
        const double off_ns = (via_mesh - direct) * kNsPerRefclk;
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {}: {:+.1f} ns off the mesh (path "
            "asymmetry; estimate precision {:.2f} ns)",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            off_ns,
            s.precision_ns);
    }
}

// The link solve: receiver refclk = sender refclk + offset + rate * (sender refclk - mean midpoint), the least-squares
// line through the rounds' (midpoint, receiver minus midpoint) points.
void LinkSolver::solve_link(const std::vector<RoundPoint>& pts, LinkSolution& out) {
    long double mid_acc = 0;
    for (const RoundPoint& p : pts) {
        mid_acc += p.mid_refclk;
    }
    const double nn = static_cast<double>(pts.size());
    const double mean = static_cast<double>(mid_acc / static_cast<long double>(pts.size()));
    double sy = 0, sxx = 0, sxy = 0;
    for (const RoundPoint& p : pts) {
        const double x = p.mid_refclk - mean, y = p.off_refclk;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double slope = sxy / sxx;
    const double offset = sy / nn;
    double ss = 0;
    for (const RoundPoint& p : pts) {
        const double res = p.off_refclk - (offset + slope * (p.mid_refclk - mean));
        ss += res * res;
    }
    const double rms = std::sqrt(ss / nn);
    out.ok = true;
    out.rate = slope;
    out.mid_refclk = mean;
    out.offset_refclk = offset;
    out.residual_rms_ns = rms * kNsPerRefclk;
    out.precision_ns = rms * kNsPerRefclk / std::sqrt(nn);
    out.rounds = pts.size();
}

struct SyncEngine::LinkErrors {
    std::vector<PlotPoint> pts;  // (sender's host placement, receiver minus sender on the root, ns)
    ErrorHistogram h;            // the same errors
    std::vector<RoundTerms> terms;
    std::vector<double> raw_x, raw_y;  // the round's sender midpoint and the receiver's offset from it, refclk
    std::vector<double> rtt, turn, path;
    size_t past_model = 0;     // rounds past an end's newest instant
    size_t before_series = 0;  // rounds before a chip's oldest kept node: the series wrapped, nothing places them
};

// The rounds, placed through the final map. A round's error is the receiver's placement on the root less the
// sender's: the links' and the map's error, the chips' models cancelling. Next to it: the sender's round trip and the
// one way and turnaround inside the stamps.
SyncEngine::LinkErrors SyncEngine::link_errors(size_t li) const {
    const CaptureContext::Link& L = ctx_.links[li];
    const std::vector<Round>& rounds = links_.rounds(li);
    LinkErrors e;
    const double rate = links_.solutions()[li].rate;
    const LocalClockModel& model_a = chips_[L.dev_a].model;
    const LocalClockModel& model_b = chips_[L.dev_b].model;
    const double until_a = static_cast<double>(model_a.frontier());
    const double until_b = static_cast<double>(model_b.frontier());
    const auto oldest = [&](uint32_t dev, uint32_t chip) {
        const int64_t at = map_.oldest_at(chip);
        return at == std::numeric_limits<int64_t>::min() ? -std::numeric_limits<double>::infinity()
                                                         : static_cast<double>(at - bases_.wall(dev));
    };
    const double oldest_a = oldest(L.dev_a, L.chip_a), oldest_b = oldest(L.dev_b, L.chip_b);
    const auto root_at = [&](uint32_t dev, double wall) {
        const double tick = std::floor(wall);
        return map_.lookup_root(ctx_.devices[dev].chip_id, bases_.wall(dev) + static_cast<int64_t>(tick), wall - tick);
    };
    for (const Round& r : rounds) {
        const double mid_a = LinkSolver::mid_a_refclk(r), mid_b = LinkSolver::mid_b_refclk(r);
        if (mid_a > until_a || mid_b > until_b) {
            e.past_model++;
            continue;
        }
        const std::optional<double> wall_a = model_a.wall_at(mid_a);
        const std::optional<double> wall_b = model_b.wall_at(mid_b);
        if (!wall_a || !wall_b) {
            continue;
        }
        const std::optional<double> root_a = root_at(L.dev_a, *wall_a);
        const std::optional<double> root_b = root_at(L.dev_b, *wall_b);
        const std::optional<double> host_a = root_a ? map_.host_tsc(*root_a) : std::nullopt;
        if (!root_b || !host_a) {
            continue;
        }
        if (*wall_a < oldest_a || *wall_b < oldest_b) {
            e.before_series++;
            continue;
        }
        const double err = (*root_b - *root_a) * kNsPerRefclk;
        e.pts.push_back(PlotPoint{map_.tsc_base() + std::llround(*host_a), err});
        e.h.add(err);
        e.terms.push_back(RoundTerms{.wall_a = *wall_a, .wall_b = *wall_b, .root_a = *root_a, .root_b = *root_b});
        e.raw_x.push_back(mid_a);
        e.raw_y.push_back(mid_b - mid_a);
        e.rtt.push_back(LinkSolver::rtt_ns(r));
        e.path.push_back(LinkSolver::path_ns(r, rate));
        e.turn.push_back(e.rtt.back() - 2.0 * e.path.back());
    }
    return e;
}

void SyncEngine::log_link_stats(const CaptureContext::Link& L, const LinkErrors& e, size_t rounds) const {
    const auto pct = [](std::vector<double> v, double q) {
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
    const std::string path = fmt::format("{}.err_{}_{}_eth{}_{}.csv", csv, L.chip_b, L.chip_a, L.eth_a.x, L.eth_a.y);
    std::FILE* ef = std::fopen(path.c_str(), "w");
    TT_FATAL(ef != nullptr, "streaming profiler: cannot write the d2d sync CSV {}", path);
    std::fprintf(ef, "host_ns,err_ns,mid_a_refclk,r1_b_refclk,wall_a,wall_b,root_a,root_b,rtt_ns,turn_ns,path_ns\n");
    for (size_t i = 0; i < e.pts.size(); i++) {
        const RoundTerms& t = e.terms[i];
        std::fprintf(
            ef,
            "%lld,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f\n",
            static_cast<long long>(steady_mono_ns(e.pts[i].tsc)),
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
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        const LocalClockModel& l = chips_[dev].model;
        if (l.pts.empty()) {
            continue;
        }
        const std::string path = fmt::format("{}.model_{}.csv", csv, ctx_.devices[dev].chip_id);
        std::FILE* f = std::fopen(path.c_str(), "w");
        TT_FATAL(f != nullptr, "streaming profiler: cannot write the d2d sync CSV {}", path);
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
void SyncEngine::publish_error_plots(const std::vector<LinkErrors>& errs) {
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        const LinkErrors& e = errs[li];
        if (e.pts.empty()) {
            continue;
        }
        log_link_stats(L, e, links_.rounds(li).size());
        write_err_csv(L, e);
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync error chip {} vs chip {} (links and map, model cancelled): {} rounds, mean "
            "{:+.2f} ns, rms {:.2f} ns, worst {:.2f} ns",
            L.chip_b,
            L.chip_a,
            e.pts.size(),
            e.h.mean(),
            e.h.rms(),
            e.h.worst);
        plot(fmt::format("d2d sync error chip{} vs chip{} (ns)", L.chip_b, L.chip_a), e.pts);
    }
}

// The drainer's anchors against each chip's model, and pooled over the links what a record of one chip is off one of a
// linked chip: the link and map term of each round with both models' errors, the chips' clocks being independent
// processes.
void SyncEngine::log_audit(const std::vector<LinkErrors>& errs) const {
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
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        const AnchorAudit& a = chips_[dev].audit;
        if (a.err.n <= 0.0 && a.pending.empty() && a.before_model + a.past_model + a.drainer_unbracketed == 0) {
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync audit chip {}: the drainer's anchors against the clock model, {}; worst {:.3f} "
            "s into "
            "the capture; {} before the model, {} past it, {} dropped for a read with no refclk update",
            ctx_.devices[dev].chip_id,
            parts(a.err),
            a.worst_r / kernel_profiler::kEthRefclkHz,
            a.before_model,
            a.past_model,
            a.drainer_unbracketed);
    }
    ErrorHistogram pooled;
    size_t links = 0;
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        const ErrorHistogram& ea = chips_[L.dev_a].audit.err;
        const ErrorHistogram& eb = chips_[L.dev_b].audit.err;
        if (ea.n <= 0.0 || eb.n <= 0.0 || errs[li].pts.empty()) {
            continue;
        }
        const ErrorHistogram total = errs[li].h.convolve(eb, 1).convolve(ea, -1);
        for (int b = 0; b < ErrorHistogram::kBins; b++) {
            pooled.bins[b] += total.bins[b];
        }
        pooled.n += 1.0;
        pooled.sum += total.sum;
        pooled.sumsq += total.sumsq;
        pooled.beyond += total.beyond;
        pooled.worst = std::max(pooled.worst, total.worst);
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
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        const uint32_t chip = ctx_.devices[dev].chip_id;
        std::vector<PlotPoint> pts;
        for (const LocalClockModel::Instant& p : chips_[dev].model.pts) {
            if (p.k8 == 0) {
                continue;
            }
            const std::optional<double> root = map_.lookup_root(chip, bases_.wall(dev) + p.wall_tick());
            const std::optional<double> tsc = root ? map_.host_tsc(*root) : std::nullopt;
            if (tsc) {
                pts.push_back(PlotPoint{map_.tsc_base() + std::llround(*tsc), aiclk_ghz(p.k8 / 8.0)});
            }
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
        tracy::Profiler::PlotDataAt(nm, p.value, p.tsc);
    }
#endif
}

void SyncEngine::on_capture_end() {
    links_.solve_final();
    publish_all();
    std::vector<LinkErrors> errs;
    errs.reserve(ctx_.links.size());
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        errs.push_back(link_errors(li));
    }
    write_model_csv();
    publish_error_plots(errs);
    publish_clock_plots();
    log_summary();
    for (Chip& c : chips_) {
        if (!c.model.pts.empty()) {
            c.audit.settle(c.model, /*final=*/true);
        }
    }
    log_audit(errs);
    // The published corrections stay for the sinks that write at process end; the next attach starts fresh.
    chips_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
