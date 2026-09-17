// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_placement_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/streaming_profiler/streaming_profiler_host_clock.hpp"
#include "tt_metal/common/indexed_ring.hpp"

#include <condition_variable>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

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
    using Node = PlacementNode<Key>;
    IndexedRing<Node> nodes{PlacementMap::kSeriesNodes};
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
    using Node = PlacementNode<Key>;
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
    std::array<Cursor<int64_t>, PlacementMap::kMaxChips> chip{};
    Cursor<double> host{};
    std::array<Composed, PlacementMap::kMaxChips> composed{};
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

struct PlacementMap::Impl {
    std::array<Log<int64_t>, kMaxChips> chips;
    Log<double> host;
    alignas(64) std::atomic<uint64_t> cover_generation{0};
};

PlacementMap::PlacementMap() : impl_(std::make_unique<Impl>()) {}
PlacementMap::~PlacementMap() = default;

void PlacementMap::append(uint32_t chip_id, SyncNode node) {
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

void PlacementMap::extend(uint32_t chip_id, int64_t cover_ticks) {
    if (chip_id < kMaxChips) {
        impl_->chips[chip_id].extend(cover_ticks);
        impl_->cover_generation.fetch_add(1, std::memory_order_release);
    }
}

void PlacementMap::finish(uint32_t chip_id) { extend(chip_id, std::numeric_limits<int64_t>::max()); }

void PlacementMap::clear(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        impl_->chips[chip_id].clear();
    }
}

void PlacementMap::append_host(HostNode node) {
    TT_FATAL(
        std::isfinite(node.at) && std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0 &&
            node.tangent < 1e4,
        "streaming profiler: host placement node at refclk {} is not a rate: tsc {} tangent {}",
        node.at,
        node.value,
        node.tangent);
    impl_->host.append(kHostSeries, node);
}

int64_t PlacementMap::cover_ticks(uint32_t chip_id) const noexcept {
    if (chip_id >= kMaxChips) {
        return std::numeric_limits<int64_t>::max();
    }
    return impl_->chips[chip_id].cover.load(std::memory_order_acquire);
}

uint64_t PlacementMap::cover_generation() const noexcept {
    return impl_->cover_generation.load(std::memory_order_acquire);
}

size_t PlacementMap::host_published() const noexcept { return impl_->host.nodes.count() - impl_->host.nodes.first(); }

double PlacementMap::lookup_root(uint32_t chip_id, int64_t wall) const noexcept {
    double root = 0.0;
    if (chip_id >= kMaxChips || !place(impl_->chips[chip_id], view_of(impl_.get()).chip[chip_id], wall, root)) {
        return 0.0;
    }
    return root;
}

double PlacementMap::host_tsc(double root) const noexcept {
    double tsc = 0.0;
    if (!place(impl_->host, view_of(impl_.get()).host, root, tsc)) {
        return 0.0;
    }
    return tsc;
}

int64_t PlacementMap::lookup_tsc(uint32_t chip_id, int64_t wall) const noexcept {
    if (chip_id >= kMaxChips) {
        return 0;
    }
    ThreadView& v = view_of(impl_.get());
    double root = 0.0, tsc = 0.0;
    if (!place(impl_->chips[chip_id], v.chip[chip_id], wall, root) || !place(impl_->host, v.host, root, tsc)) {
        return 0;
    }
    return std::llrint(tsc);
}

int64_t PlacementMap::place_host(uint32_t chip_id, int64_t wall) const noexcept {
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

namespace {
std::mutex& plots_mutex() {
    static std::mutex m;
    return m;
}
std::vector<std::pair<std::string, std::vector<SyncPlotPoint>>>& plots_store() {
    static std::vector<std::pair<std::string, std::vector<SyncPlotPoint>>> v;
    return v;
}
}  // namespace

void SyncPlots::publish(std::string name, std::vector<SyncPlotPoint> points) {
    std::lock_guard<std::mutex> g(plots_mutex());
    auto& v = plots_store();
    for (auto& e : v) {
        if (e.first == name) {
            e.second = std::move(points);
            return;
        }
    }
    v.emplace_back(std::move(name), std::move(points));
}

std::vector<std::pair<std::string, std::vector<SyncPlotPoint>>> SyncPlots::drain() {
    std::lock_guard<std::mutex> g(plots_mutex());
    auto out = std::move(plots_store());
    plots_store().clear();
    return out;
}

namespace {
std::condition_variable& plots_cv() {
    static std::condition_variable cv;
    return cv;
}
bool& plots_pending() {
    static bool pending = false;
    return pending;
}
}  // namespace

void SyncPlots::expect() {
    std::lock_guard<std::mutex> g(plots_mutex());
    plots_pending() = true;
}

void SyncPlots::complete() {
    {
        std::lock_guard<std::mutex> g(plots_mutex());
        plots_pending() = false;
    }
    plots_cv().notify_all();
}

void SyncPlots::wait_complete(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(plots_mutex());
    plots_cv().wait_for(lk, timeout, [] { return !plots_pending(); });
}

}  // namespace tt::tt_metal::streaming_profiler
