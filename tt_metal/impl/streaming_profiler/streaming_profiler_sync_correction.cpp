// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <x86intrin.h>
#include <ctime>
#include "impl/streaming_profiler/streaming_profiler_host_probe.hpp"

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

// One series: nodes in fixed chunks reached through a fixed table, so an append never moves a node a reader may be
// looking at; the chunks stay allocated across captures.
constexpr uint32_t kHostSeries = std::numeric_limits<uint32_t>::max();

template <typename Key>
struct Log {
    using Node = PlacementNode<Key>;
    static constexpr uint32_t kChunkShift = 12;
    static constexpr uint32_t kChunkNodes = 1u << kChunkShift;
    static constexpr uint32_t kChunks = 256;
    static constexpr uint32_t kCapacity = kChunks * kChunkNodes;
    std::array<std::atomic<Node*>, kChunks> chunks{};
    alignas(64) std::atomic<uint32_t> count{0};
    alignas(64) std::atomic<Key> cover{key_min<Key>()};
    alignas(64) std::atomic<uint32_t> gen{0};  // bumped by clear(): cursors built on an earlier capture miss
    bool full_warned = false;

    const Node& at(uint32_t i) const noexcept {
        return chunks[i >> kChunkShift].load(std::memory_order_relaxed)[i & (kChunkNodes - 1)];
    }
    void append(uint32_t chip_id, const Node& node) {
        const uint32_t n = count.load(std::memory_order_relaxed);
        if (n != 0 && node.at <= at(n - 1).at) {
            return;
        }
        if (n >= kCapacity) {
            if (!full_warned) {
                full_warned = true;
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] d2d sync: {} has {} placement nodes, the series' capacity; later records "
                    "convert on the newest tangent",
                    chip_id == kHostSeries ? std::string("the host") : "chip " + std::to_string(chip_id),
                    n);
            }
            extend(node.at);
            return;
        }
        Node* chunk = chunks[n >> kChunkShift].load(std::memory_order_relaxed);
        if (chunk == nullptr) {
            chunk = new Node[kChunkNodes];
            chunks[n >> kChunkShift].store(chunk, std::memory_order_release);
        }
        chunk[n & (kChunkNodes - 1)] = node;
        count.store(n + 1, std::memory_order_release);
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
        count.store(0, std::memory_order_release);
        full_warned = false;
    }
};

std::array<Log<int64_t>, SyncCorrections::kMaxChips>& logs() {
    static std::array<Log<int64_t>, SyncCorrections::kMaxChips> l;
    return l;
}
Log<double>& host_log() {
    static Log<double> l;
    return l;
}
alignas(64) std::atomic<uint64_t> g_cover_generation{0};
std::atomic<double> g_asymmetry_ns{0.0};
// The steady view: double-buffered under a generation, so a reader that sees the new generation sees the whole
// segment.
struct SteadySlots {
    SteadySegment seg[2];
    std::atomic<uint32_t> gen{0};
};
SteadySlots g_steady;

// A reader's place in one series: the segment [a, b] it last converted in and that segment's line through
// (origin, value). For the open segment b is the cover the reader last saw, so a record past that cover re-reads it.
template <typename Key>
struct Cursor {
    uint32_t gen = 0;
    uint32_t i = 0;
    Key a = key_max<Key>();
    Key b = key_min<Key>();
    Key origin{};
    double value = 0.0;
    double slope = 0.0;
    float sigma = 0.0f;  // of the segment's value: its nodes' largest, plus the freeze margin on the open tangent
};
// The margin a frontier may sit from the frozen tangent before a node is frozen (D2dSyncConsumer::kFreezeNs).
constexpr float kOpenMarginNs = 0.25f;
constinit thread_local Cursor<int64_t> t_cursors[SyncCorrections::kMaxChips];
constinit thread_local Cursor<double> t_host_cursor;

template <typename Key>
inline double on_line(const Cursor<Key>& c, Key t) noexcept {
    return c.value + c.slope * static_cast<double>(t - c.origin);
}

// The last node at or before t, from a hint; records arrive nearly in order, so a few steps usually reach it.
template <typename Key>
uint32_t locate(const Log<Key>& log, uint32_t n, uint32_t hint, Key t) noexcept {
    uint32_t i = hint < n ? hint : 0;
    for (int steps = 0; steps < 8; steps++) {
        if (i + 1 < n && log.at(i + 1).at <= t) {
            i++;
        } else if (i > 0 && log.at(i).at > t) {
            i--;
        } else {
            return i;
        }
    }
    uint32_t lo = 0, hi = n;  // first node past t
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (log.at(mid).at <= t) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

template <typename Key>
bool refill(Log<Key>& log, Cursor<Key>& c, Key t, double& value) noexcept {
    using Node = PlacementNode<Key>;
    const uint32_t gen = log.gen.load(std::memory_order_acquire);
    const Key cover = log.cover.load(std::memory_order_acquire);
    const uint32_t n = log.count.load(std::memory_order_acquire);
    if (n == 0) {
        c = Cursor<Key>{.gen = gen, .sigma = std::numeric_limits<float>::infinity()};
        return false;
    }
    const Node& first = log.at(0);
    if (t < first.at) {
        // Before the first node: back along its tangent, measured from the node itself.
        c = Cursor<Key>{gen, 0, key_min<Key>(), first.at, first.at, first.value, first.tangent, first.sigma_ns};
        value = on_line(c, t);
        return true;
    }
    const uint32_t i = locate(log, n, c.gen == gen ? c.i : 0, t);
    const Node& a = log.at(i);
    if (i + 1 < n) {
        const Node& b = log.at(i + 1);
        c = Cursor<Key>{
            gen,
            i,
            a.at,
            b.at,
            a.at,
            a.value,
            (b.value - a.value) / static_cast<double>(b.at - a.at),
            std::max(a.sigma_ns, b.sigma_ns)};
        value = on_line(c, t);
        return true;
    }
    if (t <= cover) {
        c = Cursor<Key>{gen, i, a.at, cover, a.at, a.value, a.tangent, a.sigma_ns + kOpenMarginNs};
        value = on_line(c, t);
        return true;
    }
    // Past the cover: a batch the service released before the sync covered it (counted and reported as a fault).
    // The newest tangent carries on; holding still here would collapse every such record onto one instant. Not
    // cached, the cover moves.
    c = Cursor<Key>{.gen = gen, .i = i, .sigma = a.sigma_ns + kOpenMarginNs};
    value = a.value + a.tangent * static_cast<double>(t - a.at);
    return true;
}

// Places t on the series through the cursor (refilled when it is not there); false when the series has no node.
template <typename Key>
inline bool place(Log<Key>& log, Cursor<Key>& c, Key t, double& value) noexcept {
    if (c.gen == log.gen.load(std::memory_order_relaxed) && t >= c.a && t <= c.b) {
        value = on_line(c, t);
        return true;
    }
    return refill(log, c, t, value);
}

inline bool place_root(uint32_t chip_id, int64_t wall, double& root) noexcept {
    return place(logs()[chip_id], t_cursors[chip_id], wall, root);
}

}  // namespace

void SyncCorrections::append(uint32_t chip_id, SyncNode node) {
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
    logs()[chip_id].append(chip_id, node);
    g_cover_generation.fetch_add(1, std::memory_order_release);
}

void SyncCorrections::extend(uint32_t chip_id, int64_t cover_ticks) {
    if (chip_id < kMaxChips) {
        logs()[chip_id].extend(cover_ticks);
        g_cover_generation.fetch_add(1, std::memory_order_release);
    }
}

void SyncCorrections::finish(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        logs()[chip_id].extend(std::numeric_limits<int64_t>::max());
        g_cover_generation.fetch_add(1, std::memory_order_release);
    }
}

void SyncCorrections::clear(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        logs()[chip_id].clear();
    }
}

void SyncCorrections::append_host(HostNode node) {
    TT_FATAL(
        std::isfinite(node.at) && std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0 &&
            node.tangent < 1e4,
        "streaming profiler: host placement node at refclk {} is not a rate: tsc {} tangent {}",
        node.at,
        node.value,
        node.tangent);
    host_log().append(kHostSeries, node);
}

void SyncCorrections::extend_host(double cover_root) { host_log().extend(cover_root); }

int64_t SyncCorrections::cover_ticks(uint32_t chip_id) noexcept {
    if (chip_id >= kMaxChips) {
        return std::numeric_limits<int64_t>::max();
    }
    return logs()[chip_id].cover.load(std::memory_order_acquire);
}

uint64_t SyncCorrections::cover_generation() noexcept { return g_cover_generation.load(std::memory_order_acquire); }

size_t SyncCorrections::published(uint32_t chip_id) noexcept {
    return chip_id < kMaxChips ? logs()[chip_id].count.load(std::memory_order_acquire) : 0;
}

size_t SyncCorrections::host_published() noexcept { return host_log().count.load(std::memory_order_acquire); }

std::vector<HostNode> SyncCorrections::host_nodes() {
    const Log<double>& log = host_log();
    const uint32_t n = log.count.load(std::memory_order_acquire);
    std::vector<HostNode> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        out.push_back(log.at(i));
    }
    return out;
}

double SyncCorrections::lookup_root(uint32_t chip_id, int64_t wall) noexcept {
    double root = 0.0;
    if (chip_id >= kMaxChips || !place_root(chip_id, wall, root)) {
        return 0.0;
    }
    return root;
}

int64_t SyncCorrections::lookup_tsc(uint32_t chip_id, int64_t wall) noexcept {
    double root = 0.0, tsc = 0.0;
    if (chip_id >= kMaxChips || !place_root(chip_id, wall, root) || !place(host_log(), t_host_cursor, root, tsc)) {
        return 0;
    }
    return std::llrint(tsc);
}

namespace {
// host_clock units per TSC tick.
double units_per_tsc() {
    static const double u = 10.0 / tsc_ticks_per_ns();
    return u;
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
constinit thread_local Composed t_composed[SyncCorrections::kMaxChips];
}  // namespace

int64_t SyncCorrections::place_host(uint32_t chip_id, int64_t wall) noexcept {
    if (chip_id >= kMaxChips) {
        return 0;
    }
    Composed& k = t_composed[chip_id];
    if (wall >= k.a && wall <= k.b && k.gen_c == logs()[chip_id].gen.load(std::memory_order_relaxed) &&
        k.gen_h == host_log().gen.load(std::memory_order_relaxed)) {
        return std::llround(k.value + k.slope * static_cast<double>(wall - k.origin));
    }
    double root = 0.0, tsc = 0.0;
    if (!place_root(chip_id, wall, root) || !place(host_log(), t_host_cursor, root, tsc)) {
        return 0;
    }
    const Cursor<int64_t>& cc = t_cursors[chip_id];
    const Cursor<double>& hc = t_host_cursor;
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

int64_t SyncCorrections::lookup_error_ns(uint32_t chip_id, int64_t wall) noexcept {
    double root = 0.0;
    if (chip_id >= kMaxChips || !place_root(chip_id, wall, root)) {
        return std::numeric_limits<int64_t>::max();
    }
    const Cursor<int64_t>& c = t_cursors[chip_id];
    const double e = kSigmas * static_cast<double>(c.sigma) + g_asymmetry_ns.load(std::memory_order_relaxed);
    return static_cast<int64_t>(std::ceil(e));
}

void SyncCorrections::set_asymmetry_ns(double ns) noexcept { g_asymmetry_ns.store(ns, std::memory_order_relaxed); }

void SyncCorrections::set_steady(const SteadySegment& segment) noexcept {
    const uint32_t g = g_steady.gen.load(std::memory_order_relaxed);
    g_steady.seg[(g + 1) & 1] = segment;
    g_steady.gen.store(g + 1, std::memory_order_release);
}

int64_t SyncCorrections::tsc_to_mono_ns(int64_t tsc) noexcept {
    thread_local uint32_t gen = ~0u;
    thread_local SteadySegment seg;
    const uint32_t g = g_steady.gen.load(std::memory_order_acquire);
    if (g != gen) {
        seg = g_steady.seg[g & 1];
        gen = g;
    }
    if (!seg.ok) {
        // No probe has published a segment: the pair is taken here, once per thread and generation.
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const int64_t mono = static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
        seg = SteadySegment{static_cast<int64_t>(__rdtsc()), mono, 1.0 / tsc_ticks_per_ns(), true};
    }
    return seg.mono_of(tsc);
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

namespace tt::tt_metal::experimental::streaming_profiler {
namespace {
double units_per_tsc() {
    static const double u = 10.0 / tt::tt_metal::streaming_profiler::tsc_ticks_per_ns();
    return u;
}
}  // namespace
host_clock::time_point host_clock::now() noexcept { return from_tsc(tt::tt_metal::streaming_profiler::tsc_now()); }
int64_t host_clock::tsc(time_point t) noexcept {
    return std::llround(static_cast<double>(t.time_since_epoch().count()) / units_per_tsc());
}
host_clock::time_point host_clock::from_tsc(int64_t ticks) noexcept {
    return time_point(duration(std::llround(static_cast<double>(ticks) * units_per_tsc())));
}
}  // namespace tt::tt_metal::experimental::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler::detail {
int64_t host_to_steady_ns(int64_t host) noexcept {
    return tt::tt_metal::streaming_profiler::SyncCorrections::tsc_to_mono_ns(
        host_clock::tsc(host_clock::time_point(host_clock::duration(host))));
}
}  // namespace tt::tt_metal::experimental::streaming_profiler::detail
