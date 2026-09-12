// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <condition_variable>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace tt::tt_metal::streaming_profiler {

namespace {

// One series: nodes in fixed chunks reached through a fixed table, so an append never moves a node a reader may be
// looking at; the chunks stay allocated across captures.
struct Log {
    static constexpr uint32_t kChunkShift = 12;
    static constexpr uint32_t kChunkNodes = 1u << kChunkShift;
    static constexpr uint32_t kChunks = 256;
    static constexpr uint32_t kCapacity = kChunks * kChunkNodes;
    std::array<std::atomic<SyncNode*>, kChunks> chunks{};
    alignas(64) std::atomic<uint32_t> count{0};
    alignas(64) std::atomic<int64_t> cover{std::numeric_limits<int64_t>::min()};
    alignas(64) std::atomic<uint32_t> gen{0};  // bumped by clear(): cursors built on an earlier capture miss
    bool full_warned = false;

    const SyncNode& at(uint32_t i) const noexcept {
        return chunks[i >> kChunkShift].load(std::memory_order_relaxed)[i & (kChunkNodes - 1)];
    }
    void append(uint32_t chip_id, const SyncNode& node) {
        const uint32_t n = count.load(std::memory_order_relaxed);
        if (n != 0 && node.host_ns <= at(n - 1).host_ns) {
            return;
        }
        if (n >= kCapacity) {
            if (!full_warned) {
                full_warned = true;
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] d2d sync: chip {} has {} correction nodes, the series' capacity; later "
                    "records convert on the newest tangent",
                    chip_id,
                    n);
            }
            extend(node.host_ns);
            return;
        }
        SyncNode* chunk = chunks[n >> kChunkShift].load(std::memory_order_relaxed);
        if (chunk == nullptr) {
            chunk = new SyncNode[kChunkNodes];
            chunks[n >> kChunkShift].store(chunk, std::memory_order_release);
        }
        chunk[n & (kChunkNodes - 1)] = node;
        count.store(n + 1, std::memory_order_release);
        extend(node.host_ns);
    }
    void extend(int64_t c) {
        if (c > cover.load(std::memory_order_relaxed)) {
            cover.store(c, std::memory_order_release);
        }
    }
    void clear() {
        gen.fetch_add(1, std::memory_order_release);
        cover.store(std::numeric_limits<int64_t>::min(), std::memory_order_release);
        count.store(0, std::memory_order_release);
        full_warned = false;
    }
};

struct Logs {
    Log linked, local;
    Log& of(SyncSeries s) { return s == SyncSeries::Linked ? linked : local; }
};

std::array<Logs, SyncCorrections::kMaxChips>& logs() {
    static std::array<Logs, SyncCorrections::kMaxChips> l;
    return l;
}

// A reader's place in one series: the segment [a, b) it last converted in and that segment's line. For the open
// segment b is one past the cover the reader last saw, so a record past that cover re-reads it.
struct Cursor {
    uint32_t gen = 0;
    uint32_t i = 0;
    int64_t a = 0;
    int64_t b = 0;
    double d = 0.0;
    double slope = 0.0;
};
constinit thread_local Cursor t_cursors[2][SyncCorrections::kMaxChips];

inline int64_t on_line(const Cursor& c, int64_t t) noexcept {
    return static_cast<int64_t>(c.d + c.slope * static_cast<double>(t - c.a));
}

// The last node at or before t, from a hint; records arrive nearly in order, so a few steps usually reach it.
uint32_t locate(const Log& log, uint32_t n, uint32_t hint, int64_t t) noexcept {
    uint32_t i = hint < n ? hint : 0;
    for (int steps = 0; steps < 8; steps++) {
        if (i + 1 < n && log.at(i + 1).host_ns <= t) {
            i++;
        } else if (i > 0 && log.at(i).host_ns > t) {
            i--;
        } else {
            return i;
        }
    }
    uint32_t lo = 0, hi = n;  // first node past t
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (log.at(mid).host_ns <= t) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

int64_t refill(Log& log, Cursor& c, int64_t t) noexcept {
    const uint32_t gen = log.gen.load(std::memory_order_acquire);
    const int64_t cover = log.cover.load(std::memory_order_acquire);
    const uint32_t n = log.count.load(std::memory_order_acquire);
    if (n == 0) {
        c = Cursor{.gen = gen};
        return 0;
    }
    const SyncNode& first = log.at(0);
    if (t < first.host_ns) {
        // Constant before the first node; the segment's start only has to lie below every host time a record can
        // carry without the difference overflowing.
        c = Cursor{gen, 0, std::numeric_limits<int64_t>::min() / 2, first.host_ns, first.delta_ns, 0.0};
        return on_line(c, t);
    }
    const uint32_t i = locate(log, n, c.gen == gen ? c.i : 0, t);
    const SyncNode& a = log.at(i);
    if (i + 1 < n) {
        const SyncNode& b = log.at(i + 1);
        c = Cursor{
            gen,
            i,
            a.host_ns,
            b.host_ns,
            a.delta_ns,
            (b.delta_ns - a.delta_ns) / static_cast<double>(b.host_ns - a.host_ns)};
        return on_line(c, t);
    }
    if (t <= cover) {
        const int64_t b = cover == std::numeric_limits<int64_t>::max() ? cover : cover + 1;
        c = Cursor{gen, i, a.host_ns, b, a.delta_ns, a.tangent};
        return on_line(c, t);
    }
    // Past the cover (a consumer that does not wait for the sync): the tangent, then a hold. Not cached, the cover
    // moves.
    c = Cursor{.gen = gen, .i = i};
    const int64_t tt = std::min(t, cover + SyncCorrections::kHoldNs);
    return static_cast<int64_t>(a.delta_ns + a.tangent * static_cast<double>(tt - a.host_ns));
}

inline int64_t lookup(uint32_t chip_id, int64_t t, SyncSeries series) noexcept {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return 0;
    }
    Log& log = logs()[chip_id].of(series);
    Cursor& c = t_cursors[static_cast<size_t>(series)][chip_id];
    if (c.gen == log.gen.load(std::memory_order_relaxed) && t >= c.a && t < c.b) {
        return on_line(c, t);
    }
    return refill(log, c, t);
}

}  // namespace

void SyncCorrections::append(uint32_t chip_id, SyncNode node, SyncSeries series) {
    if (chip_id >= kMaxChips) {
        return;
    }
    TT_FATAL(
        std::isfinite(node.delta_ns) && std::isfinite(node.tangent) && std::abs(node.tangent) < 1.0,
        "streaming profiler: correction node for chip {} at {} is not a slope below one: delta {} tangent {}",
        chip_id,
        node.host_ns,
        node.delta_ns,
        node.tangent);
    logs()[chip_id].of(series).append(chip_id, node);
}

void SyncCorrections::extend(uint32_t chip_id, int64_t cover_ns, SyncSeries series) {
    if (chip_id < kMaxChips) {
        logs()[chip_id].of(series).extend(cover_ns);
    }
}

void SyncCorrections::finish(uint32_t chip_id, SyncSeries series) {
    if (chip_id < kMaxChips) {
        logs()[chip_id].of(series).extend(std::numeric_limits<int64_t>::max());
    }
}

void SyncCorrections::clear(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        logs()[chip_id].linked.clear();
        logs()[chip_id].local.clear();
    }
}

int64_t SyncCorrections::cover_ns(uint32_t chip_id) noexcept {
    if (chip_id >= kMaxChips) {
        return std::numeric_limits<int64_t>::max();
    }
    return logs()[chip_id].linked.cover.load(std::memory_order_acquire);
}

size_t SyncCorrections::published(uint32_t chip_id) noexcept {
    return chip_id < kMaxChips ? logs()[chip_id].linked.count.load(std::memory_order_acquire) : 0;
}

int64_t SyncCorrections::lookup_ns(uint32_t chip_id, int64_t host_ns, SyncSeries series) noexcept {
    return lookup(chip_id, host_ns, series);
}

void SyncCorrections::lookup_span_ns(
    uint32_t chip_id, int64_t start_ns, int64_t end_ns, int64_t& d_start, int64_t& d_end) noexcept {
    d_start = lookup(chip_id, start_ns, SyncSeries::Linked);
    d_end = lookup(chip_id, end_ns, SyncSeries::Linked);
    if (end_ns + d_end < start_ns + d_start) {
        d_end = d_start;  // unreachable for a continuous series with |slope| << 1; never let an end precede its start
    }
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

namespace tt::tt_metal::experimental::streaming_profiler::detail {

int64_t sync_correction_ns(uint16_t chip_id, int64_t host_ns) noexcept {
    return tt::tt_metal::streaming_profiler::SyncCorrections::lookup_ns(chip_id, host_ns);
}

void sync_correction_span_ns(
    uint16_t chip_id, int64_t start_ns, int64_t end_ns, int64_t& d_start, int64_t& d_end) noexcept {
    tt::tt_metal::streaming_profiler::SyncCorrections::lookup_span_ns(chip_id, start_ns, end_ns, d_start, d_end);
}

}  // namespace tt::tt_metal::experimental::streaming_profiler::detail
