// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/clock_map.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <limits>
#include <numeric>
#include <thread>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace {

constexpr auto kRateSpan = std::chrono::milliseconds(100);
constexpr int kSteadyBrackets = 16;

int64_t clock_ns(clockid_t id) {
    timespec now{};
    clock_gettime(id, &now);
    return static_cast<int64_t>(now.tv_sec) * 1'000'000'000 + now.tv_nsec;
}

}  // namespace

int64_t tsc_now() noexcept {
#if defined(__x86_64__)
    _mm_lfence();
    const int64_t tsc = static_cast<int64_t>(__rdtsc());
    _mm_lfence();
    return tsc;
#else
    return clock_ns(CLOCK_MONOTONIC_RAW);
#endif
}

double ns_per_tsc_tick() {
    static const double rate = [] {
        const int64_t tsc_start = tsc_now(), raw_start = clock_ns(CLOCK_MONOTONIC_RAW);
        std::this_thread::sleep_for(kRateSpan);
        const int64_t raw_end = clock_ns(CLOCK_MONOTONIC_RAW), tsc_end = tsc_now();
        return static_cast<double>(raw_end - raw_start) / static_cast<double>(tsc_end - tsc_start);
    }();
    return rate;
}

template <typename Key>
void ClockSeries<Key>::append(const Node& node) {
    if (node.at <= last.at) {
        return;
    }
    last = node;
    nodes.writer().publish(last);
    extend(node.at);
}
template struct ClockSeries<int64_t>;
template struct ClockSeries<double>;

namespace {

// Returns the index of the first node after `key`, or nullopt if the writer overwrote a node mid-search. It gallops out
// from `hint` because consecutive lookups usually land a few nodes from the last one.
template <typename Key>
std::optional<uint64_t> first_after(
    const ClockSeries<Key>& series, uint64_t oldest, uint64_t end, Key key, uint64_t hint) {
    bool overwritten = false;
    const auto after = [&](uint64_t index) {
        typename ClockSeries<Key>::Node node{};
        overwritten = overwritten || !series.nodes.read_at(index, node);
        return node.at > key;
    };
    // Every node before `lo` is at or before `key`, and `hi` is either the end or a node after `key`.
    uint64_t lo = oldest, hi = end;
    const uint64_t start = std::clamp(hint, oldest, end - 1);
    if (after(start)) {
        hi = start;
        for (uint64_t step = 1; lo < hi; step *= 2) {
            const uint64_t probe = start - std::min(step, start - oldest);
            if (!after(probe)) {
                lo = probe + 1;
                break;
            }
            hi = probe;
        }
    } else {
        lo = start + 1;
        for (uint64_t step = 1; lo < hi; step *= 2) {
            const uint64_t probe = std::min(start + step, end - 1);
            if (after(probe)) {
                hi = probe;
                break;
            }
            lo = probe + 1;
        }
    }
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (after(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return overwritten ? std::nullopt : std::optional(lo);
}

// Returns the segment that maps `key`, given `next`, the first node after it, or nullopt if the writer overwrote one of
// its nodes.
template <typename Key>
std::optional<Segment<Key>> segment_at(
    const ClockSeries<Key>& series, uint64_t oldest, uint64_t end, uint64_t next, Key key, Key cover) {
    using Node = typename ClockSeries<Key>::Node;
    Node below{}, above{};
    if ((next > oldest && !series.nodes.read_at(next - 1, below)) ||
        (next < end && !series.nodes.read_at(next, above))) {
        return std::nullopt;
    }
    if (next == oldest) {
        return Segment<Key>{
            .from = std::numeric_limits<Key>::lowest(),
            .to = above.at,
            .origin = above.at,
            .value = above.value,
            .slope = above.tangent,
            .hint = next};
    }
    if (next < end) {
        return Segment<Key>{
            .from = below.at,
            .to = above.at,
            .origin = below.at,
            .value = below.value,
            .slope = (above.value - below.value) / static_cast<double>(above.at - below.at),
            .hint = next};
    }
    if (key <= cover) {
        return Segment<Key>{
            .from = below.at,
            .to = cover,
            .origin = below.at,
            .value = below.value,
            .slope = below.tangent,
            .hint = next};
    }
    // Keys past `cover` aren't final, because the next node can change them. This extrapolates along the newest node's
    // tangent and returns an empty segment, so the guess isn't cached.
    return Segment<Key>{.origin = below.at, .value = below.value, .slope = below.tangent, .hint = next};
}

// Maps `key` through the reader's cached segment, or through a newly found segment. If the writer overwrites a node
// mid-search, the lookup retries from the series' new oldest node.
template <typename Key>
std::optional<double> lookup(const ClockSeries<Key>& series, Segment<Key>& segment, Key key) noexcept {
    if (segment.holds(key)) {
        return segment.at(key);
    }
    while (true) {
        const Key cover = series.cover.load(std::memory_order_acquire);
        const uint64_t oldest = series.nodes.oldest();
        const uint64_t end = series.nodes.published();
        if (end == oldest) {
            segment = Segment<Key>{};
            return std::nullopt;
        }
        if (const std::optional<uint64_t> next = first_after(series, oldest, end, key, segment.hint)) {
            if (const std::optional<Segment<Key>> found = segment_at(series, oldest, end, *next, key, cover)) {
                segment = *found;
                return segment.at(key);
            }
        }
    }
}

}  // namespace

ClockMap::ClockMap(size_t devices, ClockBases bases) : chips_(devices), tsc_base_(bases.tsc) {}

ClockMap::Reader ClockMap::reader() const { return Reader(chips_.size(), tsc_base_); }

void ClockMap::append(uint32_t dev, SyncNode node) { chips_[dev].append(node); }

void ClockMap::finish(uint32_t dev) { chips_[dev].extend(std::numeric_limits<int64_t>::max()); }

void ClockMap::append_host(HostNode node, double until) {
    TT_FATAL(
        std::isfinite(node.at) && std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0,
        "streaming profiler: host placement node at refclk {} is not a rate: tsc {} tangent {}",
        node.at,
        node.value,
        node.tangent);
    host_.append(node);
    host_.extend(until);
}

bool ClockMap::is_final(Reader& reader, uint32_t dev, int64_t wall) const noexcept {
    const int64_t chip_cover = chips_[dev].cover.load(std::memory_order_acquire);
    if (chip_cover == std::numeric_limits<int64_t>::max()) {
        return true;
    }
    return wall <= chip_cover && *place_root(reader, dev, wall, 0.0) <= host_.cover.load(std::memory_order_acquire);
}

bool ClockMap::has_host_nodes() const noexcept { return host_.nodes.published() != 0; }

std::optional<double> ClockMap::place_root(
    Reader& reader, uint32_t dev, int64_t wall, double tick_fraction) const noexcept {
    Segment<int64_t>& segment = reader.chips_[dev];
    const std::optional<double> root = lookup(chips_[dev], segment, wall);
    return root ? std::optional(*root + segment.slope * tick_fraction) : std::nullopt;
}

std::optional<int64_t> ClockMap::place_tsc(Reader& reader, double root) const noexcept {
    const std::optional<double> tsc = lookup(host_, reader.host_, root);
    return tsc ? std::optional(tsc_base_ + round_nearest(*tsc)) : std::nullopt;
}

int64_t ClockMap::place_slow(Reader& reader, uint32_t dev, int64_t wall) const {
    const std::optional<double> root = lookup(chips_[dev], reader.chips_[dev], wall);
    TT_FATAL(root, "streaming profiler: device {} has no placement node for its wall tick {}", dev, wall);
    const std::optional<int64_t> tsc = place_tsc(reader, *root);
    TT_FATAL(tsc, "streaming profiler: device {}'s wall tick {} has no host placement node", dev, wall);
    return *tsc;
}

// CLOCK_MONOTONIC is slewed but never stepped, so the series is just these pairs, joined by straight lines.
void SteadyClock::sample() {
    int64_t best_gap = std::numeric_limits<int64_t>::max(), best_tsc = 0, best_mono = 0;
    for (int i = 0; i < kSteadyBrackets; i++) {
        const int64_t before = tsc_now();
        const int64_t mono = clock_ns(CLOCK_MONOTONIC);
        const int64_t after = tsc_now();
        if (after - before < best_gap) {
            best_gap = after - before;
            best_tsc = std::midpoint(before, after);
            best_mono = mono;
        }
    }
    std::lock_guard<std::mutex> lock(append_mu_);
    const ClockSeries<int64_t>::Node& last = steady_.last;
    if (best_tsc <= last.at) {
        return;
    }
    const bool first = !base_ns_;
    if (first) {
        base_ns_ = best_mono;
    }
    const double value = static_cast<double>(best_mono - *base_ns_);
    const double ns_per_tick =
        first ? ns_per_tsc_tick() : (value - last.value) / static_cast<double>(best_tsc - last.at);
    steady_.append(ClockSeries<int64_t>::Node{.at = best_tsc, .value = value, .tangent = ns_per_tick});
}

std::optional<Segment<int64_t>> SteadyClock::segment_at(int64_t tsc) const noexcept {
    Segment<int64_t> segment;
    if (!lookup(steady_, segment, tsc)) {
        return std::nullopt;
    }
    return segment;
}

int64_t SteadyClock::base_ns() const noexcept { return *base_ns_; }

}  // namespace tt::tt_metal::streaming_profiler
