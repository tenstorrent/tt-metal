// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"

#include <algorithm>
#include <array>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <condition_variable>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace tt::tt_metal::streaming_profiler {

namespace {
using Series = std::vector<SyncNode>;
using Slot = std::atomic<std::shared_ptr<const Series>>;

std::array<Slot, SyncCorrections::kMaxChips>& slots() {
    static std::array<Slot, SyncCorrections::kMaxChips> s;
    return s;
}
std::array<Slot, SyncCorrections::kMaxChips>& local_slots() {
    static std::array<Slot, SyncCorrections::kMaxChips> s;
    return s;
}
int64_t lookup_in(std::array<Slot, SyncCorrections::kMaxChips>& sl, uint32_t chip_id, int64_t host_ns) noexcept;
}  // namespace

namespace {
void publish_in(std::array<Slot, SyncCorrections::kMaxChips>& sl, uint32_t chip_id, std::vector<SyncNode> nodes) {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return;
    }
    for (size_t i = 1; i < nodes.size(); i++) {
        TT_FATAL(
            nodes[i].host_ns > nodes[i - 1].host_ns,
            "streaming profiler: sync correction nodes for chip {} are not strictly increasing at {} ({} then {})",
            chip_id,
            i,
            nodes[i - 1].host_ns,
            nodes[i].host_ns);
    }
    sl[chip_id].store(std::make_shared<const Series>(std::move(nodes)), std::memory_order_release);
}
}  // namespace

void SyncCorrections::publish(uint32_t chip_id, std::vector<SyncNode> nodes) {
    publish_in(slots(), chip_id, std::move(nodes));
}

void SyncCorrections::clear(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        slots()[chip_id].store(nullptr, std::memory_order_release);
    }
}

int64_t SyncCorrections::published_until_ns(uint32_t chip_id) noexcept {
    if (chip_id >= kMaxChips) {
        return std::numeric_limits<int64_t>::min();
    }
    const auto p = slots()[chip_id].load(std::memory_order_acquire);
    return (!p || p->empty()) ? std::numeric_limits<int64_t>::min() : p->back().host_ns;
}

size_t SyncCorrections::published(uint32_t chip_id) noexcept {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return 0;
    }
    const auto p = slots()[chip_id].load(std::memory_order_acquire);
    return p ? p->size() : 0;
}

int64_t SyncCorrections::lookup_ns(uint32_t chip_id, int64_t host_ns) noexcept {
    return lookup_in(slots(), chip_id, host_ns);
}

int64_t SyncCorrections::lookup_local_ns(uint32_t chip_id, int64_t host_ns) noexcept {
    return lookup_in(local_slots(), chip_id, host_ns);
}

void SyncCorrections::publish_local(uint32_t chip_id, std::vector<SyncNode> nodes) {
    publish_in(local_slots(), chip_id, std::move(nodes));
}

namespace {
int64_t lookup_series(const Series& s, int64_t host_ns) noexcept {
    const auto line = [](const SyncNode& a, const SyncNode& b, int64_t t) {
        return a.delta_ns + (b.delta_ns - a.delta_ns) * static_cast<double>(t - a.host_ns) /
                                static_cast<double>(b.host_ns - a.host_ns);
    };
    if (host_ns <= s.front().host_ns || s.size() == 1) {
        return static_cast<int64_t>(s.front().delta_ns);
    }
    // The first node past `host_ns`.
    const auto it = std::upper_bound(
        s.begin(), s.end(), host_ns, [](int64_t t, const SyncNode& n) { return t < n.host_ns; });
    if (it == s.end()) {
        // Past the fitted range (a live sink slightly ahead of the fit): extend the last line, then hold.
        const SyncNode& b = s.back();
        const int64_t t = b.host_ns + std::min<int64_t>(host_ns - b.host_ns, SyncCorrections::kHoldNs);
        return static_cast<int64_t>(line(s[s.size() - 2], b, t));
    }
    return static_cast<int64_t>(line(*(it - 1), *it, host_ns));
}
int64_t lookup_in(std::array<Slot, SyncCorrections::kMaxChips>& sl, uint32_t chip_id, int64_t host_ns) noexcept {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return 0;
    }
    const auto p = sl[chip_id].load(std::memory_order_acquire);
    if (!p || p->empty()) {
        return 0;
    }
    return lookup_series(*p, host_ns);
}
}  // namespace

void SyncCorrections::lookup_span_ns(
    uint32_t chip_id, int64_t start_ns, int64_t end_ns, int64_t& d_start, int64_t& d_end) noexcept {
    d_start = 0;
    d_end = 0;
    if (chip_id >= kMaxChips) {
        return;
    }
    const auto p = slots()[chip_id].load(std::memory_order_acquire);
    if (!p || p->empty()) {
        return;
    }
    d_start = lookup_series(*p, start_ns);
    d_end = lookup_series(*p, end_ns);
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

// Both ends through one correction snapshot: the correction is continuous with a slope far below one, so within a
// snapshot the corrected time is strictly increasing in the record's own time and a zone's end cannot precede its
// start; two independent lookups could straddle a publish and invert it.
void sync_correction_span_ns(
    uint16_t chip_id, int64_t start_ns, int64_t end_ns, int64_t& d_start, int64_t& d_end) noexcept {
    tt::tt_metal::streaming_profiler::SyncCorrections::lookup_span_ns(chip_id, start_ns, end_ns, d_start, d_end);
}

}  // namespace tt::tt_metal::experimental::streaming_profiler::detail
