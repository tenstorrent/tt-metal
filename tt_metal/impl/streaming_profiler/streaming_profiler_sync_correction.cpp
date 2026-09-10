// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"

#include <algorithm>
#include <array>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace {
using Series = std::vector<SyncSegment>;
using Slot = std::atomic<std::shared_ptr<const Series>>;

std::array<Slot, SyncCorrections::kMaxChips>& slots() {
    static std::array<Slot, SyncCorrections::kMaxChips> s;
    return s;
}
std::array<Slot, SyncCorrections::kMaxChips>& local_slots() {
    static std::array<Slot, SyncCorrections::kMaxChips> s;
    return s;
}
int64_t lookup_in(std::array<Slot, SyncCorrections::kMaxChips>& sl, uint32_t chip_id, uint64_t ticks) noexcept;
}  // namespace

void SyncCorrections::publish(uint32_t chip_id, std::vector<SyncSegment> segments) {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return;
    }
    std::sort(segments.begin(), segments.end(), [](const SyncSegment& a, const SyncSegment& b) {
        return a.tick_lo < b.tick_lo;
    });
    slots()[chip_id].store(std::make_shared<const Series>(std::move(segments)), std::memory_order_release);
}

void SyncCorrections::clear(uint32_t chip_id) {
    if (chip_id < kMaxChips) {
        slots()[chip_id].store(nullptr, std::memory_order_release);
    }
}

size_t SyncCorrections::published(uint32_t chip_id) noexcept {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return 0;
    }
    const auto p = slots()[chip_id].load(std::memory_order_acquire);
    return p ? p->size() : 0;
}

int64_t SyncCorrections::lookup_ns(uint32_t chip_id, uint64_t ticks) noexcept {
    return lookup_in(slots(), chip_id, ticks);
}

int64_t SyncCorrections::lookup_local_ns(uint32_t chip_id, uint64_t ticks) noexcept {
    return lookup_in(local_slots(), chip_id, ticks);
}

void SyncCorrections::publish_local(uint32_t chip_id, std::vector<SyncSegment> segments) {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return;
    }
    std::sort(segments.begin(), segments.end(), [](const SyncSegment& x, const SyncSegment& y) {
        return x.tick_lo < y.tick_lo;
    });
    local_slots()[chip_id].store(std::make_shared<const Series>(std::move(segments)), std::memory_order_release);
}

namespace {
int64_t lookup_in(std::array<Slot, SyncCorrections::kMaxChips>& sl, uint32_t chip_id, uint64_t ticks) noexcept {
    if (chip_id >= SyncCorrections::kMaxChips) {
        return 0;
    }
    const auto p = sl[chip_id].load(std::memory_order_acquire);
    if (!p || p->empty()) {
        return 0;
    }
    const Series& s = *p;
    if (ticks < s.front().tick_lo) {
        return 0;  // before the fit began: the anchor alone
    }
    // The last segment starting at or before `ticks`.
    const auto it =
        std::upper_bound(s.begin(), s.end(), ticks, [](uint64_t t, const SyncSegment& seg) { return t < seg.tick_lo; });
    const SyncSegment& seg = *(it - 1);
    uint64_t t = ticks;
    if (t > seg.tick_hi) {
        // Past the fitted range (a live sink slightly ahead of the fit, or a gap): extend the line, then hold.
        t = seg.tick_hi + std::min<uint64_t>(t - seg.tick_hi, SyncCorrections::kHoldTicks);
    }
    const double d = seg.delta_ns_lo + seg.slope_ns_per_tick * static_cast<double>(t - seg.tick_lo);
    return static_cast<int64_t>(d);
}
}  // namespace

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler::detail {

int64_t sync_correction_ns(uint16_t chip_id, uint64_t device_ticks) noexcept {
    return tt::tt_metal::streaming_profiler::SyncCorrections::lookup_ns(chip_id, device_ticks);
}

}  // namespace tt::tt_metal::experimental::streaming_profiler::detail
