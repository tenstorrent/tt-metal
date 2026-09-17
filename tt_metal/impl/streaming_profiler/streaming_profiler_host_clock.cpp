// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_host_clock.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <x86intrin.h>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

int64_t clock_ns(clockid_t id) {
    timespec ts{};
    clock_gettime(id, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

int64_t tsc_now() noexcept {
    _mm_lfence();
    const int64_t t = static_cast<int64_t>(__rdtsc());
    _mm_lfence();
    return t;
}

double tsc_ticks_per_ns() {
    static const double rate = [] {
        const int64_t t0 = tsc_now(), r0 = clock_ns(CLOCK_MONOTONIC_RAW);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int64_t r1 = clock_ns(CLOCK_MONOTONIC_RAW), t1 = tsc_now();
        return static_cast<double>(t1 - t0) / static_cast<double>(r1 - r0);
    }();
    return rate;
}

double units_per_tsc() {
    static const double u = 10.0 / tsc_ticks_per_ns();
    return u;
}

namespace {
// Double-buffered under a generation, so a reader that sees the new generation sees the whole segment.
struct SteadySlots {
    SteadySegment seg[2];
    std::atomic<uint32_t> gen{0};
};
SteadySlots g_steady;
}  // namespace

void SteadyView::set(const SteadySegment& segment) noexcept {
    const uint32_t g = g_steady.gen.load(std::memory_order_relaxed);
    g_steady.seg[(g + 1) & 1] = segment;
    g_steady.gen.store(g + 1, std::memory_order_release);
}

int64_t SteadyView::mono_ns(int64_t tsc) noexcept {
    thread_local uint32_t gen = ~0u;
    thread_local SteadySegment seg;
    const uint32_t g = g_steady.gen.load(std::memory_order_acquire);
    if (g != gen) {
        seg = g_steady.seg[g & 1];
        gen = g;
    }
    if (!seg.ok) {
        // The stand-in pair is taken once per thread and generation.
        seg = SteadySegment{static_cast<int64_t>(__rdtsc()), clock_ns(CLOCK_MONOTONIC), 1.0 / tsc_ticks_per_ns(), true};
    }
    return seg.mono_of(tsc);
}

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler {
host_clock::time_point host_clock::now() noexcept { return from_tsc(tt::tt_metal::streaming_profiler::tsc_now()); }
int64_t host_clock::tsc(time_point t) noexcept {
    return std::llround(
        static_cast<double>(t.time_since_epoch().count()) / tt::tt_metal::streaming_profiler::units_per_tsc());
}
host_clock::time_point host_clock::from_tsc(int64_t ticks) noexcept {
    return time_point(
        duration(std::llround(static_cast<double>(ticks) * tt::tt_metal::streaming_profiler::units_per_tsc())));
}
}  // namespace tt::tt_metal::experimental::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler::detail {
int64_t host_to_steady_ns(int64_t host) noexcept {
    return tt::tt_metal::streaming_profiler::SteadyView::mono_ns(
        host_clock::tsc(host_clock::time_point(host_clock::duration(host))));
}
}  // namespace tt::tt_metal::experimental::streaming_profiler::detail
