// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <thread>
#include <time.h>

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

namespace tt::tt_metal {

inline int64_t realtime_profiler_monotonic_raw_ns() noexcept {
    timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000ll + static_cast<int64_t>(ts.tv_nsec);
}

inline bool realtime_profiler_host_clock_uses_tsc() noexcept {
#if defined(__i386__) || defined(__x86_64__)
    static const bool has_invariant_tsc = [] {
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        if (__get_cpuid_max(0, nullptr) < 1 || !__get_cpuid(1, &eax, &ebx, &ecx, &edx) || (edx & bit_TSC) == 0) {
            return false;
        }
        if (__get_cpuid_max(0x80000000, nullptr) < 0x80000007 || !__get_cpuid(0x80000007, &eax, &ebx, &ecx, &edx)) {
            return false;
        }
        return (edx & (1u << 8)) != 0;
    }();
    return has_invariant_tsc;
#else
    return false;
#endif
}

inline int64_t realtime_profiler_host_timestamp() noexcept {
#if defined(__i386__) || defined(__x86_64__)
    if (realtime_profiler_host_clock_uses_tsc()) {
        return static_cast<int64_t>(__rdtsc());
    }
#endif
    return realtime_profiler_monotonic_raw_ns();
}

// Nanoseconds represented by one tick of realtime_profiler_host_timestamp().
inline double realtime_profiler_host_ns_per_tick() noexcept {
    if (!realtime_profiler_host_clock_uses_tsc()) {
        return 1.0;
    }

    static const double ratio = [] {
        const int64_t host_ns_before = realtime_profiler_monotonic_raw_ns();
        const int64_t ticks_before = realtime_profiler_host_timestamp();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const int64_t ticks_after = realtime_profiler_host_timestamp();
        const int64_t host_ns_after = realtime_profiler_monotonic_raw_ns();
        return static_cast<double>(host_ns_after - host_ns_before) / static_cast<double>(ticks_after - ticks_before);
    }();
    return ratio;
}

}  // namespace tt::tt_metal
