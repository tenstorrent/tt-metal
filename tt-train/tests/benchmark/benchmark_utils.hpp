// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <string_view>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "utils/memory_utils.hpp"

namespace ttml::benchmark_utils {

struct BenchmarkIterationConfig {
    int num_warmup_iterations = 0;
    int num_measurement_iterations = 0;
};

// Benchmark-only helper utilities.
//
// Keep these separate from test_utils: test_utils owns reusable test data construction,
// while benchmark_utils owns benchmark timing and reporting conveniences.
// Only place helpers here once they have real benchmark call sites; speculative helpers
// should stay local to the benchmark that needs them.

inline uint32_t seed_from_name(std::string_view name) {
    return static_cast<uint32_t>(std::hash<std::string_view>{}(name));
}

// Relative change from reference (%). Positive means value increased.
inline double relative_change_pct(const double value, const double reference) {
    if (reference == 0.0) {
        return 0.0;
    }
    return (value - reference) / reference * 100.0;
}

// Reduction against baseline (%). Positive means value decreased.
inline double reduction_pct(const double baseline, const double value) {
    return -relative_change_pct(value, baseline);
}

inline double speedup_x(const double baseline, const double value) {
    if (value == 0.0) {
        return 0.0;
    }
    return baseline / value;
}

inline double average(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

// Wall-clock per call with no device sync, so for device ops this is enqueue time; use
// time_device_avg_us for device work.
template <typename Fn>
inline double measure_average_iteration_time_s(const int num_iterations, Fn&& fn) {
    if (num_iterations <= 0) {
        return 0.0;
    }
    auto total_time = std::chrono::duration<double>::zero();
    for (int iter = 0; iter < num_iterations; ++iter) {
        const auto start = std::chrono::high_resolution_clock::now();
        fn();
        const auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
    }
    return total_time.count() / static_cast<double>(num_iterations);
}

// Host µs per launch of a device op: `num_warmup` launches, then `num_measure` launches between two
// device syncs.
template <typename Fn>
inline double time_device_avg_us(
    tt::tt_metal::distributed::MeshDevice& device, uint32_t num_warmup, uint32_t num_measure, Fn&& fn) {
    for (uint32_t i = 0; i < num_warmup; ++i) {
        fn();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_measure; ++i) {
        fn();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(num_measure);
}

// Peak DRAM in bytes over one invocation of `fn`, tracked under `name`.
template <typename Fn>
inline size_t capture_dram_peak(const std::string& name, Fn&& fn) {
    ttml::utils::MemoryUsageTracker::clear();
    const auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
    (void)guard;
    fn();
    ttml::utils::MemoryUsageTracker::end_capture(name);
    return static_cast<size_t>(std::max(0LL, ttml::utils::MemoryUsageTracker::get_dram_usage(name).peak));
}

}  // namespace ttml::benchmark_utils
