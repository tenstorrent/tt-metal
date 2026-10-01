// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

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

inline uint32_t seed_from_name(const std::string& name) {
    return static_cast<uint32_t>(std::hash<std::string>{}(name));
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

}  // namespace ttml::benchmark_utils
