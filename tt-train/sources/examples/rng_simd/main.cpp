// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "core/cpu_features.hpp"
#include "core/random.hpp"
#include "core/random_avx.hpp"
#include "core/random_sse.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace std::chrono;
using ttml::core::CpuFeatures;

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string name;
    std::string type;
    double time_ms{0.0};
    double throughput_gb_s{0.0};
    double elements_per_sec_m{0.0};
    double mean{0.0};
    double stddev{0.0};
};

template <typename T, typename Func>
BenchmarkResult run_benchmark(
    const std::string& name, const std::string& type_name, Func& func, size_t size, int iterations = 10) {
    BenchmarkResult result;
    result.name = name;
    result.type = type_name;

    std::vector<double> times;
    times.reserve(iterations);

    std::vector<T> data(size);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        func(data);
    }

    // Measure
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        func(data);
        auto end = high_resolution_clock::now();
        times.push_back(duration_cast<nanoseconds>(end - start).count() / 1e6);
    }

    // Statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    result.time_ms = sum / times.size();

    // Throughput
    double bytes = size * sizeof(T);
    result.throughput_gb_s = (bytes / (result.time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    result.elements_per_sec_m = (size / (result.time_ms / 1000.0)) / 1e6;

    // Distribution stats
    double data_sum = 0.0;
    for (const auto& val : data) {
        data_sum += static_cast<double>(val);
    }
    result.mean = data_sum / size;

    double var_sum = 0.0;
    for (const auto& val : data) {
        double diff = static_cast<double>(val) - result.mean;
        var_sum += diff * diff;
    }
    result.stddev = std::sqrt(var_sum / size);

    return result;
}

// ============================================================================
// Benchmark Runners
// ============================================================================

template <typename T>
void benchmark_type(const std::string& type_name, size_t size) {
    // Print header
    std::cout << "\n\n\n" << (type_name + " Performance Benchmark (" + std::to_string(size) + " elements)") << "\n";

    auto dist_factory = []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); };
    uint32_t seed = 42;

    std::vector<BenchmarkResult> results;

    auto func_seq = [&](std::vector<T>& data) { ttml::core::sequential_generate(std::span{data}, dist_factory, seed); };
    results.push_back(run_benchmark<T>("MT19937 (Sequential)", type_name, func_seq, size));

    auto func_par = [&](std::vector<T>& data) { ttml::core::parallel_generate(std::span{data}, dist_factory, seed); };
    results.push_back(run_benchmark<T>("MT19937 (Parallel)", type_name, func_par, size));

    // SSE Sequential and Parallel
    if (CpuFeatures::has_sse_support()) {
        auto func_seq = [&](std::vector<T>& data) {
            ttml::core::sse::sequential_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("SSE (Sequential)", type_name, func_seq, size));

        auto func_par = [&](std::vector<T>& data) {
            ttml::core::sse::parallel_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("SSE (Parallel)", type_name, func_par, size));
    }

    // AVX2 Sequential and Parallel
    if (CpuFeatures::has_avx2_support()) {
        auto func_seq = [&](std::vector<T>& data) {
            ttml::core::avx::sequential_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("AVX2 (Sequential)", type_name, func_seq, size));

        auto func_par = [&](std::vector<T>& data) {
            ttml::core::avx::parallel_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("AVX2 (Parallel)", type_name, func_par, size));
    }

    // Print results in vertical format
    double baseline = results.empty() ? 1.0 : results[0].time_ms;
    constexpr int label_width = 18;

    for (const auto& r : results) {
        std::cout << r.name << ":\n";

        std::cout << "  " << std::left << std::setw(label_width) << "Time (ms):" << std::right << std::fixed
                  << std::setprecision(3) << r.time_ms << "\n";

        std::cout << "  " << std::left << std::setw(label_width) << "Throughput:" << std::right << std::fixed
                  << std::setprecision(2) << r.throughput_gb_s << " GB/s\n";

        std::cout << "  " << std::left << std::setw(label_width) << "Elements/sec:" << std::right << std::fixed
                  << std::setprecision(1) << r.elements_per_sec_m << " M\n";

        std::cout << "  " << std::left << std::setw(label_width) << "Speedup:" << std::right << std::fixed
                  << std::setprecision(1) << (baseline / r.time_ms) << "x\n";

        std::cout << "  " << std::left << std::setw(label_width) << "Mean:" << std::right << std::fixed
                  << std::setprecision(6) << r.mean << "\n";

        std::cout << "  " << std::left << std::setw(label_width) << "StdDev:" << std::right << std::fixed
                  << std::setprecision(6) << r.stddev << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char** argv) {
    std::cout << "CPU Features:\n";
    std::cout << "  SSE4.2 + AES-NI: " << (CpuFeatures::has_sse_support() ? "✓" : "✗") << "\n";
    std::cout << "  AVX2 + AES-NI:   " << (CpuFeatures::has_avx2_support() ? "✓" : "✗") << "\n";
    std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << "\n";

    // Run benchmarks
    constexpr size_t default_size = 4194304;  // 4M elements
    auto const size = argc > 1 ? std::stoul(argv[1]) : default_size;

    benchmark_type<float>("float (32-bit)", size);
    benchmark_type<double>("double (64-bit)", size);
    benchmark_type<bfloat16>("bfloat16 (16-bit)", size);

    return 0;
}
