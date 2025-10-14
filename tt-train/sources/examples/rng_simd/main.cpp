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

template <typename T>
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
BenchmarkResult<T> run_benchmark(
    const std::string& name, const std::string& type_name, Func&& func, size_t size, int iterations = 10) {
    BenchmarkResult<T> result;
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
    std::cout << "\n+============================================================================+\n";
    std::cout << "| " << type_name << " Performance Benchmark (" << size << " elements)";
    int padding = 78 - 32 - type_name.length() - std::to_string(size).length();
    if (padding < 0)
        padding = 0;
    std::cout << std::string(padding, ' ') << "|\n";
    std::cout << "+============================================================================+\n\n";

    auto dist_factory = []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); };
    uint32_t seed = 42;

    std::vector<BenchmarkResult<T>> results;

    // MT19937 (skip for bfloat16 as it requires linking against full tt-metalium library)
    if constexpr (!std::same_as<T, bfloat16>) {
        auto func = [&](std::vector<T>& data) { ttml::core::sequential_generate(std::span{data}, dist_factory, seed); };
        results.push_back(run_benchmark<T>("MT19937", type_name, func, size));
    }

    // SSE
    if (CpuFeatures::has_sse_support()) {
        auto func = [&](std::vector<T>& data) {
            ttml::core::sse::sequential_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("SSE", type_name, func, size));
    }

    // AVX2
    if (CpuFeatures::has_avx2_support()) {
        auto func = [&](std::vector<T>& data) {
            ttml::core::avx::sequential_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("AVX2", type_name, func, size));
    }

    // Print results
    std::cout << std::left << std::setw(15) << "Implementation" << std::right << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput" << std::setw(15) << "Elements/sec" << std::setw(12) << "Speedup"
              << std::setw(10) << "Mean" << std::setw(10) << "StdDev"
              << "\n";
    std::cout << std::string(89, '-') << "\n";

    double baseline = results.empty() ? 1.0 : results[0].time_ms;
    for (const auto& r : results) {
        std::cout << std::left << std::setw(15) << r.name << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.time_ms << std::setw(13) << r.throughput_gb_s << " GB/s" << std::setw(13)
                  << r.elements_per_sec_m << " M" << std::setw(11) << (baseline / r.time_ms) << "x" << std::setw(10)
                  << r.mean << std::setw(10) << r.stddev << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "+============================================================================+\n";
    std::cout << "|                                                                            |\n";
    std::cout << "|      Float / Double / bfloat16 SIMD RNG Performance Comparison            |\n";
    std::cout << "|                                                                            |\n";
    std::cout << "+============================================================================+\n";

    // System info
    std::cout << "\n+============================================================================+\n";
    std::cout << "| System Information                                                         |\n";
    std::cout << "+============================================================================+\n\n";
    std::cout << "CPU Features:\n";
    std::cout << "  SSE4.2 + AES-NI: " << (CpuFeatures::has_sse_support() ? "✓" : "✗") << "\n";
    std::cout << "  AVX2 + AES-NI:   " << (CpuFeatures::has_avx2_support() ? "✓" : "✗") << "\n";
    std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << "\n";

    // Test size
    size_t size = 4194304;  // 4M elements

    benchmark_type<float>("float (32-bit)", size);
    benchmark_type<double>("double (64-bit)", size);
    benchmark_type<bfloat16>("bfloat16 (16-bit)", size);

    // Summary
    std::cout << "\n+============================================================================+\n";
    std::cout << "| Summary                                                                    |\n";
    std::cout << "+============================================================================+\n\n";

    std::cout << "Type Sizes:\n";
    std::cout << "  • float:    " << sizeof(float) << " bytes (32-bit)\n";
    std::cout << "  • double:   " << sizeof(double) << " bytes (64-bit)\n";
    std::cout << "  • bfloat16: " << sizeof(bfloat16) << " bytes (16-bit)\n\n";

    std::cout << "SIMD Width Comparison:\n";
    std::cout << "  • float:    SSE = 4 elements (128-bit), AVX2 = 8 elements (256-bit)\n";
    std::cout << "  • double:   SSE = 2 elements (128-bit), AVX2 = 4 elements (256-bit)\n";
    std::cout << "  • bfloat16: SSE = 4 elements (128-bit), AVX2 = 8 elements (256-bit)\n";
    std::cout << "              (Generated from float, then converted)\n\n";

    std::cout << "Expected Speedup vs MT19937:\n";
    std::cout << "  • float:    SSE ~100x, AVX2 ~200x\n";
    std::cout << "  • double:   SSE ~50x, AVX2 ~100x\n";
    std::cout << "  • bfloat16: SSE ~100x, AVX2 ~200x (similar to float)\n\n";

    std::cout << "Precision:\n";
    std::cout << "  • float:    23-bit mantissa (~7 decimal digits)\n";
    std::cout << "  • double:   52-bit mantissa (~15 decimal digits)\n";
    std::cout << "  • bfloat16: 7-bit mantissa (~2-3 decimal digits)\n\n";

    std::cout << "Use Cases:\n";
    std::cout << "  • float:    General ML training, neural networks\n";
    std::cout << "  • double:   Scientific computing, high precision needs\n";
    std::cout << "  • bfloat16: ML training (dynamic range of float32, less memory)\n\n";

    return 0;
}
