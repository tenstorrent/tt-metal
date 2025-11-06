// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include "core/random.hpp"
#include "tt-metalium/bfloat16.hpp"

namespace {

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct BenchmarkResult {
    std::string name;
    std::string impl;   // "Legacy" or "SSE"
    std::string dist;   // "Uniform" or "Normal"
    std::string dtype;  // "float" or "bfloat16"
    size_t size;
    double time_ms = 0.0;
    double throughput_gb_s = 0.0;
    double throughput_m_elem_s = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double ks_statistic = 0.0;
};

// ============================================================================
// Statistical Analysis
// ============================================================================

struct Statistics {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;

    void compute(std::vector<double>& data) {
        if (data.empty())
            return;

        // Mean
        mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

        // Standard deviation
        double variance = 0.0;
        for (double x : data) {
            variance += (x - mean) * (x - mean);
        }
        stddev = std::sqrt(variance / data.size());

        // Min/Max
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        min = *min_it;
        max = *max_it;
    }
};

// Kolmogorov-Smirnov test
double ks_statistic(std::vector<double>& data1, std::vector<double>& data2) {
    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());

    double max_diff = 0.0;
    size_t i = 0, j = 0;

    while (i < data1.size() && j < data2.size()) {
        double cdf1 = static_cast<double>(i) / data1.size();
        double cdf2 = static_cast<double>(j) / data2.size();
        max_diff = std::max(max_diff, std::abs(cdf1 - cdf2));

        if (data1[i] <= data2[j]) {
            i++;
        } else {
            j++;
        }
    }

    while (i < data1.size()) {
        double cdf1 = static_cast<double>(i) / data1.size();
        max_diff = std::max(max_diff, std::abs(cdf1 - 1.0));
        i++;
    }

    while (j < data2.size()) {
        double cdf2 = static_cast<double>(j) / data2.size();
        max_diff = std::max(max_diff, std::abs(1.0 - cdf2));
        j++;
    }

    return max_diff;
}

// ============================================================================
// Benchmarking Templates
// ============================================================================

template <typename T, typename Dist>
BenchmarkResult benchmark_legacy(size_t size, const Dist& dist_factory, const std::string& dist_name) {
    std::vector<T> data(size);
    const uint32_t seed = 42;

    // Warmup
    ttml::core::legacy::sequential_generate(std::span{data}, dist_factory, seed);

    // Time the generation
    auto start = Clock::now();
    ttml::core::legacy::sequential_generate(std::span{data}, dist_factory, seed);
    auto end = Clock::now();

    double time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double throughput_gb_s = (size * sizeof(T)) / (time_ms / 1000.0) / 1e9;
    double throughput_m_elem_s = (size / (time_ms / 1000.0)) / 1e6;

    // Compute statistics
    std::vector<double> data_double;
    for (auto val : data) {
        if constexpr (std::same_as<T, float>) {
            data_double.push_back(static_cast<double>(val));
        } else {
            data_double.push_back(static_cast<double>(static_cast<float>(val)));
        }
    }

    Statistics stats;
    stats.compute(data_double);

    BenchmarkResult result;
    result.name = dist_name;
    result.impl = "Legacy";
    result.dist = dist_name;
    result.dtype = std::same_as<T, float> ? "float" : "bfloat16";
    result.size = size;
    result.time_ms = time_ms;
    result.throughput_gb_s = throughput_gb_s;
    result.throughput_m_elem_s = throughput_m_elem_s;
    result.mean = stats.mean;
    result.stddev = stats.stddev;

    return result;
}

template <typename T, typename Dist>
BenchmarkResult benchmark_sse(size_t size, const Dist& dist_factory, const std::string& dist_name) {
    std::vector<T> data(size);
    const uint32_t seed = 42;

    // Warmup
    ttml::core::sse::sequential_generate(std::span{data}, dist_factory, seed);

    // Time the generation
    auto start = Clock::now();
    ttml::core::sse::sequential_generate(std::span{data}, dist_factory, seed);
    auto end = Clock::now();

    double time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double throughput_gb_s = (size * sizeof(T)) / (time_ms / 1000.0) / 1e9;
    double throughput_m_elem_s = (size / (time_ms / 1000.0)) / 1e6;

    // Compute statistics
    std::vector<double> data_double;
    for (auto val : data) {
        if constexpr (std::same_as<T, float>) {
            data_double.push_back(static_cast<double>(val));
        } else {
            data_double.push_back(static_cast<double>(static_cast<float>(val)));
        }
    }

    Statistics stats;
    stats.compute(data_double);

    BenchmarkResult result;
    result.name = dist_name;
    result.impl = "SSE";
    result.dist = dist_name;
    result.dtype = std::same_as<T, float> ? "float" : "bfloat16";
    result.size = size;
    result.time_ms = time_ms;
    result.throughput_gb_s = throughput_gb_s;
    result.throughput_m_elem_s = throughput_m_elem_s;
    result.mean = stats.mean;
    result.stddev = stats.stddev;

    return result;
}

// ============================================================================
// Comparison Benchmark
// ============================================================================

template <typename T, typename Dist>
std::pair<BenchmarkResult, BenchmarkResult> compare_distributions(
    size_t size, const Dist& dist_factory, const std::string& dist_name) {
    auto legacy_result = benchmark_legacy<T>(size, dist_factory, dist_name);
    auto sse_result = benchmark_sse<T>(size, dist_factory, dist_name);

    // Compute KS statistic (sample subset for large sizes to save memory)
    std::vector<T> legacy_data(std::min(size_t(1000000), size));
    std::vector<T> sse_data(std::min(size_t(1000000), size));

    const uint32_t seed = 42;
    ttml::core::legacy::sequential_generate(std::span{legacy_data}, dist_factory, seed);
    ttml::core::sse::sequential_generate(std::span{sse_data}, dist_factory, seed);

    std::vector<double> legacy_double;
    std::vector<double> sse_double;

    for (auto val : legacy_data) {
        if constexpr (std::same_as<T, float>) {
            legacy_double.push_back(static_cast<double>(val));
        } else {
            legacy_double.push_back(static_cast<double>(static_cast<float>(val)));
        }
    }

    for (auto val : sse_data) {
        if constexpr (std::same_as<T, float>) {
            sse_double.push_back(static_cast<double>(val));
        } else {
            sse_double.push_back(static_cast<double>(static_cast<float>(val)));
        }
    }

    double ks = ks_statistic(legacy_double, sse_double);
    legacy_result.ks_statistic = ks;
    sse_result.ks_statistic = ks;

    return {legacy_result, sse_result};
}

}  // namespace

int main() {
    std::cout << "==========================================================================\n";
    std::cout << "RNG Performance and Distribution Benchmark\n";
    std::cout << "Comparing Legacy (MT19937) vs SSE RNG Implementations\n";
    std::cout << "==========================================================================\n\n";

    // Test sizes: from 1K to 100M to see scaling
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000, 10000000};

    std::vector<BenchmarkResult> all_results;

    // ========================================================================
    // Test 1: Uniform Distribution - float
    // ========================================================================
    std::cout << "TEST 1: Uniform Distribution [0,1) - float\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t size : sizes) {
        auto dist_factory = []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); };
        auto [legacy_result, sse_result] = compare_distributions<float>(size, dist_factory, "Uniform");

        all_results.push_back(legacy_result);
        all_results.push_back(sse_result);

        std::cout << "\nSize: " << std::setw(10) << size << " elements\n";
        std::cout << "  Legacy:  " << std::fixed << std::setprecision(2) << legacy_result.throughput_gb_s << " GB/s ("
                  << legacy_result.throughput_m_elem_s << " M elem/s, " << legacy_result.time_ms << " ms)\n";
        std::cout << "  SSE:     " << std::fixed << std::setprecision(2) << sse_result.throughput_gb_s << " GB/s ("
                  << sse_result.throughput_m_elem_s << " M elem/s, " << sse_result.time_ms
                  << " ms)  [KS: " << std::scientific << std::setprecision(6) << sse_result.ks_statistic << "]\n";
    }
    std::cout << "\n";

    // ========================================================================
    // Test 2: Uniform Distribution - bfloat16
    // ========================================================================
    std::cout << "TEST 2: Uniform Distribution [0,1) - bfloat16\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t size : sizes) {
        auto dist_factory = []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); };
        auto [legacy_result, sse_result] = compare_distributions<bfloat16>(size, dist_factory, "Uniform");

        all_results.push_back(legacy_result);
        all_results.push_back(sse_result);

        std::cout << "\nSize: " << std::setw(10) << size << " elements\n";
        std::cout << "  Legacy:  " << std::fixed << std::setprecision(2) << legacy_result.throughput_gb_s << " GB/s ("
                  << legacy_result.throughput_m_elem_s << " M elem/s, " << legacy_result.time_ms << " ms)\n";
        std::cout << "  SSE:     " << std::fixed << std::setprecision(2) << sse_result.throughput_gb_s << " GB/s ("
                  << sse_result.throughput_m_elem_s << " M elem/s, " << sse_result.time_ms
                  << " ms)  [KS: " << std::scientific << std::setprecision(6) << sse_result.ks_statistic << "]\n";
    }
    std::cout << "\n";

    // ========================================================================
    // Test 3: Normal Distribution - float
    // ========================================================================
    std::cout << "TEST 3: Normal Distribution (mean=0, stddev=1) - float\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t size : sizes) {
        auto dist_factory = []() { return std::normal_distribution<float>(0.0f, 1.0f); };
        auto [legacy_result, sse_result] = compare_distributions<float>(size, dist_factory, "Normal");

        all_results.push_back(legacy_result);
        all_results.push_back(sse_result);

        std::cout << "\nSize: " << std::setw(10) << size << " elements\n";
        std::cout << "  Legacy:  " << std::fixed << std::setprecision(2) << legacy_result.throughput_gb_s << " GB/s ("
                  << legacy_result.throughput_m_elem_s << " M elem/s, " << legacy_result.time_ms << " ms)\n";
        std::cout << "  SSE:     " << std::fixed << std::setprecision(2) << sse_result.throughput_gb_s << " GB/s ("
                  << sse_result.throughput_m_elem_s << " M elem/s, " << sse_result.time_ms
                  << " ms)  [KS: " << std::scientific << std::setprecision(6) << sse_result.ks_statistic << "]\n";
    }
    std::cout << "\n";

    // ========================================================================
    // Test 4: Normal Distribution - bfloat16
    // ========================================================================
    std::cout << "TEST 4: Normal Distribution (mean=0, stddev=1) - bfloat16\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t size : sizes) {
        auto dist_factory = []() { return std::normal_distribution<float>(0.0f, 1.0f); };
        auto [legacy_result, sse_result] = compare_distributions<bfloat16>(size, dist_factory, "Normal");

        all_results.push_back(legacy_result);
        all_results.push_back(sse_result);

        std::cout << "\nSize: " << std::setw(10) << size << " elements\n";
        std::cout << "  Legacy:  " << std::fixed << std::setprecision(2) << legacy_result.throughput_gb_s << " GB/s ("
                  << legacy_result.throughput_m_elem_s << " M elem/s, " << legacy_result.time_ms << " ms)\n";
        std::cout << "  SSE:     " << std::fixed << std::setprecision(2) << sse_result.throughput_gb_s << " GB/s ("
                  << sse_result.throughput_m_elem_s << " M elem/s, " << sse_result.time_ms
                  << " ms)  [KS: " << std::scientific << std::setprecision(6) << sse_result.ks_statistic << "]\n";
    }
    std::cout << "\n";

    // ========================================================================
    // Summary Table
    // ========================================================================
    std::cout << "==========================================================================\n";
    std::cout << "PERFORMANCE SUMMARY\n";
    std::cout << "==========================================================================\n\n";

    std::cout << std::setw(12) << "Distribution" << std::setw(10) << "Type" << std::setw(15) << "Size" << std::setw(12)
              << "Legacy GB/s" << std::setw(12) << "SSE GB/s" << std::setw(12) << "Speedup" << std::setw(15)
              << "KS-Statistic\n";
    std::cout << std::string(88, '-') << "\n";

    for (size_t i = 0; i < all_results.size(); i += 2) {
        const auto& legacy = all_results[i];
        const auto& sse = all_results[i + 1];

        double speedup = sse.throughput_gb_s / legacy.throughput_gb_s;

        std::cout << std::setw(12) << legacy.dist << std::setw(10) << legacy.dtype << std::setw(15) << legacy.size
                  << std::setw(12) << std::fixed << std::setprecision(2) << legacy.throughput_gb_s << std::setw(12)
                  << sse.throughput_gb_s << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(15) << std::scientific << std::setprecision(6) << sse.ks_statistic << "\n";
    }

    std::cout << "\n==========================================================================\n";
    std::cout << "DISTRIBUTION STATISTICS (Mean and Stddev for largest size)\n";
    std::cout << "==========================================================================\n\n";

    std::cout << std::setw(12) << "Distribution" << std::setw(10) << "Type" << std::setw(15) << "Implementation"
              << std::setw(15) << "Mean" << std::setw(15) << "Stddev\n";
    std::cout << std::string(67, '-') << "\n";

    for (size_t i = all_results.size() - 8; i < all_results.size(); ++i) {
        const auto& result = all_results[i];

        std::cout << std::setw(12) << result.dist << std::setw(10) << result.dtype << std::setw(15) << result.impl
                  << std::setw(15) << std::fixed << std::setprecision(6) << result.mean << std::setw(15)
                  << result.stddev << "\n";
    }

    std::cout << "\n==========================================================================\n";
    std::cout << "Benchmark Complete\n";
    std::cout << "==========================================================================\n";

    return 0;
}
