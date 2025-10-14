// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "core/random.hpp"
#include "core/random_modern.hpp"

using namespace std::chrono;

template <typename Func>
double benchmark_fast(Func&& func, int iterations = 10) {
    // Warmup
    func();

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();

    return duration_cast<microseconds>(end - start).count() / (1000.0 * iterations);
}

void print_row(const std::string& name, size_t size, double nonsimd_ms, double simd_ms) {
    double speedup = nonsimd_ms / simd_ms;
    double throughput_nonsimd = (size * sizeof(float)) / (nonsimd_ms * 1e6);  // GB/s
    double throughput_simd = (size * sizeof(float)) / (simd_ms * 1e6);        // GB/s

    std::cout << std::left << std::setw(18) << name << std::right << std::fixed << std::setprecision(3) << std::setw(10)
              << nonsimd_ms << std::setw(10) << simd_ms << std::setw(10) << speedup << "x" << std::setw(12)
              << throughput_nonsimd << std::setw(12) << throughput_simd << "\n";
}

void benchmark_uniform_sequential() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  SEQUENTIAL UNIFORM DISTRIBUTION: non-SIMD (MT19937) vs SIMD (AES-NI)   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << std::left << std::setw(18) << "Size" << std::right << std::setw(10) << "Non-SIMD" << std::setw(10)
              << "SIMD" << std::setw(10) << "Speedup" << std::setw(12) << "Non-SIMD" << std::setw(12) << "SIMD"
              << "\n";
    std::cout << std::left << std::setw(18) << "" << std::right << std::setw(10) << "(ms)" << std::setw(10) << "(ms)"
              << std::setw(10) << "" << std::setw(12) << "GB/s" << std::setw(12) << "GB/s"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    auto dist_factory = []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); };
    uint32_t seed = 42;

    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};

    for (auto size : sizes) {
        std::vector<float> data_nonsimd(size);
        std::vector<float> data_simd(size);

        auto time_nonsimd =
            benchmark_fast([&]() { ttml::core::sequential_generate(std::span{data_nonsimd}, dist_factory, seed); });

        auto time_simd = benchmark_fast(
            [&]() { ttml::core::modern::sequential_generate(std::span{data_simd}, dist_factory, seed); });

        std::string size_str = std::to_string(size / 1024) + (size >= 1048576 ? "M" : "K");
        print_row(size_str, size, time_nonsimd, time_simd);
    }
}

void benchmark_uniform_parallel() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  PARALLEL UNIFORM DISTRIBUTION: non-SIMD (MT19937) vs SIMD (AES-NI)     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << std::left << std::setw(18) << "Size" << std::right << std::setw(10) << "Non-SIMD" << std::setw(10)
              << "SIMD" << std::setw(10) << "Speedup" << std::setw(12) << "Non-SIMD" << std::setw(12) << "SIMD"
              << "\n";
    std::cout << std::left << std::setw(18) << "" << std::right << std::setw(10) << "(ms)" << std::setw(10) << "(ms)"
              << std::setw(10) << "" << std::setw(12) << "GB/s" << std::setw(12) << "GB/s"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    auto dist_factory = []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); };
    uint32_t seed = 42;
    uint32_t threads = std::thread::hardware_concurrency();

    std::vector<size_t> sizes = {4096, 16384, 65536, 262144, 1048576, 4194304};

    for (auto size : sizes) {
        std::vector<float> data_nonsimd(size);
        std::vector<float> data_simd(size);

        auto time_nonsimd = benchmark_fast(
            [&]() { ttml::core::parallel_generate(std::span{data_nonsimd}, dist_factory, seed, threads); });

        auto time_simd = benchmark_fast(
            [&]() { ttml::core::modern::parallel_generate(std::span{data_simd}, dist_factory, seed, threads); });

        std::string size_str = std::to_string(size / 1024) + (size >= 1048576 ? "M" : "K");
        print_row(size_str, size, time_nonsimd, time_simd);
    }
}

void verify_distribution() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  DISTRIBUTION CORRECTNESS VERIFICATION                                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    size_t size = 1000000;
    std::vector<float> data_nonsimd(size);
    std::vector<float> data_simd(size);
    uint32_t seed = 42;

    auto dist_factory = []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); };

    ttml::core::sequential_generate(std::span{data_nonsimd}, dist_factory, seed);
    ttml::core::modern::sequential_generate(std::span{data_simd}, dist_factory, seed);

    auto check_stats = [](const std::vector<float>& data, const std::string& name) {
        double sum = 0.0, sum_sq = 0.0;
        float min_val = data[0], max_val = data[0];

        for (float val : data) {
            sum += val;
            sum_sq += val * val;
            if (val < min_val)
                min_val = val;
            if (val > max_val)
                max_val = val;
        }

        double mean = sum / data.size();
        double variance = (sum_sq / data.size()) - (mean * mean);
        double stddev = std::sqrt(variance);

        std::cout << name << " (1M samples):\n";
        std::cout << "  Range:     [" << std::fixed << std::setprecision(6) << min_val << ", " << max_val << "]\n";
        std::cout << "  Mean:      " << std::setw(10) << mean << " (expected: 0.0)\n";
        std::cout << "  Std Dev:   " << std::setw(10) << stddev << " (expected: 0.577)\n";
        std::cout << "  Variance:  " << std::setw(10) << variance << " (expected: 0.333)\n";

        bool valid =
            (min_val >= -1.0f) && (max_val <= 1.0f) && (std::abs(mean) < 0.01) && (std::abs(stddev - 0.577) < 0.01);
        std::cout << "  Status:    " << (valid ? "✓ PASS" : "✗ FAIL") << "\n\n";
    };

    check_stats(data_nonsimd, "Non-SIMD (MT19937)");
    check_stats(data_simd, "SIMD (AES-NI)");
}

void benchmark_thread_scaling() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  THREAD SCALING (1M elements uniform distribution)                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << std::left << std::setw(18) << "Threads" << std::right << std::setw(10) << "Non-SIMD" << std::setw(10)
              << "SIMD" << std::setw(10) << "Speedup" << std::setw(12) << "Non-SIMD" << std::setw(12) << "SIMD"
              << "\n";
    std::cout << std::left << std::setw(18) << "" << std::right << std::setw(10) << "(ms)" << std::setw(10) << "(ms)"
              << std::setw(10) << "" << std::setw(12) << "GB/s" << std::setw(12) << "GB/s"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    size_t size = 1048576;
    auto dist_factory = []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); };
    uint32_t seed = 42;

    std::vector<uint32_t> thread_counts = {1, 2, 4, 8};

    for (auto threads : thread_counts) {
        std::vector<float> data_nonsimd(size);
        std::vector<float> data_simd(size);

        auto time_nonsimd = benchmark_fast(
            [&]() { ttml::core::parallel_generate(std::span{data_nonsimd}, dist_factory, seed, threads); }, 5);

        auto time_simd = benchmark_fast(
            [&]() { ttml::core::modern::parallel_generate(std::span{data_simd}, dist_factory, seed, threads); }, 5);

        print_row(std::to_string(threads), size, time_nonsimd, time_simd);
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                          ║\n";
    std::cout << "║           SIMD vs Non-SIMD Random Generation Benchmark                  ║\n";
    std::cout << "║                                                                          ║\n";
    std::cout << "║  Comparing: ttml::core (MT19937) vs ttml::core::modern (AES-NI SIMD)   ║\n";
    std::cout << "║                                                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    std::cout << "\nSystem Configuration:\n";
    std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "  SIMD width:       128-bit (4x float32)\n";
    std::cout << "  AES-NI:           Enabled\n";
    std::cout << "  Optimization:     -O3 -march=native\n";

    verify_distribution();
    benchmark_uniform_sequential();
    benchmark_uniform_parallel();
    benchmark_thread_scaling();

    return 0;
}
