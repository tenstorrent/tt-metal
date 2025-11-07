// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "core/random.hpp"
#include "core/random_sse.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace std::chrono;

// Global configuration for CSV output
struct CsvConfig {
    bool enabled{false};
    std::string filename;
    std::ofstream file;
} g_csv;

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string name;
    std::string type;
    std::string distribution;
    size_t size{0};
    double time_ms{0.0};
    double throughput_gb_s{0.0};
    double elements_per_sec_m{0.0};
    double mean{0.0};
    double stddev{0.0};
    double median{0.0};
    double q25{0.0};  // 25th percentile
    double q75{0.0};  // 75th percentile
};

struct DistributionParams {
    std::string name;
    double expected_mean{0.0};
    double expected_stddev{0.0};
    bool has_finite_moments{true};
    double expected_median{0.0};
    double expected_q25{0.0};
    double expected_q75{0.0};
    bool has_simd_support{true};  // If false, skip SSE benchmarks
};

// ============================================================================
// CSV Output Functions
// ============================================================================

// Helper function to escape CSV fields (quote if contains comma, quote, or newline)
std::string escape_csv_field(const std::string& field) {
    bool needs_quoting = false;
    for (char c : field) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quoting = true;
            break;
        }
    }

    if (!needs_quoting) {
        return field;
    }

    // Quote the field and escape internal quotes
    std::string escaped = "\"";
    for (char c : field) {
        if (c == '"') {
            escaped += "\"\"";  // Escape quote by doubling it
        } else {
            escaped += c;
        }
    }
    escaped += "\"";
    return escaped;
}

void write_csv_header() {
    if (!g_csv.enabled || !g_csv.file.is_open())
        return;

    g_csv.file << "implementation,data_type,distribution,num_elements,"
               << "time_ms,throughput_gb_s,elements_per_sec_m,"
               << "mean,stddev,median,q25,q75\n";
}

void write_csv_row(const BenchmarkResult& result) {
    if (!g_csv.enabled || !g_csv.file.is_open())
        return;

    g_csv.file << escape_csv_field(result.name) << "," << escape_csv_field(result.type) << ","
               << escape_csv_field(result.distribution) << "," << result.size << "," << std::fixed
               << std::setprecision(6) << result.time_ms << "," << result.throughput_gb_s << ","
               << result.elements_per_sec_m << "," << result.mean << "," << result.stddev << "," << result.median << ","
               << result.q25 << "," << result.q75 << "\n";
}

// ============================================================================
// Benchmark Functions
// ============================================================================

template <typename T, typename Func>
BenchmarkResult run_benchmark(
    const std::string& name,
    const std::string& type_name,
    const std::string& dist_name,
    Func& func,
    size_t size,
    int iterations = 10) {
    BenchmarkResult result;
    result.name = name;
    result.type = type_name;
    result.distribution = dist_name;
    result.size = size;

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

    // Distribution stats - mean and variance
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

    // Calculate quantiles (need to sort a copy for this)
    std::vector<double> sorted_data(size);
    for (size_t i = 0; i < size; ++i) {
        sorted_data[i] = static_cast<double>(data[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    // Median (50th percentile)
    size_t mid = size / 2;
    if (size % 2 == 0) {
        result.median = (sorted_data[mid - 1] + sorted_data[mid]) / 2.0;
    } else {
        result.median = sorted_data[mid];
    }

    // 25th percentile
    size_t q25_idx = size / 4;
    result.q25 = sorted_data[q25_idx];

    // 75th percentile
    size_t q75_idx = (3 * size) / 4;
    result.q75 = sorted_data[q75_idx];

    return result;
}

// ============================================================================
// Benchmark Runners
// ============================================================================

template <typename T, typename DistFactory>
void benchmark_distribution(
    const std::string& type_name, DistFactory dist_factory, size_t size, const DistributionParams& params) {
    // Print header (skip in CSV-only mode)
    if (!g_csv.enabled) {
        std::cout << "\n\n" << (type_name + " - " + params.name + " (" + std::to_string(size) + " elements)") << "\n";
        std::cout << std::string(80, '=') << "\n";

        if (!params.has_simd_support) {
            std::cout << "Note: SIMD benchmarks disabled (no optimization available for this distribution)\n";
        }
    }

    uint32_t seed = 42;

    std::vector<BenchmarkResult> results;

    auto func_seq = [&](std::vector<T>& data) { ttml::core::sequential_generate(std::span{data}, dist_factory, seed); };
    results.push_back(run_benchmark<T>("MT19937 (Sequential)", type_name, params.name, func_seq, size));

    auto func_par = [&](std::vector<T>& data) { ttml::core::parallel_generate(std::span{data}, dist_factory, seed); };
    results.push_back(run_benchmark<T>("MT19937 (Parallel)", type_name, params.name, func_par, size));

    // SSE Sequential and Parallel - only if SIMD is supported for this distribution
    if (params.has_simd_support) {
        auto func_seq = [&](std::vector<T>& data) {
            ttml::core::sse::sequential_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("SSE (Sequential)", type_name, params.name, func_seq, size));

        auto func_par = [&](std::vector<T>& data) {
            ttml::core::sse::parallel_generate(std::span{data}, dist_factory, seed);
        };
        results.push_back(run_benchmark<T>("SSE (Parallel)", type_name, params.name, func_par, size));
    }

    // Write CSV rows if enabled
    for (const auto& r : results) {
        write_csv_row(r);
    }

    // Print results in vertical format with verification for each run (skip in CSV-only mode)
    if (g_csv.enabled) {
        return;  // Skip console output when CSV is enabled
    }

    double baseline = results.empty() ? 1.0 : results[0].time_ms;
    constexpr int label_width = 18;

    // Tolerances (these are reasonable for large sample sizes)
    constexpr double mean_tolerance = 1.0;    // 1% relative error
    constexpr double stddev_tolerance = 2.0;  // 2% relative error

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
                  << std::setprecision(6) << r.mean;

        // Add verification for mean
        if (params.has_finite_moments) {
            double mean_error = std::abs(r.mean - params.expected_mean);
            double mean_rel_error = params.expected_mean != 0.0 ? (mean_error / std::abs(params.expected_mean)) * 100.0
                                                                : mean_error * 100.0;
            bool mean_ok = mean_rel_error < mean_tolerance;
            std::cout << " (" << (mean_ok ? "✓" : "✗") << " " << std::setprecision(2) << mean_rel_error << "%)";
        }
        std::cout << "\n";

        std::cout << "  " << std::left << std::setw(label_width) << "StdDev:" << std::right << std::fixed
                  << std::setprecision(6) << r.stddev;

        // Add verification for stddev
        if (params.has_finite_moments) {
            double stddev_error = std::abs(r.stddev - params.expected_stddev);
            double stddev_rel_error =
                params.expected_stddev != 0.0 ? (stddev_error / params.expected_stddev) * 100.0 : stddev_error * 100.0;
            bool stddev_ok = stddev_rel_error < stddev_tolerance;
            std::cout << " (" << (stddev_ok ? "✓" : "✗") << " " << std::setprecision(2) << stddev_rel_error << "%)";
        }
        std::cout << "\n";

        // For distributions without finite moments, show quantiles
        if (!params.has_finite_moments) {
            std::cout << "  " << std::left << std::setw(label_width) << "Median:" << std::right << std::fixed
                      << std::setprecision(6) << r.median;
            double median_error = std::abs(r.median - params.expected_median);
            double median_abs_error = median_error;
            bool median_ok = median_abs_error < 0.1;  // absolute tolerance for median
            std::cout << " (" << (median_ok ? "✓" : "✗") << " Δ=" << std::setprecision(4) << median_error << ")\n";

            std::cout << "  " << std::left << std::setw(label_width) << "Q25 (25th %ile):" << std::right << std::fixed
                      << std::setprecision(6) << r.q25;
            double q25_error = std::abs(r.q25 - params.expected_q25);
            bool q25_ok = std::abs((r.q25 - params.expected_q25) / params.expected_q25) < 0.05;  // 5% tolerance
            std::cout << " (" << (q25_ok ? "✓" : "✗") << " Δ=" << std::setprecision(4) << q25_error << ")\n";

            std::cout << "  " << std::left << std::setw(label_width) << "Q75 (75th %ile):" << std::right << std::fixed
                      << std::setprecision(6) << r.q75;
            double q75_error = std::abs(r.q75 - params.expected_q75);
            bool q75_ok = std::abs((r.q75 - params.expected_q75) / params.expected_q75) < 0.05;  // 5% tolerance
            std::cout << " (" << (q75_ok ? "✓" : "✗") << " Δ=" << std::setprecision(4) << q75_error << ")\n";

            double iqr = r.q75 - r.q25;
            double expected_iqr = params.expected_q75 - params.expected_q25;
            std::cout << "  " << std::left << std::setw(label_width) << "IQR:" << std::right << std::fixed
                      << std::setprecision(6) << iqr;
            double iqr_error = std::abs(iqr - expected_iqr);
            bool iqr_ok = std::abs((iqr - expected_iqr) / expected_iqr) < 0.05;  // 5% tolerance
            std::cout << " (" << (iqr_ok ? "✓" : "✗") << " Δ=" << std::setprecision(4) << iqr_error << ")\n";
        }
    }

    // Summary section
    if (!params.has_finite_moments) {
        std::cout << "\nNote: This distribution has undefined mean/variance.\n";
        std::cout << "Verification uses robust statistics (median, IQR, quantiles).\n";
        std::cout << "Expected: Median=" << std::fixed << std::setprecision(4) << params.expected_median
                  << ", IQR=" << (params.expected_q75 - params.expected_q25) << "\n";
    } else {
        std::cout << "\nExpected: Mean=" << std::fixed << std::setprecision(4) << params.expected_mean
                  << ", StdDev=" << params.expected_stddev << "\n";
    }
}

template <typename T>
void benchmark_type(const std::string& type_name, size_t size) {
    // Uniform distribution [-1, 1]
    // E[X] = (a+b)/2 = 0, Var[X] = (b-a)^2/12, StdDev = (b-a)/sqrt(12) = 2/sqrt(12)
    {
        // For float and double, use matching distribution type; for bfloat16, use float
        using DistType = std::conditional_t<std::same_as<T, double>, double, float>;
        auto uniform_factory = []() { return std::uniform_real_distribution<DistType>(-1.0, 1.0); };
        DistributionParams params{
            .name = "Uniform[-1,1]",
            .expected_mean = 0.0,
            .expected_stddev = 2.0 / std::sqrt(12.0),
            .has_finite_moments = true,
            .expected_median = 0.0,
            .expected_q25 = 0.0,
            .expected_q75 = 0.0,
            .has_simd_support = true  // SIMD optimized
        };
        benchmark_distribution<T>(type_name, uniform_factory, size, params);
    }

    // Normal distribution (mean=0, stddev=1)
    // E[X] = μ = 0, StdDev = σ = 1
    {
        using DistType = std::conditional_t<std::same_as<T, double>, double, float>;
        auto normal_factory = []() { return std::normal_distribution<DistType>(0.0, 1.0); };
        DistributionParams params{
            .name = "Normal(0,1)",
            .expected_mean = 0.0,
            .expected_stddev = 1.0,
            .has_finite_moments = true,
            .expected_median = 0.0,
            .expected_q25 = 0.0,
            .expected_q75 = 0.0,
            .has_simd_support = std::same_as<T, float> || std::same_as<T, bfloat16>  // SIMD for float/bfloat16 only
        };
        benchmark_distribution<T>(type_name, normal_factory, size, params);
    }

    // Exponential distribution (lambda=1.0)
    // E[X] = 1/λ = 1, StdDev = 1/λ = 1
    {
        using DistType = std::conditional_t<std::same_as<T, double>, double, float>;
        auto exponential_factory = []() { return std::exponential_distribution<DistType>(1.0); };
        DistributionParams params{
            .name = "Exponential(1.0)",
            .expected_mean = 1.0,
            .expected_stddev = 1.0,
            .has_finite_moments = true,
            .expected_median = 0.0,
            .expected_q25 = 0.0,
            .expected_q75 = 0.0,
            .has_simd_support = false  // No SIMD optimization
        };
        benchmark_distribution<T>(type_name, exponential_factory, size, params);
    }

    // Log-normal distribution (μ=0, σ=1)
    // E[X] = exp(μ + σ²/2), Var[X] = (exp(σ²)-1)*exp(2μ+σ²)
    {
        using DistType = std::conditional_t<std::same_as<T, double>, double, float>;
        auto lognormal_factory = []() { return std::lognormal_distribution<DistType>(0.0, 1.0); };
        DistributionParams params{
            .name = "LogNormal(0,1)",
            .expected_mean = std::exp(0.5),  // exp(0 + 1/2)
            .expected_stddev = std::sqrt((std::exp(1.0) - 1.0) * std::exp(1.0)),
            .has_finite_moments = true,
            .expected_median = 0.0,
            .expected_q25 = 0.0,
            .expected_q75 = 0.0,
            .has_simd_support = false  // No SIMD optimization
        };
        benchmark_distribution<T>(type_name, lognormal_factory, size, params);
    }

    // Cauchy distribution (location=0, scale=1)
    // Mean and variance are undefined (infinite)
    // Median = x0 = 0, Q1 = x0 - γ = -1, Q3 = x0 + γ = 1, IQR = 2γ = 2
    {
        using DistType = std::conditional_t<std::same_as<T, double>, double, float>;
        auto cauchy_factory = []() { return std::cauchy_distribution<DistType>(0.0, 1.0); };
        DistributionParams params{
            .name = "Cauchy(0,1)",
            .expected_mean = 0.0,    // not used
            .expected_stddev = 1.0,  // not used
            .has_finite_moments = false,
            .expected_median = 0.0,
            .expected_q25 = -1.0,
            .expected_q75 = 1.0,
            .has_simd_support = false  // No SIMD optimization
        };
        benchmark_distribution<T>(type_name, cauchy_factory, size, params);
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [SIZE]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --csv FILE    Write results to CSV file (e.g., --csv=results.csv)\n";
    std::cout << "  --help        Show this help message\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  SIZE          Number of elements to generate (default: 4194304)\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " 1000000\n";
    std::cout << "  " << program_name << " --csv=benchmark.csv 2000000\n";
}

int main(int argc, const char** argv) {
    // Parse command-line arguments
    constexpr size_t default_size = 4194304;  // 4M elements
    size_t size = default_size;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.starts_with("--csv=")) {
            g_csv.enabled = true;
            g_csv.filename = std::string(arg.substr(6));
            g_csv.file.open(g_csv.filename);
            if (!g_csv.file.is_open()) {
                std::cerr << "Error: Failed to open CSV file: " << g_csv.filename << "\n";
                return 1;
            }
        } else if (arg.starts_with("--csv")) {
            std::cerr << "Error: --csv requires a filename (e.g., --csv=results.csv)\n";
            return 1;
        } else {
            // Try to parse as size
            try {
                size = std::stoul(std::string(arg));
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid argument: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    // Write CSV header if CSV output is enabled
    if (g_csv.enabled) {
        write_csv_header();
        std::cerr << "Writing benchmark results to: " << g_csv.filename << "\n";
    } else {
        // Print system info only in console mode
        std::cout << "CPU Features:\n";
        std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    }

    // Run benchmarks
    benchmark_type<float>("float", size);
    benchmark_type<double>("double", size);
    benchmark_type<bfloat16>("bfloat16", size);

    // Close CSV file if opened
    if (g_csv.enabled && g_csv.file.is_open()) {
        g_csv.file.close();
        std::cerr << "Benchmark results written to: " << g_csv.filename << "\n";
    }

    return 0;
}
