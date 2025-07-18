// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <init/cpu_initializers.hpp>
#include <init/xoshiro.hpp>
#include <random>

#include "autograd/auto_context.hpp"

class InitTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Set a fixed seed for reproducible tests
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
    }
};

TEST_F(InitTests, ParallelUniformInitDeterminism) {
    // Test that parallel initialization produces deterministic results
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    // Create two vectors for comparison
    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::init::UniformRange range{-1.0f, 1.0f};

    // Reset seed before each initialization
    ttml::autograd::ctx().set_seed(42);
    std::uniform_real_distribution<float> dist(range.a, range.b);
    ttml::init::parallel_init(vec1, dist);

    ttml::autograd::ctx().set_seed(42);
    std::uniform_real_distribution<float> dist2(range.a, range.b);
    ttml::init::parallel_init(vec2, dist2);

    // Results should be identical
    EXPECT_EQ(vec1, vec2);
}

TEST_F(InitTests, UniformInitsGoodMeanAndRange) {
    // Test that parallel and sequential produce different but valid results
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> parallel_vec(size);
    std::vector<float> sequential_vec(size);

    ttml::init::UniformRange range{-1.0f, 1.0f};

    // Initialize with parallel method
    ttml::autograd::ctx().set_seed(42);
    std::uniform_real_distribution<float> parallel_dist(range.a, range.b);
    ttml::init::parallel_init(parallel_vec, parallel_dist);

    // Initialize with sequential method
    ttml::autograd::ctx().set_seed(42);
    ttml::init::uniform_init(sequential_vec, range);

    // Results will be different due to different generation patterns
    // But both should be valid uniform distributions

    // Check that both vectors contain values in the expected range
    auto check_range = [&](const std::vector<float>& vec) {
        return std::all_of(vec.begin(), vec.end(), [&](float val) { return val >= range.a && val <= range.b; });
    };

    EXPECT_TRUE(check_range(parallel_vec));
    EXPECT_TRUE(check_range(sequential_vec));

    // Check that both have reasonable statistical properties
    auto compute_mean = [](const std::vector<float>& vec) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        return sum / vec.size();
    };

    double parallel_mean = compute_mean(parallel_vec);
    double sequential_mean = compute_mean(sequential_vec);

    // Both means should be close to 0 (midpoint of [-1, 1])
    EXPECT_NEAR(parallel_mean, 0.0, 0.01);
    EXPECT_NEAR(sequential_mean, 0.0, 0.01);
}

TEST_F(InitTests, ParallelInitPerformance) {
    // Performance comparison test
    constexpr size_t nano_gpt_size = 1024 * 256 * 384;  // ~100M elements
    auto bench_it = [](auto& size) {
        std::vector<float> parallel_vec(size);
        std::vector<float> sequential_vec(size);

        ttml::init::UniformRange range{-1.0f, 1.0f};

        constexpr int num_runs = 10;
        std::vector<std::chrono::nanoseconds> parallel_times;
        std::vector<std::chrono::nanoseconds> parallel_times_xoshiro;
        std::vector<std::chrono::nanoseconds> sequential_times;
        ttml::autograd::ctx().set_seed(42);

        parallel_times.reserve(num_runs);
        sequential_times.reserve(num_runs);
        parallel_times_xoshiro.reserve(num_runs);

        std::uniform_real_distribution<float> dist(range.a, range.b);
        // Run multiple times to get median
        for (int run = 0; run < num_runs; ++run) {
            // Time parallel initialization
            auto start = std::chrono::high_resolution_clock::now();
            ttml::init::parallel_init(parallel_vec, dist);
            auto end = std::chrono::high_resolution_clock::now();

            parallel_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));

            // time parallel initialization with xoshiro
            auto start_xoshiro = std::chrono::high_resolution_clock::now();
            auto xoshiro = ttml::init::Xoshiro128Plus(42);
            ttml::init::parallel_init(parallel_vec, dist, xoshiro);
            auto end_xoshiro = std::chrono::high_resolution_clock::now();
            parallel_times_xoshiro.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_xoshiro - start_xoshiro));

            // Time sequential initialization
            start = std::chrono::high_resolution_clock::now();
            ttml::init::uniform_init(sequential_vec, range);
            end = std::chrono::high_resolution_clock::now();

            sequential_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));
        }

        // Calculate median times
        auto calculate_median = [](std::vector<std::chrono::nanoseconds>& times) {
            std::sort(times.begin(), times.end());
            size_t n = times.size();
            if (n % 2 == 0) {
                return (times[n / 2 - 1] + times[n / 2]) / 2;
            } else {
                return times[n / 2];
            }
        };

        auto parallel_median = calculate_median(parallel_times);
        auto parallel_median_xoshiro = calculate_median(parallel_times_xoshiro);
        auto sequential_median = calculate_median(sequential_times);

        std::cout << "Parallel initialization with xoshiro took: " << parallel_median_xoshiro.count() << " ns"
                  << std::endl;
        std::cout << "Parallel initialization with xoshiro is "
                  << ((float)sequential_median.count() / (float)parallel_median_xoshiro.count()) << "x faster"
                  << std::endl;

        // Print timing results for manual inspection
        std::cout << "Size: " << size << std::endl;
        std::cout << "Parallel initialization took: " << parallel_median.count() << " ns" << std::endl;
        std::cout << "Sequential initialization took: " << sequential_median.count() << " ns" << std::endl;
        std::cout << "Parallel initialization is "
                  << ((float)sequential_median.count() / (float)parallel_median.count()) << "x faster" << std::endl;
    };

    bench_it(nano_gpt_size);

    // for (size_t expt = 10; expt <= 16; expt++) {
    //     size_t size = 1 << expt;
    //     bench_it(size);
    // }
}
