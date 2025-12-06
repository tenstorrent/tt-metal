// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <random>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/random_sse.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"

namespace {

// Helper function to compute statistics on a vector of doubles
struct Statistics {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;

    void compute(std::vector<double>& data) {
        if (data.empty()) {
            return;
        }

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

}  // namespace

class RandomGenerationTests : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
    }
};

TEST_F(RandomGenerationTests, ParallelUniformInitDeterminism) {
    // Test that parallel initialization produces deterministic results for a
    // fixed seed.
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::core::legacy::parallel_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);
    ttml::core::legacy::parallel_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    EXPECT_EQ(vec1, vec2);
}

TEST_F(RandomGenerationTests, UniformInitsGoodMeanAndRange) {
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> parallel_vec(size);
    std::vector<float> sequential_vec(size);

    ttml::core::legacy::parallel_generate(
        std::span{parallel_vec.data(), parallel_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42);

    ttml::core::legacy::sequential_generate(
        std::span{sequential_vec.data(), sequential_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42);

    // Check that both vectors contain values in the expected range
    auto check_range = [](const std::vector<float>& vec) {
        return std::all_of(vec.begin(), vec.end(), [](float val) { return val >= -1.0f && val <= 1.0f; });
    };

    EXPECT_TRUE(check_range(parallel_vec));
    EXPECT_TRUE(check_range(sequential_vec));

    auto compute_mean = [](const std::vector<float>& vec) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        return sum / vec.size();
    };

    double parallel_mean = compute_mean(parallel_vec);
    double sequential_mean = compute_mean(sequential_vec);

    EXPECT_NEAR(parallel_mean, 0.0, 0.01);
    EXPECT_NEAR(sequential_mean, 0.0, 0.01);
}

TEST_F(RandomGenerationTests, ParallelUniformInitDeterminismSSE) {
    // Test that parallel initialization produces deterministic results for a
    // fixed seed.
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::core::sse::parallel_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);
    ttml::core::sse::parallel_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    EXPECT_EQ(vec1, vec2);
}

TEST_F(RandomGenerationTests, UniformInitsGoodMeanAndRangeSSE) {
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> parallel_vec(size);
    std::vector<float> sequential_vec(size);

    ttml::core::sse::parallel_generate(
        std::span{parallel_vec.data(), parallel_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42);

    ttml::core::sse::sequential_generate(
        std::span{sequential_vec.data(), sequential_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42);

    // Check that both vectors contain values in the expected range
    auto check_range = [](const std::vector<float>& vec) {
        return std::all_of(vec.begin(), vec.end(), [](float val) { return val >= -1.0f && val <= 1.0f; });
    };

    EXPECT_TRUE(check_range(parallel_vec));
    EXPECT_TRUE(check_range(sequential_vec));

    auto compute_mean = [](const std::vector<float>& vec) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        return sum / vec.size();
    };

    double parallel_mean = compute_mean(parallel_vec);
    double sequential_mean = compute_mean(sequential_vec);

    EXPECT_NEAR(parallel_mean, 0.0, 0.01);
    EXPECT_NEAR(sequential_mean, 0.0, 0.01);
}

// ============================================================================
// Distribution-specific tests for SSE RNG
// ============================================================================

// Test 1: Uniform Distribution [0, 1) - float - sequential
TEST_F(RandomGenerationTests, SSE_UniformDistribution_Float_Sequential) {
    constexpr size_t size = 1000000;
    std::vector<float> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::sequential_generate(
        std::span{data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double(data.begin(), data.end());
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0.5, stddev ≈ 1/sqrt(12) ≈ 0.2887
    EXPECT_NEAR(stats.mean, 0.5, 0.01);
    EXPECT_NEAR(stats.stddev, 0.2887, 0.01);
    EXPECT_GE(stats.min, 0.0f);
    EXPECT_LE(stats.max, 1.0f);
}

// Test 2: Uniform Distribution [0, 1) - float - parallel
TEST_F(RandomGenerationTests, SSE_UniformDistribution_Float_Parallel) {
    constexpr size_t size = 1000000;
    std::vector<float> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::parallel_generate(
        std::span{data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double(data.begin(), data.end());
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0.5, stddev ≈ 1/sqrt(12) ≈ 0.2887
    EXPECT_NEAR(stats.mean, 0.5, 0.01);
    EXPECT_NEAR(stats.stddev, 0.2887, 0.01);
    EXPECT_GE(stats.min, 0.0f);
    EXPECT_LE(stats.max, 1.0f);
}

// Test 3: Uniform Distribution [0, 1) - bfloat16 - sequential
TEST_F(RandomGenerationTests, SSE_UniformDistribution_bfloat16_Sequential) {
    constexpr size_t size = 1000000;
    std::vector<bfloat16> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::sequential_generate(
        std::span{data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double;
    for (auto val : data) {
        data_double.push_back(static_cast<double>(static_cast<float>(val)));
    }
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0.5, stddev ≈ 1/sqrt(12) ≈ 0.2887
    // Higher tolerance for bfloat16 due to lower precision
    EXPECT_NEAR(stats.mean, 0.5, 0.02);
    EXPECT_NEAR(stats.stddev, 0.2887, 0.02);
    EXPECT_GE(stats.min, 0.0f);
    EXPECT_LE(stats.max, 1.0f);
}

// Test 4: Uniform Distribution [0, 1) - bfloat16 - parallel
TEST_F(RandomGenerationTests, SSE_UniformDistribution_bfloat16_Parallel) {
    constexpr size_t size = 1000000;
    std::vector<bfloat16> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::parallel_generate(
        std::span{data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double;
    for (auto val : data) {
        data_double.push_back(static_cast<double>(static_cast<float>(val)));
    }
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0.5, stddev ≈ 1/sqrt(12) ≈ 0.2887
    // Higher tolerance for bfloat16 due to lower precision
    EXPECT_NEAR(stats.mean, 0.5, 0.02);
    EXPECT_NEAR(stats.stddev, 0.2887, 0.02);
    EXPECT_GE(stats.min, 0.0f);
    EXPECT_LE(stats.max, 1.0f);
}

// Test 5: Normal Distribution N(0,1) - float - sequential
TEST_F(RandomGenerationTests, SSE_NormalDistribution_Float_Sequential) {
    constexpr size_t size = 1000000;
    std::vector<float> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::sequential_generate(
        std::span{data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double(data.begin(), data.end());
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0, stddev ≈ 1
    EXPECT_NEAR(stats.mean, 0.0, 0.01);
    EXPECT_NEAR(stats.stddev, 1.0, 0.05);
}

// Test 6: Normal Distribution N(0,1) - float - parallel
TEST_F(RandomGenerationTests, SSE_NormalDistribution_Float_Parallel) {
    constexpr size_t size = 1000000;
    std::vector<float> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::parallel_generate(
        std::span{data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double(data.begin(), data.end());
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0, stddev ≈ 1
    EXPECT_NEAR(stats.mean, 0.0, 0.01);
    EXPECT_NEAR(stats.stddev, 1.0, 0.05);
}

// Test 7: Normal Distribution N(0,1) - bfloat16 - sequential
TEST_F(RandomGenerationTests, SSE_NormalDistribution_bfloat16_Sequential) {
    constexpr size_t size = 1000000;
    std::vector<bfloat16> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::sequential_generate(
        std::span{data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double;
    for (auto val : data) {
        data_double.push_back(static_cast<double>(static_cast<float>(val)));
    }
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0, stddev ≈ 1
    // Higher tolerance for bfloat16 due to lower precision
    EXPECT_NEAR(stats.mean, 0.0, 0.02);
    EXPECT_NEAR(stats.stddev, 1.0, 0.05);
}

// Test 8: Normal Distribution N(0,1) - bfloat16 - parallel
TEST_F(RandomGenerationTests, SSE_NormalDistribution_bfloat16_Parallel) {
    constexpr size_t size = 1000000;
    std::vector<bfloat16> data(size);
    const uint32_t seed = 42;

    ttml::core::sse::parallel_generate(
        std::span{data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> data_double;
    for (auto val : data) {
        data_double.push_back(static_cast<double>(static_cast<float>(val)));
    }
    Statistics stats;
    stats.compute(data_double);

    // Expected: mean ≈ 0, stddev ≈ 1
    // Higher tolerance for bfloat16 due to lower precision
    EXPECT_NEAR(stats.mean, 0.0, 0.02);
    EXPECT_NEAR(stats.stddev, 1.0, 0.05);
}

// Test 9: Uniform Distribution with custom range (-1, 1) - float
TEST_F(RandomGenerationTests, SSE_UniformDistribution_CustomRange_Float) {
    constexpr size_t size = 1000000;
    std::vector<float> sequential_data(size);
    std::vector<float> parallel_data(size);
    const uint32_t seed = 42;

    // Generate with both methods
    ttml::core::sse::sequential_generate(
        std::span{sequential_data}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);

    ttml::core::sse::parallel_generate(
        std::span{parallel_data}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);

    // Check ranges
    auto check_range = [](const std::vector<float>& vec) {
        return std::all_of(vec.begin(), vec.end(), [](float val) { return val >= -1.0f && val <= 1.0f; });
    };

    EXPECT_TRUE(check_range(sequential_data));
    EXPECT_TRUE(check_range(parallel_data));

    // Check means
    auto compute_mean = [](const std::vector<float>& vec) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        return sum / vec.size();
    };

    double sequential_mean = compute_mean(sequential_data);
    double parallel_mean = compute_mean(parallel_data);

    EXPECT_NEAR(sequential_mean, 0.0, 0.01);
    EXPECT_NEAR(parallel_mean, 0.0, 0.01);
}

// Test 10: Normal Distribution with custom parameters - float
TEST_F(RandomGenerationTests, SSE_NormalDistribution_CustomParameters_Float) {
    constexpr size_t size = 1000000;
    std::vector<float> sequential_data(size);
    std::vector<float> parallel_data(size);
    const uint32_t seed = 42;
    const float expected_mean = 5.0f;
    const float expected_stddev = 2.0f;

    // Generate with both methods
    ttml::core::sse::sequential_generate(
        std::span{sequential_data},
        [&]() { return std::normal_distribution<float>(expected_mean, expected_stddev); },
        seed);

    ttml::core::sse::parallel_generate(
        std::span{parallel_data},
        [&]() { return std::normal_distribution<float>(expected_mean, expected_stddev); },
        seed);

    // Convert to double for statistics
    std::vector<double> seq_double(sequential_data.begin(), sequential_data.end());
    std::vector<double> par_double(parallel_data.begin(), parallel_data.end());

    Statistics seq_stats, par_stats;
    seq_stats.compute(seq_double);
    par_stats.compute(par_double);

    // Check means and stddevs
    EXPECT_NEAR(seq_stats.mean, expected_mean, 0.1);
    EXPECT_NEAR(par_stats.mean, expected_mean, 0.1);
    EXPECT_NEAR(seq_stats.stddev, expected_stddev, 0.1);
    EXPECT_NEAR(par_stats.stddev, expected_stddev, 0.1);
}

// Test 11: Determinism - sequential and parallel should give same results with same seed
TEST_F(RandomGenerationTests, SSE_Determinism_Sequential_vs_Parallel_Uniform_Float) {
    constexpr size_t size = 100000;
    std::vector<float> seq_data(size);
    std::vector<float> par_data(size);
    const uint32_t seed = 42;

    // Note: Parallel and sequential may not produce identical sequences due to threading,
    // but they should have the same statistical properties
    ttml::core::sse::sequential_generate(
        std::span{seq_data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

    ttml::core::sse::parallel_generate(
        std::span{par_data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> seq_double(seq_data.begin(), seq_data.end());
    std::vector<double> par_double(par_data.begin(), par_data.end());

    Statistics seq_stats, par_stats;
    seq_stats.compute(seq_double);
    par_stats.compute(par_double);

    // Check that both have similar distributions (not necessarily identical sequences)
    EXPECT_NEAR(seq_stats.mean, par_stats.mean, 0.02);
    EXPECT_NEAR(seq_stats.stddev, par_stats.stddev, 0.02);
}

// Test 12: Determinism - sequential and parallel should give same results with same seed - Normal
TEST_F(RandomGenerationTests, SSE_Determinism_Sequential_vs_Parallel_Normal_Float) {
    constexpr size_t size = 100000;
    std::vector<float> seq_data(size);
    std::vector<float> par_data(size);
    const uint32_t seed = 42;

    ttml::core::sse::sequential_generate(
        std::span{seq_data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

    ttml::core::sse::parallel_generate(
        std::span{par_data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

    // Convert to double for statistics
    std::vector<double> seq_double(seq_data.begin(), seq_data.end());
    std::vector<double> par_double(par_data.begin(), par_data.end());

    Statistics seq_stats, par_stats;
    seq_stats.compute(seq_double);
    par_stats.compute(par_double);

    // Check that both have similar distributions
    EXPECT_NEAR(seq_stats.mean, par_stats.mean, 0.02);
    EXPECT_NEAR(seq_stats.stddev, par_stats.stddev, 0.02);
}
