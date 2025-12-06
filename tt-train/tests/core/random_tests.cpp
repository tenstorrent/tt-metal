// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <random>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/random_sse.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"

// Test parameters: size and tolerance
struct TestParams {
    size_t size;
    double tolerance;
};

class RandomGenerationTests : public ::testing::Test, public ::testing::WithParamInterface<TestParams> {
protected:
    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
    }

    size_t get_size() const {
        return GetParam().size;
    }
    double get_tolerance() const {
        return GetParam().tolerance;
    }
};

TEST_P(RandomGenerationTests, ParallelUniformInitDeterminism) {
    // Test that parallel initialization produces deterministic results for a
    // fixed seed.
    const size_t size = get_size();

    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::core::legacy::parallel_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);
    ttml::core::legacy::parallel_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    EXPECT_EQ(vec1, vec2);
}

TEST_P(RandomGenerationTests, UniformInitsGoodMeanAndRange) {
    const size_t size = get_size();

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

    // Tightened tolerance: with 100M samples, standard error is even smaller
    EXPECT_NEAR(parallel_mean, 0.0, 0.003);
    EXPECT_NEAR(sequential_mean, 0.0, 0.003);
}

TEST_P(RandomGenerationTests, ParallelUniformInitDeterminismSSE) {
    // Test that parallel initialization produces deterministic results for a
    // fixed seed.
    const size_t size = get_size();

    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::core::sse::parallel_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);
    ttml::core::sse::parallel_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    EXPECT_EQ(vec1, vec2);
}

TEST_P(RandomGenerationTests, UniformInitsGoodMeanAndRangeSSE) {
    const size_t size = get_size();

    std::vector<float> parallel_vec(size);
    std::vector<float> sequential_vec(size);

    {
        std::jthread sequential_thread([&sequential_vec]() {
            ttml::core::sse::sequential_generate(
                std::span{sequential_vec.data(), sequential_vec.size()},
                []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
                42);
        });

        ttml::core::sse::parallel_generate(
            std::span{parallel_vec.data(), parallel_vec.size()},
            []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
            42);
    }

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

    // Tightened tolerance: with 100M samples, standard error is even smaller
    EXPECT_NEAR(parallel_mean, 0.0, 0.003);
    EXPECT_NEAR(sequential_mean, 0.0, 0.003);
}

// ============================================================================
// SIMD RNG Distribution Support Tests
// ============================================================================

TEST_P(RandomGenerationTests, UniformDistributionDifferentParametersSSE) {
    // Test uniform distribution with various parameter ranges
    const size_t size = get_size();

    struct TestCase {
        float min;
        float max;
        double expected_mean;
        double expected_stddev;
    };

    const std::array<TestCase, 5> test_cases = {{
        {0.0f, 1.0f, 0.5, std::sqrt(1.0 / 12.0)},      // [0, 1]
        {-1.0f, 1.0f, 0.0, 2.0 / std::sqrt(12.0)},     // [-1, 1]
        {-10.0f, 10.0f, 0.0, 20.0 / std::sqrt(12.0)},  // [-10, 10]
        {5.0f, 15.0f, 10.0, 10.0 / std::sqrt(12.0)},   // [5, 15]
        {-0.5f, 0.5f, 0.0, 1.0 / std::sqrt(12.0)},     // [-0.5, 0.5]
    }};

    for (const auto& test_case : test_cases) {
        std::vector<float> vec(size);

        ttml::core::sse::parallel_generate(
            std::span{vec.data(), vec.size()},
            [&]() { return std::uniform_real_distribution<float>(test_case.min, test_case.max); },
            42);

        // Check range
        for (float val : vec) {
            EXPECT_GE(val, test_case.min);
            EXPECT_LE(val, test_case.max);
        }

        // Check mean
        // Tolerance is parameterized based on sample size
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double mean = sum / size;
        EXPECT_NEAR(mean, test_case.expected_mean, get_tolerance())
            << "Mean mismatch for uniform[" << test_case.min << ", " << test_case.max << "]";

        // Check standard deviation
        // Use parameterized tolerance for stddev as well
        double var_sum = 0.0;
        for (float val : vec) {
            double diff = static_cast<double>(val) - mean;
            var_sum += diff * diff;
        }
        double stddev = std::sqrt(var_sum / size);
        EXPECT_NEAR(stddev, test_case.expected_stddev, get_tolerance())
            << "StdDev mismatch for uniform[" << test_case.min << ", " << test_case.max << "]";
    }
}

TEST_P(RandomGenerationTests, NormalDistributionMeanAndStddevSSE) {
    // Test normal distribution with various parameters
    const size_t size = get_size();

    struct TestCase {
        float mean;
        float stddev;
    };

    constexpr std::array<TestCase, 4> test_cases = {{
        {0.0f, 1.0f},   // Standard normal
        {5.0f, 2.0f},   // Mean=5, StdDev=2
        {-3.0f, 0.5f},  // Mean=-3, StdDev=0.5
        {10.0f, 3.0f},  // Mean=10, StdDev=3
    }};

    for (const auto& test_case : test_cases) {
        std::vector<float> vec(size);

        ttml::core::sse::sequential_generate(
            std::span{vec.data(), vec.size()},
            [&]() { return std::normal_distribution<float>(test_case.mean, test_case.stddev); },
            42);

        // Check mean
        // Tolerance is parameterized based on sample size
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double mean = sum / size;
        EXPECT_NEAR(mean, test_case.mean, get_tolerance())
            << "Mean mismatch for normal(mean=" << test_case.mean << ", stddev=" << test_case.stddev << ")";

        // Check standard deviation (use sample stddev for better estimate)
        // With 1M samples, standard error of stddev ≈ stddev/sqrt(2*n) ≈ 0.0007 for stddev=1
        // Use relative tolerance: 0.3% is reasonable with 1024-rectangle Ziggurat and double precision
        double var_sum = 0.0;
        for (float val : vec) {
            double diff = static_cast<double>(val) - mean;
            var_sum += diff * diff;
        }
        // Use sample standard deviation (divide by n-1) for better estimate
        double stddev = std::sqrt(var_sum / (size - 1));
        // Tightened tolerance: 0.3% relative with minimum absolute tolerance
        double tolerance = std::max(0.003 * test_case.stddev, 0.002);
        EXPECT_NEAR(stddev, test_case.stddev, tolerance)
            << "StdDev mismatch for normal(mean=" << test_case.mean << ", stddev=" << test_case.stddev << ")";
    }
}

TEST_P(RandomGenerationTests, NormalDistributionRangeSSE) {
    // Test that normal distribution values are within reasonable bounds
    // For a normal distribution, ~99.7% of values should be within 3 standard deviations
    const size_t size = get_size();
    constexpr float mean = 0.0f;
    constexpr float stddev = 1.0f;
    constexpr float tolerance_stddevs = 4.0f;  // Allow up to 4 stddevs

    std::vector<float> vec(size);

    ttml::core::sse::sequential_generate(
        std::span{vec.data(), vec.size()}, [&]() { return std::normal_distribution<float>(mean, stddev); }, 42);

    // Check that values are within reasonable bounds
    size_t out_of_bounds = 0;
    for (float val : vec) {
        if (std::abs(val - mean) > tolerance_stddevs * stddev) {
            out_of_bounds++;
        }
    }

    // Should have very few values outside 4 standard deviations (< 0.01%)
    // Allow slightly more tolerance due to statistical variance
    double out_of_bounds_ratio = static_cast<double>(out_of_bounds) / size;
    EXPECT_LT(out_of_bounds_ratio, 0.0002) << "Too many values outside reasonable bounds";
}

TEST_P(RandomGenerationTests, UniformDistributionVarianceSSE) {
    // Test variance calculation for uniform distribution
    const size_t size = get_size();
    constexpr float min = -1.0f;
    constexpr float max = 1.0f;
    constexpr double expected_variance = (max - min) * (max - min) / 12.0;  // (b-a)^2/12

    std::vector<float> vec(size);

    ttml::core::sse::sequential_generate(
        std::span{vec.data(), vec.size()}, [&]() { return std::uniform_real_distribution<float>(min, max); }, 42);

    // Calculate sample variance
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    double mean = sum / size;

    double var_sum = 0.0;
    for (float val : vec) {
        double diff = static_cast<double>(val) - mean;
        var_sum += diff * diff;
    }
    double variance = var_sum / size;

    // Variance tolerance: with 1M samples, can tighten to 0.003-0.005
    EXPECT_NEAR(variance, expected_variance, 0.005) << "Variance mismatch for uniform distribution";
}

TEST_P(RandomGenerationTests, UniformDistributionBfloat16SSE) {
    // Test uniform distribution with bfloat16
    const size_t size = get_size();
    constexpr float min = -1.0f;
    constexpr float max = 1.0f;

    std::vector<bfloat16> vec(size);

    ttml::core::sse::sequential_generate(
        std::span{vec.data(), vec.size()}, [&]() { return std::uniform_real_distribution<float>(min, max); }, 42);

    // Check range
    for (bfloat16 val : vec) {
        float val_float = static_cast<float>(val);
        EXPECT_GE(val_float, min);
        EXPECT_LE(val_float, max);
    }

    // Check mean
    double sum = 0.0;
    for (bfloat16 val : vec) {
        sum += static_cast<double>(static_cast<float>(val));
    }
    double mean = sum / size;
    // bfloat16 has reduced precision, but with 1M samples can still use tighter tolerance
    EXPECT_NEAR(mean, 0.0, 0.005) << "Mean mismatch for bfloat16 uniform distribution";
}

TEST_P(RandomGenerationTests, NormalDistributionBfloat16SSE) {
    // Test normal distribution with bfloat16
    const size_t size = get_size();
    constexpr float mean = 0.0f;
    constexpr float stddev = 1.0f;

    std::vector<bfloat16> vec(size);

    ttml::core::sse::sequential_generate(
        std::span{vec.data(), vec.size()}, [&]() { return std::normal_distribution<float>(mean, stddev); }, 42);

    // Check mean
    double sum = 0.0;
    for (bfloat16 val : vec) {
        sum += static_cast<double>(static_cast<float>(val));
    }
    double computed_mean = sum / size;
    // bfloat16 has reduced precision, but with 1M samples can still use tighter tolerance
    EXPECT_NEAR(computed_mean, mean, 0.005) << "Mean mismatch for bfloat16 normal distribution";

    // Check standard deviation (use sample stddev for better estimate)
    double var_sum = 0.0;
    for (bfloat16 val : vec) {
        double diff = static_cast<double>(static_cast<float>(val)) - computed_mean;
        var_sum += diff * diff;
    }
    // Use sample standard deviation (divide by n-1) for better estimate
    double computed_stddev = std::sqrt(var_sum / (size - 1));
    // Tightened tolerance: bfloat16 has reduced precision but 1024-rectangle Ziggurat helps
    // Use 0.5% relative tolerance with minimum absolute tolerance
    double tolerance = std::max(0.005 * stddev, 0.003);
    EXPECT_NEAR(computed_stddev, stddev, tolerance) << "StdDev mismatch for bfloat16 normal distribution";
}

TEST_P(RandomGenerationTests, SequentialVsParallelConsistencySSE) {
    // Test that sequential and parallel generate produce consistent results
    // Note: They won't be identical due to different thread seeds, but should have
    // similar statistical properties
    const size_t size = get_size();

    std::vector<float> sequential_vec(size);
    std::vector<float> parallel_vec(size);

    ttml::core::sse::sequential_generate(
        std::span{sequential_vec.data(), sequential_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42);

    ttml::core::sse::parallel_generate(
        std::span{parallel_vec.data(), parallel_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42);

    // Check that both have similar statistical properties
    auto compute_stats = [](const std::vector<float>& vec) -> std::pair<double, double> {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double mean = sum / vec.size();

        double var_sum = 0.0;
        for (float val : vec) {
            double diff = static_cast<double>(val) - mean;
            var_sum += diff * diff;
        }
        double stddev = std::sqrt(var_sum / vec.size());
        return {mean, stddev};
    };

    auto [seq_mean, seq_stddev] = compute_stats(sequential_vec);
    auto [par_mean, par_stddev] = compute_stats(parallel_vec);

    // Sequential and parallel should have very similar statistics with same seed
    // Tightened tolerance: with 1M samples and deterministic seeding
    EXPECT_NEAR(seq_mean, par_mean, 0.005) << "Mean mismatch between sequential and parallel";
    EXPECT_NEAR(seq_stddev, par_stddev, 0.005) << "StdDev mismatch between sequential and parallel";
}

TEST_P(RandomGenerationTests, DifferentSeedsProduceDifferentResultsSSE) {
    // Test that different seeds produce different sequences
    const size_t size = get_size();

    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::core::sse::sequential_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    ttml::core::sse::sequential_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 123);

    // Vectors should be different (very unlikely to be identical)
    EXPECT_NE(vec1, vec2) << "Different seeds produced identical sequences";
}

TEST_F(RandomGenerationTests, EdgeCaseSmallSizesSSE) {
    // Test edge cases with small sizes
    constexpr std::array<size_t, 17> sizes = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65};

    for (size_t size : sizes) {
        std::vector<float> vec(size);

        ttml::core::sse::sequential_generate(
            std::span{vec.data(), vec.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

        // Check range
        for (float val : vec) {
            EXPECT_GE(val, -1.0f);
            EXPECT_LE(val, 1.0f);
        }

        // For sizes >= 16, check mean is reasonable (skip for very small sizes due to high variance)
        if (size >= 16) {
            double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
            double mean = sum / size;
            // Use more lenient tolerance for small samples
            double tolerance = std::max(0.2, 0.5 / std::sqrt(static_cast<double>(size)));
            EXPECT_NEAR(mean, 0.0, tolerance) << "Mean mismatch for size " << size;
        }
    }
}

TEST_P(RandomGenerationTests, NormalDistributionQuantilesSSE) {
    // Test that normal distribution has correct quantiles
    // For standard normal: Q25 ≈ -0.674, Q50 (median) = 0, Q75 ≈ 0.674
    const size_t size = get_size();
    constexpr float mean = 0.0f;
    constexpr float stddev = 1.0f;

    std::vector<float> vec(size);

    ttml::core::sse::sequential_generate(
        std::span{vec.data(), vec.size()}, [&]() { return std::normal_distribution<float>(mean, stddev); }, 42);

    // Sort to find quantiles
    std::vector<float> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    size_t q25_idx = size / 4;
    size_t q50_idx = size / 2;
    size_t q75_idx = (3 * size) / 4;

    double q25 = sorted_vec[q25_idx];
    double q50 = sorted_vec[q50_idx];
    double q75 = sorted_vec[q75_idx];

    // Check median (should be close to mean for normal distribution)
    // With 1M samples, median standard error ≈ 0.001, so 0.003-0.005 is reasonable
    EXPECT_NEAR(q50, mean, 0.005) << "Median mismatch for normal distribution";

    // Check that Q25 and Q75 are approximately symmetric around mean
    // Quantiles have higher variance, but with 1M samples and 1024-rectangle Ziggurat,
    // symmetry should be within 0.01-0.02
    EXPECT_NEAR(q25, -q75, 0.02) << "Q25 and Q75 not symmetric for normal distribution";

    // Check approximate values (using tolerance for quantiles)
    // With 1M samples, quantile standard error ≈ 0.0016, so 0.005-0.01 is reasonable
    // With 1024-rectangle Ziggurat and double precision, can use tighter tolerance
    EXPECT_NEAR(q25, -0.674, 0.01) << "Q25 mismatch for standard normal";
    EXPECT_NEAR(q75, 0.674, 0.01) << "Q75 mismatch for standard normal";
}

TEST_P(RandomGenerationTests, UniformDistributionDeterminismParallelSSE) {
    // Test that parallel generation with same seed is deterministic
    const size_t size = get_size();

    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::core::sse::parallel_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    ttml::core::sse::parallel_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    EXPECT_EQ(vec1, vec2) << "Parallel generation not deterministic with same seed";
}

TEST_P(RandomGenerationTests, NormalDistributionDeterminismParallelSSE) {
    // Test that parallel normal generation with same seed is deterministic
    const size_t size = get_size();

    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::core::sse::parallel_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, 42);

    ttml::core::sse::parallel_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, 42);

    EXPECT_EQ(vec1, vec2) << "Parallel normal generation not deterministic with same seed";
}

TEST_P(RandomGenerationTests, UniformDistributionStatisticalPropertiesSSE) {
    // More rigorous statistical test for uniform distribution
    // Check that the distribution is approximately uniform across bins
    constexpr size_t num_bins = 100;
    constexpr float min = -1.0f;
    constexpr float max = 1.0f;
    constexpr double bin_width = (max - min) / num_bins;
    constexpr double tolerance = 0.1;  // 10% tolerance
    const size_t size = get_size();
    const double expected_count_per_bin = static_cast<double>(size) / num_bins;

    std::vector<float> vec(size);

    ttml::core::sse::sequential_generate(
        std::span{vec.data(), vec.size()}, [&]() { return std::uniform_real_distribution<float>(min, max); }, 42);

    // Count values in each bin
    std::vector<size_t> bin_counts(num_bins, 0);
    for (float val : vec) {
        int bin_idx = static_cast<int>((val - min) / bin_width);
        bin_idx = std::max(0, std::min(bin_idx, static_cast<int>(num_bins) - 1));
        bin_counts[bin_idx]++;
    }

    // Check that bin counts are approximately uniform
    for (size_t i = 0; i < num_bins; ++i) {
        double count_ratio = static_cast<double>(bin_counts[i]) / expected_count_per_bin;
        EXPECT_NEAR(count_ratio, 1.0, tolerance)
            << "Bin " << i << " has non-uniform distribution (ratio: " << count_ratio << ")";
    }
}

// Instantiate parameterized tests with different sizes and tolerances
INSTANTIATE_TEST_SUITE_P(
    RandomGeneration,
    RandomGenerationTests,
    ::testing::Values(
        TestParams{1024 * 256, 0.03},        // 256K elements, tolerance 0.03
        TestParams{1024 * 1024, 0.015},      // 1M elements, tolerance 0.015
        TestParams{10 * 1024 * 1024, 0.005}  // 10M elements, tolerance 0.005
        ),
    [](const ::testing::TestParamInfo<TestParams>& info) {
        const TestParams& params = info.param;
        if (params.size == 1024 * 256) {
            return "256K";
        } else if (params.size == 1024 * 1024) {
            return "1M";
        } else if (params.size == 10 * 1024 * 1024) {
            return "10M";
        }
        return "Custom";
    });
