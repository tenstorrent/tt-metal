// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/random_sse.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"

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

TEST_F(RandomGenerationTests, ParallelUniformThreadAgnostic) {
    // Ensure that parlallel RNG gives same results regardless of number of threads.
    // Here, we compare outputs with 7 threads and all threads against sequential RNG.
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> sequential_vec(size);
    std::vector<float> parallel_vec(size);
    std::vector<float> parallel_all_threads_vec(size);

    const uint32_t num_threads = std::min<uint32_t>(7u, std::thread::hardware_concurrency());
    const uint32_t all_threads = std::thread::hardware_concurrency();

    ttml::core::legacy::parallel_generate(
        std::span{sequential_vec.data(), sequential_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42,
        1);
    ttml::core::legacy::parallel_generate(
        std::span{parallel_vec.data(), parallel_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42,
        num_threads);
    ttml::core::legacy::parallel_generate(
        std::span{parallel_all_threads_vec.data(), parallel_all_threads_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42,
        all_threads);

    EXPECT_EQ(sequential_vec, parallel_vec);
    EXPECT_EQ(sequential_vec, parallel_all_threads_vec);
}

TEST_F(RandomGenerationTests, ParallelNormalThreadAgnostic) {
    // Same thread-agnostic property as above, for the normal distribution.
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> sequential_vec(size);
    std::vector<float> parallel_vec(size);
    std::vector<float> parallel_all_threads_vec(size);

    const uint32_t num_threads = std::min<uint32_t>(7u, std::thread::hardware_concurrency());
    const uint32_t all_threads = std::thread::hardware_concurrency();
    ttml::core::legacy::parallel_generate(
        std::span{sequential_vec.data(), sequential_vec.size()},
        []() { return std::normal_distribution<float>(0.0f, 1.0f); },
        42,
        1);
    ttml::core::legacy::parallel_generate(
        std::span{parallel_vec.data(), parallel_vec.size()},
        []() { return std::normal_distribution<float>(0.0f, 1.0f); },
        42,
        num_threads);
    ttml::core::legacy::parallel_generate(
        std::span{parallel_all_threads_vec.data(), parallel_all_threads_vec.size()},
        []() { return std::normal_distribution<float>(0.0f, 1.0f); },
        42,
        all_threads);

    EXPECT_EQ(sequential_vec, parallel_vec);
    EXPECT_EQ(sequential_vec, parallel_all_threads_vec);
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

namespace {

// Generates 4099 elements, not a multiple of the SIMD batch size, with use_simd on, into a vector of sentinels that lie
// outside [lo, hi]. Checks that the sequential and the parallel generator each overwrite every sentinel.
template <typename T, typename DistGenFunc>
void expect_simd_generators_fill_every_element(const DistGenFunc& dist_factory, float lo, float hi) {
    for (const bool parallel : {false, true}) {
        std::vector<T> seq(4099, static_cast<T>(hi + 1.0F));
        if (parallel) {
            ttml::core::parallel_generate(
                std::span{seq}, dist_factory, 42, std::thread::hardware_concurrency(), /*use_simd=*/true);
        } else {
            ttml::core::sequential_generate(std::span{seq}, dist_factory, 42, /*use_simd=*/true);
        }
        for (size_t i = 0; i < seq.size(); ++i) {
            const auto value = static_cast<float>(seq[i]);
            ASSERT_TRUE(value >= lo && value <= hi)
                << (parallel ? "parallel" : "sequential") << " element " << i << " is " << value;
        }
    }
}

}  // namespace

TEST_F(RandomGenerationTests, SimdRngFillsEveryElement) {
    // The SIMD generator's fast paths.
    const auto uniform = []() { return std::uniform_real_distribution<float>(0.5F, 1.5F); };
    const auto normal = []() { return std::normal_distribution<float>(0.0F, 1.0F); };
    expect_simd_generators_fill_every_element<float>(uniform, 0.5F, 1.5F);
    expect_simd_generators_fill_every_element<bfloat16>(uniform, 0.5F, 1.5F);
    expect_simd_generators_fill_every_element<float>(normal, -100.0F, 100.0F);
    expect_simd_generators_fill_every_element<bfloat16>(normal, -100.0F, 100.0F);

    // The SIMD generator would leave these unwritten, so they fall back to the scalar one.
    expect_simd_generators_fill_every_element<uint32_t>(
        []() { return std::uniform_int_distribution<uint32_t>(0U, 63U); }, 0.0F, 63.0F);
    expect_simd_generators_fill_every_element<int32_t>(
        []() { return std::uniform_int_distribution<int32_t>(-8, 8); }, -8.0F, 8.0F);
    expect_simd_generators_fill_every_element<double>(
        []() { return std::uniform_real_distribution<double>(0.5, 1.5); }, 0.5F, 1.5F);
}
