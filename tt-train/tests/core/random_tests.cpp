// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
#include "models/gpt2.hpp"
#include "models/llama.hpp"

class RandomGenerationTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Set a fixed seed for reproducible tests
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
    }
};

TEST_F(RandomGenerationTests, ParallelUniformInitDeterminism) {
    // Test that parallel initialization produces deterministic results for a
    // fixed seed.
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    // Create two vectors for comparison
    std::vector<float> vec1(size);
    std::vector<float> vec2(size);

    ttml::init::UniformRange range{-1.0f, 1.0f};

    auto uniform_factory = [&]() { return std::uniform_real_distribution<float>(range.a, range.b); };
    ttml::core::random::parallel_generate(std::span{vec1.data(), vec1.size()}, uniform_factory, 42);
    ttml::core::random::parallel_generate(std::span{vec2.data(), vec2.size()}, uniform_factory, 42);

    // Results should be identical
    EXPECT_EQ(vec1, vec2);
}

TEST_F(RandomGenerationTests, UniformInitsGoodMeanAndRange) {
    // Test that parallel and sequential produce different but valid results
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> parallel_vec(size);
    std::vector<float> sequential_vec(size);

    ttml::init::UniformRange range{-1.0f, 1.0f};

    auto uniform_factory = [&]() { return std::uniform_real_distribution<float>(range.a, range.b); };
    ttml::core::random::parallel_generate(std::span{parallel_vec.data(), parallel_vec.size()}, uniform_factory, 42);

    // Initialize with sequential method
    ttml::core::random::sequential_generate(
        std::span{sequential_vec.data(), sequential_vec.size()}, uniform_factory, 42);

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

    // Both means should be close to 0
    EXPECT_NEAR(parallel_mean, 0.0, 0.01);
    EXPECT_NEAR(sequential_mean, 0.0, 0.01);
}
