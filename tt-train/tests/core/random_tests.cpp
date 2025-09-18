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

    ttml::core::parallel_generate(
        std::span{vec1.data(), vec1.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);
    ttml::core::parallel_generate(
        std::span{vec2.data(), vec2.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, 42);

    EXPECT_EQ(vec1, vec2);
}

TEST_F(RandomGenerationTests, UniformInitsGoodMeanAndRange) {
    constexpr size_t size = 1024 * 256 * 384;  // ~100M elements

    std::vector<float> parallel_vec(size);
    std::vector<float> sequential_vec(size);

    ttml::core::parallel_generate(
        std::span{parallel_vec.data(), parallel_vec.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        42);

    ttml::core::sequential_generate(
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
