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

TEST_F(RandomGenerationTests, ParallelGenerateRelativePerformance) {
    auto bench_it = [](size_t expt, size_t max_threads) -> float {
        size_t size = 1 << expt;
        std::vector<float> parallel_vec(size);
        std::vector<float> sequential_vec(size);

        ttml::init::UniformRange range{-1.0f, 1.0f};

        constexpr int num_runs = 10;
        std::vector<std::chrono::nanoseconds> parallel_times;
        std::vector<std::chrono::nanoseconds> sequential_times;

        parallel_times.reserve(num_runs);
        sequential_times.reserve(num_runs);

        auto uniform_factory = [&]() { return std::uniform_real_distribution<float>(range.a, range.b); };
        // Run multiple times to get median (skip first run to warm up)
        for (int run = 0; run < num_runs + 1; ++run) {
            // Time parallel initialization
            auto start = std::chrono::high_resolution_clock::now();
            ttml::core::random::parallel_generate(
                std::span{parallel_vec.data(), parallel_vec.size()}, uniform_factory, 42, max_threads);
            auto end = std::chrono::high_resolution_clock::now();

            if (run > 0) {
                parallel_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));
            }

            // Time sequential initialization
            start = std::chrono::high_resolution_clock::now();
            ttml::core::random::sequential_generate(
                std::span{sequential_vec.data(), sequential_vec.size()}, uniform_factory, 42);
            end = std::chrono::high_resolution_clock::now();

            if (run > 0) {
                sequential_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));
            }
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
        auto sequential_median = calculate_median(sequential_times);

        float speedup = (float)sequential_median.count() / (float)parallel_median.count();
        return speedup;
    };

    auto sizes = std::vector<size_t>();
    for (int i = 0; i < 20; i++) {
        sizes.push_back(i);
    }

    auto thread_limits = std::vector<size_t>();
    for (int i = 2; i <= std::thread::hardware_concurrency(); i++) {
        thread_limits.push_back(i);
    }

    auto speedup_map = std::map<size_t, std::pair<size_t, float>>();

    for (auto size : sizes) {
        for (auto thread_limit : thread_limits) {
            auto speedup = bench_it(size, thread_limit);
            if (speedup > 1.0) {
                fmt::println("first speedup observed for expt: {}", size);
                fmt::println("max_threads: {}, speedup: {}", thread_limit, speedup);
                speedup_map[size] = {thread_limit, speedup};
                break;
            }
        }
    }
}

TEST_F(RandomGenerationTests, ModelInitBench) {
    tt::tt_metal::distributed::MeshShape mesh_shape{1, 2};
    std::vector<int> device_ids{0, 1};
    ttml::autograd::ctx().open_device(mesh_shape, device_ids);
    auto benchmark_config = [](const std::string& config_path, const std::string& config_name) -> double {
        auto yaml_config = YAML::LoadFile(config_path);
        auto training_config = yaml_config["training_config"];

        // Extract transformer config
        auto transformer_config_node = training_config["transformer_config"];
        auto model_type = training_config["model_type"].as<std::string>();

        std::vector<std::chrono::nanoseconds> init_times;
        init_times.reserve(10);

        for (int i = 0; i < 10; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            if (model_type == "llama") {
                auto config = ttml::models::llama::read_config(transformer_config_node);
                auto model = ttml::models::llama::create(config);
            } else if (model_type == "gpt2") {
                auto config = ttml::models::gpt2::read_config(transformer_config_node);
                auto model = ttml::models::gpt2::create(config);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            init_times.push_back(duration);

            // Reset autograd context for next iteration
            ttml::autograd::ctx().reset_graph();
        }

        // Calculate median time
        std::sort(init_times.begin(), init_times.end());
        auto median_time = init_times[init_times.size() / 2];
        float median_time_ms = median_time.count() / 1e6F;

        std::cout << config_name << " model initialization median time: " << median_time_ms << " ms" << std::endl;
        return median_time_ms;
    };

    auto nanollama_time = benchmark_config(
        "/home/j/worktrees/parallel_init/tt-train/configs/training_shakespeare_nanollama3.yaml", "nanollama");
    auto tinyllama_time = benchmark_config(
        "/home/j/worktrees/parallel_init/tt-train/configs/training_shakespeare_tinyllama.yaml", "tinyllama");
    fmt::println("nanollama: {}ms", nanollama_time);
    fmt::println("tinyllama: {}ms", tinyllama_time);
    ttml::autograd::ctx().close_device();
}
