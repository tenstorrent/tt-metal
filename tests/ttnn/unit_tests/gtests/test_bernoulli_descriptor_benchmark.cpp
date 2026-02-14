// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark & correctness test: Original Bernoulli (ProgramFactoryConcept)
// vs BernoulliNew (ProgramDescriptorFactoryConcept).
//
// 1. Verifies both ops produce identical results on cache miss and cache hit.
// 2. Runs each operation N times and compares wall-clock dispatch time.

#include <chrono>
#include <iostream>

#include "gtest/gtest.h"
#include "ttnn_test_fixtures.hpp"

#include <tt-metalium/distributed.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/bernoulli/bernoulli.hpp"
#include "ttnn/operations/bernoulli_new/bernoulli_new.hpp"

namespace {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

// Helper: call original bernoulli with all required args
inline Tensor call_old(const Tensor& input, uint32_t seed) {
    return ttnn::bernoulli(input, seed, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

// Helper: call new bernoulli with all required args
inline Tensor call_new(const Tensor& input, uint32_t seed) {
    return ttnn::bernoulli_new(input, seed, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

class BernoulliDescriptorBenchmark : public ttnn::TTNNFixtureWithDevice {};

// ---------------------------------------------------------------
// Correctness: verify old and new produce identical output
// on cache-miss (first call with a given config).
// ---------------------------------------------------------------
TEST_F(BernoulliDescriptorBenchmark, CorrectnessNonCached) {
    constexpr uint32_t seed = 12345;
    ttnn::Shape shape{1, 1, 32, 32};
    Tensor input = ttnn::random::random(shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);

    // First invocation -- cache miss for both ops.
    // Bernoulli defaults output dtype to FLOAT32.
    Tensor out_old = call_old(input, seed).cpu();
    Tensor out_new = call_new(input, seed).cpu();

    bool match = ttnn::allclose<float>(out_old, out_new);
    ASSERT_TRUE(match) << "Cache-miss outputs differ between Bernoulli and BernoulliNew";
}

// ---------------------------------------------------------------
// Correctness: verify old and new produce identical output
// on cache-hit (second call, program cache is warm).
// ---------------------------------------------------------------
TEST_F(BernoulliDescriptorBenchmark, CorrectnessCached) {
    constexpr uint32_t seed = 12345;
    ttnn::Shape shape{1, 1, 32, 32};
    Tensor input = ttnn::random::random(shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);

    // Warm up -- populate caches for both ops.
    {
        auto _ = call_old(input, seed).cpu();
    }
    {
        auto _ = call_new(input, seed).cpu();
    }

    // Second invocation -- cache hit for both ops.
    constexpr uint32_t seed2 = 99999;
    Tensor out_old = call_old(input, seed2).cpu();
    Tensor out_new = call_new(input, seed2).cpu();

    bool match = ttnn::allclose<float>(out_old, out_new);
    ASSERT_TRUE(match) << "Cache-hit outputs differ between Bernoulli and BernoulliNew";
}

// ---------------------------------------------------------------
// Performance: dispatch N ops, compare wall-clock time.
// ---------------------------------------------------------------
TEST_F(BernoulliDescriptorBenchmark, DispatchPerformance) {
    constexpr uint32_t N = 1000000;
    constexpr uint32_t seed = 42;

    ttnn::Shape shape{1, 1, 32, 32};
    Tensor input = ttnn::random::random(shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);

    // Warm up: trigger compilation + caching for both.
    {
        auto a = call_old(input, seed).cpu();
        auto b = call_new(input, seed).cpu();
    }

    // ---- Benchmark original Bernoulli ----
    auto t0 = Clock::now();
    for (uint32_t i = 0; i < N; ++i) {
        auto out = call_old(input, seed + i);
        (void)out;
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});
    auto t1 = Clock::now();
    Duration old_time = t1 - t0;

    // ---- Benchmark BernoulliNew ----
    auto t2 = Clock::now();
    for (uint32_t i = 0; i < N; ++i) {
        auto out = call_new(input, seed + i);
        (void)out;
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});
    auto t3 = Clock::now();
    Duration new_time = t3 - t2;

    double old_per_op = old_time.count() / N;
    double new_per_op = new_time.count() / N;
    double overhead_pct = ((new_per_op - old_per_op) / old_per_op) * 100.0;

    std::cout << "\n=== Bernoulli ProgramDescriptor Benchmark ===\n";
    std::cout << "  Iterations:          " << N << "\n";
    std::cout << "  Original Bernoulli:  " << old_time.count() << " ms total, " << old_per_op << " ms/op\n";
    std::cout << "  BernoulliNew (desc): " << new_time.count() << " ms total, " << new_per_op << " ms/op\n";
    std::cout << "  Overhead:            " << overhead_pct << " %\n";
    std::cout << "=============================================\n\n";

    // Fail if BernoulliNew is more than 3% slower than original.
    EXPECT_LT(new_per_op, old_per_op * 1.03) << "BernoulliNew is more than 3% slower than original Bernoulli.";
}

}  // namespace
