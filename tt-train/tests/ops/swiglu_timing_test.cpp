// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file swiglu_timing_test.cpp
 * @brief Simple wall-clock timing comparison between fused and composite SwiGLU.
 *
 * Build with: cmake --build build_Release --target swiglu_timing_test -j$(nproc)
 * Run with: ./build_Release/tt-train/tests/swiglu_timing_test
 */

#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <iostream>
#include <vector>
#include <xtensor/containers/xarray.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/swiglu_op.hpp"

namespace {

struct BenchmarkConfig {
    std::vector<uint32_t> input_shape;
    uint32_t hidden_dim;
    uint32_t warmup_iterations;
    uint32_t benchmark_iterations;
    std::string name;
};

xt::xarray<float> create_random_data(const std::vector<uint32_t>& shape, uint32_t seed) {
    const float bound = 1.0f;
    xt::xarray<float> data = xt::empty<float>(shape);
    ttml::core::parallel_generate<float>(
        data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, seed);
    return data;
}

void run_benchmark(const BenchmarkConfig& config) {
    using namespace ttml;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "\n========================================\n";
    std::cout << "Benchmark: " << config.name << "\n";
    std::cout << "Input shape: [" << config.input_shape[0] << ", " << config.input_shape[1] << ", "
              << config.input_shape[2] << ", " << config.input_shape[3] << "]\n";
    std::cout << "Hidden dim: " << config.hidden_dim << "\n";
    std::cout << "========================================\n";

    auto& device = autograd::ctx().get_device();
    auto* device_ptr = &device;

    const uint32_t input_dim = config.input_shape.back();
    const uint32_t hidden_dim = config.hidden_dim;

    auto& rng = autograd::ctx().get_generator();

    auto input_data = create_random_data(config.input_shape, rng());
    auto w1_data = create_random_data({1, 1, input_dim, hidden_dim}, rng());
    auto w2_data_normal = create_random_data({1, 1, hidden_dim, input_dim}, rng());
    xt::xarray<float> w2_data_transposed = xt::transpose(w2_data_normal, {0, 1, 3, 2});
    auto w3_data = create_random_data({1, 1, input_dim, hidden_dim}, rng());

    auto input = autograd::create_tensor(core::from_xtensor(input_data, device_ptr));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_data, device_ptr));
    auto w2_normal = autograd::create_tensor(core::from_xtensor(w2_data_normal, device_ptr));
    auto w2_transposed = autograd::create_tensor(core::from_xtensor(w2_data_transposed, device_ptr));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_data, device_ptr));

    // ========== Warmup Phase ==========
    std::cout << "Warmup (" << config.warmup_iterations << " iterations)...\n";
    for (uint32_t i = 0; i < config.warmup_iterations; ++i) {
        auto fused_result = ops::swiglu(input, w1, w2_transposed, w3);
        fused_result->get_value();

        auto xw1 = ttnn::matmul(input->get_value(), w1->get_value());
        auto silu_xw1 = ttnn::silu(xw1);
        auto xw3 = ttnn::matmul(input->get_value(), w3->get_value());
        auto gated = ttnn::multiply(silu_xw1, xw3);
        [[maybe_unused]] auto composite_result = ttnn::matmul(gated, w2_normal->get_value());
    }

    // ========== Benchmark Fused SwiGLU ==========
    std::cout << "Benchmarking fused SwiGLU (" << config.benchmark_iterations << " iterations)...\n";
    auto fused_start = Clock::now();
    for (uint32_t i = 0; i < config.benchmark_iterations; ++i) {
        auto result = ops::swiglu(input, w1, w2_transposed, w3);
        result->get_value();
    }
    auto fused_end = Clock::now();
    auto fused_us = std::chrono::duration_cast<std::chrono::microseconds>(fused_end - fused_start).count();
    double fused_per_iter = static_cast<double>(fused_us) / config.benchmark_iterations;

    // ========== Benchmark Composite SwiGLU ==========
    std::cout << "Benchmarking composite SwiGLU (" << config.benchmark_iterations << " iterations)...\n";
    auto composite_start = Clock::now();
    for (uint32_t i = 0; i < config.benchmark_iterations; ++i) {
        auto xw1 = ttnn::matmul(input->get_value(), w1->get_value());
        auto silu_xw1 = ttnn::silu(xw1);
        auto xw3 = ttnn::matmul(input->get_value(), w3->get_value());
        auto gated = ttnn::multiply(silu_xw1, xw3);
        [[maybe_unused]] auto result = ttnn::matmul(gated, w2_normal->get_value());
    }
    auto composite_end = Clock::now();
    auto composite_us = std::chrono::duration_cast<std::chrono::microseconds>(composite_end - composite_start).count();
    double composite_per_iter = static_cast<double>(composite_us) / config.benchmark_iterations;

    // ========== Print Results ==========
    std::cout << "\n--- Results ---\n";
    std::cout << "Fused SwiGLU:     " << fused_per_iter << " µs/iter\n";
    std::cout << "Composite SwiGLU: " << composite_per_iter << " µs/iter\n";
    if (fused_per_iter < composite_per_iter) {
        double speedup = composite_per_iter / fused_per_iter;
        std::cout << "Fused is " << speedup << "x FASTER ✓\n";
    } else {
        double slowdown = fused_per_iter / composite_per_iter;
        std::cout << "Fused is " << slowdown << "x SLOWER ✗\n";
    }
}

}  // namespace

int main() {
    using namespace ttml;

    std::cout << "SwiGLU Timing Benchmark\n";
    std::cout << "========================\n";

    autograd::ctx().open_device();

    std::vector<BenchmarkConfig> configs = {
        // Small case (dimensions aligned for batching)
        {
            .input_shape = {4, 1, 128, 128},
            .hidden_dim = 256,
            .warmup_iterations = 3,
            .benchmark_iterations = 20,
            .name = "Small Aligned (4x1x128x128, H=256)",
        },
        // NanoLlama-like dimensions (M fits in L1 case)
        {
            .input_shape = {32, 1, 128, 384},
            .hidden_dim = 1024,
            .warmup_iterations = 3,
            .benchmark_iterations = 10,
            .name = "NanoLlama-like (32x1x128x384, H=1024)",
        },
        // Larger batch test
        {
            .input_shape = {64, 1, 256, 384},
            .hidden_dim = 1024,
            .warmup_iterations = 2,
            .benchmark_iterations = 5,
            .name = "Large Batch (64x1x256x384, H=1024)",
        },
    };

    for (const auto& config : configs) {
        run_benchmark(config);
    }

    autograd::ctx().close_device();

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
