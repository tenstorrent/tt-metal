// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file swiglu_profiler_benchmark.cpp
 * @brief Profiling benchmark for SwiGLU forward operation using device profiler markers.
 *
 * This benchmark uses ProfilerNoopOperation markers to measure DEVICE KERNEL DURATION
 * for fair comparison between fused and composite SwiGLU operations.
 *
 * Usage:
 *   # Build with Tracy enabled:
 *   ./build_metal.sh --build-tt-train --enable-tracy
 *
 *   # Run with profiling:
 *   python3 -m tracy -r -v -p -m './build_Release/tt-train/tests/swiglu_profiler_benchmark'
 *
 *   # Export CSV from Tracy and analyze with:
 *   python3 analyze_tracy_output.py profile.csv --mode markers \
 *       --begin-marker fused_swiglu_begin --end-marker fused_swiglu_end
 *
 *   python3 analyze_tracy_output.py profile.csv --mode markers \
 *       --begin-marker composite_swiglu_begin --end-marker composite_swiglu_end
 *
 *   # Or analyze fused operation directly:
 *   python3 analyze_tracy_output.py profile.csv --mode operations \
 *       --op-name "ttml_swiglu_fw"
 */

#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <iostream>
#include <vector>
#include <xtensor/containers/xarray.hpp>

// Tracy profiler - for GUI visualization
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_profiler.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/swiglu_op.hpp"

namespace {

// Number of profiler markers to emit (helps with grouping in analysis)
constexpr size_t NUM_MARKER_NOOPS = 10;

/**
 * Configuration for benchmark runs
 */
struct BenchmarkConfig {
    std::vector<uint32_t> input_shape;
    uint32_t hidden_dim;
    uint32_t warmup_iterations;
    uint32_t benchmark_iterations;
    std::string name;
};

/**
 * Create random tensor data using the project's standard approach
 */
xt::xarray<float> create_random_data(const std::vector<uint32_t>& shape, uint32_t seed) {
    const float bound = 1.0f;
    xt::xarray<float> data = xt::empty<float>(shape);
    ttml::core::parallel_generate<float>(
        data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, seed);
    return data;
}

/**
 * Emit profiler markers using ProfilerNoopOperation
 */
void emit_marker(ttnn::distributed::MeshDevice* device, const std::string& marker_name) {
    auto fake_tensor = ttml::core::from_vector({1.F}, ttnn::Shape({1, 1, 1, 1}), device, ttnn::Layout::ROW_MAJOR);
    for (size_t i = 0; i < NUM_MARKER_NOOPS; ++i) {
        [[maybe_unused]] auto _ = ttml::metal::profiler_no_op(fake_tensor, marker_name);
    }
}

/**
 * Run fused SwiGLU kernel benchmark with profiler markers
 */
void benchmark_fused_swiglu(
    ttnn::distributed::MeshDevice* device,
    const ttml::autograd::TensorPtr& input,
    const ttml::autograd::TensorPtr& w1,
    const ttml::autograd::TensorPtr& w2,
    const ttml::autograd::TensorPtr& w3,
    uint32_t iterations) {
    using namespace ttml;

    for (uint32_t i = 0; i < iterations; ++i) {
#ifdef TRACY_ENABLE
        ZoneScopedN("fused_swiglu");
        ZoneColor(0x00FF00);  // Green
#endif
        // Emit begin marker
        emit_marker(device, "fused_swiglu_begin");

        // Fused SwiGLU operation
        auto result = ops::swiglu(input, w1, w2, w3);

        // Force completion (get_value ensures the tensor is computed)
        result->get_value();

        // Emit end marker
        emit_marker(device, "fused_swiglu_end");
    }
}

/**
 * Run composite SwiGLU operation (mimics llama_block.cpp implementation)
 * This is the baseline: silu(x @ w1) * (x @ w3) @ w2
 */
void benchmark_composite_swiglu(
    ttnn::distributed::MeshDevice* device,
    const ttml::autograd::TensorPtr& input,
    const ttml::autograd::TensorPtr& w1,
    const ttml::autograd::TensorPtr& w2,
    const ttml::autograd::TensorPtr& w3,
    uint32_t iterations) {
    using namespace ttml;

    for (uint32_t i = 0; i < iterations; ++i) {
#ifdef TRACY_ENABLE
        ZoneScopedN("composite_swiglu");
        ZoneColor(0xFF0000);  // Red
#endif
        // Emit begin marker
        emit_marker(device, "composite_swiglu_begin");

        // Composite SwiGLU: silu(x @ w1) * (x @ w3) @ w2
        // x @ w1
        auto xw1 = ttnn::matmul(input->get_value(), w1->get_value());
        // silu(x @ w1)
        auto silu_xw1 = ttnn::silu(xw1);
        // x @ w3
        auto xw3 = ttnn::matmul(input->get_value(), w3->get_value());
        // silu(x @ w1) * (x @ w3)
        auto gated = ttnn::multiply(silu_xw1, xw3);
        // result @ w2
        [[maybe_unused]] auto result = ttnn::matmul(gated, w2->get_value());

        // Emit end marker
        emit_marker(device, "composite_swiglu_end");
    }
}

/**
 * Run a single benchmark configuration
 */
void run_benchmark(const BenchmarkConfig& config) {
    using namespace ttml;

    std::cout << "\n========================================\n";
    std::cout << "Benchmark: " << config.name << "\n";
    std::cout << "Input shape: [" << config.input_shape[0] << ", " << config.input_shape[1] << ", "
              << config.input_shape[2] << ", " << config.input_shape[3] << "]\n";
    std::cout << "Hidden dim: " << config.hidden_dim << "\n";
    std::cout << "Warmup iterations: " << config.warmup_iterations << "\n";
    std::cout << "Benchmark iterations: " << config.benchmark_iterations << "\n";
    std::cout << "========================================\n";

    auto& device = autograd::ctx().get_device();
    auto* device_ptr = &device;

    // Create test tensors
    const uint32_t input_dim = config.input_shape.back();
    const uint32_t hidden_dim = config.hidden_dim;

    auto& rng = autograd::ctx().get_generator();

    auto input_data = create_random_data(config.input_shape, rng());
    auto w1_data = create_random_data({1, 1, input_dim, hidden_dim}, rng());
    // W2 data: create as [H, C] then transpose for fused kernel
    // Note: Fused kernel expects W2 transposed [C, H] for efficient row reads
    auto w2_data_normal = create_random_data({1, 1, hidden_dim, input_dim}, rng());
    // Transpose W2 for fused kernel: [1, 1, H, C] -> [1, 1, C, H]
    // Use xt::eval to create a concrete array from the view
    xt::xarray<float> w2_data_transposed = xt::transpose(w2_data_normal, {0, 1, 3, 2});
    auto w3_data = create_random_data({1, 1, input_dim, hidden_dim}, rng());

    auto input = autograd::create_tensor(core::from_xtensor(input_data, device_ptr));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_data, device_ptr));
    // W2 normal layout for composite benchmark (uses matmul with no transpose)
    auto w2_normal = autograd::create_tensor(core::from_xtensor(w2_data_normal, device_ptr));
    // W2 transposed layout for fused benchmark (enables efficient row reads)
    auto w2_transposed = autograd::create_tensor(core::from_xtensor(w2_data_transposed, device_ptr));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_data, device_ptr));

    // ========== Warmup Phase ==========
    std::cout << "Running warmup (fused)...\n";
    benchmark_fused_swiglu(device_ptr, input, w1, w2_transposed, w3, config.warmup_iterations);

    std::cout << "Running warmup (composite)...\n";
    benchmark_composite_swiglu(device_ptr, input, w1, w2_normal, w3, config.warmup_iterations);

    // ========== Benchmark Phase ==========
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Benchmarking fused SwiGLU...\n";
    auto fused_start = Clock::now();
    benchmark_fused_swiglu(device_ptr, input, w1, w2_transposed, w3, config.benchmark_iterations);
    auto fused_end = Clock::now();
    auto fused_us = std::chrono::duration_cast<std::chrono::microseconds>(fused_end - fused_start).count();

    std::cout << "Benchmarking composite SwiGLU...\n";
    auto composite_start = Clock::now();
    benchmark_composite_swiglu(device_ptr, input, w1, w2_normal, w3, config.benchmark_iterations);
    auto composite_end = Clock::now();
    auto composite_us = std::chrono::duration_cast<std::chrono::microseconds>(composite_end - composite_start).count();

    // Print timing results
    double fused_per_iter = static_cast<double>(fused_us) / config.benchmark_iterations;
    double composite_per_iter = static_cast<double>(composite_us) / config.benchmark_iterations;

    std::cout << "\n--- Wall-Clock Timing Results ---\n";
    std::cout << "Fused SwiGLU:     " << fused_per_iter << " µs/iter\n";
    std::cout << "Composite SwiGLU: " << composite_per_iter << " µs/iter\n";
    if (fused_per_iter < composite_per_iter) {
        double speedup = composite_per_iter / fused_per_iter;
        std::cout << "Fused is " << speedup << "x FASTER ✓\n";
    } else {
        double slowdown = fused_per_iter / composite_per_iter;
        std::cout << "Fused is " << slowdown << "x SLOWER ✗\n";
    }

    std::cout << "Benchmark iteration complete.\n";
}

}  // namespace

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    using namespace ttml;

    std::cout << "SwiGLU Profiler Benchmark (Device Kernel Duration)\n";
    std::cout << "===================================================\n";
    std::cout << "This benchmark uses ProfilerNoopOperation markers.\n";
    std::cout << "Analyze with: python3 analyze_tracy_output.py <csv> --mode markers\n\n";

    // Open device
    autograd::ctx().open_device();

    // Define benchmark configurations
    // Use NanoLlama-like dimensions for production-relevant comparison
    std::vector<BenchmarkConfig> configs = {
        // NanoLlama-like dimensions (M fits in L1 case)
        {
            .input_shape = {32, 1, 128, 384},
            .hidden_dim = 1024,
            .warmup_iterations = 2,
            .benchmark_iterations = 10,
            .name = "NanoLlama-like (32x1x128x384, H=1024)",
        },
    };

    // Run benchmarks
    for (const auto& config : configs) {
        run_benchmark(config);
    }

    // Close device
    autograd::ctx().close_device();

    std::cout << "\nBenchmark complete.\n";
    std::cout << "\nTo analyze results:\n";
    std::cout << "1. Export CSV from Tracy\n";
    std::cout << "2. Run: python3 analyze_tracy_output.py <csv> --mode markers \\\n";
    std::cout << "        --begin-marker fused_swiglu_begin --end-marker fused_swiglu_end\n";
    std::cout << "3. Run: python3 analyze_tracy_output.py <csv> --mode markers \\\n";
    std::cout << "        --begin-marker composite_swiglu_begin --end-marker composite_swiglu_end\n";

    return 0;
}
