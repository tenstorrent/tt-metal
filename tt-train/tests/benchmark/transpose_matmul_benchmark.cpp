// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Measures transpose-flag overhead in variable_matmul.
// B is square (K == N) so A@B and A@B^T have the same output shape; the
// only difference between the runs is the transpose flag.

#include <benchmark/benchmark.h>

#include <chrono>
#include <tt-metalium/device.hpp>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "metal/operations.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

using namespace tt::tt_metal;
using ttml::metal::VariableMatmulConfig;

// HiFi4 + FP32 dest: 2 cycles per tile matmul
// 120 cores * 1350 MHz * (1/2 tile/cycle) * 2048 FLOPs/tile = 165.9 TFLOPS
constexpr double kPeakTflops = 165.9;

const VariableMatmulConfig kBaseConfig{
    .M_block_size = 4,
    .K_block_size = 8,
    .N_block_size = 8,
    .subblock_h = 2,
    .subblock_w = 2,
    .compute_with_storage_grid_size = {10, 10},
};

constexpr int kWarmupIterations = 3;
constexpr int kMeasureIterations = 10;

ttnn::Tensor create_random_tensor(
    const ttnn::Shape& shape, ttnn::DataType dtype, uint32_t seed, ttnn::device::MeshDevice* device) {
    std::vector<float> data(shape.volume());
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-0.1f, 0.1f); }, seed);
    return ttnn::Tensor::from_vector(
        data, ttnn::TensorSpec(shape, TensorLayout(dtype, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)), device);
}

void report_counters(benchmark::State& state, double time_us, uint32_t M, uint32_t K, uint32_t N) {
    const double flops = 2.0 * M * K * N;
    const double tflops = flops / (time_us * 1e6);
    state.counters["time_us"] = time_us;
    state.counters["tflops"] = tflops;
    state.counters["util_pct"] = (tflops / kPeakTflops) * 100.0;
}

// Run a single benchmark loop with the given transpose flags. Matmul dimensions are always
// M x K x N where N == K (square B), regardless of transpose configuration. The stored
// shape of A and B is adjusted so that the matmul is logically the same operation under
// each configuration.
void run_matmul_bench(benchmark::State& state, bool transpose_a, bool transpose_b) {
    auto device = ttnn::device::open_mesh_device(0);
    device->enable_program_cache();

    const uint32_t M = static_cast<uint32_t>(state.range(0));
    const uint32_t K = static_cast<uint32_t>(state.range(1));
    const uint32_t N = K;  // B is square

    // For A^T@B: A is stored as [K, M] but interpreted as [M, K].
    auto input_shape = transpose_a ? ttnn::Shape({1, 1, K, M}) : ttnn::Shape({1, 1, M, K});
    auto input = create_random_tensor(input_shape, DataType::BFLOAT16, 42, device.get());
    // For A@B^T: B is stored as [N, K] but interpreted as [K, N].
    auto weight_shape = transpose_b ? ttnn::Shape({1, 1, N, K}) : ttnn::Shape({1, 1, K, N});
    auto weight = create_random_tensor(weight_shape, DataType::BFLOAT16, 43, device.get());

    const auto& cfg = kBaseConfig;

    for (int i = 0; i < kWarmupIterations; ++i) {
        auto out = ttml::metal::variable_matmul(input, weight, cfg, transpose_a, transpose_b);
        distributed::Synchronize(device.get(), std::nullopt);
        out.deallocate();
    }

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttml::metal::variable_matmul(input, weight, cfg, transpose_a, transpose_b);
        distributed::Synchronize(device.get(), std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        double time_us = std::chrono::duration<double, std::micro>(end - start).count();
        state.SetIterationTime(time_us / 1e6);
        report_counters(state, time_us, M, K, N);
        out.deallocate();
    }

    input.deallocate();
    weight.deallocate();
    device->close();
}

}  // namespace

// ============================================================================
// A @ B (no transpose) — baseline
// ============================================================================
static void BM_AB(benchmark::State& state) {
    run_matmul_bench(state, /*transpose_a=*/false, /*transpose_b=*/false);
}

// ============================================================================
// A @ B^T (transpose_b=true)
// ============================================================================
static void BM_ABT(benchmark::State& state) {
    run_matmul_bench(state, /*transpose_a=*/false, /*transpose_b=*/true);
}

// ============================================================================
// A^T @ B (transpose_a=true) — exercises the compute-side transpose_wh_tile pass
// ============================================================================
static void BM_ATB(benchmark::State& state) {
    run_matmul_bench(state, /*transpose_a=*/true, /*transpose_b=*/false);
}

// Shapes: {M, K} with N == K (square B). Cover a range so we can see how the
// overhead scales with shape size.
#define REGISTER_SHAPES(name)            \
    BENCHMARK(name)                      \
        ->Args({128, 512})               \
        ->Args({512, 512})               \
        ->Args({128, 1024})              \
        ->Args({1024, 1024})             \
        ->Args({128, 2048})              \
        ->Args({1024, 2048})             \
        ->Args({128, 4096})              \
        ->Args({1024, 4096})             \
        ->Args({4096, 4096})             \
        ->ArgNames({"M", "K=N"})         \
        ->UseManualTime()                \
        ->Iterations(kMeasureIterations) \
        ->Unit(benchmark::kMicrosecond);

REGISTER_SHAPES(BM_AB)
REGISTER_SHAPES(BM_ABT)
REGISTER_SHAPES(BM_ATB)

BENCHMARK_MAIN();
