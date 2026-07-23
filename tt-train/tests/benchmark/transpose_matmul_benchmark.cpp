// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

using namespace tt::tt_metal;
using ttml::metal::VariableMatmulConfig;

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
        data, tt::tt_metal::TensorSpec(shape, TensorLayout(dtype, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)), device);
}

void report_counters(benchmark::State& state, double time_us, uint32_t M, uint32_t K, uint32_t N) {
    const double flops = 2.0 * M * K * N;
    const double tflops = flops / (time_us * 1e6);
    state.counters["time_us"] = time_us;
    state.counters["tflops"] = tflops;
}

// Run one benchmark loop. Dimensions are always M x K x N with N == K (square B), so the matmul
// is the same logical op under each transpose config — only the stored shapes of A/B change.
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

    // EP-only API: build a trivial offsets tensor [0, M] and a pre-allocated output, then call
    // through InputAndOutputRow with the full M range. Equivalent to a non-EP matmul.
    const std::vector<uint32_t> offsets_host = {0U, M};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device.get(), ttnn::Layout::ROW_MAJOR);
    auto output = create_random_tensor(ttnn::Shape({1, 1, M, N}), DataType::BFLOAT16, 44, device.get());
    const uint32_t M_tiles = M / 32U;

    auto run_once = [&]() {
        ttml::metal::variable_matmul_into_rows(
            /*input_tensor=*/input,
            /*weight_tensor=*/weight,
            /*config=*/cfg,
            /*offsets_tensor=*/offsets,
            /*output_tensor=*/output,
            /*offsets_start_index=*/0,
            /*expected_M_tiles=*/M_tiles,
            /*transpose_a=*/transpose_a,
            /*transpose_b=*/transpose_b);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
        distributed::Synchronize(device.get(), std::nullopt);
    }

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        run_once();
        distributed::Synchronize(device.get(), std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        double time_us = std::chrono::duration<double, std::micro>(end - start).count();
        state.SetIterationTime(time_us / 1e6);
        report_counters(state, time_us, M, K, N);
    }
    output.deallocate();
    offsets.deallocate();

    input.deallocate();
    weight.deallocate();
    device->close();
}

}  // namespace

static void BM_AB(benchmark::State& state) {
    run_matmul_bench(state, /*transpose_a=*/false, /*transpose_b=*/false);
}

static void BM_ABT(benchmark::State& state) {
    run_matmul_bench(state, /*transpose_a=*/false, /*transpose_b=*/true);
}

// transpose_a adds a compute-side transpose_wh_tile pass — extra work beyond the stored-shape
// change that transpose_b alone makes (see run_matmul_bench).
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
