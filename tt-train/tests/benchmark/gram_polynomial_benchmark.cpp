// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <llrt/tt_cluster.hpp>
#include <random>
#include <span>
#include <string>
#include <tracy/Tracy.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "impl/context/metal_context.hpp"
#include "metal/operations.hpp"
#include "metal/ops/gram_polynomial/device/gram_polynomial_device_operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

struct BenchShape {
    uint32_t M;
    uint32_t K;
    std::string name;
};

struct BenchResult {
    std::string impl_name;
    double time_us = std::numeric_limits<double>::quiet_NaN();
    double tflops = 0.0;
    double utilization_pct = 0.0;
};

struct TestConfig {
    int num_warmup_iterations = 3;
    int num_measurement_iterations = 100;
};

const std::vector<BenchShape> shapes = {
    {2048, 5632, "2048x5632"},
};

const TestConfig test_config = {
    .num_warmup_iterations = 3,
    .num_measurement_iterations = 100,
};

namespace {

ttnn::Tensor make_random_tensor(
    const ttnn::Shape& shape, const std::shared_ptr<ttnn::device::MeshDevice>& device, uint32_t seed) {
    std::vector<float> data(shape.volume());
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device.get());
}

// Benchmark gram_matmul only (Phase 1: G = X @ X^T)
BenchResult bench_gram_matmul(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& x,
    int device_id) {
    auto* dev_ptr = device.get();

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto out = ttml::metal::gram_matmul(x);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        out.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttml::metal::gram_matmul(x);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        out.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    double flops = 2.0 * M * M * K;
    double tflops = flops / 1e12 / avg_s;

    auto compute_grid_size = device->compute_with_storage_grid_size();
    constexpr int HiFi4_cycles_per_tile = 64;
    int num_cores = compute_grid_size.x * compute_grid_size.y;
    double num_tile_ops = static_cast<double>(M) * M * K / (32.0 * 32.0 * 32.0);
    double ideal_cycles = num_tile_ops * HiFi4_cycles_per_tile / num_cores;
    int freq_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_id);
    double actual_cycles = avg_s * freq_mhz * 1e6;
    double utilization = ideal_cycles / actual_cycles * 100.0;

    return BenchResult{
        .impl_name = "gram_matmul",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark Phase 2 only (H = cG^2 + bG, given pre-computed G)
BenchResult bench_phase2(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& G,
    float b,
    float c,
    int device_id) {
    auto* dev_ptr = device.get();

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto out =
            ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        out.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out =
            ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        out.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    // Phase 2 FLOPs: G^2 matmul (2*M*M*M) + epilogue (3*M*M for mul+mul+add)
    uint64_t M = shape.M;
    double flops = 2.0 * M * M * M + 3.0 * M * M;
    double tflops = flops / 1e12 / avg_s;

    auto compute_grid_size = device->compute_with_storage_grid_size();
    constexpr int HiFi4_cycles_per_tile = 64;
    int num_cores = compute_grid_size.x * compute_grid_size.y;
    double num_tile_ops = static_cast<double>(M) * M * M / (32.0 * 32.0 * 32.0);
    double ideal_cycles = num_tile_ops * HiFi4_cycles_per_tile / num_cores;
    int freq_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_id);
    double actual_cycles = avg_s * freq_mhz * 1e6;
    double utilization = ideal_cycles / actual_cycles * 100.0;

    return BenchResult{
        .impl_name = "phase2 (cG²+bG)",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark G^2 = G @ G using minimal_matmul (matmul-only, no epilogue)
BenchResult bench_gsq_minimal_matmul(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& G,
    int device_id) {
    auto* dev_ptr = device.get();

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto out = ttnn::experimental::minimal_matmul(
            G, G, /*bias_tensor=*/std::nullopt, /*fused_activation=*/std::nullopt, /*config=*/std::nullopt);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        out.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttnn::experimental::minimal_matmul(
            G, G, /*bias_tensor=*/std::nullopt, /*fused_activation=*/std::nullopt, /*config=*/std::nullopt);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        out.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    double flops = 2.0 * M * M * M;
    double tflops = flops / 1e12 / avg_s;

    auto compute_grid_size = device->compute_with_storage_grid_size();
    constexpr int HiFi4_cycles_per_tile = 64;
    int num_cores = compute_grid_size.x * compute_grid_size.y;
    double num_tile_ops = static_cast<double>(M) * M * M / (32.0 * 32.0 * 32.0);
    double ideal_cycles = num_tile_ops * HiFi4_cycles_per_tile / num_cores;
    int freq_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_id);
    double actual_cycles = avg_s * freq_mhz * 1e6;
    double utilization = ideal_cycles / actual_cycles * 100.0;

    return BenchResult{
        .impl_name = "G² minimal_mm",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark full gram_polynomial (Phase 1 + Phase 2)
BenchResult bench_gram_polynomial(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& x,
    float b,
    float c,
    int device_id) {
    auto* dev_ptr = device.get();

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto out = ttml::metal::gram_polynomial(x, b, c);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        out.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttml::metal::gram_polynomial(x, b, c);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        out.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    // Total FLOPs: Phase 1 (2*M*M*K) + Phase 2 (2*M*M*M + 3*M*M)
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    double flops = 2.0 * M * M * K + 2.0 * M * M * M + 3.0 * M * M;
    double tflops = flops / 1e12 / avg_s;

    return BenchResult{
        .impl_name = "gram_polynomial",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = 0.0,  // combined, not meaningful
    };
}

// Benchmark reference: separate ttnn ops (matmul + matmul + mul + mul + add)
BenchResult bench_reference_separate(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& x,
    float b,
    float c,
    int device_id) {
    auto* dev_ptr = device.get();
    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();
    auto compute_grid_size = device->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(compute_grid_size.x, compute_grid_size.y);

    auto xt_tensor = ttnn::transpose(x, -2, -1);

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto G = ttnn::matmul(
            x,
            xt_tensor,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto G2 = ttnn::matmul(
            G,
            G,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto cG2 = ttnn::multiply(G2, c);
        auto bG = ttnn::multiply(G, b);
        auto H = ttnn::add(cG2, bG);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        G.deallocate();
        G2.deallocate();
        cG2.deallocate();
        bG.deallocate();
        H.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto G = ttnn::matmul(
            x,
            xt_tensor,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto G2 = ttnn::matmul(
            G,
            G,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto cG2 = ttnn::multiply(G2, c);
        auto bG = ttnn::multiply(G, b);
        auto H = ttnn::add(cG2, bG);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        G.deallocate();
        G2.deallocate();
        cG2.deallocate();
        bG.deallocate();
        H.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    double flops = 2.0 * M * M * K + 2.0 * M * M * M + 3.0 * M * M;
    double tflops = flops / 1e12 / avg_s;

    return BenchResult{
        .impl_name = "ttnn separate",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = 0.0,
    };
}

// Benchmark reference: separate minimal_matmul ops (minimal_matmul + minimal_matmul + mul + mul + add)
BenchResult bench_reference_minimal(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& x,
    float b,
    float c,
    int device_id) {
    auto* dev_ptr = device.get();

    auto xt_tensor = ttnn::transpose(x, -2, -1);

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto G = ttnn::experimental::minimal_matmul(
            x, xt_tensor, /*bias_tensor=*/std::nullopt, /*fused_activation=*/std::nullopt, /*config=*/std::nullopt);
        auto G2 = ttnn::experimental::minimal_matmul(
            G, G, /*bias_tensor=*/std::nullopt, /*fused_activation=*/std::nullopt, /*config=*/std::nullopt);
        auto cG2 = ttnn::multiply(G2, c);
        auto bG = ttnn::multiply(G, b);
        auto H = ttnn::add(cG2, bG);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        G.deallocate();
        G2.deallocate();
        cG2.deallocate();
        bG.deallocate();
        H.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto G = ttnn::experimental::minimal_matmul(
            x, xt_tensor, /*bias_tensor=*/std::nullopt, /*fused_activation=*/std::nullopt, /*config=*/std::nullopt);
        auto G2 = ttnn::experimental::minimal_matmul(
            G, G, /*bias_tensor=*/std::nullopt, /*fused_activation=*/std::nullopt, /*config=*/std::nullopt);
        auto cG2 = ttnn::multiply(G2, c);
        auto bG = ttnn::multiply(G, b);
        auto H = ttnn::add(cG2, bG);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        G.deallocate();
        G2.deallocate();
        cG2.deallocate();
        bG.deallocate();
        H.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    double flops = 2.0 * M * M * K + 2.0 * M * M * M + 3.0 * M * M;
    double tflops = flops / 1e12 / avg_s;

    return BenchResult{
        .impl_name = "minimal separate",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = 0.0,
    };
}

void PrintComparisonTable(const std::string& shape_name, const std::vector<BenchResult>& results) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Shape: " << std::left << std::setw(63) << shape_name << " ║\n";
    std::cout << "╠══════════════════════╦═══════════════╦═══════════════╦═══════════════╣\n";
    std::cout << "║ Implementation       ║   Time (us)   ║    TFLOPs     ║  Utilization  ║\n";
    std::cout << "╠══════════════════════╬═══════════════╬═══════════════╬═══════════════╣\n";

    for (const auto& result : results) {
        std::cout << "║ " << std::setw(20) << std::left << result.impl_name << " ║ " << std::setw(13) << std::right
                  << std::fixed << std::setprecision(1) << result.time_us << " ║ " << std::setw(13) << std::fixed
                  << std::setprecision(2) << result.tflops << " ║ " << std::setw(12) << std::fixed
                  << std::setprecision(2) << result.utilization_pct << "% ║\n";
    }

    std::cout << "╚══════════════════════╩═══════════════╩═══════════════╩═══════════════╝\n";
}

void BM_GramPolynomial(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const BenchShape& shape = shapes[shape_index];

    const int device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id, /*l1_small_size=*/200000, /*trace_region_size=*/1048576);
    device->enable_program_cache();

    const float b = 0.5F;
    const float c = 0.3F;

    for (auto _ : state) {
        const uint32_t seed = 42;

        ttnn::Shape x_shape({1, 1, shape.M, shape.K});
        auto x = make_random_tensor(x_shape, device, seed);

        // Pre-compute G for Phase 2 isolated benchmark
        auto G = ttml::metal::gram_matmul(x);
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);

        std::vector<BenchResult> results;
        results.push_back(bench_gram_matmul(shape, device, x, device_id));
        results.push_back(bench_phase2(shape, device, G, b, c, device_id));
        results.push_back(bench_gsq_minimal_matmul(shape, device, G, device_id));
        results.push_back(bench_gram_polynomial(shape, device, x, b, c, device_id));
        results.push_back(bench_reference_separate(shape, device, x, b, c, device_id));
        results.push_back(bench_reference_minimal(shape, device, x, b, c, device_id));

        PrintComparisonTable(shape.name, results);

        const auto& poly_result = results[3];
        state.SetIterationTime(poly_result.time_us / 1e6);
        state.counters["phase1_us"] = results[0].time_us;
        state.counters["phase2_us"] = results[1].time_us;
        state.counters["gsq_minimal_us"] = results[2].time_us;
        state.counters["total_us"] = results[3].time_us;
        state.counters["ref_ttnn_us"] = results[4].time_us;
        state.counters["ref_minimal_us"] = results[5].time_us;

        x.deallocate();
        G.deallocate();
    }

    device->close();
}

}  // namespace

BENCHMARK(BM_GramPolynomial)
    ->DenseRange(0, static_cast<int>(shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("GramPolynomial");

BENCHMARK_MAIN();
