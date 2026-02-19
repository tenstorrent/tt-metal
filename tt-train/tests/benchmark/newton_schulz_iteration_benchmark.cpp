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
    {4096, 11008, "4096x11008"},
};

const TestConfig test_config = {
    .num_warmup_iterations = 3,
    .num_measurement_iterations = 100,
};

namespace {

double compute_utilization(
    double avg_s, double matmul_flops, const std::shared_ptr<ttnn::device::MeshDevice>& device, int device_id) {
    auto compute_grid_size = device->compute_with_storage_grid_size();
    constexpr int HiFi4_cycles_per_tile = 64;
    int num_cores = compute_grid_size.x * compute_grid_size.y;
    double num_tile_ops = matmul_flops / (2.0 * 32.0 * 32.0 * 32.0);
    double ideal_cycles = num_tile_ops * HiFi4_cycles_per_tile / num_cores;
    int freq_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_id);
    double actual_cycles = avg_s * freq_mhz * 1e6;
    return ideal_cycles / actual_cycles * 100.0;
}

// Total matmul FLOPs for the full Muon preconditioner:
//   Phase 1: 2*M*M*K (G = X @ X^T)
//   Phase 2: 2*M*M*M (G^2 in cG^2+bG)
//   Phase 3: 2*M*M*K (H @ X in HX+aX)
double total_matmul_flops(uint64_t M, uint64_t K) {
    return 2.0 * M * M * K + 2.0 * M * M * M + 2.0 * M * M * K;
}

// Total FLOPs including epilogue ops
double total_flops(uint64_t M, uint64_t K) {
    // Phase 1: 2*M*M*K
    // Phase 2: 2*M*M*M + 3*M*M (mul+mul+add per element for cG^2+bG)
    // Phase 3: 2*M*M*K + 2*M*K (mul+add per element for aX+HX)
    return 2.0 * M * M * K + 2.0 * M * M * M + 3.0 * M * M + 2.0 * M * M * K + 2.0 * M * K;
}

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

// Benchmark fused newton_schulz_iteration (Phase 1 + Phase 2 + Phase 3)
BenchResult bench_newton_schulz_iteration(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& x,
    float a,
    float b,
    float c,
    int device_id) {
    auto* dev_ptr = device.get();

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto out = ttml::metal::newton_schulz_iteration(x, a, b, c);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        out.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttml::metal::newton_schulz_iteration(x, a, b, c);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        out.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    double flops = total_flops(M, K);
    double tflops = flops / 1e12 / avg_s;
    double utilization = compute_utilization(avg_s, total_matmul_flops(M, K), device, device_id);

    return BenchResult{
        .impl_name = "newton_schulz_iteration",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark Phase 3 only (X' = HX + aX, given pre-computed H and X)
BenchResult bench_phase3(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& H,
    const ttnn::Tensor& x,
    float a,
    int device_id) {
    auto* dev_ptr = device.get();

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto out = ttnn::prim::ttml_hx_plus_ax(H, x, a, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        out.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttnn::prim::ttml_hx_plus_ax(H, x, a, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        out.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    // Phase 3 FLOPs: H@X matmul (2*M*M*K) + epilogue (2*M*K for mul+add)
    double flops = 2.0 * M * M * K + 2.0 * M * K;
    double tflops = flops / 1e12 / avg_s;
    double matmul_flops = 2.0 * M * M * K;
    double utilization = compute_utilization(avg_s, matmul_flops, device, device_id);

    return BenchResult{
        .impl_name = "phase3 (HX+aX)",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark Phase 1 only (G = X @ X^T)
BenchResult bench_phase1(
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
    double utilization = compute_utilization(avg_s, flops, device, device_id);

    return BenchResult{
        .impl_name = "phase1 (G=XX^T)",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark Phase 2 only (H = cG^2 + bG)
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
    uint64_t M = shape.M;
    double flops = 2.0 * M * M * M + 3.0 * M * M;
    double tflops = flops / 1e12 / avg_s;
    double matmul_flops = 2.0 * M * M * M;
    double utilization = compute_utilization(avg_s, matmul_flops, device, device_id);

    return BenchResult{
        .impl_name = "phase2 (cG²+bG)",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark composite reference: separate ttnn ops for the full pipeline
// G = matmul(X, X^T), G2 = matmul(G, G), H = c*G2 + b*G, HX = matmul(H, X), X' = HX + a*X
BenchResult bench_composite_ttnn(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& x,
    float a,
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
        auto HX = ttnn::matmul(
            H,
            x,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto aX = ttnn::multiply(x, a);
        auto X_prime = ttnn::add(HX, aX);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        G.deallocate();
        G2.deallocate();
        cG2.deallocate();
        bG.deallocate();
        H.deallocate();
        HX.deallocate();
        aX.deallocate();
        X_prime.deallocate();
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
        auto HX = ttnn::matmul(
            H,
            x,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto aX = ttnn::multiply(x, a);
        auto X_prime = ttnn::add(HX, aX);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        G.deallocate();
        G2.deallocate();
        cG2.deallocate();
        bG.deallocate();
        H.deallocate();
        HX.deallocate();
        aX.deallocate();
        X_prime.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    double flops = total_flops(M, K);
    double tflops = flops / 1e12 / avg_s;
    double utilization = compute_utilization(avg_s, total_matmul_flops(M, K), device, device_id);

    return BenchResult{
        .impl_name = "ttnn composite",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

// Benchmark hybrid: use fused Phase 1+2 (gram_polynomial) + separate ttnn for Phase 3
BenchResult bench_hybrid_gram_poly_plus_ttnn(
    const BenchShape& shape,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& x,
    float a,
    float b,
    float c,
    int device_id) {
    auto* dev_ptr = device.get();
    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();
    auto compute_grid_size = device->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(compute_grid_size.x, compute_grid_size.y);

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto H = ttml::metal::gram_polynomial(x, b, c);
        auto HX = ttnn::matmul(
            H,
            x,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto aX = ttnn::multiply(x, a);
        auto X_prime = ttnn::add(HX, aX);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        H.deallocate();
        HX.deallocate();
        aX.deallocate();
        X_prime.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    for (int i = 0; i < test_config.num_measurement_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto H = ttml::metal::gram_polynomial(x, b, c);
        auto HX = ttnn::matmul(
            H,
            x,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto aX = ttnn::multiply(x, a);
        auto X_prime = ttnn::add(HX, aX);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
        H.deallocate();
        HX.deallocate();
        aX.deallocate();
        X_prime.deallocate();
    }

    double avg_s = total_time.count() / test_config.num_measurement_iterations;
    uint64_t M = shape.M;
    uint64_t K = shape.K;
    double flops = total_flops(M, K);
    double tflops = flops / 1e12 / avg_s;
    double utilization = compute_utilization(avg_s, total_matmul_flops(M, K), device, device_id);

    return BenchResult{
        .impl_name = "gram_poly+ttnn P3",
        .time_us = avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization,
    };
}

void PrintComparisonTable(const std::string& shape_name, const std::vector<BenchResult>& results) {
    std::cout << "\n";
    std::cout << "╔══════════════════════╦═══════════════╦═══════════════╦═════════════════╗\n";
    std::cout << "║ " << std::setw(20) << std::left << ("Shape: " + shape_name)
              << " ║               ║               ║                 ║\n";
    std::cout << "╠══════════════════════╬═══════════════╬═══════════════╬═════════════════╣\n";
    std::cout << "║ Implementation       ║   Time (us)   ║    TFLOPs     ║   Utilization   ║\n";
    std::cout << "╠══════════════════════╬═══════════════╬═══════════════╬═════════════════╣\n";

    for (const auto& result : results) {
        std::cout << "║ " << std::setw(20) << std::left << result.impl_name << " ║ " << std::setw(13) << std::right
                  << std::fixed << std::setprecision(1) << result.time_us << " ║ " << std::setw(13) << std::fixed
                  << std::setprecision(2) << result.tflops << " ║ " << std::setw(14) << std::fixed
                  << std::setprecision(1) << result.utilization_pct << "% ║\n";
    }

    std::cout << "╚══════════════════════╩═══════════════╩═══════════════╩═════════════════╝\n";

    // Print speedup summary
    if (results.size() >= 2) {
        double fused_us = results[0].time_us;  // newton_schulz_iteration is first
        std::cout << "\n  Speedups vs newton_schulz_iteration (" << std::fixed << std::setprecision(1) << fused_us
                  << " us):\n";
        for (size_t i = 1; i < results.size(); ++i) {
            if (results[i].impl_name.find("phase") != std::string::npos) {
                continue;  // Skip individual phase entries
            }
            double speedup = results[i].time_us / fused_us;
            std::cout << "    vs " << std::setw(20) << std::left << results[i].impl_name << ": " << std::fixed
                      << std::setprecision(2) << speedup << "x\n";
        }
    }
}

void BM_NewtonSchulzIteration(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const BenchShape& shape = shapes[shape_index];

    const int device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id, /*l1_small_size=*/200000, /*trace_region_size=*/1048576);
    device->enable_program_cache();

    const float a = 0.9F;
    const float b = 0.5F;
    const float c = 0.3F;

    for (auto _ : state) {
        const uint32_t seed = 42;

        ttnn::Shape x_shape({1, 1, shape.M, shape.K});
        auto x = make_random_tensor(x_shape, device, seed);

        // Pre-compute G and H for isolated phase benchmarks
        auto G = ttml::metal::gram_matmul(x);
        auto H =
            ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);

        std::vector<BenchResult> results;

        // Full fused pipeline
        results.push_back(bench_newton_schulz_iteration(shape, device, x, a, b, c, device_id));

        // Individual phases
        results.push_back(bench_phase1(shape, device, x, device_id));
        results.push_back(bench_phase2(shape, device, G, b, c, device_id));
        results.push_back(bench_phase3(shape, device, H, x, a, device_id));

        // Composite baselines
        results.push_back(bench_composite_ttnn(shape, device, x, a, b, c, device_id));
        results.push_back(bench_hybrid_gram_poly_plus_ttnn(shape, device, x, a, b, c, device_id));

        PrintComparisonTable(shape.name, results);

        const auto& fused_result = results[0];
        state.SetIterationTime(fused_result.time_us / 1e6);
        state.counters["fused_us"] = results[0].time_us;
        state.counters["phase1_us"] = results[1].time_us;
        state.counters["phase2_us"] = results[2].time_us;
        state.counters["phase3_us"] = results[3].time_us;
        state.counters["composite_ttnn_us"] = results[4].time_us;
        state.counters["hybrid_us"] = results[5].time_us;

        x.deallocate();
        G.deallocate();
        H.deallocate();
    }

    device->close();
}

}  // namespace

BENCHMARK(BM_NewtonSchulzIteration)
    ->DenseRange(0, static_cast<int>(shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("NewtonSchulzIteration");

BENCHMARK_MAIN();
