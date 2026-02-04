// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <llrt/tt_cluster.hpp>
#include <memory>
#include <optional>
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
#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

struct CoreGridConfig {
    std::optional<ttnn::CoreGrid> core_grid;
    std::string name;

    bool is_supported(const tt::tt_metal::CoreCoord& device_grid) const {
        if (!core_grid.has_value()) {
            return true;
        }
        return static_cast<int>(core_grid->x) <= device_grid.x && static_cast<int>(core_grid->y) <= device_grid.y;
    }
};

struct MatmulTestShape {
    std::vector<uint32_t> a_shape;
    std::vector<uint32_t> b_shape;
    bool transpose_a;
    bool transpose_b;
    std::string name;
};

struct BenchmarkResult {
    std::string grid_name;
    double time_us = std::numeric_limits<double>::quiet_NaN();
    double tflops = 0.0;
    double utilization_pct = 0.0;
    bool skipped = false;
};

struct TestConfig {
    int num_warmup_iterations = 2;
    int num_measurement_iterations = 25;
};

// CoreGrid configurations to compare
const std::vector<CoreGridConfig> core_grid_configs = {
    {std::nullopt, "std::nullopt"},
    {ttnn::CoreGrid{7, 8}, "7x8"},
    {ttnn::CoreGrid{8, 8}, "8x8"},
    {ttnn::CoreGrid{10, 10}, "10x10"},
    {ttnn::CoreGrid{11, 10}, "11x10"},
    {ttnn::CoreGrid{12, 10}, "12x10"},
    {ttnn::CoreGrid{13, 10}, "13x10"},
};

// Test shapes from tt-train workloads
const std::vector<MatmulTestShape> tt_train_shapes = {
    // Attention matmuls
    {{64, 6, 256, 64}, {64, 6, 256, 64}, false, true, "attn_64x6x256x64_x_64x6x256x64_Tb"},
    {{64, 6, 256, 256}, {64, 6, 256, 64}, false, false, "attn_64x6x256x256_x_64x6x256x64"},
    {{64, 6, 256, 256}, {64, 6, 256, 64}, true, false, "attn_64x6x256x256_x_64x6x256x64_Ta"},
    {{4, 12, 1024, 64}, {4, 12, 1024, 64}, false, true, "attn_4x12x1024x64_x_4x12x1024x64_Tb"},
    {{4, 12, 1024, 1024}, {4, 12, 1024, 64}, false, false, "attn_4x12x1024x1024_x_4x12x1024x64"},
    {{4, 12, 1024, 1024}, {4, 12, 1024, 64}, true, false, "attn_4x12x1024x1024_x_4x12x1024x64_Ta"},

    // Linear layer forward matmuls (2D x 4D broadcast)
    {{16384, 96}, {1, 1, 96, 384}, false, false, "fwd_16384x96_x_96x384"},
    {{16384, 384}, {1, 1, 384, 1536}, false, false, "fwd_16384x384_x_384x1536"},
    {{16384, 1536}, {1, 1, 1536, 384}, false, false, "fwd_16384x1536_x_1536x384"},
    {{16384, 384}, {1, 1, 384, 384}, false, false, "fwd_16384x384_x_384x384"},
    {{16384, 1152}, {1, 1, 1152, 384}, false, false, "fwd_16384x1152_x_1152x384"},
    {{4096, 96}, {1, 1, 96, 768}, false, false, "fwd_4096x96_x_96x768"},
    {{4096, 768}, {1, 1, 768, 3072}, false, false, "fwd_4096x768_x_768x3072"},
    {{4096, 3072}, {1, 1, 3072, 768}, false, false, "fwd_4096x3072_x_3072x768"},
    {{4096, 768}, {1, 1, 768, 768}, false, false, "fwd_4096x768_x_768x768"},
    {{4096, 2304}, {1, 1, 2304, 768}, false, false, "fwd_4096x2304_x_2304x768"},

    // Linear layer backward matmuls (2D x 2D with transpose_a)
    {{16384, 96}, {16384, 384}, true, false, "bwd_16384x96_x_16384x384_Ta"},
    {{16384, 384}, {16384, 1536}, true, false, "bwd_16384x384_x_16384x1536_Ta"},
    {{16384, 1536}, {16384, 384}, true, false, "bwd_16384x1536_x_16384x384_Ta"},
    {{16384, 384}, {16384, 384}, true, false, "bwd_16384x384_x_16384x384_Ta"},
    {{16384, 1152}, {16384, 384}, true, false, "bwd_16384x1152_x_16384x384_Ta"},
    {{4096, 96}, {4096, 768}, true, false, "bwd_4096x96_x_4096x768_Ta"},
    {{4096, 768}, {4096, 3072}, true, false, "bwd_4096x768_x_4096x3072_Ta"},
    {{4096, 3072}, {4096, 768}, true, false, "bwd_4096x3072_x_4096x768_Ta"},
    {{4096, 768}, {4096, 768}, true, false, "bwd_4096x768_x_4096x768_Ta"},
    {{4096, 2304}, {4096, 768}, true, false, "bwd_4096x2304_x_4096x768_Ta"},
};

const TestConfig test_config = {
    .num_warmup_iterations = 2,
    .num_measurement_iterations = 10,
};

namespace {

// Get effective M, K, N for TFLOPS calculation
std::tuple<uint64_t, uint64_t, uint64_t> get_mkn_from_shapes(
    const ttnn::Tensor& input_a, const ttnn::Tensor& input_b, bool transpose_a, bool transpose_b) {
    // For matmul A[..., M, K] x B[..., K, N] = C[..., M, N]
    // If transpose_a: A[..., K, M] -> A^T[..., M, K]
    // If transpose_b: B[..., N, K] -> B^T[..., K, N]
    const auto& a_shape = input_a.logical_shape();
    const auto& b_shape = input_b.logical_shape();

    uint64_t M = transpose_a ? a_shape[-1] : a_shape[-2];
    uint64_t K = transpose_a ? a_shape[-2] : a_shape[-1];
    uint64_t N = transpose_b ? b_shape[-2] : b_shape[-1];

    uint64_t batch = ttnn::get_batch_size(a_shape);

    return {batch * M, K, N};
}

BenchmarkResult RunSingleMatmulBenchmark(
    const MatmulTestShape& matmul_shape,
    const CoreGridConfig& grid_config,
    const std::shared_ptr<ttnn::device::MeshDevice>& device,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const int device_id = 0) {
    const int num_warmup_iterations = test_config.num_warmup_iterations;
    const int num_measurement_iterations = test_config.num_measurement_iterations;

    auto* dev_ptr = device.get();
    auto compute_grid_size = device->compute_with_storage_grid_size();

    if (!grid_config.is_supported(compute_grid_size)) {
        return BenchmarkResult{.grid_name = grid_config.name, .skipped = true};
    }

    // Get tt-train compute kernel config
    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();

    ttnn::Tensor output_tensor;

    // Warmup iterations
    for (int iter = 0; iter < num_warmup_iterations; ++iter) {
        output_tensor = ttnn::matmul(
            input_tensor_a,
            input_tensor_b,
            /*transpose_a=*/matmul_shape.transpose_a,
            /*transpose_b=*/matmul_shape.transpose_b,
            /*memory_config=*/std::nullopt,
            /*dtype=*/std::nullopt,
            /*program_config=*/std::nullopt,
            /*activation=*/std::nullopt,
            /*compute_kernel_config=*/compute_kernel_config,
            /*core_grid=*/grid_config.core_grid,
            /*output_tile=*/std::nullopt);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        output_tensor.deallocate();
    }

    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();

    // Performance measurement iterations
    {
        ZoneScopedN("TTTrain Matmul iterations");
        for (int iter = 0; iter < num_measurement_iterations; ++iter) {
            auto start_time = std::chrono::high_resolution_clock::now();
            output_tensor = ttnn::matmul(
                input_tensor_a,
                input_tensor_b,
                /*transpose_a=*/matmul_shape.transpose_a,
                /*transpose_b=*/matmul_shape.transpose_b,
                /*memory_config=*/std::nullopt,
                /*dtype=*/std::nullopt,
                /*program_config=*/std::nullopt,
                /*activation=*/std::nullopt,
                /*compute_kernel_config=*/compute_kernel_config,
                /*core_grid=*/grid_config.core_grid,
                /*output_tile=*/std::nullopt);
            tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
            auto end_time = std::chrono::high_resolution_clock::now();
            total_time += end_time - start_time;
            output_tensor.deallocate();
        }
    }

    const double inference_time_avg_s = total_time.count() / num_measurement_iterations;

    // Calculate TFLOPS
    auto [M, K, N] =
        get_mkn_from_shapes(input_tensor_a, input_tensor_b, matmul_shape.transpose_a, matmul_shape.transpose_b);
    double tflops = 2.0 * M * K * N / 1e12 / inference_time_avg_s;

    // Calculate utilization against full grid
    constexpr int HiFi4_cycles_per_tile = 64;
    int num_cores_full_grid = compute_grid_size.x * compute_grid_size.y;

    const double num_tile_ops = static_cast<double>(M) * K * N / (32.0 * 32.0 * 32.0);
    double ideal_cycle_full_grid = num_tile_ops * HiFi4_cycles_per_tile / num_cores_full_grid;

    const int freq_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_id);
    double inference_cycle = inference_time_avg_s * freq_mhz * 1e6;

    double utilization_full_grid = ideal_cycle_full_grid / inference_cycle;

    return BenchmarkResult{
        .grid_name = grid_config.name,
        .time_us = inference_time_avg_s * 1e6,
        .tflops = tflops,
        .utilization_pct = utilization_full_grid * 100,
    };
}

void PrintComparisonTable(const std::string& shape_name, const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Shape: " << std::left << std::setw(53) << shape_name << " ║\n";
    std::cout << "╠══════════════╦═══════════════╦═══════════════╦═══════════════╣\n";
    std::cout << "║   CoreGrid   ║   Time (us)   ║    TFLOPs     ║  Utilization  ║\n";
    std::cout << "╠══════════════╬═══════════════╬═══════════════╬═══════════════╣\n";

    for (const auto& result : results) {
        std::cout << "║ " << std::setw(12) << std::left << result.grid_name << " ║ ";
        if (result.skipped) {
            std::cout << std::setw(13) << std::right << "SKIP"
                      << " ║ " << std::setw(13) << std::right << "-"
                      << " ║ " << std::setw(13) << std::right << "-"
                      << " ║\n";
        } else {
            std::cout << std::setw(13) << std::right << std::fixed << std::setprecision(1) << result.time_us << " ║ "
                      << std::setw(13) << std::fixed << std::setprecision(2) << result.tflops << " ║ " << std::setw(12)
                      << std::fixed << std::setprecision(2) << result.utilization_pct << "% ║\n";
        }
    }

    std::cout << "╚══════════════╩═══════════════╩═══════════════╩═══════════════╝\n";
}

// Benchmark function - for each shape, compares all grid configs and prints table
void BM_TTTrainMatmulComparison(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const MatmulTestShape& matmul_shape = tt_train_shapes[shape_index];

    // Open device
    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id, /*l1_small_size=*/200000, /*trace_region_size=*/1048576);
    device->enable_program_cache();

    std::vector<BenchmarkResult> results;

    for (auto _ : state) {
        results.clear();

        // Create inputs once per shape so each grid config sees identical tensors
        const auto dtype = ttnn::DataType::BFLOAT16;
        const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(matmul_shape.name));

        ttnn::Shape a_shape(matmul_shape.a_shape);
        ttnn::Shape b_shape(matmul_shape.b_shape);

        std::vector<float> data_a(a_shape.volume());
        ttml::core::parallel_generate(
            std::span{data_a.data(), data_a.size()},
            []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
            seed);
        ttnn::Tensor input_tensor_a = ttnn::Tensor::from_vector(
            data_a,
            ttnn::TensorSpec(
                a_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
            device.get());

        std::vector<float> data_b(b_shape.volume());
        ttml::core::parallel_generate(
            std::span{data_b.data(), data_b.size()},
            []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
            seed + 1);
        ttnn::Tensor input_tensor_b = ttnn::Tensor::from_vector(
            data_b,
            ttnn::TensorSpec(
                b_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
            device.get());

        // Run benchmark for each grid configuration
        for (const auto& grid_config : core_grid_configs) {
            auto result =
                RunSingleMatmulBenchmark(matmul_shape, grid_config, device, input_tensor_a, input_tensor_b, device_id);
            results.push_back(result);
        }

        // Print comparison table
        PrintComparisonTable(matmul_shape.name, results);

        // Report the best time to benchmark framework
        auto fastest = std::min_element(results.begin(), results.end(), [](const auto& a, const auto& b) {
            if (a.skipped != b.skipped) {
                return !a.skipped;  // non-skipped beats skipped
            }
            if (a.skipped && b.skipped) {
                return false;
            }
            return a.time_us < b.time_us;
        });

        if (fastest == results.end() || fastest->skipped) {
            state.SkipWithError("No valid core grid configs for this device");
            input_tensor_a.deallocate();
            input_tensor_b.deallocate();
            continue;
        }

        state.SetIterationTime(fastest->time_us / 1e6);
        state.counters["Best_TFLOPs"] = fastest->tflops;
        state.counters["Best_Time_us"] = fastest->time_us;

        state.counters["Best_Grid"] = std::distance(results.begin(), fastest);

        // Cleanup inputs
        input_tensor_a.deallocate();
        input_tensor_b.deallocate();
    }

    // Close device
    device->close();
}

}  // namespace

// Register one benchmark per shape - each will compare all grid configs
BENCHMARK(BM_TTTrainMatmulComparison)
    ->DenseRange(0, static_cast<int>(tt_train_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("TTTrainMatmul");

BENCHMARK_MAIN();
