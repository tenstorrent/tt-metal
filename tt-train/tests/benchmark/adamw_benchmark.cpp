// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/optimizers/adamw/adamw.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

struct AdamWShape {
    std::vector<uint32_t> shape;
    std::string name;
};

struct TestConfig {
    int num_warmup_iterations = 3;
    int num_measurement_iterations = 20;
};

const TestConfig test_config = {
    .num_warmup_iterations = 5,
    .num_measurement_iterations = 50,
};

// Shapes from Llama training + larger shapes
const std::vector<AdamWShape> adamw_shapes = {
    {{1, 1, 2048, 2048}, "2048x2048"},
    {{1, 1, 5632, 2048}, "5632x2048"},
    {{1, 1, 2048, 5632}, "2048x5632"},
    {{1, 1, 512, 2048}, "512x2048"},
    {{1, 1, 1, 2048}, "1x2048"},
    {{1, 1, 8192, 8192}, "8192x8192"},
    {{1, 1, 16384, 2048}, "16384x2048"},
};

ttnn::Tensor make_random_tensor(
    const ttnn::Shape& shape, ttnn::DataType dtype, ttnn::distributed::MeshDevice* device, uint32_t seed) {
    std::vector<float> data(shape.volume());
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

ttnn::Tensor make_positive_tensor(
    const ttnn::Shape& shape, ttnn::DataType dtype, ttnn::distributed::MeshDevice* device, uint32_t seed) {
    std::vector<float> data(shape.volume());
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

void PrintResultTable(const std::string& shape_name, double time_us, double gb_per_s, uint64_t tensor_bytes) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Shape: " << std::left << std::setw(51) << shape_name << " ║\n";
    std::cout << "╠═══════════════╦═══════════════╦══════════════════════════╣\n";
    std::cout << "║   Time (us)   ║    GB/s       ║  Tensor bytes            ║\n";
    std::cout << "╠═══════════════╬═══════════════╬══════════════════════════╣\n";
    std::cout << "║ " << std::setw(13) << std::right << std::fixed << std::setprecision(1) << time_us << " ║ "
              << std::setw(13) << std::fixed << std::setprecision(2) << gb_per_s << " ║ " << std::setw(24) << std::right
              << tensor_bytes << " ║\n";
    std::cout << "╚═══════════════╩═══════════════╩══════════════════════════╝\n";
}

// ── Sweep results collector ──────────────────────────────────────────────────
struct SweepResult {
    std::string depth_label;
    double time_us;
    double gb_per_s;
};

// shape_name -> list of results (one per depth), insertion order preserved by vector
std::map<std::string, std::vector<SweepResult>> g_sweep_results;

void PrintSweepSummary() {
    if (g_sweep_results.empty()) {
        return;
    }

    std::cout << "\n\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Pipeline Depth Sweep — Summary                       ║\n";
    std::cout << "╠══════════════════╦═══════════╦═══════════════╦═══════════════╦═══════════╣\n";
    std::cout << "║ Shape            ║ Depth     ║   Time (us)   ║    GB/s       ║ vs best   ║\n";
    std::cout << "╠══════════════════╬═══════════╬═══════════════╬═══════════════╬═══════════╣\n";

    for (auto& [shape_name, results] : g_sweep_results) {
        // Find best GB/s for this shape
        double best_gb_per_s = 0.0;
        for (const auto& r : results) {
            best_gb_per_s = std::max(best_gb_per_s, r.gb_per_s);
        }

        bool first_row = true;
        for (const auto& r : results) {
            double ratio = (best_gb_per_s > 0.0) ? r.gb_per_s / best_gb_per_s : 0.0;
            std::string marker = (r.gb_per_s == best_gb_per_s) ? " *best*" : "";

            std::cout << "║ " << std::left << std::setw(17) << (first_row ? shape_name : "") << "║ " << std::left
                      << std::setw(10) << r.depth_label << "║ " << std::right << std::setw(13) << std::fixed
                      << std::setprecision(1) << r.time_us << " ║ " << std::right << std::setw(13) << std::fixed
                      << std::setprecision(2) << r.gb_per_s << " ║ " << std::right << std::setw(5) << std::fixed
                      << std::setprecision(2) << ratio << marker
                      << std::setw(static_cast<int>(std::max(0, 4 - static_cast<int>(marker.size())))) << ""
                      << " ║\n";
            first_row = false;
        }
        std::cout << "╠══════════════════╬═══════════╬═══════════════╬═══════════════╬═══════════╣\n";
    }

    // Replace last separator with bottom border
    std::cout << "\033[1A";  // move cursor up one line
    std::cout << "╚══════════════════╩═══════════╩═══════════════╩═══════════════╩═══════════╝\n";
    std::cout << std::endl;
}

bool g_sweep_summary_registered = []() {
    std::atexit(PrintSweepSummary);
    return true;
}();

// ─────────────────────────────────────────────────────────────────────────────

void BM_AdamW(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& adamw_shape = adamw_shapes[shape_index];

    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id, /*l1_small_size=*/0, /*trace_region_size=*/1048576);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape shape(adamw_shape.shape);
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(adamw_shape.name));

    // bf16 no-AMSGrad: reads 4 tensors (param, grad, exp_avg, exp_avg_sq), writes 3 (param_out, exp_avg_out,
    // exp_avg_sq_out)
    const uint64_t tensor_bytes = static_cast<uint64_t>(shape.volume()) * sizeof(uint16_t);  // bf16 = 2 bytes
    const uint64_t total_dram_bytes = 7ULL * tensor_bytes;

    auto param = make_random_tensor(shape, dtype, device.get(), seed);
    auto grad = make_random_tensor(shape, dtype, device.get(), seed + 1);
    auto exp_avg = make_random_tensor(shape, dtype, device.get(), seed + 2);
    auto exp_avg_sq = make_positive_tensor(shape, dtype, device.get(), seed + 3);

    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float beta1_pow = std::pow(beta1, 10.0f);
    const float beta2_pow = std::pow(beta2, 10.0f);
    const float epsilon = 1e-8f;
    const float weight_decay = 0.01f;

    // Warmup
    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto result = ttml::metal::adamw(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            std::nullopt,
            lr,
            beta1,
            beta2,
            beta1_pow,
            beta2_pow,
            epsilon,
            weight_decay);
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
        result.deallocate();
    }

    for ([[maybe_unused]] auto _ : state) {
        auto total_time = std::chrono::duration<double>::zero();

        for (int iter = 0; iter < test_config.num_measurement_iterations; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = ttml::metal::adamw(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                std::nullopt,
                lr,
                beta1,
                beta2,
                beta1_pow,
                beta2_pow,
                epsilon,
                weight_decay);
            tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
            result.deallocate();
        }

        double avg_time_s = total_time.count() / test_config.num_measurement_iterations;
        double time_us = avg_time_s * 1e6;
        double gb_per_s = static_cast<double>(total_dram_bytes) / avg_time_s / 1e9;

        PrintResultTable(adamw_shape.name, time_us, gb_per_s, tensor_bytes);

        state.SetIterationTime(avg_time_s);
        state.counters["Time_us"] = time_us;
        state.counters["GB_per_s"] = gb_per_s;
        state.counters["Tensor_MB"] = static_cast<double>(tensor_bytes) / 1e6;
    }

    param.deallocate();
    grad.deallocate();
    exp_avg.deallocate();
    exp_avg_sq.deallocate();

    device->close();
}

// Shapes for pipeline depth sweep
const std::vector<AdamWShape> sweep_shapes = {
    {{1, 1, 2048, 2048}, "2048x2048"},
    {{1, 1, 2048, 5632}, "2048x5632"},
    {{1, 1, 8192, 8192}, "8192x8192"},
    {{1, 1, 16384, 2048}, "16384x2048"},
};

// Depth multipliers: actual depth = multiplier * block_size (bf16 block_size=4)
// 0 means "auto" (existing L1-budget heuristic)
const std::vector<int> depth_multipliers = {0, 1, 2, 4, 8, 16, 32, 64};

void BM_AdamW_PipelineDepthSweep(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const int depth_mult = static_cast<int>(state.range(1));
    const auto& adamw_shape = sweep_shapes[shape_index];

    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id, /*l1_small_size=*/0, /*trace_region_size=*/1048576);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape shape(adamw_shape.shape);
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(adamw_shape.name));

    constexpr uint32_t kBlockSize = 2U;
    const uint32_t pipeline_depth_tiles = (depth_mult == 0) ? 0 : static_cast<uint32_t>(depth_mult) * kBlockSize;

    const uint64_t tensor_bytes = static_cast<uint64_t>(shape.volume()) * sizeof(uint16_t);
    const uint64_t total_dram_bytes = 7ULL * tensor_bytes;

    auto param = make_random_tensor(shape, dtype, device.get(), seed);
    auto grad = make_random_tensor(shape, dtype, device.get(), seed + 1);
    auto exp_avg = make_random_tensor(shape, dtype, device.get(), seed + 2);
    auto exp_avg_sq = make_positive_tensor(shape, dtype, device.get(), seed + 3);

    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float beta1_pow = std::pow(beta1, 10.0f);
    const float beta2_pow = std::pow(beta2, 10.0f);
    const float epsilon = 1e-8f;
    const float weight_decay = 0.01f;

    // Warmup
    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto result = ttml::metal::adamw(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            std::nullopt,
            lr,
            beta1,
            beta2,
            beta1_pow,
            beta2_pow,
            epsilon,
            weight_decay,
            ttml::metal::StochasticRounding::Disabled,
            pipeline_depth_tiles);
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
        result.deallocate();
    }

    for ([[maybe_unused]] auto _ : state) {
        auto total_time = std::chrono::duration<double>::zero();

        for (int iter = 0; iter < test_config.num_measurement_iterations; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = ttml::metal::adamw(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                std::nullopt,
                lr,
                beta1,
                beta2,
                beta1_pow,
                beta2_pow,
                epsilon,
                weight_decay,
                ttml::metal::StochasticRounding::Disabled,
                pipeline_depth_tiles);
            tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
            result.deallocate();
        }

        double avg_time_s = total_time.count() / test_config.num_measurement_iterations;
        double time_us = avg_time_s * 1e6;
        double gb_per_s = static_cast<double>(total_dram_bytes) / avg_time_s / 1e9;

        std::string depth_label = (depth_mult == 0) ? "auto" : (std::to_string(depth_mult) + "x bs");
        std::string label = adamw_shape.name + " depth=" + depth_label;
        PrintResultTable(label, time_us, gb_per_s, tensor_bytes);

        g_sweep_results[adamw_shape.name].push_back({depth_label, time_us, gb_per_s});

        state.SetIterationTime(avg_time_s);
        state.counters["Time_us"] = time_us;
        state.counters["GB_per_s"] = gb_per_s;
        state.counters["Tensor_MB"] = static_cast<double>(tensor_bytes) / 1e6;
        state.counters["Depth_tiles"] = static_cast<double>(pipeline_depth_tiles);
    }

    param.deallocate();
    grad.deallocate();
    exp_avg.deallocate();
    exp_avg_sq.deallocate();

    device->close();
}

}  // namespace

BENCHMARK(BM_AdamW)
    ->DenseRange(0, static_cast<int>(adamw_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("AdamW");

BENCHMARK(BM_AdamW_PipelineDepthSweep)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(sweep_shapes.size()) - 1, 1),
        {0, 1, 2, 4, 8, 16, 32, 64},
    })
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("AdamW/PipelineDepthSweep");

BENCHMARK_MAIN();
