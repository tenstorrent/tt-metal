// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>

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

void BM_AdamW(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& adamw_shape = adamw_shapes[shape_index];

    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape shape(adamw_shape.shape);
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(adamw_shape.name));

    // bf16 no-AMSGrad: reads 4 tensors (param, grad, exp_avg, exp_avg_sq), writes 3 (param_out, exp_avg_out,
    // exp_avg_sq_out)
    const uint64_t tensor_bytes = shape.volume() * sizeof(uint16_t);  // bf16 = 2 bytes
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

        state.SetIterationTime(avg_time_s);
        state.SetLabel(adamw_shape.name);
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

}  // namespace

BENCHMARK(BM_AdamW)
    ->DenseRange(0, static_cast<int>(adamw_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("AdamW");

BENCHMARK_MAIN();
