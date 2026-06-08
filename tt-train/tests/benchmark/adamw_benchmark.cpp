// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include "benchmark_utils.hpp"
#include "metal/optimizers/adamw/adamw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/types.hpp"

namespace {

struct AdamWShape {
    std::vector<uint32_t> shape;
    std::string name;
};

constexpr ttml::benchmark_utils::BenchmarkIterationConfig test_config = {
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

void BM_AdamW(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& adamw_shape = adamw_shapes[shape_index];

    constexpr int device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape shape(adamw_shape.shape);
    const uint32_t seed = ttml::benchmark_utils::seed_from_name(adamw_shape.name);
    const auto tensor_spec = ttnn::TensorSpec(
        shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));

    const auto make_random_tensor = [&](float min, float max, uint32_t tensor_seed) {
        auto data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), min, max, tensor_seed);
        return ttnn::Tensor::from_vector(data, tensor_spec, device.get());
    };

    // bf16 no-AMSGrad: reads 4 tensors (param, grad, exp_avg, exp_avg_sq), writes 3 (param_out, exp_avg_out,
    // exp_avg_sq_out)
    const uint64_t tensor_bytes = shape.volume() * sizeof(uint16_t);  // bf16 = 2 bytes
    const uint64_t total_dram_bytes = 7ULL * tensor_bytes;

    auto param = make_random_tensor(-1.0F, 1.0F, seed);
    auto grad = make_random_tensor(-1.0F, 1.0F, seed + 1);
    auto exp_avg = make_random_tensor(-1.0F, 1.0F, seed + 2);
    auto exp_avg_sq = make_random_tensor(0.0F, 1.0F, seed + 3);

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
        const double avg_time_s =
            ttml::benchmark_utils::measure_average_iteration_time_s(test_config.num_measurement_iterations, [&]() {
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
            });
        const double time_us = avg_time_s * 1e6;
        const double gb_per_s = static_cast<double>(total_dram_bytes) / avg_time_s / 1e9;

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
