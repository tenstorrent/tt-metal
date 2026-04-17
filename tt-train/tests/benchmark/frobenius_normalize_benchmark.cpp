// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>

#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

struct FrobeniusShape {
    std::vector<uint32_t> shape;
    std::string name;
};

struct TestConfig {
    int num_warmup_iterations = 5;
    int num_measurement_iterations = 20;
};

const TestConfig test_config;

const std::vector<FrobeniusShape> frobenius_shapes = {
    {{1, 1, 2048, 5632}, "2048x5632"},
    {{1, 1, 8192, 8192}, "8192x8192"},
};

constexpr float kEps = 1e-7f;

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

ttnn::Tensor composite_frobenius_normalize(const ttnn::Tensor& input, float eps) {
    auto squares = ttnn::square(input);
    auto sum_squares = ttnn::sum(squares, ttsl::SmallVector<int>{-2, -1}, true);
    auto norm_tensor = ttnn::sqrt(sum_squares);
    auto norm_plus_eps = ttnn::add(norm_tensor, eps);
    return ttnn::divide(input, norm_plus_eps);
}

void BM_FrobeniusNormalize_Fused(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& frobenius_shape = frobenius_shapes[shape_index];

    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape shape(frobenius_shape.shape);
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(frobenius_shape.name));

    auto input = make_random_tensor(shape, dtype, device.get(), seed);

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto result = ttml::metal::frobenius_normalize(input, kEps);
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
        result.deallocate();
    }

    for ([[maybe_unused]] auto _ : state) {
        auto total_time = std::chrono::duration<double>::zero();

        for (int iter = 0; iter < test_config.num_measurement_iterations; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = ttml::metal::frobenius_normalize(input, kEps);
            tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
            result.deallocate();
        }

        const double avg_time_s = total_time.count() / test_config.num_measurement_iterations;
        state.SetIterationTime(avg_time_s);
        state.SetLabel(frobenius_shape.name);
        state.counters["Time_us"] = avg_time_s * 1e6;
    }

    input.deallocate();
    device->close();
}

void BM_FrobeniusNormalize_Composite(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& frobenius_shape = frobenius_shapes[shape_index];

    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape shape(frobenius_shape.shape);
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(frobenius_shape.name));

    auto input = make_random_tensor(shape, dtype, device.get(), seed);

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto result = composite_frobenius_normalize(input, kEps);
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
        result.deallocate();
    }

    for ([[maybe_unused]] auto _ : state) {
        auto total_time = std::chrono::duration<double>::zero();

        for (int iter = 0; iter < test_config.num_measurement_iterations; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = composite_frobenius_normalize(input, kEps);
            tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
            result.deallocate();
        }

        const double avg_time_s = total_time.count() / test_config.num_measurement_iterations;
        state.SetIterationTime(avg_time_s);
        state.SetLabel(frobenius_shape.name);
        state.counters["Time_us"] = avg_time_s * 1e6;
    }

    input.deallocate();
    device->close();
}

}  // namespace

BENCHMARK(BM_FrobeniusNormalize_Fused)
    ->DenseRange(0, static_cast<int>(frobenius_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("FrobeniusNormalize_Fused");

BENCHMARK(BM_FrobeniusNormalize_Composite)
    ->DenseRange(0, static_cast<int>(frobenius_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("FrobeniusNormalize_Composite");

BENCHMARK_MAIN();
