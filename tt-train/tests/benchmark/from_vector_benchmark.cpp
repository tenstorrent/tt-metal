// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <type_traits>
#include <vector>

#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

ttnn::distributed::MeshDevice* g_device = nullptr;

template <typename T>
std::vector<T> make_benchmark_input_data(size_t volume, unsigned seed) {
    if constexpr (std::is_same_v<T, float>) {
        return ttml::test_utils::make_uniform_vector<float>(volume, -1.0F, 1.0F, seed);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        return ttml::test_utils::make_uniform_vector<bfloat16>(volume, bfloat16{-1.0F}, bfloat16{1.0F}, seed);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return ttml::test_utils::make_uniform_vector<uint32_t>(volume, 0U, 1'000'000U, seed);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return ttml::test_utils::make_uniform_vector<int32_t>(volume, -1'000'000, 1'000'000, seed);
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported random data type");
    }
}

void BM_FromVector_Bfloat16_BFLOAT16(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_benchmark_input_data<bfloat16>(volume, 42u);
    for ([[maybe_unused]] auto _ : state) {
        auto tensor =
            ttml::core::from_vector<bfloat16, ttnn::DataType::BFLOAT16>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Float_BFLOAT16(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_benchmark_input_data<float>(volume, 42u);
    for ([[maybe_unused]] auto _ : state) {
        auto tensor =
            ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Float_FLOAT32(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_benchmark_input_data<float>(volume, 42u);
    for ([[maybe_unused]] auto _ : state) {
        auto tensor =
            ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Uint32_UINT32(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_benchmark_input_data<uint32_t>(volume, 42u);
    for ([[maybe_unused]] auto _ : state) {
        auto tensor =
            ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Int32_INT32(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_benchmark_input_data<int32_t>(volume, 42u);
    for ([[maybe_unused]] auto _ : state) {
        auto tensor =
            ttml::core::from_vector<int32_t, ttnn::DataType::INT32>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

}  // namespace

BENCHMARK(BM_FromVector_Bfloat16_BFLOAT16)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1 << 16, 1 << 20);

BENCHMARK(BM_FromVector_Float_BFLOAT16)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1 << 16, 1 << 20);

BENCHMARK(BM_FromVector_Float_FLOAT32)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1 << 16, 1 << 20);

BENCHMARK(BM_FromVector_Uint32_UINT32)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1 << 16, 1 << 20);

BENCHMARK(BM_FromVector_Int32_INT32)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1 << 16, 1 << 20);

int main(int argc, char** argv) {
    auto device = ttnn::device::open_mesh_device(0, /*l1_small_size=*/0, /*trace_region_size=*/1048576);
    g_device = device.get();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    device->close();
    return 0;
}
