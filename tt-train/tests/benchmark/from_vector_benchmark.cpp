// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "core/tt_tensor_utils.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

ttnn::distributed::MeshDevice* g_device = nullptr;

template <typename T>
std::vector<T> make_random_data(size_t volume, unsigned seed) {
    std::vector<T> data(volume);
    std::mt19937 gen(seed);
    if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& x : data) x = dist(gen);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& x : data) x = bfloat16(dist(gen));
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        std::uniform_int_distribution<uint32_t> dist(0, 1000000u);
        for (auto& x : data) x = dist(gen);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
        for (auto& x : data) x = dist(gen);
    }
    return data;
}

void BM_FromVector_Bfloat16_BFLOAT16(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_random_data<bfloat16>(volume, 42u);
    for (auto _ : state) {
        auto tensor =
            ttml::core::from_vector<bfloat16, ttnn::DataType::BFLOAT16>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Float_BFLOAT16(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_random_data<float>(volume, 42u);
    for (auto _ : state) {
        auto tensor =
            ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Float_FLOAT32(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_random_data<float>(volume, 42u);
    for (auto _ : state) {
        auto tensor =
            ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Uint32_UINT32(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_random_data<uint32_t>(volume, 42u);
    for (auto _ : state) {
        auto tensor =
            ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(data, shape, g_device, ttnn::Layout::TILE);
        benchmark::DoNotOptimize(tensor);
    }
}

void BM_FromVector_Int32_INT32(benchmark::State& state) {
    const size_t volume = static_cast<size_t>(state.range(0));
    const ttnn::Shape shape({1, 1, 1, volume});
    auto data = make_random_data<int32_t>(volume, 42u);
    for (auto _ : state) {
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
