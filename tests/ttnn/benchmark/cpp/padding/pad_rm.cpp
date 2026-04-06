// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt_stl/indestructible.hpp>
#include <ttnn/tensor/storage.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt_stl/small_vector.hpp>

namespace {
template <typename T>
ttnn::Tensor GenInputTensor(const ttnn::SmallVector<uint32_t>& shape) {
    using namespace tt::tt_metal;
    static std::mt19937 gen(42);  // fixed seed for reproducibility

    std::vector<T> input_data;
    // Get volume
    size_t volume = 1;
    for (const auto& dim : shape) {
        volume *= dim;
    }
    input_data.reserve(volume);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (size_t i = 0; i < volume; ++i) {
        input_data.push_back(static_cast<T>(dist(gen)));
    }
    return Tensor(HostBuffer{input_data}, ttnn::Shape(shape), DataType::BFLOAT16, Layout::ROW_MAJOR);
}

void BM_pad_rm_2d_last_dim_right(benchmark::State& state) {
    auto input_tensor = GenInputTensor<bfloat16>({8192, 8100});
    ttnn::SmallVector<uint32_t> padded_shape = {8192, 8192};
    ttnn::SmallVector<uint32_t> tensor_start = {0, 0};

    for ([[maybe_unused]] auto _ : state) {
        auto out = input_tensor.pad(
            ttnn::Shape(padded_shape),
            ttnn::Shape(tensor_start),
            0.0f);  // Pad the tensor to the same shape as input_tensor
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
}

void BM_pad_rm_2d_last_dim_left_right(benchmark::State& state) {
    auto input_tensor = GenInputTensor<bfloat16>({8192, 8100});
    ttnn::SmallVector<uint32_t> padded_shape = {8192, 8192};
    ttnn::SmallVector<uint32_t> tensor_start = {0, 92};

    for ([[maybe_unused]] auto _ : state) {
        auto out = input_tensor.pad(
            ttnn::Shape(padded_shape),
            ttnn::Shape(tensor_start),
            0.0f);  // Pad the tensor to the same shape as input_tensor
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
}

void BM_pad_rm_4d_last_dim_left_right(benchmark::State& state) {
    auto input_tensor = GenInputTensor<bfloat16>({16, 20, 512, 500});
    ttnn::SmallVector<uint32_t> padded_shape = {16, 20 + 12, 512 + 30, 500 + 30};
    ttnn::SmallVector<uint32_t> tensor_start = {0, 1, 3, 4};

    for ([[maybe_unused]] auto _ : state) {
        auto out = input_tensor.pad(
            ttnn::Shape(padded_shape),
            ttnn::Shape(tensor_start),
            0.0f);  // Pad the tensor to the same shape as input_tensor
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
}

void BM_pad_rm_2d_scaling(benchmark::State& state) {
    int N = state.range(0);
    int N_padded = N + (2 * 100);

    auto input_tensor = GenInputTensor<bfloat16>({8192, N});
    ttnn::SmallVector<uint32_t> padded_shape = {8192, N_padded};
    ttnn::SmallVector<uint32_t> tensor_start = {0, 100};

    for ([[maybe_unused]] auto _ : state) {
        auto out = input_tensor.pad(
            ttnn::Shape(padded_shape),
            ttnn::Shape(tensor_start),
            0.0f);  // Pad the tensor to the same shape as input_tensor
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(N_padded);
}

BENCHMARK(BM_pad_rm_2d_last_dim_right)->Unit(benchmark::kMillisecond)->MinTime(5.0);
BENCHMARK(BM_pad_rm_2d_last_dim_left_right)->Unit(benchmark::kMillisecond)->MinTime(5.0);
BENCHMARK(BM_pad_rm_4d_last_dim_left_right)->Unit(benchmark::kMillisecond)->MinTime(5.0);
BENCHMARK(BM_pad_rm_2d_scaling)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(128, 8192)->Complexity();
}  // namespace

BENCHMARK_MAIN();
