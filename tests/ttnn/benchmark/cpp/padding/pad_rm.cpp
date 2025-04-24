// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt_stl/indestructible.hpp>
#include <ttnn/tensor/storage.hpp>
#include <ttnn/tensor/host_buffer/owned_buffer.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include "small_vector.hpp"

// static std::vector<float> input_data;

template <typename T>
std::vector<T> GenInputData(const ttnn::SmallVector<uint32_t>& shape) {
    std::vector<T> input_data;
    // Get volume
    uint32_t volume = 1;
    for (const auto& dim : shape) {
        volume *= dim;
    }
    input_data.resize(volume);
    std::mt19937 gen(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (auto& val : input_data) {
        val = static_cast<T>(dist(gen));
    }
    return input_data;
}

static void BM_pad_rm_2d_last_dim_right(benchmark::State& state) {
    using namespace tt::tt_metal;
    using T = bfloat16;

    ttnn::SmallVector<uint32_t> input_shape = {8192, 8100};
    ttnn::SmallVector<uint32_t> padded_shape = {8192, 8192};
    ttnn::SmallVector<uint32_t> tensor_start = {0, 0};

    auto input_data = GenInputData<T>(input_shape);
    auto input_buffer = owned_buffer::create<T>(std::move(input_data));

    auto input_tensor =
        Tensor(OwnedStorage{input_buffer}, ttnn::Shape(input_shape), DataType::BFLOAT16, Layout::ROW_MAJOR);
    for (auto _ : state) {
        auto out = input_tensor.pad(
            ttnn::Shape(padded_shape),
            ttnn::Shape(tensor_start),
            0.0f);  // Pad the tensor to the same shape as input_tensor
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
}

static void BM_pad_rm_2d_last_dim_left_right(benchmark::State& state) {
    using namespace tt::tt_metal;
    using T = bfloat16;

    ttnn::SmallVector<uint32_t> input_shape = {8192, 8100};
    ttnn::SmallVector<uint32_t> padded_shape = {8192, 8192};
    ttnn::SmallVector<uint32_t> tensor_start = {0, 92};

    auto input_data = GenInputData<T>(input_shape);
    auto input_buffer = owned_buffer::create<T>(std::move(input_data));

    auto input_tensor =
        Tensor(OwnedStorage{input_buffer}, ttnn::Shape(input_shape), DataType::BFLOAT16, Layout::ROW_MAJOR);
    for (auto _ : state) {
        auto out = input_tensor.pad(
            ttnn::Shape(padded_shape),
            ttnn::Shape(tensor_start),
            0.0f);  // Pad the tensor to the same shape as input_tensor
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
}

static void BM_pad_rm_4d_last_dim_left_right(benchmark::State& state) {
    using namespace tt::tt_metal;
    using T = bfloat16;

    ttnn::SmallVector<uint32_t> input_shape = {16, 20, 512, 500};
    ttnn::SmallVector<uint32_t> padded_shape = {16, 20 + 12, 512 + 30, 500 + 30};
    ttnn::SmallVector<uint32_t> tensor_start = {0, 1, 3, 4};

    auto input_data = GenInputData<T>(input_shape);
    auto input_buffer = owned_buffer::create<T>(std::move(input_data));

    auto input_tensor =
        Tensor(OwnedStorage{input_buffer}, ttnn::Shape(input_shape), DataType::BFLOAT16, Layout::ROW_MAJOR);
    for (auto _ : state) {
        auto out = input_tensor.pad(
            ttnn::Shape(padded_shape),
            ttnn::Shape(tensor_start),
            0.0f);  // Pad the tensor to the same shape as input_tensor
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_pad_rm_2d_last_dim_right)->Unit(benchmark::kMillisecond)->MinTime(5.0);
BENCHMARK(BM_pad_rm_2d_last_dim_left_right)->Unit(benchmark::kMillisecond)->MinTime(5.0);
BENCHMARK(BM_pad_rm_4d_last_dim_left_right)->Unit(benchmark::kMillisecond)->MinTime(5.0);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
