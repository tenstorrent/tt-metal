// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include "ttnn/operations/experimental/where/where.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/device.hpp"

#include "ttnn/operations/functions.hpp"

#include <thread>

namespace {

// much faster than ttnn::random::random which uses uniform_real_distribution for floats
template <typename ElemType>
static tt::tt_metal::Tensor genSmallRandomTensor(const ttnn::Shape& shape, const tt::tt_metal::Layout layout) {
    constexpr ttnn::DataType data_type = tt::tt_metal::convert_to_data_type<ElemType>();

    ttnn::TensorSpec spec(shape, ttnn::TensorLayout(data_type, ttnn::PageConfig(layout), tt::tt_metal::MemoryConfig{}));
    auto output_buffer = std::vector<ElemType>(spec.padded_shape().volume());

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 100);
    for (size_t i = 0; i < output_buffer.size(); ++i) {
        output_buffer[i] = ElemType(dist(rng));
    }

    return tt::tt_metal::Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), spec).to_layout(layout);
}

template <typename ElemType>
static tt::tt_metal::Tensor genRandomTensor(const ttnn::Shape& shape, const tt::tt_metal::Layout layout) {
    const unsigned num_threads = std::thread::hardware_concurrency();

    constexpr ttnn::DataType data_type = tt::tt_metal::convert_to_data_type<ElemType>();
    ttnn::TensorSpec spec(shape, ttnn::TensorLayout(data_type, ttnn::PageConfig(layout), tt::tt_metal::MemoryConfig{}));
    auto output_buffer = std::vector<ElemType>(spec.padded_shape().volume());

    const size_t total = output_buffer.size();
    if (total < 2048 * 2048) {
        return genSmallRandomTensor<ElemType>(shape, layout);
    } else {
        const size_t chunk_size = (total + num_threads - 1) / num_threads;
        std::vector<std::jthread> threads;

        for (unsigned t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, total);

            threads.emplace_back([start, end, &output_buffer]() {
                std::mt19937 rng(std::random_device{}());
                std::uniform_int_distribution<int> dist(0, 100);
                for (size_t i = start; i < end; ++i) {
                    output_buffer[i] = ElemType(dist(rng));
                }
            });
        }
    }

    return tt::tt_metal::Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), spec).to_layout(layout);
}

void BM_where_experimental_ttt(benchmark::State& state) {
    using namespace ttnn::types;

    auto shape = ttnn::Shape({state.range(0), state.range(0)});
    auto dtype = DataType::BFLOAT16;
    auto layout = Layout::TILE;
    auto device_id = 0;

    auto device = ttnn::device::open_mesh_device(device_id);

    auto host_condition = genRandomTensor<::bfloat16>(shape, layout);
    auto host_true_values = genRandomTensor<::bfloat16>(shape, layout);
    auto host_false_values = genRandomTensor<::bfloat16>(shape, layout);

    auto dev_ptr = device.get();
    auto cond_tensor = host_condition.to_device(dev_ptr);
    auto true_value_tensor = host_true_values.to_device(dev_ptr);
    auto false_value_tensor = host_false_values.to_device(dev_ptr);

    auto output = ttnn::operations::ternary::experimental::where(cond_tensor, true_value_tensor, false_value_tensor);

    for (auto _ : state) {
        auto out = ttnn::operations::ternary::experimental::where(cond_tensor, true_value_tensor, false_value_tensor);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(3 * shape[0] * shape[1]);
    ttnn::device::close_device(*device);
}

void BM_where_ttt(benchmark::State& state) {
    using namespace ttnn::types;

    auto shape = ttnn::Shape({state.range(0), state.range(0)});
    auto dtype = DataType::BFLOAT16;
    auto layout = Layout::TILE;
    auto device_id = 0;

    auto device = ttnn::device::open_mesh_device(device_id);

    auto host_condition = genRandomTensor<::bfloat16>(shape, layout);
    auto host_true_values = genRandomTensor<::bfloat16>(shape, layout);
    auto host_false_values = genRandomTensor<::bfloat16>(shape, layout);

    auto dev_ptr = device.get();
    auto cond_tensor = host_condition.to_device(dev_ptr);
    auto true_value_tensor = host_true_values.to_device(dev_ptr);
    auto false_value_tensor = host_false_values.to_device(dev_ptr);

    auto output = ttnn::where(cond_tensor, true_value_tensor, false_value_tensor);

    for (auto _ : state) {
        auto out = ttnn::where(cond_tensor, true_value_tensor, false_value_tensor);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(3 * shape[0] * shape[1]);
    ttnn::device::close_device(*device);
}

BENCHMARK(BM_where_experimental_ttt)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(32, 16384)->Complexity();
BENCHMARK(BM_where_ttt)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(32, 16384)->Complexity();

}  // namespace

BENCHMARK_MAIN();
