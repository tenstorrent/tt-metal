// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include "ttnn/operations/experimental/where/where.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/device.hpp"

#include "ttnn/operations/functions.hpp"

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace {

// Will be replaced with ttnn::rand
template <typename ElemType>
tt::tt_metal::Tensor genRandomTensor(const ttnn::Shape& shape, const tt::tt_metal::Layout layout) {
    constexpr ttnn::DataType data_type = tt::tt_metal::convert_to_data_type<ElemType>();
    ttnn::TensorSpec spec(shape, ttnn::TensorLayout(data_type, ttnn::PageConfig(layout), tt::tt_metal::MemoryConfig{}));
    auto output_buffer = std::vector<ElemType>(spec.padded_shape().volume());

    auto init_rand_elem = [](auto& elem) {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, 100);
        elem = ElemType(dist(rng));
    };

    const size_t total = output_buffer.size();
    if (total < 256 * 256) {
        std::ranges::for_each(output_buffer, init_rand_elem);
        return tt::tt_metal::Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), spec);
    } else {
        tf::Executor executor;
        tf::Taskflow taskflow;

        taskflow.for_each(output_buffer.begin(), output_buffer.end(), init_rand_elem);
        executor.run(taskflow).wait();
    }

    return tt::tt_metal::Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), spec);
}

void BM_where_experimental_bf16_ttt(benchmark::State& state) {
    using namespace ttnn::types;

    auto shape = ttnn::Shape({state.range(0), state.range(0)});
    auto layout = Layout::TILE;
    auto device_id = 0;

    auto device = ttnn::device::open_mesh_device(device_id);

    auto host_condition = genRandomTensor<::bfloat16>(shape, layout);
    auto host_true_values = genRandomTensor<::bfloat16>(shape, layout);
    auto host_false_values = genRandomTensor<::bfloat16>(shape, layout);

    auto* dev_ptr = device.get();
    auto cond_tensor = host_condition.to_device(dev_ptr);
    auto true_value_tensor = host_true_values.to_device(dev_ptr);
    auto false_value_tensor = host_false_values.to_device(dev_ptr);

    auto output = ttnn::operations::experimental::ternary::where(cond_tensor, true_value_tensor, false_value_tensor);

    for ([[maybe_unused]] auto _ : state) {
        auto out = ttnn::operations::experimental::ternary::where(cond_tensor, true_value_tensor, false_value_tensor);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(3 * shape[0] * shape[1]);
    ttnn::device::close_device(*device);
}

void BM_where_bf16_ttt(benchmark::State& state) {
    using namespace ttnn::types;

    auto shape = ttnn::Shape({state.range(0), state.range(0)});
    auto layout = Layout::TILE;
    auto device_id = 0;

    auto device = ttnn::device::open_mesh_device(device_id);

    auto host_condition = genRandomTensor<::bfloat16>(shape, layout);
    auto host_true_values = genRandomTensor<::bfloat16>(shape, layout);
    auto host_false_values = genRandomTensor<::bfloat16>(shape, layout);

    auto* dev_ptr = device.get();
    auto cond_tensor = host_condition.to_device(dev_ptr);
    auto true_value_tensor = host_true_values.to_device(dev_ptr);
    auto false_value_tensor = host_false_values.to_device(dev_ptr);

    auto output = ttnn::where(cond_tensor, true_value_tensor, false_value_tensor);

    for ([[maybe_unused]] auto _ : state) {
        auto out = ttnn::where(cond_tensor, true_value_tensor, false_value_tensor);
        tt::tt_metal::distributed::Synchronize(dev_ptr, std::nullopt);
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(3 * shape[0] * shape[1]);
    ttnn::device::close_device(*device);
}

BENCHMARK(BM_where_experimental_bf16_ttt)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(32, 16384)
    ->Complexity();
BENCHMARK(BM_where_bf16_ttt)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(32, 16384)->Complexity();

}  // namespace

BENCHMARK_MAIN();
