// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include "ttnn/operations/experimental/where/where.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/device.hpp"

#include "ttnn/operations/functions.hpp"

namespace {

void BM_where_experimental_ttt(benchmark::State& state) {
    using namespace ttnn::types;

    auto shape = ttnn::Shape({state.range(0), state.range(0)});
    auto dtype = DataType::BFLOAT16;
    auto layout = Layout::TILE;
    auto device_id = 0;

    auto device = ttnn::device::open_mesh_device(device_id);

    auto host_condition = ttnn::random::random(shape, dtype, layout);
    auto host_true_values = ttnn::random::random(shape, dtype, layout);
    auto host_false_values = ttnn::random::random(shape, dtype, layout);

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
}

void BM_where_ttt(benchmark::State& state) {
    using namespace ttnn::types;

    auto shape = ttnn::Shape({state.range(0), state.range(0)});
    auto dtype = DataType::BFLOAT16;
    auto layout = Layout::TILE;
    auto device_id = 0;

    auto device = ttnn::device::open_mesh_device(device_id);

    auto host_condition = ttnn::random::random(shape, dtype, layout);
    auto host_true_values = ttnn::random::random(shape, dtype, layout);
    auto host_false_values = ttnn::random::random(shape, dtype, layout);

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
}

BENCHMARK(BM_where_experimental_ttt)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(32, 4096)->Complexity();
BENCHMARK(BM_where_ttt)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(32, 4096)->Complexity();  // 8192

}  // namespace

BENCHMARK_MAIN();
