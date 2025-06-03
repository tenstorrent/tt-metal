// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include "ttnn/operations/experimental/where/where.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/functions.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>

#include "small_vector.hpp"

namespace {

void BM_where_ttt(benchmark::State& state) {
    using namespace ttnn::types;

    auto shape = ttnn::Shape({state.range(0), state.range(1)});
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
}

static void CustomShapes(benchmark::internal::Benchmark* b) {
    b->Args({32, 32});
    b->Args({64, 64});
    b->Args({256, 256});
    b->Args({512, 512});
    b->Args({1024, 1024});
    b->Args({2048, 2048});
}

BENCHMARK(BM_where_ttt)->Unit(benchmark::kMillisecond)->Apply(CustomShapes);

}  // namespace

BENCHMARK_MAIN();
