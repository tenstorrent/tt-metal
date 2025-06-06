// SPDX - FileCopyrightText : © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>
#include <omp.h>

#include <tt_stl/span.hpp>
#include <ttnn/tensor/storage.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <tt-metalium/host_buffer.hpp>
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace tt::tt_metal {
namespace {

void BM_parallel_to_layout(benchmark::State& state) {
    static std::mt19937 gen(42);
    const int num_shards = state.range(0);

    std::vector<Tensor> tensor_shards;
    for (int i = 0; i < num_shards; ++i) {
        ttnn::Shape shape({1, 1, 1024, 1024});
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        std::vector<float> input_data;
        input_data.reserve(shape.volume());
        for (size_t i = 0; i < shape.volume(); ++i) {
            input_data.push_back(dist(gen));
        }

        tensor_shards.push_back(Tensor(HostBuffer{std::move(input_data)}, shape, DataType::FLOAT32, Layout::ROW_MAJOR));
    }

    auto mult_device_tensor = ttnn::distributed::aggregate_as_tensor(tensor_shards, tt::tt_metal::AllGatherTensor{});

    for (auto _ : state) {
        auto out = tt::tt_metal::transform(
            mult_device_tensor,
            [](const auto& shard) { return shard.to_layout(Layout::TILE); },
            DeviceShardExecutionPolicy::PARALLEL);
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_parallel_to_layout)->Unit(benchmark::kMillisecond)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

}  // namespace
}  // namespace tt::tt_metal

BENCHMARK_MAIN();
