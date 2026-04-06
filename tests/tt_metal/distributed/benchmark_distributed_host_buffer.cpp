// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <tt-metalium/distributed_host_buffer.hpp>

#include <vector>
#include <numeric>
#include <algorithm>

namespace tt::tt_metal {
namespace {

DistributedHostBuffer create_distributed_host_buffer(int num_shards) {
    constexpr size_t kShardSize = 1 << 20;

    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(num_shards));
    std::vector<int> data(kShardSize);
    std::iota(data.begin(), data.end(), 0);
    for (size_t i = 0; i < num_shards; ++i) {
        buffer.emplace_shard(distributed::MeshCoordinate(i), [&data]() { return HostBuffer(data); });
    }

    return buffer;
}

std::function<HostBuffer(const HostBuffer&)> get_transform_fn() {
    return [](const HostBuffer& buffer) {
        auto span = buffer.view_as<int>();
        std::vector<int> new_data;
        new_data.reserve(span.size());
        std::transform(span.begin(), span.end(), std::back_inserter(new_data), [](int v) { return v * 2; });
        return HostBuffer(std::move(new_data));
    };
}

void BM_DistributedHostBufferSequentialTransform(benchmark::State& state) {
    auto buffer = create_distributed_host_buffer(state.range(0));
    auto transform_fn = get_transform_fn();

    for ([[maybe_unused]] auto _ : state) {
        auto transformed_buffer =
            buffer.transform(transform_fn, DistributedHostBuffer::ProcessShardExecutionPolicy::SEQUENTIAL);
        benchmark::DoNotOptimize(transformed_buffer);
    }
}

void BM_DistributedHostBufferParallelTransform(benchmark::State& state) {
    auto buffer = create_distributed_host_buffer(state.range(0));
    auto transform_fn = get_transform_fn();

    for ([[maybe_unused]] auto _ : state) {
        auto transformed_buffer =
            buffer.transform(transform_fn, DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
        benchmark::DoNotOptimize(transformed_buffer);
    }
}

BENCHMARK(BM_DistributedHostBufferSequentialTransform)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);
BENCHMARK(BM_DistributedHostBufferParallelTransform)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);

}  // namespace
}  // namespace tt::tt_metal
