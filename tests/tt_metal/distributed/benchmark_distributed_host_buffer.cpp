#include <benchmark/benchmark.h>

#include <tt-metalium/distributed_host_buffer.hpp>

#include <vector>
#include <numeric>
#include <algorithm>

namespace tt::tt_metal {
namespace {

constexpr size_t kNumShards = 64;
constexpr size_t kShardSize = 1 << 20;

void BM_DistributedHostBufferParallelTransform(benchmark::State& state) {
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(kNumShards));
    std::vector<int> data(kShardSize);
    std::iota(data.begin(), data.end(), 0);

    for (size_t i = 0; i < kNumShards; ++i) {
        buffer.emplace_shard(distributed::MeshCoordinate(i), [&data]() { return HostBuffer(data); });
    }

    for (auto _ : state) {
        buffer.transform(
            [](const HostBuffer& buffer) {
                auto span = buffer.view_as<int>();
                std::vector<int> new_data;
                new_data.reserve(span.size());
                std::transform(span.begin(), span.end(), std::back_inserter(new_data), [](int val) { return val * 2; });
                return HostBuffer(std::move(new_data));
            },
            DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
    }
}

BENCHMARK(BM_DistributedHostBufferParallelTransform)->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace tt::tt_metal
