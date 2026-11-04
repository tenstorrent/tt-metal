// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host cost of the MeshCommandQueue operations that fan out over the local devices through the dispatch and
// reader thread pools: event records, replicated and per-shard buffer writes, and per-shard reads. Opens the
// whole mesh.

#include <benchmark/benchmark.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shard_data_transfer.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

MeshDevice& mesh_device() {
    static std::shared_ptr<MeshDevice> device = MeshDevice::create(MeshDeviceConfig(std::nullopt));
    return *device;
}

constexpr uint32_t PAGE_SIZE = 1024;
constexpr uint32_t DATUM_SIZE = 4;

std::shared_ptr<MeshBuffer> replicated_buffer(uint32_t bytes_per_device) {
    const DeviceLocalBufferConfig local{.page_size = PAGE_SIZE, .buffer_type = BufferType::DRAM};
    return MeshBuffer::create(ReplicatedBufferConfig{.size = bytes_per_device}, local, &mesh_device());
}

std::shared_ptr<MeshBuffer> sharded_buffer(uint32_t bytes_per_device) {
    const auto& shape = mesh_device().shape();
    const uint32_t shard_rows = 32;
    const uint32_t shard_cols = bytes_per_device / (shard_rows * DATUM_SIZE);
    const distributed::ShardedBufferConfig config{
        .global_size = static_cast<DeviceAddr>(bytes_per_device) * mesh_device().num_devices(),
        .global_buffer_shape = {shard_rows * shape[0], shard_cols * shape[1]},
        .shard_shape = {shard_rows, shard_cols}};
    const DeviceLocalBufferConfig local{.page_size = PAGE_SIZE, .buffer_type = BufferType::DRAM};
    return MeshBuffer::create(config, local, &mesh_device());
}

std::vector<ShardDataTransfer> transfers_for_every_device(void* host_data) {
    std::vector<ShardDataTransfer> transfers;
    for (const auto& coord : MeshCoordinateRange(mesh_device().shape())) {
        transfers.push_back(ShardDataTransfer{coord}.host_data(host_data));
    }
    return transfers;
}

void BM_RecordEvent(benchmark::State& state) {
    auto& cq = mesh_device().mesh_command_queue();
    for ([[maybe_unused]] auto _ : state) {
        benchmark::DoNotOptimize(cq.enqueue_record_event());
    }
    Finish(cq);
}

void BM_WriteReplicated(benchmark::State& state) {
    const auto bytes = static_cast<uint32_t>(state.range(0));
    auto buffer = replicated_buffer(bytes);
    std::vector<uint32_t> data(bytes / sizeof(uint32_t), 1);
    auto& cq = mesh_device().mesh_command_queue();
    for ([[maybe_unused]] auto _ : state) {
        EnqueueWriteMeshBuffer(cq, buffer, data, /*blocking=*/false);
    }
    Finish(cq);
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes) * mesh_device().num_devices());
}

void BM_WriteShards(benchmark::State& state) {
    const auto bytes = static_cast<uint32_t>(state.range(0));
    auto buffer = sharded_buffer(bytes);
    std::vector<uint32_t> data(bytes / sizeof(uint32_t), 1);
    const auto transfers = transfers_for_every_device(data.data());
    auto& cq = mesh_device().mesh_command_queue();
    for ([[maybe_unused]] auto _ : state) {
        cq.enqueue_write_shards(buffer, transfers, /*blocking=*/false);
    }
    Finish(cq);
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes) * mesh_device().num_devices());
}

// Blocking, so this includes the device round trip.
void BM_ReadShards(benchmark::State& state) {
    const auto bytes = static_cast<uint32_t>(state.range(0));
    auto buffer = sharded_buffer(bytes);
    std::vector<std::vector<uint32_t>> data(
        mesh_device().num_devices(), std::vector<uint32_t>(bytes / sizeof(uint32_t)));
    std::vector<ShardDataTransfer> transfers;
    uint32_t i = 0;
    for (const auto& coord : MeshCoordinateRange(mesh_device().shape())) {
        transfers.push_back(ShardDataTransfer{coord}.host_data(data[i++].data()));
    }
    auto& cq = mesh_device().mesh_command_queue();
    for ([[maybe_unused]] auto _ : state) {
        cq.enqueue_read_shards(transfers, buffer, /*blocking=*/true);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes) * mesh_device().num_devices());
}

BENCHMARK(BM_RecordEvent)
    ->Name("BM_MeshFanOut/RecordEvent")
    ->Iterations(20000)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
// Model-sized (a few KB per device) and larger writes.
BENCHMARK(BM_WriteReplicated)
    ->Name("BM_MeshFanOut/WriteReplicated")
    ->ArgName("bytes_per_device")
    ->Arg(4 << 10)
    ->Arg(64 << 10)
    ->Iterations(5000)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_WriteShards)
    ->Name("BM_MeshFanOut/WriteShards")
    ->ArgName("bytes_per_device")
    ->Arg(4 << 10)
    ->Arg(64 << 10)
    ->Iterations(5000)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ReadShards)
    ->Name("BM_MeshFanOut/ReadShards")
    ->ArgName("bytes_per_device")
    ->Arg(4 << 10)
    ->Arg(64 << 10)
    ->Iterations(1000)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace
