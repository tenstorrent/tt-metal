// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <fmt/format.h>
#include <stdint.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include <benchmark/benchmark.h>
#include "context/metal_context.hpp"
#include "mesh_coord.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of host-to-device data transfer and
// device-to-host data transfer. It uses EnqueueWriteMeshBuffer and
// ReadShard APIs to transfer the data. The device memory object
// (mesh buffer) will be in DRAM.
//
// Benchmark Matrix:
// Read & Write
// Page Size (Bytes): 32, 64, 128, 256, 512, 1024, 2048
// Transfer Size: 64 MB
// Buffer Type: DRAM, L1
// Sharding: Default (interleave), Horizontal
// Device: 0 (local), 1 (remote) (when possible)
//
////////////////////////////////////////////////////////////////////////////////

/*
 * The upper bound of the page size & transfer size is derived experimentally on July 31st, 2025.
 * They are set at the point where diminishing returns are observed.
 *
 * Link to data sheet as follows:
 * https://docs.google.com/spreadsheets/d/1zy1teJtgf7hsMMdgy5uIOtcuI73AGVqy4lnyYwL7YFQ/edit
 */

static const auto KB = 1024;
static const auto MB = 1024 * KB;

static const int64_t TRANSFER_SIZE = 64 * MB;

static const std::vector<int64_t> PAGE_SIZE_ARGS = benchmark::CreateRange(32, 2048, 2);

static constexpr std::array<BufferType, 2> BUFFER_TYPES = {BufferType::DRAM, BufferType::L1};
static const std::vector<int64_t> BUFFER_TYPE_ARGS = {0, 1};

static const std::vector<int64_t> SHARD_ENABLE_ARGS = {false, true};

static BufferShardingArgs get_sharding_args(std::shared_ptr<MeshDevice> mesh_device, BufferType buffer_type) {
    // These are pulled out of TTNN for WH,
    // This is modeled with a 2048 x 1024 tensor,
    // which if containing uint32, would occupy 64MB (our transfer size).
    static const std::array<uint32_t, 2> PAGE_SHAPE = {32, 32};
    static const std::array<uint32_t, 2> TENSOR2D_SHAPE_IN_PAGES = {64, 32};

    static const std::array<uint32_t, 2> L1_SHARD_SHAPE = {32, 1024};
    static const std::array<uint32_t, 2> DRAM_SHARD_SHAPE = {192, 1024};

    auto device_grid_size =
        buffer_type == BufferType::L1 ? mesh_device->compute_with_storage_grid_size() : mesh_device->dram_grid_size();
    CoreRangeSet core_range_set(
        {CoreRange(CoreCoord(0, 0), CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))});
    ShardSpec shard_spec(core_range_set, buffer_type == BufferType::L1 ? L1_SHARD_SHAPE : DRAM_SHARD_SHAPE);
    ShardSpecBuffer shard_spec_buffer(shard_spec, PAGE_SHAPE, TENSOR2D_SHAPE_IN_PAGES);

    return BufferShardingArgs(shard_spec_buffer, TensorMemoryLayout::HEIGHT_SHARDED);
}

static void BM_write(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto buffer_type = BUFFER_TYPES[state.range(1)];
    auto shard_enable = state.range(2);
    auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Write Benchmark for Page Size: {}, Buffer Type: {}, Shard Enable: {}, Device ID: {}",
        page_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        shard_enable,
        device_id);

    auto random_buffer_seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto host_buffer = create_random_vector_of_bfloat16(TRANSFER_SIZE, 1000, random_buffer_seed);

    BufferShardingArgs sharding_args;
    if (shard_enable) {
        sharding_args = get_sharding_args(mesh_device, buffer_type);
    }

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{TRANSFER_SIZE},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type, .sharding_args = sharding_args},
        mesh_device.get());

    for (auto _ : state) {
        EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), device_buffer, host_buffer, true);
    }

    state.SetBytesProcessed(TRANSFER_SIZE * state.iterations());
}

static void BM_read(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto buffer_type = BUFFER_TYPES[state.range(1)];
    auto shard_enable = state.range(2);
    auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Read Benchmark for Page Size: {}, Buffer Type: {}, Shard Enable: {}, Device ID: {}",
        page_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        shard_enable,
        device_id);

    BufferShardingArgs sharding_args;
    if (shard_enable) {
        sharding_args = get_sharding_args(mesh_device, buffer_type);
    }

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{TRANSFER_SIZE},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type, .sharding_args = sharding_args},
        mesh_device.get());
    std::vector<uint32_t> host_buffer;

    for (auto _ : state) {
        // EnqueueReadMeshBuffer cannot read from a replicated buffer yet, have to use ReadShard
        ReadShard(mesh_device->mesh_command_queue(), host_buffer, device_buffer, MeshCoordinate(0, 0), true);
    }

    state.SetBytesProcessed(TRANSFER_SIZE * state.iterations());
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    auto available_device_ids = MetalContext::instance().get_cluster().all_chip_ids();

    TT_ASSERT(available_device_ids.contains(0));
    std::vector<chip_id_t> device_ids = {0};

    if (available_device_ids.contains(1)) {
        log_info(LogTest, "Device 1 available, enable testing on device 1 assuming it's a remote device");
        device_ids.push_back(1);
    } else {
        log_info(LogTest, "Device 1 is not available");
    }

    auto devices = MeshDevice::create_unit_meshes(device_ids);
    for (auto [device_id, device] : devices) {
        // Device ID embedded here for extraction
        auto benchmark_args = {PAGE_SIZE_ARGS, BUFFER_TYPE_ARGS, SHARD_ENABLE_ARGS, {device_id}};
        // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
        // benchmark
        benchmark::RegisterBenchmark("Write", BM_write, device)->ArgsProduct(benchmark_args)->UseRealTime();
        benchmark::RegisterBenchmark("Read", BM_read, device)->ArgsProduct(benchmark_args)->UseRealTime();
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
