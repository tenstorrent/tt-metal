// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <fmt/format.h>
#include <stdint.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <enchantum/enchantum.hpp>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
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

// This is page sizes for non-sharded (interleaved) buffers
static const std::vector<int64_t> PAGE_SIZES = benchmark::CreateRange(32, 2048, 2);
// This is page sizes we test for sharded buffers
static const std::vector<int64_t> SHARDED_PAGE_SIZES = {4096};
static constexpr std::array<BufferType, 2> BUFFER_TYPES = {BufferType::DRAM, BufferType::L1};
// Interleaved is the default sharding config.
static constexpr std::array<TensorMemoryLayout, 3> SHARD_ORIENTATIONS = {
    TensorMemoryLayout::INTERLEAVED, TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED};

// Short-hand for page dimension = 32x32
static constexpr std::uint32_t PAGE_SIDE = 32;

/**
 * Compute the element side after sharding across num_cores.
 *
 * The result needs to round up to a multiple of PAGE_SIDE.
 */
static constexpr auto compute_shard_side(auto element_side, auto num_cores) {
    auto shard_side = element_side / num_cores;
    shard_side += (element_side % num_cores > 0 ? 1 : 0);
    // Rounding up to the nearest multiple of PAGE_HEIGHT
    shard_side += (PAGE_SIDE - shard_side % PAGE_SIDE);
    return shard_side;
}

// Quick test case
static_assert(compute_shard_side(4096, 12) == 352);

static std::shared_ptr<MeshBuffer> create_device_buffer(
    std::shared_ptr<MeshDevice> mesh_device, int64_t page_size, BufferType buffer_type, TensorMemoryLayout sharding) {
    // float32 of 4096x4096 is 64MB (transfer size)
    static constexpr std::uint32_t ELEMENT_SHAPE_SIDE = 4096;
    static_assert(ELEMENT_SHAPE_SIDE * ELEMENT_SHAPE_SIDE * sizeof(float) == TRANSFER_SIZE);

    BufferShardingArgs sharding_args;

    if (sharding != TensorMemoryLayout::INTERLEAVED) {
        auto grid_size = buffer_type == BufferType::L1 ? mesh_device->compute_with_storage_grid_size()
                                                       : mesh_device->dram_grid_size();
        CoreRangeSet core_range_set({CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))});
        auto total_num_cores = core_range_set.num_cores();
        auto shard_side = compute_shard_side(ELEMENT_SHAPE_SIDE, total_num_cores);

        std::array<uint32_t, 2> shard_shape;
        if (sharding == TensorMemoryLayout::HEIGHT_SHARDED) {
            shard_shape = {shard_side, ELEMENT_SHAPE_SIDE};
        } else {  // width sharded
            shard_shape = {ELEMENT_SHAPE_SIDE, shard_side};
        }

        ShardSpec shard_spec(core_range_set, shard_shape);

        log_info(LogTest, "shard_shape: {}", shard_shape);

        std::array<uint32_t, 2> tensor2d_shape_in_pages{ELEMENT_SHAPE_SIDE / PAGE_SIDE, ELEMENT_SHAPE_SIDE / PAGE_SIDE};
        ShardSpecBuffer shard_spec_buffer(shard_spec, {PAGE_SIDE, PAGE_SIDE}, tensor2d_shape_in_pages);

        log_info(LogTest, "tensor2d_shape_in_pages: {}", tensor2d_shape_in_pages);

        sharding_args = BufferShardingArgs(shard_spec_buffer, TensorMemoryLayout::HEIGHT_SHARDED);
    }

    DeviceLocalBufferConfig device_local_config{
        .page_size = page_size, .buffer_type = buffer_type, .sharding_args = sharding_args};

    return MeshBuffer::create(ReplicatedBufferConfig{TRANSFER_SIZE}, device_local_config, mesh_device.get());
}

static void BM_write(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto buffer_type = BUFFER_TYPES[state.range(1)];
    auto sharding = SHARD_ORIENTATIONS[state.range(2)];
    auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Write Benchmark for Page Size: {}, Buffer Type: {}, Sharding: {}, Device ID: {}",
        page_size,
        enchantum::to_string(buffer_type),
        enchantum::to_string(sharding),
        device_id);

    auto random_buffer_seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto host_buffer = create_random_vector_of_bfloat16(TRANSFER_SIZE, 1000, random_buffer_seed);

    auto device_buffer = create_device_buffer(mesh_device, page_size, buffer_type, sharding);

    for (auto _ : state) {
        EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), device_buffer, host_buffer, true);
    }

    state.SetBytesProcessed(TRANSFER_SIZE * state.iterations());
}

static void BM_read(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto buffer_type = BUFFER_TYPES[state.range(1)];
    auto sharding = SHARD_ORIENTATIONS[state.range(2)];
    auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Read Benchmark for Page Size: {}, Buffer Type: {}, Sharding: {}, Device ID: {}",
        page_size,
        enchantum::to_string(buffer_type),
        enchantum::to_string(sharding),
        device_id);

    auto device_buffer = create_device_buffer(mesh_device, page_size, buffer_type, sharding);
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

    auto buffer_type_args = benchmark::CreateDenseRange(0, BUFFER_TYPES.size() - 1, 1);

    // Benchmark with no special sharding config
    for (auto [device_id, device] : devices) {
        auto benchmark_args = {
            PAGE_SIZES,
            buffer_type_args,
            // We only consider interleaved buffer here
            {0},
            {device_id}};
        // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
        // benchmark
        benchmark::RegisterBenchmark("Write", BM_write, device)->ArgsProduct(benchmark_args)->UseRealTime();
        benchmark::RegisterBenchmark("Read", BM_read, device)->ArgsProduct(benchmark_args)->UseRealTime();
    }

    // Benchmark with customized sharding config
    for (auto [device_id, device] : devices) {
        auto sharding_args = benchmark::CreateDenseRange(0, SHARD_ORIENTATIONS.size() - 1, 1);
        auto benchmark_args = {SHARDED_PAGE_SIZES, buffer_type_args, sharding_args, {device_id}};

        benchmark::RegisterBenchmark("Write", BM_write, device)->ArgsProduct(benchmark_args)->UseRealTime();
        benchmark::RegisterBenchmark("Read", BM_read, device)->ArgsProduct(benchmark_args)->UseRealTime();
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
