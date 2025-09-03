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

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/math.hpp>
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
// Transfer Size: 32 MB
// Buffer Type: DRAM, L1
// Sharding: Default (interleave), Height Sharded, Width Sharded
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

static constexpr auto KB = 1024;
static constexpr auto MB = 1024 * KB;

// This is the maximum transfer size we can test for on L1 for sharded cases, note the bandwidth improvement gained by
// increasing this transfer size is minimal (see spreadsheet above)
static constexpr int64_t TRANSFER_SIZE = 32 * MB;

// You can only pass std::vector<int64_t> to Google Benchmark's ArgsProduct, so we have to use const
// std::vector<int64_t> for some of the parameters.

// This is page sizes for non-sharded (interleaved) buffers
static const std::vector<int64_t> PAGE_SIZES = benchmark::CreateRange(32, 2048, 2);
// This is page sizes we test for sharded buffers
static const std::vector<int64_t> SHARDED_PAGE_SIZES = {2048, 4096};
static constexpr std::array<BufferType, 2> BUFFER_TYPES = {BufferType::DRAM, BufferType::L1};
// Interleaved is the default sharding config.
static constexpr std::array<TensorMemoryLayout, 3> SHARD_ORIENTATIONS = {
    TensorMemoryLayout::INTERLEAVED, TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED};

/**
 * Sharding related constants:
 *
 * At metal API level, we need to compute the BufferShardingArgs.
 * Which communicates how the buffer would be sliced at a per-core level as a tensor,
 * this involves injecting meta data about the tensor that lives in the buffer.
 *
 * Principly we test the throughput of arbitrary buffer transfer between host and device,
 * but to test for sharding, we must "makeup" the tensor's shape.
 *
 * A buffer is consisted of pages (this is the page size we are varying on),
 * a page is assumed acrossed the system to be built with 32x32 element tiles,
 * where a single element is any support datatype (e.g. float32, uint16, etc.).
 * This means a page is at least 32x32 elements large.
 *
 * In this context, novel page size like 32B would be unconstructable with sharding
 * (this means a single unit of data is smaller than a byte).
 *
 * A reasonable place to start for page size would be a page of a single tile of elements of the smallest datatype.
 * In this benchmark, we start with uint16, which would result in a 32x32x2B=2kb page size.
 * We test 4kb page size in addition to 2kb page size,
 * as this is farily commonly used and would be constructed using a tile of float32 values.
 *
 * Note that page size's impact on bandwidth exihibits diminishing returns at about 1kb as per the spreadsheet above.
 *
 * Given we've come up with the datatype,
 * we can define the shape of the tensor that is being transferred easily from TRANSFER_SIZE,
 * which would be:
 *  - a tall matrix of 2048x4096 of float32
 *  - a wide matrix of 4096x4096 of uint16
 *
 */

// Page shape
static constexpr std::uint32_t PAGE_SIDE = 32;
static constexpr std::array<std::uint32_t, 2> PAGE_SHAPE = {PAGE_SIDE, PAGE_SIDE};

// float32 of 2048x4096 is 32MB (transfer size)
static constexpr std::array<std::uint32_t, 2> ELEMENT_SHAPE_4K = {2048, 4096};
static_assert(sizeof(float) * ELEMENT_SHAPE_4K[0] * ELEMENT_SHAPE_4K[1] == TRANSFER_SIZE);
static_assert(sizeof(float) * PAGE_SIDE * PAGE_SIDE == 4096);
static_assert(ELEMENT_SHAPE_4K[0] % PAGE_SIDE == 0 && ELEMENT_SHAPE_4K[1] % PAGE_SIDE == 0);

// uint16 of 4096x4096 is 32MB (transfer size)
static constexpr std::array<std::uint32_t, 2> ELEMENT_SHAPE_2K = {4096, 4096};
static_assert(sizeof(std::uint16_t) * ELEMENT_SHAPE_2K[0] * ELEMENT_SHAPE_2K[1] == TRANSFER_SIZE);
static_assert(sizeof(std::uint16_t) * PAGE_SIDE * PAGE_SIDE == 2048);
static_assert(ELEMENT_SHAPE_2K[0] % PAGE_SIDE == 0 && ELEMENT_SHAPE_2K[1] % PAGE_SIDE == 0);

/**
 * Compute the element side after sharding across num_cores.
 *
 * The result needs to round up to a multiple of PAGE_SIDE.
 *
 * e.g. if a slice of 4096 needs to be split between 12 cores,
 * each core needs at least 341.3 elements,
 * that is a slice that needs to be at least 342 elements long.
 * Then we need to round it up to the nearest multiple of PAGE_SIDE, which would be 352.
 */
static constexpr auto compute_sharded_side_size(auto element_side, auto num_cores) {
    auto element_per_core = div_up(element_side, num_cores);
    auto pages_per_core = div_up(element_per_core, PAGE_SIDE);
    return pages_per_core * PAGE_SIDE;
}

// Quick test case
static_assert(compute_sharded_side_size(4096, 12) == 352);
static_assert(compute_sharded_side_size(PAGE_SIDE * 2 , 2) == PAGE_SIDE);

static constexpr auto compute_shard_shape(auto element_shape, auto num_cores, TensorMemoryLayout sharding)
    -> decltype(element_shape) {
    auto element_div_side = sharding == TensorMemoryLayout::HEIGHT_SHARDED ? element_shape[0] : element_shape[1];
    auto shard_side = compute_sharded_side_size(element_div_side, num_cores);
    auto shard_shape = element_shape;
    if (sharding == TensorMemoryLayout::HEIGHT_SHARDED) {
        shard_shape[0] = shard_side;
    } else {  // width sharded
        shard_shape[1] = shard_side;
    }
    return shard_shape;
}

static BufferShardingArgs create_sharding_args(
    auto mesh_device, auto page_size, BufferType buffer_type, TensorMemoryLayout sharding) {
    // Default behavior
    if (sharding == TensorMemoryLayout::INTERLEAVED) {
        return {};
    }

    auto grid_size =
        buffer_type == BufferType::L1 ? mesh_device->compute_with_storage_grid_size() : mesh_device->dram_grid_size();
    CoreRangeSet core_range_set{CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))};
    auto element_shape = page_size == 4096 ? ELEMENT_SHAPE_4K : ELEMENT_SHAPE_2K;

    ShardSpec shard_spec{core_range_set, compute_shard_shape(element_shape, core_range_set.num_cores(), sharding)};

    std::array<uint32_t, 2> tensor2d_shape_in_pages{element_shape[0] / PAGE_SIDE, element_shape[1] / PAGE_SIDE};
    ShardSpecBuffer shard_spec_buffer(shard_spec, PAGE_SHAPE, tensor2d_shape_in_pages);

    return {shard_spec_buffer, sharding};
}

static std::shared_ptr<MeshBuffer> create_device_buffer(
    std::shared_ptr<MeshDevice> mesh_device, int64_t page_size, BufferType buffer_type, TensorMemoryLayout sharding) {
    DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = buffer_type,
        .sharding_args = create_sharding_args(mesh_device, page_size, buffer_type, sharding)};
    return MeshBuffer::create(ReplicatedBufferConfig{TRANSFER_SIZE}, device_local_config, mesh_device.get());
}

// Google Benchmark entry point
static void BM_write(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto buffer_type = BUFFER_TYPES[state.range(1)];
    auto sharding = SHARD_ORIENTATIONS[state.range(2)];
    auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Write Benchmark for Page Size: {}, Buffer Type: {}, Sharding: {}, Device ID: {}",
        page_size,
        buffer_type,
        sharding,
        device_id);

    auto random_buffer_seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto host_buffer = create_random_vector_of_bfloat16(TRANSFER_SIZE, 1000, random_buffer_seed);

    auto device_buffer = create_device_buffer(mesh_device, page_size, buffer_type, sharding);

    for (auto _ : state) {
        EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), device_buffer, host_buffer, true);
    }

    state.SetBytesProcessed(TRANSFER_SIZE * state.iterations());
}

// Google Benchmark entry point
static void BM_read(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto buffer_type = BUFFER_TYPES[state.range(1)];
    auto sharding = SHARD_ORIENTATIONS[state.range(2)];
    auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Read Benchmark for Page Size: {}, Buffer Type: {}, Sharding: {}, Device ID: {}",
        page_size,
        buffer_type,
        sharding,
        device_id);

    auto device_buffer = create_device_buffer(mesh_device, page_size, buffer_type, sharding);
    // Zero-initialize to avoid memory allocation for this vector during benchmark
    std::vector<uint32_t> host_buffer(TRANSFER_SIZE / sizeof(uint32_t), 0);

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

    log_info(LogTest, "Available device IDs: {}", available_device_ids);
    log_info(LogTest, "Testing on device IDs: {}", device_ids);

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

        benchmark::RegisterBenchmark("sharded_Write", BM_write, device)->ArgsProduct(benchmark_args)->UseRealTime();
        benchmark::RegisterBenchmark("sharded_Read", BM_read, device)->ArgsProduct(benchmark_args)->UseRealTime();
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
