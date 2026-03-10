// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <fmt/format.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <memory>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <tt-logger/tt-logger.hpp>
#include <benchmark/benchmark.h>
#include "context/metal_context.hpp"
#include "mesh_coord.hpp"
#include <llrt/tt_cluster.hpp>

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

static constexpr uint64_t KB = 1024;
static constexpr uint64_t MB = 1024 * KB;
static constexpr uint64_t GB = 1024 * MB;
using ElementType = uint32_t;
static constexpr uint32_t ElementSize = sizeof(ElementType);

static const std::vector<int64_t> PAGE_SIZE_ARGS = benchmark::CreateRange(32, 2048, 2);
static constexpr uint64_t max_transfer_size{8 * GB};
static const std::vector<int64_t> TRANSFER_SIZE_ARGS = {64 * MB};

static constexpr std::array<BufferType, 2> BUFFER_TYPES = {BufferType::DRAM, BufferType::L1};
static const std::vector<int64_t> BUFFER_TYPE_ARGS = {0, 1};

// For sharded benchmarks: fixed page size and contiguity control
static constexpr int64_t FIXED_PAGE_SIZE = 1024;
static const std::vector<int64_t> CONTIGUITY_ARGS = {1, 2, 4, 8, 16, 32, 64, 128};

static void BM_write(
    benchmark::State& state,
    const std::shared_ptr<MeshDevice>& mesh_device,
    const std::vector<ElementType>& host_buffer) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    [[maybe_unused]] auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Write Benchmark for Page Size: {}, Transfer Size: {}, Buffer Type: {}, Device ID: {}",
        page_size,
        transfer_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        device_id);

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{transfer_size},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        mesh_device.get());

    for ([[maybe_unused]] auto _ : state) {
        EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), device_buffer, host_buffer, true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

static void BM_write_pinned_memory(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    [[maybe_unused]] auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running WritePinnedMemory Benchmark for Page Size: {}, Transfer Size: {}, Buffer Type: {}, Device ID: {}",
        page_size,
        transfer_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        device_id);

    // Check if memory pinning with NOC mapping is supported
    if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
        state.SkipWithError("Memory pinning with NOC mapping is not supported on this device");
        return;
    }

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{transfer_size},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        mesh_device.get());

    // Allocate destination host buffer with 16-byte alignment
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    TT_ASSERT(
        device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0,
        "Source vector alignment {} must be divisible by PCIE read alignment {}",
        device_read_align,
        hal.get_read_alignment(HalMemType::HOST));
    auto src_storage = std::make_shared<std::vector<uint8_t, tt::stl::aligned_allocator<uint8_t, device_read_align>>>(
        static_cast<std::size_t>(transfer_size));
    void* aligned_ptr = reinterpret_cast<void*>(src_storage->data());

    // Create HostBuffer on top of aligned memory
    HostBuffer host_buffer(
        tt::stl::Span<std::uint8_t>(src_storage->data(), static_cast<std::size_t>(transfer_size)),
        MemoryPin(src_storage));

    // Pin the aligned host memory region for the shard
    auto coord = MeshCoordinate(0, 0);
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_mem =
        experimental::PinnedMemory::Create(*mesh_device, coordinate_range_set, host_buffer, /*map_to_noc=*/true);

    // Prepare the read transfer using pinned memory
    auto write_transfer = distributed::ShardDataTransfer(coord)
                              .host_data(aligned_ptr)
                              .region(BufferRegion(0, static_cast<std::size_t>(transfer_size)));
    experimental::ShardDataTransferSetPinnedMemory(write_transfer, pinned_mem);

    for (auto _ : state) {
        bool blocking = true;
        mesh_device->mesh_command_queue().enqueue_write_shards(device_buffer, {write_transfer}, blocking);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

// Helper: create a WIDTH_SHARDED MeshBuffer with controlled contiguity.
// Returns {device_buffer, actual_buf_size}, or {nullptr, 0} on skip.
//
// WIDTH_SHARDED: each core owns a vertical strip of width = contiguity pages.
//
// tensor2d = {rows, cols} where host_page = row * cols + col
//   rows  = pages_per_core / c        (number of contiguous chunks per core)
//   cols  = num_cores * c              (total width)
//
// shard_in_pages = shard_shape / page_shape = {rows, c}
//   -> core k gets columns [k*c, (k+1)*c) across all rows
//   -> each row contributes c contiguous host pages to that core
//   -> c * page_size contiguous bytes per chunk
//
static std::pair<std::shared_ptr<MeshBuffer>, uint64_t> create_sharded_buffer(
    benchmark::State& state,
    const std::shared_ptr<MeshDevice>& mesh_device,
    int64_t transfer_size,
    int64_t contiguity_factor,
    BufferType buffer_type) {
    auto page_size = FIXED_PAGE_SIZE;

    CoreCoord core_grid_size = (buffer_type == BufferType::DRAM) ? mesh_device->dram_grid_size()
                                                                 : mesh_device->compute_with_storage_grid_size();
    uint64_t total_pages = transfer_size / page_size;
    uint64_t num_cores = core_grid_size.x * core_grid_size.y;
    uint64_t pages_per_core = total_pages / num_cores;

    if (pages_per_core == 0) {
        state.SkipWithError("Transfer size too small for the core grid");
        return {nullptr, 0};
    }

    if (contiguity_factor <= 0) {
        state.SkipWithError("Contiguity factor must be positive");
        return {nullptr, 0};
    }

    uint64_t effective_contiguity = std::min(static_cast<uint64_t>(contiguity_factor), pages_per_core);

    // Round pages_per_core down so it's divisible by effective_contiguity
    pages_per_core = (pages_per_core / effective_contiguity) * effective_contiguity;
    if (pages_per_core == 0) {
        state.SkipWithError("Contiguity factor too large for available pages per core");
        return {nullptr, 0};
    }

    uint64_t actual_buf_size = pages_per_core * num_cores * page_size;

    uint32_t shard_height = pages_per_core / effective_contiguity;
    uint32_t tensor_width = num_cores * effective_contiguity;

    CoreRangeSet core_sets({CoreRange(CoreCoord(0, 0), CoreCoord(core_grid_size.x - 1, core_grid_size.y - 1))});
    std::array<uint32_t, 2> shard_shape = {shard_height, static_cast<uint32_t>(effective_contiguity * page_size)};
    std::array<uint32_t, 2> page_shape_array = {1, static_cast<uint32_t>(page_size)};
    std::array<uint32_t, 2> tensor2d_shape_in_pages = {shard_height, tensor_width};

    log_debug(
        tt::LogTest,
        "Sharded buffer parameters: shard_shape=[{}, {}], tensor2d_shape_in_pages=[{}, {}], "
        "page_size={}, pages_per_core={}, num_cores={}, actual_buf_size={}, "
        "contiguity_factor={}, effective_contiguity={}",
        shard_shape[0],
        shard_shape[1],
        tensor2d_shape_in_pages[0],
        tensor2d_shape_in_pages[1],
        page_size,
        pages_per_core,
        num_cores,
        actual_buf_size,
        contiguity_factor,
        effective_contiguity);

    ShardSpecBuffer shard_spec(
        core_sets, shard_shape, ShardOrientation::ROW_MAJOR, page_shape_array, tensor2d_shape_in_pages);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = page_size,
        .buffer_type = buffer_type,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::WIDTH_SHARDED),
        .bottom_up = false};

    ReplicatedBufferConfig global_buffer_config{.size = actual_buf_size};

    auto device_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
    return {device_buffer, actual_buf_size};
}

static void BM_write_sharded(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto transfer_size = state.range(0);
    auto contiguity_factor = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];

    auto [device_buffer, actual_buf_size] =
        create_sharded_buffer(state, mesh_device, transfer_size, contiguity_factor, buffer_type);
    if (!device_buffer) {
        return;
    }

    std::vector<uint8_t> host_buffer(static_cast<std::size_t>(actual_buf_size));

    auto write_transfer = distributed::ShardDataTransfer(MeshCoordinate(0, 0))
                              .host_data(host_buffer.data())
                              .region(BufferRegion(0, static_cast<std::size_t>(actual_buf_size)));

    for (auto _ : state) {
        mesh_device->mesh_command_queue().enqueue_write_shards(device_buffer, {write_transfer}, /*blocking=*/true);
    }

    state.SetBytesProcessed(actual_buf_size * state.iterations());
}

static void BM_write_pinned_memory_sharded(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto transfer_size = state.range(0);
    auto contiguity_factor = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];

    if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
        state.SkipWithError("Memory pinning with NOC mapping is not supported on this device");
        return;
    }

    auto [device_buffer, actual_buf_size] =
        create_sharded_buffer(state, mesh_device, transfer_size, contiguity_factor, buffer_type);
    if (!device_buffer) {
        return;
    }

    // Allocate source host buffer with 64-byte alignment
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    TT_ASSERT(
        device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0,
        "Source vector alignment {} must be divisible by PCIE read alignment {}",
        device_read_align,
        hal.get_read_alignment(HalMemType::HOST));

    auto src_storage = std::make_shared<std::vector<uint8_t, tt::stl::aligned_allocator<uint8_t, device_read_align>>>(
        static_cast<std::size_t>(actual_buf_size));
    void* aligned_ptr = reinterpret_cast<void*>(src_storage->data());

    // Create HostBuffer on top of aligned memory
    HostBuffer host_buffer(
        tt::stl::Span<std::uint8_t>(src_storage->data(), static_cast<std::size_t>(actual_buf_size)),
        MemoryPin(src_storage));

    // Pin the aligned host memory region
    auto coord = MeshCoordinate(0, 0);
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_mem =
        experimental::PinnedMemory::Create(*mesh_device, coordinate_range_set, host_buffer, /*map_to_noc=*/true);

    auto write_transfer = distributed::ShardDataTransfer(coord)
                              .host_data(aligned_ptr)
                              .region(BufferRegion(0, static_cast<std::size_t>(actual_buf_size)));
    experimental::ShardDataTransferSetPinnedMemory(write_transfer, pinned_mem);

    bool used_pinned_memory = false;
    for (auto _ : state) {
        mesh_device->mesh_command_queue().enqueue_write_shards(device_buffer, {write_transfer}, /*blocking=*/false);
        used_pinned_memory = pinned_mem->lock_may_block();
        mesh_device->mesh_command_queue().finish();
    }

    state.SetLabel(used_pinned_memory ? "PinnedMemory" : "NoPinnedMemory");

    state.SetBytesProcessed(actual_buf_size * state.iterations());
}

static void BM_read(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto page_size = state.range(0);
    uint64_t transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    [[maybe_unused]] auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Read Benchmark for Page Size: {}, Transfer Size: {}, Buffer Type: {}, Device ID: {}",
        page_size,
        transfer_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        device_id);

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{transfer_size},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        mesh_device.get());
    std::vector<ElementType> host_buffer;

    for ([[maybe_unused]] auto _ : state) {
        // EnqueueReadMeshBuffer cannot read from a replicated buffer yet, have to use ReadShard
        ReadShard(mesh_device->mesh_command_queue(), host_buffer, device_buffer, MeshCoordinate(0, 0), true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

static void BM_read_pinned_memory(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    [[maybe_unused]] auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running ReadPinnedMemory Benchmark for Page Size: {}, Transfer Size: {}, Buffer Type: {}, Device ID: {}",
        page_size,
        transfer_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        device_id);

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{transfer_size},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        mesh_device.get());

    // Allocate destination host buffer with 16-byte alignment
    auto dst_storage = std::make_shared<vector_aligned<std::uint8_t>>(static_cast<std::size_t>(transfer_size), 0);
    void* aligned_ptr = reinterpret_cast<void*>(dst_storage->data());

    // Create HostBuffer on top of aligned memory
    HostBuffer host_buffer(
        tt::stl::Span<std::uint8_t>(dst_storage->data(), static_cast<std::size_t>(transfer_size)),
        MemoryPin(dst_storage));

    // Pin the aligned host memory region for the shard
    auto coord = MeshCoordinate(0, 0);
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_mem =
        experimental::PinnedMemory::Create(*mesh_device, coordinate_range_set, host_buffer, /*map_to_noc=*/true);

    // Prepare the read transfer using pinned memory
    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(aligned_ptr)
                             .region(BufferRegion(0, static_cast<std::size_t>(transfer_size)));
    experimental::ShardDataTransferSetPinnedMemory(read_transfer, pinned_mem);

    for (auto _ : state) {
        mesh_device->mesh_command_queue().enqueue_read_shards({read_transfer}, device_buffer, /*blocking=*/true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    // no need to initialize for bandwidth measurement, saves test initialization time
    std::vector<ElementType> host_buffer_max(max_transfer_size / ElementSize);
    auto available_device_ids = MetalContext::instance().get_cluster().all_chip_ids();

    TT_FATAL(available_device_ids.contains(0), "Device 0 not available");
    std::vector<ChipId> device_ids = {0};

    if (available_device_ids.contains(1)) {
        log_info(LogTest, "Device 1 available, enable testing on device 1 assuming it's a remote device");
        device_ids.push_back(1);
    } else {
        log_info(LogTest, "Device 1 is not available");
    }

    auto devices = MeshDevice::create_unit_meshes(device_ids);

    for (auto [device_id, device] : devices) {
        // Device ID embedded here for extraction
        auto benchmark_args = {PAGE_SIZE_ARGS, TRANSFER_SIZE_ARGS, BUFFER_TYPE_ARGS, {device_id}};
        auto sharded_benchmark_args = {TRANSFER_SIZE_ARGS, CONTIGUITY_ARGS, BUFFER_TYPE_ARGS, {device_id}};
        std::vector<std::string> benchmark_arg_names = {"page_size", "size", "type", "device"};
        std::vector<std::string> sharded_benchmark_arg_names = {"size", "contiguity", "type", "device"};
        auto compute_min = [](const std::vector<double>& v) -> double { return *std::min_element(v.begin(), v.end()); };
        auto compute_max = [](const std::vector<double>& v) -> double { return *std::max_element(v.begin(), v.end()); };
        // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
        // benchmark
        benchmark::RegisterBenchmark("Write", BM_write, device, host_buffer_max)
            ->ArgsProduct(benchmark_args)
            ->ArgNames(benchmark_arg_names)
            ->UseRealTime()
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);

        benchmark::RegisterBenchmark("Read", BM_read, device)
            ->ArgsProduct(benchmark_args)
            ->ArgNames(benchmark_arg_names)
            ->UseRealTime()
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);
        benchmark::RegisterBenchmark("WriteSharded", BM_write_sharded, device)
            ->ArgsProduct(sharded_benchmark_args)
            ->ArgNames(sharded_benchmark_arg_names)
            ->UseRealTime()
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);
        bool can_map_to_noc = experimental::GetMemoryPinningParameters(*devices[0]).can_map_to_noc;

        if (can_map_to_noc) {
            benchmark::RegisterBenchmark("ReadPinnedMemory", BM_read_pinned_memory, device)
                ->ArgsProduct(benchmark_args)
                ->ArgNames(benchmark_arg_names)
                ->UseRealTime()
                ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
                ->ComputeStatistics("min", compute_min)
                ->ComputeStatistics("max", compute_max);
            benchmark::RegisterBenchmark("WritePinnedMemory", BM_write_pinned_memory, device)
                ->ArgsProduct(benchmark_args)
                ->ArgNames(benchmark_arg_names)
                ->UseRealTime()
                ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
                ->ComputeStatistics("min", compute_min)
                ->ComputeStatistics("max", compute_max);
            benchmark::RegisterBenchmark("WritePinnedMemorySharded", BM_write_pinned_memory_sharded, device)
                ->ArgsProduct(sharded_benchmark_args)
                ->ArgNames(sharded_benchmark_arg_names)
                ->UseRealTime()
                ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
                ->ComputeStatistics("min", compute_min)
                ->ComputeStatistics("max", compute_max);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
