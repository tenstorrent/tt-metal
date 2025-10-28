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

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/pinned_memory.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/vector_aligned.hpp>
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

static constexpr auto num_test_repetitions = 11;

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

    for (auto _ : state) {
        EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), device_buffer, host_buffer, true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
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

    for (auto _ : state) {
        // EnqueueReadMeshBuffer cannot read from a replicated buffer yet, have to use ReadShard
        ReadShard(mesh_device->mesh_command_queue(), host_buffer, device_buffer, MeshCoordinate(0, 0), true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

static void BM_read_pinned_memory(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
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

    // Check if memory pinning with NOC mapping is supported
    if (!mesh_device->get_memory_pinning_parameters().can_map_to_noc) {
        state.SkipWithError("Memory pinning with NOC mapping is not supported on this device");
        return;
    }

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{transfer_size},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        mesh_device.get());

    // Allocate destination host buffer with 16-byte alignment
    auto dst_storage = std::make_shared<vector_aligned<std::uint8_t>>(static_cast<std::size_t>(transfer_size), 0);
    auto aligned_ptr = reinterpret_cast<void*>(dst_storage->data());

    // Create HostBuffer on top of aligned memory
    HostBuffer host_buffer(
        tt::stl::Span<std::uint8_t>(dst_storage->data(), static_cast<std::size_t>(transfer_size)),
        MemoryPin(dst_storage));

    // Pin the aligned host memory region for the shard
    auto coord = MeshCoordinate(0, 0);
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_unique = mesh_device->pin_memory(coordinate_range_set, host_buffer, /*map_to_noc=*/true);
    std::shared_ptr<PinnedMemory> pinned_mem = std::move(pinned_unique);

    // Prepare the read transfer using pinned memory
    MeshCommandQueue::ShardDataTransfer read_transfer = {
        .shard_coord = coord,
        .host_data = aligned_ptr,
        .pinned_memory = pinned_mem,
        .region = BufferRegion(0, static_cast<std::size_t>(transfer_size)),
    };

    for (auto _ : state) {
        bool blocking = true;
        mesh_device->mesh_command_queue().enqueue_read_shards({read_transfer}, device_buffer, blocking);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

static void BM_write_pinned_memory(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
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
    if (!mesh_device->get_memory_pinning_parameters().can_map_to_noc) {
        state.SkipWithError("Memory pinning with NOC mapping is not supported on this device");
        return;
    }

    auto device_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{transfer_size},
        DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        mesh_device.get());

    // Allocate destination host buffer with 16-byte alignment
    auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{32};
    TT_ASSERT(
        device_read_align == hal.get_read_alignment(HalMemType::HOST),
        "Source vector alignment must be equal to PCIE read alignment: {}",
        hal.get_read_alignment(HalMemType::HOST));
    auto src_storage = std::make_shared<std::vector<uint8_t, tt::stl::aligned_allocator<uint8_t, device_read_align>>>(
        static_cast<std::size_t>(transfer_size));
    auto aligned_ptr = reinterpret_cast<void*>(src_storage->data());

    // Create HostBuffer on top of aligned memory
    HostBuffer host_buffer(
        tt::stl::Span<std::uint8_t>(src_storage->data(), static_cast<std::size_t>(transfer_size)),
        MemoryPin(src_storage));

    // Pin the aligned host memory region for the shard
    auto coord = MeshCoordinate(0, 0);
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto pinned_unique = mesh_device->pin_memory(coordinate_range_set, host_buffer, /*map_to_noc=*/true);
    std::shared_ptr<PinnedMemory> pinned_mem = std::move(pinned_unique);

    // Prepare the read transfer using pinned memory
    MeshCommandQueue::ShardDataTransfer write_transfer = {
        .shard_coord = coord,
        .host_data = aligned_ptr,
        .pinned_memory = pinned_mem,
        .region = BufferRegion(0, static_cast<std::size_t>(transfer_size)),
    };

    for (auto _ : state) {
        bool blocking = true;
        mesh_device->mesh_command_queue().enqueue_write_shards(device_buffer, {write_transfer}, blocking);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // no need to initialize for bandwidth measurement, saves test initialization time
    std::vector<ElementType> host_buffer_max(max_transfer_size / ElementSize);
    auto available_device_ids = MetalContext::instance().get_cluster().all_chip_ids();

    TT_ASSERT(available_device_ids.contains(0));
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
        auto compute_min = [](const std::vector<double>& v) -> double { return *std::min_element(v.begin(), v.end()); };
        auto compute_max = [](const std::vector<double>& v) -> double { return *std::max_element(v.begin(), v.end()); };
        // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
        // benchmark
        benchmark::RegisterBenchmark("Write", BM_write, device, host_buffer_max)
            ->ArgsProduct(benchmark_args)
            ->UseRealTime()
            ->Repetitions(num_test_repetitions)
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);

        benchmark::RegisterBenchmark("Read", BM_read, device)
            ->ArgsProduct(benchmark_args)
            ->UseRealTime()
            ->Repetitions(num_test_repetitions)
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);

        benchmark::RegisterBenchmark("ReadPinnedMemory", BM_read_pinned_memory, device)
            ->ArgsProduct(benchmark_args)
            ->UseRealTime()
            ->Repetitions(num_test_repetitions)
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);

        benchmark::RegisterBenchmark("WritePinnedMemory", BM_write_pinned_memory, device)
            ->ArgsProduct(benchmark_args)
            ->UseRealTime()
            ->Repetitions(num_test_repetitions)
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
