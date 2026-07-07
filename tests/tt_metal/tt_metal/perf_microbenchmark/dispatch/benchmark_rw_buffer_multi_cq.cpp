// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <exception>
#include <fmt/base.h>
#include <fmt/format.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
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
// Multi-CQ variant of benchmark_rw_buffer.
//
// This test measures the aggregate bandwidth of host<->device data transfer when
// the *exact same* transfer is issued on two command queues (CQ 0 and CQ 1)
// concurrently. Each iteration enqueues the transfer non-blocking on both CQs and
// then waits for both to finish, so the two transfers overlap in flight. The
// reported bytes_per_second is the combined throughput of both CQs.
//
// It mirrors the Read/Write matrix of benchmark_rw_buffer:
//   Page Size (Bytes): 32, 64, 128, 256, 512, 1024, 2048
//   Transfer Size (per CQ): 64 MB
//   Buffer Type: DRAM, L1
//   Device: 0 (local), 1 (remote) (when possible)
//
// The device is opened with 2 hardware command queues. If the target hardware
// does not expose a second CQ, the benchmarks are skipped.
////////////////////////////////////////////////////////////////////////////////

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

// Create a replicated device buffer, or skip the benchmark on allocation failure
// (e.g. two large L1 buffers may not fit).
static std::shared_ptr<MeshBuffer> try_create_buffer(
    benchmark::State& state,
    const std::shared_ptr<MeshDevice>& mesh_device,
    int64_t transfer_size,
    int64_t page_size,
    BufferType buffer_type) {
    try {
        return MeshBuffer::create(
            ReplicatedBufferConfig{static_cast<uint64_t>(transfer_size)},
            DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
            mesh_device.get());
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return nullptr;
    }
}

static void BM_write_dual_cq(
    benchmark::State& state,
    const std::shared_ptr<MeshDevice>& mesh_device,
    const std::vector<ElementType>& host_buffer) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    [[maybe_unused]] auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Dual-CQ Write Benchmark for Page Size: {}, Transfer Size: {}, Buffer Type: {}, Device ID: {}",
        page_size,
        transfer_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        device_id);

    // One buffer per CQ so the two transfers are fully independent.
    auto device_buffer_0 = try_create_buffer(state, mesh_device, transfer_size, page_size, buffer_type);
    auto device_buffer_1 = try_create_buffer(state, mesh_device, transfer_size, page_size, buffer_type);
    if (!device_buffer_0 || !device_buffer_1) {
        return;
    }

    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);

    for ([[maybe_unused]] auto _ : state) {
        // Issue the identical write on both CQs non-blocking so they overlap, then wait for both.
        EnqueueWriteMeshBuffer(cq0, device_buffer_0, host_buffer, /*blocking=*/false);
        EnqueueWriteMeshBuffer(cq1, device_buffer_1, host_buffer, /*blocking=*/false);
        cq0.finish();
        cq1.finish();
    }

    // Both CQs each move transfer_size bytes per iteration.
    state.SetBytesProcessed(2 * transfer_size * state.iterations());
}

static void BM_read_dual_cq(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto page_size = state.range(0);
    uint64_t transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    [[maybe_unused]] auto device_id = state.range(3);

    log_debug(
        LogTest,
        "Running Dual-CQ Read Benchmark for Page Size: {}, Transfer Size: {}, Buffer Type: {}, Device ID: {}",
        page_size,
        transfer_size,
        buffer_type == BufferType::DRAM ? "DRAM" : "L1",
        device_id);

    auto device_buffer_0 = try_create_buffer(state, mesh_device, transfer_size, page_size, buffer_type);
    auto device_buffer_1 = try_create_buffer(state, mesh_device, transfer_size, page_size, buffer_type);
    if (!device_buffer_0 || !device_buffer_1) {
        return;
    }

    std::vector<ElementType> host_buffer_0(transfer_size / ElementSize);
    std::vector<ElementType> host_buffer_1(transfer_size / ElementSize);

    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);
    auto coord = MeshCoordinate(0, 0);

    for ([[maybe_unused]] auto _ : state) {
        // EnqueueReadMeshBuffer cannot read from a replicated buffer yet, have to use ReadShard.
        // Issue the identical read on both CQs non-blocking so they overlap, then wait for both.
        ReadShard(cq0, host_buffer_0, device_buffer_0, coord, /*blocking=*/false);
        ReadShard(cq1, host_buffer_1, device_buffer_1, coord, /*blocking=*/false);
        cq0.finish();
        cq1.finish();
    }

    state.SetBytesProcessed(2 * transfer_size * state.iterations());
}

// Holds everything needed to drive one CQ's pinned-memory transfer.
struct PinnedShard {
    std::shared_ptr<MeshBuffer> device_buffer;
    std::shared_ptr<std::vector<uint8_t, tt::stl::aligned_allocator<uint8_t, 64>>> src_storage;
    std::shared_ptr<experimental::PinnedMemory> pinned_mem;
    distributed::ShardDataTransfer transfer{MeshCoordinate(0, 0)};
};

// Build a pinned-memory shard for host<->device transfer. Returns std::nullopt on skip.
static std::optional<PinnedShard> make_pinned_shard(
    benchmark::State& state,
    const std::shared_ptr<MeshDevice>& mesh_device,
    int64_t transfer_size,
    int64_t page_size,
    BufferType buffer_type) {
    auto device_buffer = try_create_buffer(state, mesh_device, transfer_size, page_size, buffer_type);
    if (!device_buffer) {
        return std::nullopt;
    }

    // Allocate host storage with 64-byte alignment (divisible by the PCIE read alignment).
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    constexpr int device_read_align{64};
    TT_ASSERT(
        device_read_align % hal.get_read_alignment(HalMemType::HOST) == 0,
        "Host vector alignment {} must be divisible by PCIE read alignment {}",
        device_read_align,
        hal.get_read_alignment(HalMemType::HOST));

    PinnedShard shard;
    shard.device_buffer = device_buffer;
    shard.src_storage = std::make_shared<std::vector<uint8_t, tt::stl::aligned_allocator<uint8_t, device_read_align>>>(
        static_cast<std::size_t>(transfer_size));
    void* aligned_ptr = reinterpret_cast<void*>(shard.src_storage->data());

    // Create HostBuffer on top of the aligned memory and pin it, mapping it to the NoC.
    HostBuffer host_buffer(
        tt::stl::Span<std::uint8_t>(shard.src_storage->data(), static_cast<std::size_t>(transfer_size)),
        MemoryPin(shard.src_storage));

    auto coord = MeshCoordinate(0, 0);
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    shard.pinned_mem =
        experimental::PinnedMemory::Create(*mesh_device, coordinate_range_set, host_buffer, /*map_to_noc=*/true);

    shard.transfer = distributed::ShardDataTransfer(coord)
                         .host_data(aligned_ptr)
                         .region(BufferRegion(0, static_cast<std::size_t>(transfer_size)));
    experimental::ShardDataTransferSetPinnedMemory(shard.transfer, shard.pinned_mem);
    return shard;
}

static void BM_write_pinned_memory_dual_cq(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];

    if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
        state.SkipWithError("Memory pinning with NOC mapping is not supported on this device");
        return;
    }

    // One independent pinned region + device buffer per CQ.
    auto shard0 = make_pinned_shard(state, mesh_device, transfer_size, page_size, buffer_type);
    auto shard1 = make_pinned_shard(state, mesh_device, transfer_size, page_size, buffer_type);
    if (!shard0 || !shard1) {
        return;
    }

    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);

    for ([[maybe_unused]] auto _ : state) {
        cq0.enqueue_write_shards(shard0->device_buffer, {shard0->transfer}, /*blocking=*/false);
        cq1.enqueue_write_shards(shard1->device_buffer, {shard1->transfer}, /*blocking=*/false);
        cq0.finish();
        cq1.finish();
    }

    state.SetBytesProcessed(2 * transfer_size * state.iterations());
}

static void BM_read_pinned_memory_dual_cq(benchmark::State& state, const std::shared_ptr<MeshDevice>& mesh_device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];

    if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
        state.SkipWithError("Memory pinning with NOC mapping is not supported on this device");
        return;
    }

    auto shard0 = make_pinned_shard(state, mesh_device, transfer_size, page_size, buffer_type);
    auto shard1 = make_pinned_shard(state, mesh_device, transfer_size, page_size, buffer_type);
    if (!shard0 || !shard1) {
        return;
    }

    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);

    for ([[maybe_unused]] auto _ : state) {
        cq0.enqueue_read_shards({shard0->transfer}, shard0->device_buffer, /*blocking=*/false);
        cq1.enqueue_read_shards({shard1->transfer}, shard1->device_buffer, /*blocking=*/false);
        cq0.finish();
        cq1.finish();
    }

    state.SetBytesProcessed(2 * transfer_size * state.iterations());
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

    // Open every device with 2 hardware command queues so we can drive both concurrently.
    constexpr size_t kNumCommandQueues = 2;
    auto devices =
        MeshDevice::create_unit_meshes(device_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, kNumCommandQueues);

    for (auto [device_id, device] : devices) {
        TT_FATAL(
            device->num_hw_cqs() >= kNumCommandQueues,
            "Dual-CQ benchmark requires >= {} hardware command queues, but device {} exposes {}",
            kNumCommandQueues,
            device_id,
            device->num_hw_cqs());

        // Device ID embedded here for extraction
        auto benchmark_args = {PAGE_SIZE_ARGS, TRANSFER_SIZE_ARGS, BUFFER_TYPE_ARGS, {device_id}};
        std::vector<std::string> benchmark_arg_names = {"page_size", "size", "type", "device"};
        auto compute_min = [](const std::vector<double>& v) -> double { return *std::min_element(v.begin(), v.end()); };
        auto compute_max = [](const std::vector<double>& v) -> double { return *std::max_element(v.begin(), v.end()); };
        // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
        // benchmark
        benchmark::RegisterBenchmark("WriteDualCQ", BM_write_dual_cq, device, host_buffer_max)
            ->ArgsProduct(benchmark_args)
            ->ArgNames(benchmark_arg_names)
            ->UseRealTime()
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);

        benchmark::RegisterBenchmark("ReadDualCQ", BM_read_dual_cq, device)
            ->ArgsProduct(benchmark_args)
            ->ArgNames(benchmark_arg_names)
            ->UseRealTime()
            ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
            ->ComputeStatistics("min", compute_min)
            ->ComputeStatistics("max", compute_max);

        if (experimental::GetMemoryPinningParameters(*device).can_map_to_noc) {
            benchmark::RegisterBenchmark("ReadPinnedMemoryDualCQ", BM_read_pinned_memory_dual_cq, device)
                ->ArgsProduct(benchmark_args)
                ->ArgNames(benchmark_arg_names)
                ->UseRealTime()
                ->ReportAggregatesOnly(true)  // Only show aggregated results (cv, min, max)
                ->ComputeStatistics("min", compute_min)
                ->ComputeStatistics("max", compute_max);
            auto pinned_write_page_sizes = PAGE_SIZE_ARGS;
            pinned_write_page_sizes.push_back(4096);
            auto pinned_write_args = {pinned_write_page_sizes, TRANSFER_SIZE_ARGS, BUFFER_TYPE_ARGS, {device_id}};
            benchmark::RegisterBenchmark("WritePinnedMemoryDualCQ", BM_write_pinned_memory_dual_cq, device)
                ->ArgsProduct(pinned_write_args)
                ->ArgNames(benchmark_arg_names)
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
