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
#include <functional>

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

static constexpr auto num_test_repetitions = 101;

static constexpr auto KB = 1024;
static constexpr auto MB = 1024 * KB;
static constexpr auto GB = 1024 * MB;

// static const std::vector<int64_t> PAGE_SIZE_ARGS{32, 128, 512, 1024, 4096, 8192, 512 * KB, 1 * MB, 16 * MB};
static const std::vector<int64_t> PAGE_SIZE_ARGS{32, 128, 512, 1024};
constexpr uint64_t max_transfer_size{1 * GB /*4294934528*/ /*(4ul * GB) - (32 * KB)*/};
// static const std::vector<int64_t> TRANSFER_SIZE_ARGS = {16 * MB, 32 * MB, 64 * MB, 128 * MB, 512 * MB, 1 * GB, 2 *
// GB, (4ul * GB) - (16 * MB), max_transfer_size};
static const std::vector<int64_t> TRANSFER_SIZE_ARGS = {1 * GB};

static constexpr std::array<BufferType, 2> BUFFER_TYPES = {BufferType::DRAM, BufferType::L1};
static const std::vector<int64_t> BUFFER_TYPE_ARGS = {0};

static void BM_write(
    benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device, const std::vector<uint32_t>& host_buffer) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    auto device_id = state.range(3);

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

static void BM_read(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    size_t transfer_size = state.range(1);
    auto buffer_type = BUFFER_TYPES[state.range(2)];
    auto device_id = state.range(3);

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
    std::vector<uint32_t> host_buffer;

    for (auto _ : state) {
        // EnqueueReadMeshBuffer cannot read from a replicated buffer yet, have to use ReadShard
        ReadShard(mesh_device->mesh_command_queue(), host_buffer, device_buffer, MeshCoordinate(0, 0), true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // no need to initialize for bandwidth measurement, saves test initialization time
    std::vector<uint32_t> host_buffer_max(max_transfer_size);
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
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
