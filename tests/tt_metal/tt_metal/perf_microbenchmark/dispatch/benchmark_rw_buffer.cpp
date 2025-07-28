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
#include <cmath>
#include <cstring>
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
// Page Size: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
// Transfer Size: 32k, 512M
// Read & Write
// Device: 0 (local), 1 (remote) (when possible)
//
////////////////////////////////////////////////////////////////////////////////

static const auto KB = 1024;
static const auto MB = 1024 * KB;

static const BufferType TARGET_BUFFER_TYPE = tt_metal::BufferType::DRAM;

static const auto PAGE_SIZE_ARGS = benchmark::CreateRange(32, 32 * KB, 2);
static const std::vector<int64_t> TRANSFER_SIZE_ARGS{32 * KB, 512 * MB};
static const auto BENCHMARK_ARGS = {PAGE_SIZE_ARGS, TRANSFER_SIZE_ARGS};

// Create a buffer of total transfer_size big that is paged with page_size
std::shared_ptr<MeshBuffer> create_buffer(int page_size, int transfer_size, std::shared_ptr<MeshDevice> device) {
    using DataType = uint32_t;
    auto num_data = transfer_size / sizeof(DataType);

    // Need to use Replicated Buffer instead of Sharded Buffer as we operate on a unit mesh.
    ReplicatedBufferConfig mesh_buffer_config{transfer_size};
    DeviceLocalBufferConfig device_local_config{.page_size = page_size, .buffer_type = TARGET_BUFFER_TYPE};

    return MeshBuffer::create(mesh_buffer_config, device_local_config, device.get());
}

static void BM_write(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);

    auto random_buffer_seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto host_buffer = create_random_vector_of_bfloat16(transfer_size, 1000, random_buffer_seed);

    auto device_buffer = create_buffer(page_size, transfer_size, mesh_device);

    for (auto _ : state) {
        EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), device_buffer, host_buffer, true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

static void BM_read(benchmark::State& state, std::shared_ptr<MeshDevice> mesh_device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);

    auto device_buffer = create_buffer(page_size, transfer_size, mesh_device);
    std::vector<uint32_t> host_buffer;

    for (auto _ : state) {
        // EnqueueReadMeshBuffer cannot read from a replicated buffer yet, have to use ReadShard
        ReadShard(mesh_device->mesh_command_queue(), host_buffer, device_buffer, MeshCoordinate(0, 0), true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
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
        auto benchmark_args = {PAGE_SIZE_ARGS, TRANSFER_SIZE_ARGS, {device_id}};
        // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
        // benchmark
        benchmark::RegisterBenchmark("Write", BM_write, device)->ArgsProduct(benchmark_args)->UseRealTime();
        benchmark::RegisterBenchmark("Read", BM_read, device)->ArgsProduct(benchmark_args)->UseRealTime();
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
