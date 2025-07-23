// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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
#include <variant>
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
#include "command_queue.hpp"
#include "shape2d.hpp"
#include "test_common.hpp"
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of host-to-device data transfer and
// device-to-host data transfer. It uses EnqueueWriteMeshBuffer and
// EnqueueReadMeshBuffer APIs to transfer the data. The device memory object
// (mesh buffer) will be in DRAM.
//
// Benchmark Matrix:
// Page Size: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
// Transfer Size: 32k, 512M
// Read & Write
// Device: 0 (local), 1 (remote) (when possible)
//
////////////////////////////////////////////////////////////////////////////////

static const BufferType TARGET_BUFFER_TYPE = tt_metal::BufferType::DRAM;

static const auto PAGE_SIZE_ARGS = benchmark::CreateRange(32, 32768, 2);
static const std::vector<int64_t> TRANSFER_SIZE_ARGS{32 * 1024, 512 * 1024 * 1024};

// Create a buffer of total transfer_size big that is paged with page_size
std::shared_ptr<distributed::MeshBuffer> create_buffer(
    int page_size, int transfer_size, std::shared_ptr<MeshDevice> device) {
    using DataType = uint32_t;
    auto num_data = transfer_size / sizeof(DataType);

    // Effectively a tall vector.
    Shape2D buffer_shape{num_data, 1};

    // This is the config of buffer to be created, as this is sent to a single device, it's global shape is the same as
    // it's shard shape.
    distributed::ShardedBufferConfig mesh_buffer_config{
        .global_size = transfer_size, .global_buffer_shape = buffer_shape, .shard_shape = buffer_shape};
    DeviceLocalBufferConfig device_local_config{.page_size = page_size, .buffer_type = TARGET_BUFFER_TYPE};

    TT_ASSERT(mesh_buffer_config.compute_datum_size_bytes() == sizeof(DataType));

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
    auto device = DevicePool::instance().get_active_device(state.range(2));

    auto device_buffer = create_buffer(page_size, transfer_size, mesh_device);
    std::vector<uint32_t> host_buffer;

    for (auto _ : state) {
        EnqueueReadMeshBuffer(mesh_device->mesh_command_queue(), host_buffer, device_buffer, true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

std::vector<chip_id_t> setup_device_pool() {
    auto available_device_ids = MetalContext::instance().get_cluster().all_chip_ids();
    TT_ASSERT(available_device_ids.contains(0));

    std::vector<chip_id_t> device_ids = {0};
    log_info(tt::LogTest, "Device 1 available, enable testing on device 1 assuming it's a remote device");
    if (available_device_ids.contains(1)) {
        device_ids.push_back(1);
    }

    DevicePool::initialize(device_ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, DispatchCoreConfig{});

    return device_ids;
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // TODO: Test Across Multiple devices.
    auto device_id = 0;
    auto mesh_device = MeshDevice::create_unit_mesh(device_id);

    if (!mesh_device->using_fast_dispatch()) {
        log_info(LogTest, "Skip! This test needs to be run with fast dispatch enabled");
        return 1;
    }

    // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
    // benchmark
    benchmark::RegisterBenchmark("EnqueueWriteMeshBuffer", BM_write, mesh_device)
        ->ArgsProduct(BENCHMARK_ARGS)
        ->UseRealTime();
    benchmark::RegisterBenchmark("EnqueueReadMeshBuffer", BM_read, mesh_device)
        ->ArgsProduct(BENCHMARK_ARGS)
        ->UseRealTime();

    benchmark::RunSpecifiedBenchmarks();
    mesh_device->close();
    benchmark::Shutdown();

    return 0;
}
