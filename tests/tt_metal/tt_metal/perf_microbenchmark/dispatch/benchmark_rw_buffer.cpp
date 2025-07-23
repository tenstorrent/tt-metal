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
#include <tt-logger/tt-logger.hpp>
#include <benchmark/benchmark.h>
#include "device_pool.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "hostdevcommon/common_values.hpp"
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of host-to-device data transfer and
// device-to-host data transfer. It uses EnqueueReadBuffer and
// EnqueueWriteBuffer APIs to transfer the data. The device memory object
// (buffer) will be in DRAM.
//
// Benchmark Matrix:
// Page Size: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
// Transfer Size: 32k, 512M
// Read & Write
// Device: 0 (local), 1 (remote) (when possible)
//
////////////////////////////////////////////////////////////////////////////////

static const auto PAGE_SIZE_ARGS = benchmark::CreateRange(32, 32768, 2);
static const std::vector<int64_t> TRANSFER_SIZE_ARGS{32 * 1024, 512 * 1024 * 1024};

static void BM_write(benchmark::State& state) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto device = DevicePool::instance().get_active_device(state.range(2));

    auto random_buffer_seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto device_buffer = Buffer::create(device, transfer_size, page_size, BufferType::DRAM);
    auto host_buffer = create_random_vector_of_bfloat16(transfer_size, 1000, random_buffer_seed);

    for (auto _ : state) {
        EnqueueWriteBuffer(device->command_queue(), device_buffer, host_buffer, false);
        Finish(device->command_queue());
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

static void BM_read(benchmark::State& state) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto device = DevicePool::instance().get_active_device(state.range(2));

    auto device_buffer = Buffer::create(device, transfer_size, page_size, BufferType::DRAM);
    std::vector<uint32_t> host_buffer;

    for (auto _ : state) {
        EnqueueReadBuffer(device->command_queue(), device_buffer, host_buffer, true);
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

    auto device_args = setup_device_pool();
    auto benchmark_args = {PAGE_SIZE_ARGS, TRANSFER_SIZE_ARGS, {device_args.begin(), device_args.end()}};

    // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
    // benchmark
    benchmark::RegisterBenchmark("EnqueueWriteBuffer", BM_write)->ArgsProduct(benchmark_args)->UseRealTime();
    benchmark::RegisterBenchmark("EnqueueReadBuffer", BM_read)->ArgsProduct(benchmark_args)->UseRealTime();

    benchmark::RunSpecifiedBenchmarks();
    DevicePool::instance().close_devices(DevicePool::instance().get_all_active_devices());
    benchmark::Shutdown();

    return 0;
}
