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
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <benchmark/benchmark.h>
#include "command_queue.hpp"
#include "test_common.hpp"
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
// Device: 0 (local), 1 (remote)
//
////////////////////////////////////////////////////////////////////////////////

static const auto PAGE_SIZE_ARGS = benchmark::CreateRange(32, 32768, 2);
static const std::vector<int64_t> TRANSFER_SIZE_ARGS{32 * 1024, 512 * 1024 * 1024};
static const auto BENCHMARK_ARGS = {PAGE_SIZE_ARGS, TRANSFER_SIZE_ARGS};

static void BM_write(benchmark::State& state, tt_metal::IDevice* device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);
    auto random_buffer_seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto device_buffer = Buffer::create(device, transfer_size, page_size, BufferType::DRAM);
    auto host_buffer = create_random_vector_of_bfloat16(transfer_size, 1000, random_buffer_seed);

    for (auto _ : state) {
        EnqueueWriteBuffer(device->command_queue(), device_buffer, host_buffer, false);
        Finish(device->command_queue());
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

static void BM_read(benchmark::State& state, tt_metal::IDevice* device) {
    auto page_size = state.range(0);
    auto transfer_size = state.range(1);

    auto device_buffer = Buffer::create(device, transfer_size, page_size, BufferType::DRAM);
    std::vector<uint32_t> host_buffer;

    for (auto _ : state) {
        EnqueueReadBuffer(device->command_queue(), device_buffer, host_buffer, true);
    }

    state.SetBytesProcessed(transfer_size * state.iterations());
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // TODO: Test Across Multiple devices.
    auto device_id = 0;
    if (device_id >= MetalContext::instance().get_cluster().number_of_devices()) {
        log_info(LogTest, "Skip! Device id {} is not applicable on this system", device_id);
        return 1;
    }

    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    if (!device->using_fast_dispatch()) {
        log_info(LogTest, "Skip! This test needs to be run with fast dispatch enabled");
        return 1;
    }

    benchmark::RegisterBenchmark("EnqueueWriteBuffer", BM_write, device)
        ->ArgsProduct(BENCHMARK_ARGS)
        // Google Benchmark uses CPU time to calculate throughput by default, which is not suitable for this
        // benchmark
        ->UseRealTime();

    benchmark::RegisterBenchmark("EnqueueReadBuffer", BM_read, device)->ArgsProduct(BENCHMARK_ARGS)->UseRealTime();

    benchmark::RunSpecifiedBenchmarks();
    tt_metal::CloseDevice(device);
    benchmark::Shutdown();

    return 0;
}
