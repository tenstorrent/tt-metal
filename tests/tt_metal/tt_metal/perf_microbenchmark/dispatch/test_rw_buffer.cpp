// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <stdint.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
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
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of host-to-device data transfer and
// device-to-host data transfer. It uses EnqueueReadBuffer and
// EnqueueWriteBuffer APIs to transfer the data. The device memory object
// (buffer) can be resident in DRAM or L1.
//
// Usage example:
//   ./test_rw_buffer
//     --buffer-type <0 for DRAM, 1 for L1>
//     --transfer-size <size in bytes>
//     --page-size <size in bytes>
//     --num-tests <count of tests>
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

struct CommandArg {
    tt_metal::BufferType buffer_type;
    uint32_t transfer_size, page_size, num_tests, device_id;
    bool skip_read, skip_write, bypass_check;
};

CommandArg parseArgs(int argc, char** argv) {
    CommandArg args;
    std::vector<std::string> input_args(argv, argv + argc);

    int32_t buffer_type_it;
    std::tie(buffer_type_it, input_args) =
        test_args::get_command_option_int32_and_remaining_args(input_args, "--buffer-type", 0);
    args.buffer_type = buffer_type_it == 0 ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1;

    std::tie(args.transfer_size, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--transfer-size", 512 * 1024 * 1024);

    std::tie(args.page_size, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--page-size", 2048);

    std::tie(args.num_tests, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

    std::tie(args.bypass_check, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

    std::tie(args.skip_read, input_args) = test_args::has_command_option_and_remaining_args(input_args, "--skip-read");

    std::tie(args.skip_write, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--skip-write");

    std::tie(args.device_id, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--device");

    test_args::validate_remaining_args(input_args);

    TT_ASSERT(
        args.page_size == 0 ? args.transfer_size == 0 : args.transfer_size % args.page_size == 0,
        "Transfer size {}B should be divisible by page size {}B",
        transfer_size,
        page_size);

    return args;
}

struct BenchmarkParam {
    CommandQueue& command_queue;
    std::shared_ptr<Buffer> device_buffer;
    std::vector<uint32_t>& host_buffer;
    uint32_t transfer_size;
};

// BenchmarkParam have to be const& to comply with Google Benchmark driver
static void BM_write(benchmark::State& state, const BenchmarkParam& para) {
    for (auto _ : state) {
        // TODO: Is blocking off here correct?
        EnqueueWriteBuffer(para.command_queue, para.device_buffer, para.host_buffer, false);
        Finish(para.command_queue);
    }

    state.SetBytesProcessed(para.transfer_size * state.iterations());
}

static void BM_read(benchmark::State& state, const BenchmarkParam& para) {
    for (auto _ : state) {
        EnqueueReadBuffer(para.command_queue, para.device_buffer, para.host_buffer, true);
    }

    state.SetBytesProcessed(para.transfer_size * state.iterations());
}

int main(int argc, char** argv) {
    CommandArg args = parseArgs(argc, argv);

    if (args.device_id >= tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices()) {
        log_info(LogTest, "Skip! Device id {} is not applicable on this system", args.device_id);
        return 1;
    }

    tt_metal::IDevice* device = tt_metal::CreateDevice(args.device_id);

    if (!device->using_fast_dispatch()) {
        log_info(LogTest, "Skip! This test needs to be run with fast dispatch enabled");
        return 1;
    }

    auto device_buffer = tt_metal::Buffer::create(device, args.transfer_size, args.page_size, args.buffer_type);
    auto host_random_buffer = create_random_vector_of_bfloat16(
        args.transfer_size, 1000, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<std::uint32_t> host_reception_buffer;

    if (!args.skip_write) {
        benchmark::RegisterBenchmark(
            "Write",
            BM_write,
            BenchmarkParam{device->command_queue(), device_buffer, host_random_buffer, args.transfer_size})
            // TODO: do we really want to preserve this? Google Benchmark has it's own mathmatical model of figuring out
            // how many iterations are needed that might be more useful here.
            ->Iterations(args.num_tests);
    }

    if (!args.skip_read) {
        benchmark::RegisterBenchmark(
            "Read",
            BM_read,
            BenchmarkParam{device->command_queue(), device_buffer, host_reception_buffer, args.transfer_size})
            ->Iterations(args.num_tests);
    }

    benchmark::RunSpecifiedBenchmarks();
    tt_metal::CloseDevice(device);
    benchmark::Shutdown();

    // TODO: Verify performance numbers

    return 0;
}
