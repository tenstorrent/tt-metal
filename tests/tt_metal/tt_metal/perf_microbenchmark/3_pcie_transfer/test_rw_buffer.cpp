// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
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

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of host-to-device data transfer and
// device-to-host data transfer. It uses EnqueueReadMeshBuffer and
// EnqueueWriteMeshBuffer APIs to transfer the data. The device memory object
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

int main(int argc, char** argv) {
    bool pass = true;
    bool bypass_check = false;
    bool skip_read = false;
    bool skip_write = false;
    bool device_is_mmio = false;  // MMIO devices should have higher perf
    std::vector<double> h2d_bandwidth;
    std::vector<double> d2h_bandwidth;
    int32_t buffer_type = 0;
    uint32_t transfer_size;
    uint32_t page_size;
    uint32_t device_id = 0;

    try {
        // Input arguments parsing
        std::vector<std::string> input_args(argv, argv + argc);

        uint32_t num_tests = 10;
        try {
            std::tie(buffer_type, input_args) =
                test_args::get_command_option_int32_and_remaining_args(input_args, "--buffer-type", 0);

            std::tie(transfer_size, input_args) = test_args::get_command_option_uint32_and_remaining_args(
                input_args, "--transfer-size", 512 * 1024 * 1024);

            std::tie(page_size, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--page-size", 2048);

            std::tie(num_tests, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

            std::tie(bypass_check, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

            std::tie(skip_read, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--skip-read");

            std::tie(skip_write, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--skip-write");

            std::tie(device_id, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--device");

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
        }

        TT_FATAL(
            page_size == 0 ? transfer_size == 0 : transfer_size % page_size == 0,
            "Transfer size {}B should be divisible by page size {}B",
            transfer_size,
            page_size);

        // Device setup
        if (device_id >= tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices()) {
            log_info(LogTest, "Skip! Device id {} is not applicable on this system", device_id);
            return 1;
        }

        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        device_is_mmio = device->get_devices()[0]->is_mmio_capable();

        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
            log_info(LogTest, "Skip! This test needs to be run with fast dispatch enabled");
            return 1;
        }

        // Application setup: MeshBuffer
        tt_metal::distributed::DeviceLocalBufferConfig device_local{
            .page_size = page_size,
            .buffer_type = buffer_type == 0 ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1,
        };
        tt_metal::distributed::ReplicatedBufferConfig global_buf{.size = transfer_size};
        auto buffer = tt_metal::distributed::MeshBuffer::create(global_buf, device_local, device.get());

        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            transfer_size, 1000, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<uint32_t> result_vec;

        log_info(
            LogTest,
            "Measuring host-to-device and device-to-host bandwidth for "
            "buffer_type={}, transfer_size={} bytes, page_size={} bytes",
            buffer_type == 0 ? "DRAM" : "L1",
            transfer_size,
            page_size);

        log_info(LogTest, "Num tests {}", num_tests);
        float best_write_bw = 0.0f;
        float best_read_bw = 0.0f;
        for (uint32_t i = 0; i < num_tests; ++i) {
            // Execute application
            if (!skip_write) {
                auto t_begin = std::chrono::steady_clock::now();
                tt_metal::distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), buffer, src_vec, false);
                tt_metal::distributed::Finish(device->mesh_command_queue());
                auto t_end = std::chrono::steady_clock::now();
                auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
                float write_bw = transfer_size / (elapsed_us * 1000.0);
                h2d_bandwidth.push_back(write_bw);
                best_write_bw = std::fmax(best_write_bw, write_bw);
                log_info(
                    LogTest,
                    "EnqueueWriteMeshBuffer to {} (H2D): {:.3f}ms, {:.3f}GB/s",
                    buffer_type == 0 ? "DRAM" : "L1",
                    elapsed_us / 1000.0,
                    h2d_bandwidth[i]);
            }

            if (!skip_read) {
                auto t_begin = std::chrono::steady_clock::now();
                tt_metal::distributed::ReadShard(
                    device->mesh_command_queue(),
                    result_vec,
                    buffer,
                    tt_metal::distributed::MeshCoordinate(0, 0),
                    true);
                auto t_end = std::chrono::steady_clock::now();
                auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
                float read_bw = transfer_size / (elapsed_us * 1000.0);
                d2h_bandwidth.push_back(read_bw);
                best_read_bw = std::fmax(best_read_bw, read_bw);
                log_info(
                    LogTest,
                    "EnqueueReadMeshBuffer from {} (D2H): {:.3f}ms, {:.3f}GB/s",
                    buffer_type == 0 ? "DRAM" : "L1",
                    elapsed_us / 1000.0,
                    d2h_bandwidth[i]);
            }
        }

        if (!skip_write) {
            log_info(LogTest, "Best write: {} GB/s", best_write_bw);
        }
        if (!skip_read) {
            log_info(LogTest, "Best read: {} GB/s", best_read_bw);
        }

        // Validation & teardown
        // Data check is only valid if both read and write are enabled
        if (!skip_read && !skip_write && !(src_vec == result_vec)) {
            log_error(tt::LogTest, "Read data mismatch");
            pass = false;
        }
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    auto avg_h2d_bandwidth = calculate_average(h2d_bandwidth);
    auto avg_d2h_bandwidth = calculate_average(d2h_bandwidth);
    if (pass && !bypass_check) {
        // TODO: check the theoritical peak of wormhole
        static constexpr double k_PcieMax = 16.0;  // GB/s
        double target_read_bandwidth;
        double target_write_bandwidth;

        if (device_is_mmio) {
            // MMIO
            target_read_bandwidth = k_PcieMax * 0.5;    // 50%
            target_write_bandwidth = k_PcieMax * 0.75;  // 80%
        } else {
            // Remote
            target_read_bandwidth = k_PcieMax * 0.15;   // 15%
            target_write_bandwidth = k_PcieMax * 0.35;  // 35%
        }

        if (!skip_write && avg_h2d_bandwidth < target_write_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The host-to-device bandwidth does not meet the criteria. "
                "Current: {:.3f}GB/s, goal: {:.3f}GB/s",
                avg_h2d_bandwidth,
                target_write_bandwidth);
        }

        if (!skip_read && avg_d2h_bandwidth < target_read_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The device-to-host bandwidth does not meet the criteria. "
                "Current: {:.3f}GB/s, goal: {:.3f}GB/s",
                avg_d2h_bandwidth,
                target_read_bandwidth);
        }
    }

    // for csv
    log_info(tt::LogTest, "CSV_MICROBENCHMARK:title:test_rw_buffer");
    log_info(
        tt::LogTest,
        "CSV_INPUT:buffer-type:{}:transfer-size:{}",
        BUFFER_TYPEToString(static_cast<BUFFER_TYPE>(buffer_type)),
        transfer_size);
    log_info(
        tt::LogTest,
        "CSV_OUTPUT:H2D_Bandwidth(GB/s):{:.3f}:D2H_Bandwidth(GB/s):{:.3f}",
        avg_h2d_bandwidth,
        avg_d2h_bandwidth);
    log_info(tt::LogTest, "CSV_RESULT:pass:{}", pass);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
