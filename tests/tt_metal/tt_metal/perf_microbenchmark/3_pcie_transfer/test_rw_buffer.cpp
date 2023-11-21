// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

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
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    bool pass = true;
    bool bypass_check = false;
    double h2d_bandwidth = 0;
    double d2h_bandwidth = 0;

    try {
        // Input arguments parsing
        std::vector<std::string> input_args(argv, argv + argc);

        int32_t buffer_type = 0;
        uint32_t transfer_size;
        try {
            std::tie(buffer_type, input_args) =
                test_args::get_command_option_int32_and_remaining_args(input_args, "--buffer-type", 0);

            std::tie(transfer_size, input_args) = test_args::get_command_option_uint32_and_remaining_args(
                input_args, "--transfer-size", 512 * 1024 * 1024);

            std::tie(bypass_check, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
        }

        // Device setup
        int device_id = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);
        CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;

        // Application setup
        uint32_t single_tile_size = 2 * 1024;
        auto page_size = single_tile_size;
        auto buffer = tt_metal::Buffer(
            device, transfer_size, page_size, buffer_type == 0 ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1);

        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            transfer_size, 1000, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<uint32_t> result_vec;

        log_info(
            LogTest,
            "Measuring host-to-device and device-to-host bandwidth for "
            "buffer_type={}, transfer_size={}bytes ",
            buffer_type == 0 ? "DRAM" : "L1",
            transfer_size);

        // Execute application
        {
            auto t_begin = std::chrono::steady_clock::now();
            EnqueueWriteBuffer(cq, buffer, src_vec, false);
            Finish(cq);
            auto t_end = std::chrono::steady_clock::now();
            auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            h2d_bandwidth = (transfer_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0);
            log_info(
                LogTest,
                "EnqueueWriteBuffer to {} (H2D): {:.3f}ms, {:.3f}GB/s",
                buffer_type == 0 ? "DRAM" : "L1",
                elapsed_us / 1000.0,
                h2d_bandwidth);
        }

        {
            auto t_begin = std::chrono::steady_clock::now();
            EnqueueReadBuffer(cq, buffer, result_vec, true);
            auto t_end = std::chrono::steady_clock::now();
            auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            d2h_bandwidth = (transfer_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0);
            log_info(
                LogTest,
                "EnqueueReadBuffer from {} (D2H): {:.3f}ms, {:.3f}GB/s",
                buffer_type == 0 ? "DRAM" : "L1",
                elapsed_us / 1000.0,
                d2h_bandwidth);
        }

        // Validation & teardown
        pass &= (src_vec == result_vec);
        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    if (pass && bypass_check == false) {
        // goal is 70% of PCI-e Gen3 x16 for grayskull
        // TODO: check the theoritical peak of wormhole
        double target_bandwidth = 16.0 * 0.7;

        if (h2d_bandwidth < target_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The host-to-device bandwidth does not meet the criteria. "
                "Current: {:.3f}GB/s, goal: {:.3f}GB/s",
                h2d_bandwidth,
                target_bandwidth);
        } else if (d2h_bandwidth < target_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The device-to-host bandwidth does not meet the criteria. "
                "Current: {:.3f}GB/s, goal: {:.3f}GB/s",
                d2h_bandwidth,
                target_bandwidth);
        }
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
