// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

int main(int argc, char** argv) {
    bool pass = true;
    try {
        // Initial Runtime Args Parse
        std::vector<std::string> input_args(argv, argv + argc);
        uint32_t iter = 1;
        std::tie(iter, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--iter", 1);

        uint32_t device_id = 0;
        std::tie(device_id, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--device_id", 0);

        uint64_t page_size = 1024;
        std::tie(page_size, input_args) =
            test_args::get_command_option_uint64_and_remaining_args(input_args, "--page_size", 1024);

        uint32_t buffer_type = 0;
        try {
            std::tie(buffer_type, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--buffer_type");
        } catch (const std::exception& e) {
            TT_THROW(
                "Please input type of the buffer with \"--buffer_type <0: "
                "DRAM, 1: L1>\"",
                e.what());
        }

        uint64_t buffer_size = 2048;
        try {
            std::tie(buffer_size, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--size");
        } catch (const std::exception& e) {
            TT_THROW("Provide buffer size with option --buffer_size");
        }

        // Device Setup
        log_info(LogTest, "Running test using device ID {}", device_id);
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);
        CommandQueue& cq = device->command_queue();

        log_info(
            LogTest,
            "Measuring performance for buffer_type={}, size={} bytes, page_size={} bytes",
            buffer_type == 0 ? "DRAM" : "L1",
            buffer_size,
            page_size);

        const BufferType buff_type = buffer_type == 0 ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1;
        tt_metal::InterleavedBufferConfig buff_config{
            .device = device, .size = buffer_size, .page_size = page_size, .buffer_type = buff_type};
        auto buffer = CreateBuffer(buff_config);

        // Execute Application
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        {
            auto begin = std::chrono::steady_clock::now();
            auto end = std::chrono::steady_clock::now();
            auto elapsed_sum = end - begin;

            for (int i = 0; i < iter; i++) {
                begin = std::chrono::steady_clock::now();
                EnqueueWriteBuffer(cq, buffer, src_vec, false);
                Finish(cq);
                end = std::chrono::steady_clock::now();
                elapsed_sum += end - begin;
            }

            auto elapsed_us = duration_cast<microseconds>(elapsed_sum / iter).count();
            auto bw = (buffer_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0);
            log_info(
                LogTest,
                "EnqueueWriteBuffer to {}: {:.3f}ms, {:.3f}GB/s",
                buffer_type == 0 ? "DRAM" : "L1",
                elapsed_us / 1000.0,
                bw);
        }

        std::vector<uint32_t> result_vec;
        {
            auto begin = std::chrono::steady_clock::now();
            auto end = std::chrono::steady_clock::now();
            auto elapsed_sum = end - begin;

            for (int i = 0; i < iter; i++) {
                begin = std::chrono::steady_clock::now();
                EnqueueReadBuffer(cq, buffer, result_vec, true);
                end = std::chrono::steady_clock::now();
                elapsed_sum += end - begin;
            }

            auto elapsed_us = duration_cast<microseconds>(elapsed_sum / iter).count();
            auto bw = (buffer_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0);
            log_info(
                LogTest,
                "EnqueueReadBuffer from {}: {:.3f}ms, {:.3f}GB/s",
                buffer_type == 0 ? "DRAM" : "L1",
                elapsed_us / 1000.0,
                bw);
        }
        // Validation & Teardown
        pass &= (src_vec == result_vec);
        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
