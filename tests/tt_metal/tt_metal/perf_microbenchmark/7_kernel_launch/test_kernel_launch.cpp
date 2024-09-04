// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the time for executing a program that contains empty data
// movement kernels and compute kernel.
//
// Usage example:
//   ./test_kernel_launch
//     --cores-r <number of cores in a row>
//     --cores-c <number of cores in a column>
//     --core-groups <number of core groups where each core group executes
//                    different kernel binaries>
//     --num-tests <count of tests>
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    std::vector<unsigned long> elapsed_us;

    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    uint32_t num_cores_r = 0;
    uint32_t num_cores_c = 0;
    uint32_t num_core_groups;
    uint32_t num_tests = 10;
    bool bypass_check = false;
    try {
        std::tie(num_cores_r, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--cores-r", 0);
        std::tie(num_cores_c, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--cores-c", 0);

        std::tie(num_core_groups, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--core-groups", 4);

        std::tie(num_tests, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

        std::tie(bypass_check, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

        test_args::validate_remaining_args(input_args);
    } catch (const std::exception& e) {
        log_error(LogTest, "Command line arguments found exception", e.what());
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);

    auto grid_coord = device->compute_with_storage_grid_size();
    num_cores_c = (num_cores_c == 0) ? grid_coord.x : num_cores_c;
    num_cores_r = (num_cores_r == 0) ? grid_coord.y : num_cores_r;

    if (num_cores_r < num_core_groups) {
        log_error(
            LogTest,
            "The number of cores in a row ({}) must be bigger than or equal than "
            "the number of core groups ({})",
            num_cores_r,
            num_core_groups);
    }

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();
        uint32_t single_tile_size = 2 * 1024;

        for (int core_group_idx = 0; core_group_idx < num_core_groups; ++core_group_idx) {
            CoreCoord start_core = {0, (num_cores_r / num_core_groups) * core_group_idx};
            CoreCoord end_core = {
                (std::size_t)num_cores_c - 1,
                (core_group_idx == num_core_groups - 1) ? (std::size_t)num_cores_r - 1
                                                        : (num_cores_r / num_core_groups) * (core_group_idx + 1) - 1};
            CoreRange group_of_cores(start_core, end_core);

            log_info(
                LogTest,
                "Setting kernels for core group {}, cores ({},{}) ~ ({},{})",
                core_group_idx,
                start_core.x,
                start_core.y,
                end_core.x,
                end_core.y);

            for (int i = start_core.y; i <= end_core.y; i++) {
                for (int j = start_core.x; j <= end_core.x; j++) {
                    CoreCoord core = {(std::size_t)j, (std::size_t)i};
                    uint32_t cb_index = 0;
                    uint32_t cb_tiles = 8;
                    tt_metal::CircularBufferConfig cb_config =
                        tt_metal::CircularBufferConfig(
                            cb_tiles * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
                            .set_page_size(cb_index, single_tile_size);
                    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_config);
                }
            }

            vector<uint32_t> reader_compile_args = {uint32_t(core_group_idx)};
            auto reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/7_kernel_launch/"
                "kernels/"
                "reader.cpp",
                group_of_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = reader_compile_args});

            vector<uint32_t> writer_compile_args = {uint32_t(core_group_idx)};
            auto writer_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/7_kernel_launch/"
                "kernels/"
                "writer.cpp",
                group_of_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = writer_compile_args});

            vector<uint32_t> compute_compile_args = {uint32_t(core_group_idx)};
            auto compute_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/7_kernel_launch/"
                "kernels/"
                "compute.cpp",
                group_of_cores,
                tt_metal::ComputeConfig{.compile_args = compute_compile_args});

            for (int i = start_core.y; i <= end_core.y; i++) {
                for (int j = start_core.x; j <= end_core.x; j++) {
                    CoreCoord core = {(std::size_t)j, (std::size_t)i};
                    int core_index = i * num_cores_c + j;

                    vector<uint32_t> reader_runtime_args;
                    vector<uint32_t> writer_runtime_args;
                    for (uint32_t k = 0; k < 255; ++k) {
                        reader_runtime_args.push_back(core_index + k);
                        writer_runtime_args.push_back(core_index + k);
                    }

                    SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);
                    SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::detail::CompileProgram(device, program);

        log_info(LogTest, "Num tests {}", num_tests);
        for (uint32_t i = 0; i < num_tests; ++i) {
            auto t_begin = std::chrono::steady_clock::now();
            EnqueueProgram(device->command_queue(), &program, false);
            Finish(device->command_queue());
            auto t_end = std::chrono::steady_clock::now();
            elapsed_us.push_back(duration_cast<microseconds>(t_end - t_begin).count());

            log_info(LogTest, "Time elapsed for executing empty kernels: {}us", elapsed_us[i]);
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    auto avg_elapsed_us = calculate_average(elapsed_us);
    if (pass && bypass_check == false) {
        // goal is under 10us
        long target_us = 10;

        if (avg_elapsed_us > target_us) {
            pass = false;
            log_error(
                LogTest,
                "The kernel launch overhead does not meet the criteria. "
                "Current: {}us, goal: <{}us",
                avg_elapsed_us,
                target_us);
        }
    }

    // for csv
    log_info("CSV_MICROBENCHMARK:title:test_kernel_launch");
    log_info("CSV_INPUT:num-cores-r:{}:num-cores-c:{}:core-groups:{}", num_cores_r, num_cores_c, num_core_groups);
    log_info("CSV_OUTPUT:ElapsedTime(us):{}", avg_elapsed_us);
    log_info("CSV_RESULT:pass:{}", pass);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
