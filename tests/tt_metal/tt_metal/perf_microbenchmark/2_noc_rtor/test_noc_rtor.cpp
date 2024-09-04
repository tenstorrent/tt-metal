// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <string>
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
// This test measures the performance of random-to-random NOC data transfer. It
// creates a L1 buffer and every Tensix cores read from or write to the L1
// buffer. The memory access pattern of each Tensix core is a stride where the
// stride length is equal to the number of Tensix cores multiplied by page size.
// Since each page of the L1 buffer is scattered across all the Tensix cores and
// their mapping is random, the random-to-random data transfer between NOCs
// occurs.
//
// TODO: Currently, this benchmark uses only 1 NOC for communication.
// Ultimately, it should use two NOCs together through Metal's channel grouping
// interface
//
// Usage example:
//   ./test_noc_rtor
//     --cores-r <number of cores in a row>
//     --cores-c <number of cores in a column>
//     --num-tiles <number of tiles each core accesses>
//     --noc-index <NOC index to use>
//     --access-type <0 for read access, 1 for write access>
//     --use-device-profiler (set to use device profiler for measurement)
//     --num-tests <count of tests>
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    std::vector<unsigned long> elapsed_us;
    unsigned long elapsed_cc;

    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    uint32_t num_cores_r = 0;
    uint32_t num_cores_c = 0;
    uint32_t num_tiles = 204800;
    uint32_t noc_index = 0;
    uint32_t access_type = 0;
    uint32_t num_tests = 10;
    bool use_device_profiler = false;
    bool bypass_check = false;
    try {
        std::tie(num_cores_r, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--cores-r", 0);
        std::tie(num_cores_c, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--cores-c", 0);

        std::tie(num_tiles, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tiles", 204800);

        std::tie(noc_index, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--noc-index", 0);

        std::tie(access_type, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--access-type", 0);

        std::tie(num_tests, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

        std::tie(use_device_profiler, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--use-device-profiler");

        std::tie(bypass_check, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

        test_args::validate_remaining_args(input_args);
    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Command line arguments found exception", e.what());
    }

    if (use_device_profiler) {
#if !defined(TRACY_ENABLE)
        log_error(
            LogTest,
            "Metal library and test code should be build with "
            "profiler option using ./scripts/build_scripts/build_with_profiler_opt.sh");
#endif
        auto device_profiler = getenv("TT_METAL_DEVICE_PROFILER");
        TT_FATAL(
            device_profiler,
            "Before running the program, do one of the following in a shell: "
            "either export the environment variable by executing export TT_METAL_DEVICE_PROFILER=1, "
            "or run the program with TT_METAL_DEVICE_PROFILER=1 prefixed to the command");
    }

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);

        int clock_freq_mhz = get_tt_npu_clock(device);
        auto grid_coord = device->compute_with_storage_grid_size();
        num_cores_c = (num_cores_c == 0) ? grid_coord.x : num_cores_c;
        num_cores_r = (num_cores_r == 0) ? grid_coord.y : num_cores_r;

        uint32_t single_tile_size = 2 * 1024;
        uint32_t page_size = single_tile_size;

        // limit size of the L1 buffer to do not exceed global L1 size
        uint32_t l1_buffer_size = num_cores_r * num_cores_c * (num_tiles > 256 ? 256 : num_tiles) * page_size;
        auto l1_buffer = tt_metal::Buffer(device, l1_buffer_size, page_size, tt_metal::BufferType::L1);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord start_core = {0, 0};
        CoreCoord end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};
        CoreRange all_cores(start_core, end_core);

        for (int i = 0; i < num_cores_r; i++) {
            for (int j = 0; j < num_cores_c; j++) {
                int core_index = i * num_cores_c + j;
                CoreCoord core = {(std::size_t)j, (std::size_t)i};
                uint32_t cb_index = 0;
                uint32_t cb_tiles = 32;
                tt_metal::CircularBufferConfig cb_src0_config =
                    tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
                        .set_page_size(cb_index, single_tile_size);
                auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
            }
        }

        auto noc_kernel = tt_metal::CreateKernel(
            program,
            (access_type == 0) ? "tests/tt_metal/tt_metal/perf_microbenchmark/"
                                 "2_noc_rtor/kernels/noc_read.cpp"
                               : "tests/tt_metal/tt_metal/perf_microbenchmark/"
                                 "2_noc_rtor/kernels/noc_write.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = (noc_index == 0) ? tt_metal::DataMovementProcessor::RISCV_0
                                              : tt_metal::DataMovementProcessor::RISCV_1,
                .noc = (noc_index == 0) ? tt_metal::NOC::RISCV_0_default : tt_metal::NOC::RISCV_1_default});

        for (int i = 0; i < num_cores_r; i++) {
            for (int j = 0; j < num_cores_c; j++) {
                CoreCoord core = {(std::size_t)j, (std::size_t)i};
                uint32_t core_index = i * num_cores_c + j;
                uint32_t l1_buffer_addr = l1_buffer.address();

                vector<uint32_t> noc_runtime_args = {core_index, l1_buffer_addr, num_tiles, num_cores_r * num_cores_c};
                SetRuntimeArgs(program, noc_kernel, core, noc_runtime_args);
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

            log_info(LogTest, "Time elapsed for NOC transfers: {}us", elapsed_us[i]);

            if (use_device_profiler) {
                elapsed_cc = get_t0_to_any_riscfw_end_cycle(device, program);
                elapsed_us.push_back((double)elapsed_cc / clock_freq_mhz);
                log_info(LogTest, "Time elapsed uisng device profiler: {}us ({}cycles)", elapsed_us[i], elapsed_cc);
            }
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
        // TODO: Numbers are TBD in SoW
        ;
    }

    // for csv
    log_info("CSV_MICROBENCHMARK:title:test_noc_rtor");
    log_info(
        "CSV_INPUT:num-cores-r:{}:num-cores-c:{}:num-tiles:{}:noc-index:{}:"
        "access-type:{}:use-device-profiler:{}",
        num_cores_r,
        num_cores_c,
        num_tiles,
        NOC_INDEXToString(static_cast<NOC_INDEX>(noc_index)),
        ACCESS_TYPEToString(static_cast<ACCESS_TYPE>(access_type)),
        use_device_profiler);
    log_info("CSV_OUTPUT:ElapsedTime(us):{}", avg_elapsed_us);
    log_info("CSV_RESULT:pass:{}", pass);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
