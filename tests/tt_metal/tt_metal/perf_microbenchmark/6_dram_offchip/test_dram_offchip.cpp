// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <chrono>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>

#include "common/bfloat16.hpp"
#include "common/tt_backend_api_types.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of DRAM.
// It uses EnqueueProgram API to launch a reader kernel to read the data from DRAM to L1.
//
// Usage example:
//   ./test_dram_offchip --input-size <size in bytes>
//
////////////////////////////////////////////////////////////////////////////////

std::tuple<uint32_t, uint32_t, uint32_t> get_num_cores_for_given_input(
    tt_metal::Device *device, const uint64_t &input_size, const uint32_t &num_tiles);

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16(
    uint64_t num_bytes, int rand_max_float, int seed, float offset = 0.0f);

template <typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n);

std::tuple<tt_metal::Program, tt_metal::KernelID, uint32_t> create_program(
    tt_metal::Device *device,
    const int &num_cores_y,
    const int &num_cores_x,
    const uint32_t &num_reqs_at_a_time,
    const uint32_t &single_tile_size,
    const tt::DataFormat &tile_format);

bool assign_runtime_args_to_program(
    tt_metal::Device *device,
    tt_metal::Program &program,
    const int &num_cores_y,
    const int &num_cores_x,
    const tt_metal::KernelID &reader_kernel,
    const uint32_t &input_buffer_addr,
    const uint32_t &num_tiles_per_core,
    const uint32_t &num_reqs_at_a_time,
    const uint32_t &single_tile_size,
    const tt::DataFormat &tile_format);

int main(int argc, char **argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_fatal("Test not supported w/ slow dispatch, exiting");
    }
    bool pass = true;
    double dram_bandwidth = 0.0f;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        uint64_t input_size;
        uint32_t access_type;
        uint32_t num_reqs_at_a_time = 1;
        try {
            std::tie(input_size, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--input-size", 512 * 1024 * 1024);
            /* std::tie(num_reqs_at_a_time, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-reqs-at-a-time", 1); */
        } catch (const std::exception &e) {
            log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
            TT_ASSERT(false);
        }

        TT_ASSERT(input_size != 0, "--input-size should not be zero");

        tt::DataFormat tile_format = tt::DataFormat::Float16_b;
        uint32_t single_tile_size = tt_metal::detail::TileSize(tile_format);
        if (input_size % single_tile_size != 0) {
            auto align_to_single_tile = [=](uint64_t value) -> uint64_t {
                return ((value + (single_tile_size - 1)) / single_tile_size) * single_tile_size;
            };

            auto input_size_aligned = align_to_single_tile(input_size);
            log_info(LogTest, "read size {} is aligned to {} bytes", input_size, input_size_aligned);
            input_size = input_size_aligned;
        }
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        uint32_t num_tiles = static_cast<uint32_t>((input_size + single_tile_size - 1) / single_tile_size);
        auto [num_cores_x, num_cores_y, num_cores] = get_num_cores_for_given_input(device, input_size, num_tiles);

        uint32_t num_tiles_per_core = num_tiles / num_cores;
        if (num_tiles_per_core % num_reqs_at_a_time != 0) {
            log_error(
                LogTest,
                "{} input tiles per core should be divided by --num-of-reqs-at-a-time {} value",
                num_tiles_per_core,
                num_reqs_at_a_time);
            TT_ASSERT(false);
        }

        log_info(
            LogTest,
            "Measuring DRAM read bandwidth for input_size = {} bytes ({} tiles), using {} cores",
            input_size,
            num_tiles,
            num_cores);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            input_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::Buffer input_buffer(
            device, input_vec.size() * sizeof(u32), single_tile_size, tt_metal::BufferType::DRAM);
        tt_metal::WriteToBuffer(input_buffer, input_vec);

        auto [program, reader_kernel, cb_addr] =
            create_program(device, num_cores_y, num_cores_x, num_reqs_at_a_time, single_tile_size, tile_format);
        pass &= assign_runtime_args_to_program(
            device,
            program,
            num_cores_y,
            num_cores_x,
            reader_kernel,
            input_buffer.address(),
            num_tiles_per_core,
            num_reqs_at_a_time,
            single_tile_size,
            tile_format);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execution Application
        ////////////////////////////////////////////////////////////////////////////
        auto t_begin = std::chrono::steady_clock::now();
        tt_metal::EnqueueProgram(*::detail::GLOBAL_CQ, program, false);
        tt_metal::Finish(*::detail::GLOBAL_CQ);
        auto t_end = std::chrono::steady_clock::now();
        auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
        dram_bandwidth = (input_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0);
        log_info(LogTest, "EnqueueProgram : {:.3f}ms, {:.3f}GB/s", elapsed_us / 1000.0, dram_bandwidth);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto input_bf16 = unpack_uint32_vec_into_bfloat16_vec(input_vec);
        for (int y = 0; y < num_cores_y; ++y) {
            for (int x = 0; x < num_cores_x; ++x) {
                std::vector<uint32_t> result_vec;
                CoreCoord core = {(size_t)x, (size_t)y};
                tt_metal::detail::ReadFromDeviceL1(
                    device, core, cb_addr, num_reqs_at_a_time * single_tile_size, result_vec);
                auto result_bf16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);

                auto input_offset = num_tiles_per_core * (y * num_cores_x + x + 1);
                auto sliced_input = slice(
                    input_bf16,
                    (input_offset - num_reqs_at_a_time) * constants::TILE_HW,
                    input_offset * constants::TILE_HW - 1);

                if (!(sliced_input == result_bf16)) {
                    pass = false;
                }
            }
        }
    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    if (pass) {
        // goal is 90% of peak DRAM bandwidth performance for WH
        constexpr double WH_DRAM_BANDWIDTH = 384;
        double target_bandwidth = WH_DRAM_BANDWIDTH * 0.9;
        if (dram_bandwidth < target_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The DRAM bandwidth does not meet the criteria. "
                "Current: {:.3f}GB/s, goal: {:.3f}GB/s",
                dram_bandwidth,
                target_bandwidth);
        }
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}

std::tuple<uint32_t, uint32_t, uint32_t> get_num_cores_for_given_input(
    tt_metal::Device *device, const uint64_t &input_size, const uint32_t &num_tiles) {
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;

    if (num_tiles % num_cores != 0) {
        std::vector<std::tuple<uint32_t, uint32_t>> core_candidates;
        for (uint32_t y = 1; y <= num_cores_y; ++y) {
            for (uint32_t x = 1; x <= num_cores_x; ++x) {
                if (num_tiles % (x * y) == 0) {
                    core_candidates.push_back({y, x});
                }
            }
        }
        uint32_t num_proper_cores = 1;
        uint32_t num_proper_cores_x = 1;
        uint32_t num_proper_cores_y = 1;
        for (auto &core : core_candidates) {
            uint32_t y = std::get<0>(core);
            uint32_t x = std::get<1>(core);
            if (x * y >= num_proper_cores) {
                num_proper_cores_x = x;
                num_proper_cores_y = y;
                num_proper_cores = x * y;
            }
        }
        TT_ASSERT(num_proper_cores != 0, "input tiles cannot bt divided");
        log_warning(
            LogTest,
            "{} input tiles should be divided by {} cores. This run use {} x {} = {} cores",
            num_tiles,
            num_cores,
            num_proper_cores_y,
            num_proper_cores_x,
            num_proper_cores);
        num_cores_x = num_proper_cores_x;
        num_cores_y = num_proper_cores_y;
        num_cores = num_proper_cores;
    }

    return {num_cores_x, num_cores_y, num_cores};
}

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16(
    uint64_t num_bytes, int rand_max_float, int seed, float offset) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<std::uint32_t> vec(num_bytes / sizeof(std::uint32_t), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = rand_float() + offset;
        float num_2_float = rand_float() + offset;

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        // pack 2 uint16 into uint32
        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }
    return vec;
}

template <typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

std::tuple<tt_metal::Program, tt_metal::KernelID, uint32_t> create_program(
    tt_metal::Device *device,
    const int &num_cores_y,
    const int &num_cores_x,
    const uint32_t &num_reqs_at_a_time,
    const uint32_t &single_tile_size,
    const tt::DataFormat &tile_format) {
    tt_metal::Program program = tt_metal::Program();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
    CoreRange all_cores{.start = start_core, .end = end_core};

    uint32_t cb_addr = L1_UNRESERVED_BASE;
    uint32_t cb_tiles = num_reqs_at_a_time;
    uint32_t cb_size = cb_tiles * single_tile_size;
    for (int y = 0; y < num_cores_y; y++) {
        for (int x = 0; x < num_cores_x; x++) {
            int core_index = y * num_cores_x + x;
            CoreCoord core = {(std::size_t)x, (std::size_t)y};
            uint32_t cb_index = 0;
            auto cb = tt_metal::CreateCircularBuffer(program, cb_index, core, cb_tiles, cb_size, tile_format, cb_addr);
        }
    }

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/6_dram_offchip/kernels/"
        "reader_dram.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});
    return {std::move(program), reader_kernel, cb_addr};
}

bool assign_runtime_args_to_program(
    tt_metal::Device *device,
    tt_metal::Program &program,
    const int &num_cores_y,
    const int &num_cores_x,
    const tt_metal::KernelID &reader_kernel,
    const uint32_t &input_buffer_addr,
    const uint32_t &num_tiles_per_core,
    const uint32_t &num_reqs_at_a_time,
    const uint32_t &single_tile_size,
    const tt::DataFormat &tile_format) {
    bool pass = true;

    uint32_t num_blocks = num_tiles_per_core / num_reqs_at_a_time;
    for (int core_idx_y = 0; core_idx_y < num_cores_y; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_x; core_idx_x++) {
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            std::vector<uint32_t> reader_args = {
                (std::uint32_t)input_buffer_addr,
                (std::uint32_t)(num_tiles_per_core * (core_idx_y * num_cores_x + core_idx_x)),
                (std::uint32_t)num_blocks,
                (std::uint32_t)num_reqs_at_a_time};

            tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_args);
        }
    }

    return pass;
}
