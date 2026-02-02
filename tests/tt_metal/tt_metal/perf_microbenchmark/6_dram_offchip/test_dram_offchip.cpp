// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>

using namespace tt;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of DRAM accesses of Tensix cores. It creates
// a bfloat16 format DRAM buffer of a given input size. Every Tensix cores read
// from or write to the buffer whrere the amount of each core accesses is
// determined by split_work_to_cores function.
//
// Disclaimer:
//   - This benchmark is designed to support an input size larger than 4GB. But
//   current tt-metal does not seem to support buffer allocation larger than 4GB
//   yet.
//   - Also, detail::ReadFromBuffer API used in DRAM write test may take a long time if
//   the input size is large.
//
// Usage example:
//   ./test_dram_offchip
//     --input-size <size in bytes>
//     --access-type <0 for read access, 1 for write access>
//     --use-device-profiler (set to use device profiler for measurement)
//     --num-tests <count of tests>
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::vector<T> slice_vec(std::vector<T> const& v, int m, int n);

std::tuple<tt_metal::Program, tt_metal::KernelHandle, uint32_t> create_program(
    tt_metal::distributed::MeshDevice* device,
    const CoreRangeSet& all_cores,
    const uint32_t& num_reqs_at_a_time,
    const uint32_t& single_tile_size,
    const tt::DataFormat& tile_format,
    const uint32_t& access_type);

bool assign_runtime_args_to_program(
    tt_metal::Program& program,
    const uint32_t& num_cores,
    const uint32_t& num_cores_y,
    const uint32_t& num_cores_x,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const tt_metal::KernelHandle& kernel,
    const uint32_t& input_buffer_addr,
    const uint32_t& num_reqs_at_a_time,
    const uint32_t& single_tile_size,
    const tt::DataFormat& tile_format);

bool validation(
    tt_metal::distributed::MeshDevice* device,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& input_buffer,
    std::vector<uint32_t>& input_vec,
    const uint32_t& num_cores,
    const uint32_t& num_cores_y,
    const uint32_t& num_cores_x,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const uint32_t& cb_addr,
    const uint32_t& num_reqs_at_a_time,
    const uint32_t& single_tile_size,
    const uint32_t& access_type);

uint32_t get_dram_bandwidth(tt::ARCH arch);

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error(tt::LogTest, "Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    bool use_device_profiler;
    bool bypass_check = false;
    std::vector<double> dram_bandwidth;
    uint32_t num_tests = 10;
    uint64_t input_size;
    uint32_t access_type;
    uint32_t dram_bandwidth_spec = 0;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        try {
            std::tie(input_size, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--input-size", 512 * 1024 * 1024);

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
            TT_FATAL(false, "Command line arguments found exception: {}", e.what());
        }
        TT_FATAL(input_size != 0, "--input-size should not be zero");

        if (use_device_profiler) {
            bool device_profiler = tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled();
            TT_FATAL(
                device_profiler,
                "Before running the program, do one of the following in a shell: "
                "either export the environment variable by executing export TT_METAL_DEVICE_PROFILER=1, "
                "or run the program with TT_METAL_DEVICE_PROFILER=1 prefixed to the command");
        }

        tt::DataFormat tile_format = tt::DataFormat::Float16_b;
        uint32_t single_tile_size = tt::tile_size(tile_format);
        if (input_size % single_tile_size != 0) {
            auto align_to_single_tile = [=](uint64_t value) -> uint64_t {
                return ((value + (single_tile_size - 1)) / single_tile_size) * single_tile_size;
            };

            auto input_size_aligned = align_to_single_tile(input_size);
            log_info(LogTest, "input size {} is aligned to {} bytes", input_size, input_size_aligned);
            input_size = input_size_aligned;
        }
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        dram_bandwidth_spec = get_dram_bandwidth(device->arch());

        int clock_freq_mhz = get_tt_npu_clock(device->get_devices()[0]);

        uint32_t num_tiles = static_cast<uint32_t>((input_size + single_tile_size - 1) / single_tile_size);
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                split_work_to_cores(compute_with_storage_grid_size, num_tiles);

        log_info(
            LogTest,
            "Measuring DRAM bandwidth for input_size = {} bytes ({:.3f} MB, "
            "{} tiles), using {} cores",
            input_size,
            static_cast<double>(input_size) / 1024 / 1024,
            num_tiles,
            num_cores);

        ////////////////////////////////////////////////////////////////////////////
        //                      Input Setup
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            input_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

        // Create MeshBuffer for DRAM
        tt_metal::distributed::DeviceLocalBufferConfig device_local{
            .page_size = single_tile_size,
            .buffer_type = tt_metal::BufferType::DRAM,
        };
        tt_metal::distributed::ReplicatedBufferConfig global_buf{.size = input_vec.size() * sizeof(uint32_t)};
        auto input_buffer = tt_metal::distributed::MeshBuffer::create(global_buf, device_local, device.get());

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        uint32_t num_reqs_at_a_time = 1;
        auto [program, kernel, cb_addr] =
            create_program(device.get(), all_cores, num_reqs_at_a_time, single_tile_size, tile_format, access_type);
        pass &= assign_runtime_args_to_program(
            program,
            num_cores,
            num_cores_y,
            num_cores_x,
            core_group_1,
            core_group_2,
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_2,
            kernel,
            input_buffer->address(),
            num_reqs_at_a_time,
            single_tile_size,
            tile_format);

        ////////////////////////////////////////////////////////////////////////////
        //                      Copy Input To DRAM or L1
        ////////////////////////////////////////////////////////////////////////////
        if (access_type == 0) {
            tt_metal::distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), input_buffer, input_vec, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        } else {
            uint64_t input_offset = 0;
            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};
                uint32_t num_tiles_per_core = 0;
                if (core_group_1.contains(core)) {
                    num_tiles_per_core = num_tiles_per_core_group_1;
                } else if (core_group_2.contains(core)) {
                    num_tiles_per_core = num_tiles_per_core_group_2;
                } else {
                    TT_FATAL(false, "Core not in specified core ranges");
                }
                auto write_size = num_reqs_at_a_time * 512;
                auto sliced_input = slice_vec(input_vec, input_offset, input_offset + write_size - 1);
                tt_metal::detail::WriteToDeviceL1(device->get_devices()[0], core, cb_addr, sliced_input);
                input_offset += (num_tiles_per_core) * 512;
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execution Application
        ////////////////////////////////////////////////////////////////////////////
        log_info(LogTest, "Num tests {}", num_tests);
        auto mesh_workload = tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));

        for (uint32_t i = 0; i < num_tests; ++i) {
            auto t_begin = std::chrono::steady_clock::now();
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
            auto t_end = std::chrono::steady_clock::now();
            auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            dram_bandwidth.push_back((input_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0));
            log_info(
                LogTest,
                "Time elapsed for DRAM accesses: {:.3f}ms ({:.3f}GB/s)",
                elapsed_us / 1000.0,
                dram_bandwidth[i]);

            if (use_device_profiler) {
                unsigned long elapsed_cc = get_t0_to_any_riscfw_end_cycle(
                    device->get_devices()[0], mesh_workload.get_programs().begin()->second);
                elapsed_us = (double)elapsed_cc / clock_freq_mhz;
                dram_bandwidth.push_back((input_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0));
                log_info(
                    LogTest,
                    "Time elapsed using device profiler: {:.3f}ms ({:.3f}GB/s)",
                    elapsed_us / 1000.0,
                    dram_bandwidth[i]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass = validation(
            device.get(),
            input_buffer,
            input_vec,
            num_cores,
            num_cores_y,
            num_cores_x,
            core_group_1,
            core_group_2,
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_2,
            cb_addr,
            num_reqs_at_a_time,
            single_tile_size,
            access_type);

        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    auto avg_dram_bandwidth = calculate_average(dram_bandwidth);
    if (pass && !bypass_check) {
        // goal is 90% of peak DRAM bandwidth performance
        double target_bandwidth = static_cast<double>(dram_bandwidth_spec) * 0.9;
        if (avg_dram_bandwidth < target_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The DRAM bandwidth does not meet the criteria. "
                "Current: {:.3f}GB/s, goal: {:.3f}GB/s",
                avg_dram_bandwidth,
                target_bandwidth);
        }
    }

    // for csv
    log_info(tt::LogTest, "CSV_MICROBENCHMARK:title:test_dram_offchip");
    log_info(
        tt::LogTest,
        "CSV_INPUT:input-size:{}:access-type:{}:use-device-profiler:{}",
        input_size,
        ACCESS_TYPEToString(static_cast<ACCESS_TYPE>(access_type)),
        use_device_profiler);
    log_info(tt::LogTest, "CSV_OUTPUT:Bandwidth(GB/s):{:.3f}", avg_dram_bandwidth);
    log_info(tt::LogTest, "CSV_RESULT:pass:{}", pass);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16(
    uint64_t num_bytes, int rand_max_float, int seed, float offset) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<std::uint32_t> vec(num_bytes / sizeof(std::uint32_t), 0);
    for (unsigned int& elem : vec) {
        float num_1_float = rand_float() + offset;
        float num_2_float = rand_float() + offset;

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        // pack 2 uint16 into uint32
        elem = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }
    return vec;
}

template <typename T>
std::vector<T> slice_vec(std::vector<T> const& v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

std::tuple<tt_metal::Program, tt_metal::KernelHandle, uint32_t> create_program(
    tt_metal::distributed::MeshDevice* device,
    const CoreRangeSet& all_cores,
    const uint32_t& num_reqs_at_a_time,
    const uint32_t& single_tile_size,
    const tt::DataFormat& tile_format,
    const uint32_t& access_type) {
    tt_metal::Program program = tt_metal::Program();

    uint32_t cb_index = 0;
    uint32_t cb_tiles = num_reqs_at_a_time;
    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    tt_metal::CircularBufferConfig cb_config =
        tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{cb_index, tile_format}})
            .set_page_size(cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        (access_type == 0) ? "tests/tt_metal/tt_metal/perf_microbenchmark/"
                             "6_dram_offchip/kernels/reader_dram.cpp"
                           : "tests/tt_metal/tt_metal/perf_microbenchmark/"
                             "6_dram_offchip/kernels/writer_dram.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = (access_type == 0) ? tt_metal::DataMovementProcessor::RISCV_1
                                            : tt_metal::DataMovementProcessor::RISCV_0,
            .noc = (access_type == 0) ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default});
    return {std::move(program), reader_kernel, cb_addr};
}

bool assign_runtime_args_to_program(
    tt_metal::Program& program,
    const uint32_t& num_cores,
    const uint32_t& num_cores_y,
    const uint32_t& /*num_cores_x*/,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const tt_metal::KernelHandle& kernel,
    const uint32_t& input_buffer_addr,
    const uint32_t& num_reqs_at_a_time,
    const uint32_t& /*single_tile_size*/,
    const tt::DataFormat& /*tile_format*/) {
    bool pass = true;
    for (uint32_t i = 0, num_tiles_used = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }
        uint32_t num_blocks = num_tiles_per_core / num_reqs_at_a_time;
        const std::array kernel_args = {
            (std::uint32_t)input_buffer_addr,
            (std::uint32_t)(num_tiles_used),
            (std::uint32_t)num_blocks,
            (std::uint32_t)num_reqs_at_a_time};

        tt_metal::SetRuntimeArgs(program, kernel, core, kernel_args);
        num_tiles_used += num_tiles_per_core;
    }
    return pass;
}

bool validation(
    tt_metal::distributed::MeshDevice* device,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& input_buffer,
    std::vector<uint32_t>& input_vec,
    const uint32_t& num_cores,
    const uint32_t& num_cores_y,
    const uint32_t& /*num_cores_x*/,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const uint32_t& cb_addr,
    const uint32_t& num_reqs_at_a_time,
    const uint32_t& single_tile_size,
    const uint32_t& access_type) {
    if (access_type == 0) {
        auto input_bf16 = unpack_uint32_vec_into_bfloat16_vec(input_vec);
        for (uint32_t i = 0, input_offset = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_tiles_per_core = 0;
            if (core_group_1.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                TT_FATAL(false, "Core not in specified core ranges");
            }

            std::vector<uint32_t> result_vec;
            tt_metal::detail::ReadFromDeviceL1(
                device->get_devices()[0], core, cb_addr, num_reqs_at_a_time * single_tile_size, result_vec);
            auto result_bf16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
            auto sliced_input = slice_vec(
                input_bf16,
                (input_offset + num_tiles_per_core - num_reqs_at_a_time) * constants::TILE_HW,
                ((input_offset + num_tiles_per_core) * constants::TILE_HW) - 1);

            if (!(sliced_input == result_bf16)) {
                return false;
            }

            input_offset += num_tiles_per_core;
        }
    } else {
        std::vector<uint32_t> result_vec;
        log_info(LogTest, "ReadShard API may take a long time if the input size is large");
        tt_metal::distributed::ReadShard(
            device->mesh_command_queue(), result_vec, input_buffer, tt_metal::distributed::MeshCoordinate(0, 0), true);
        log_info(LogTest, "ReadShard API done");

        for (uint32_t i = 0, input_offset = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_tiles_per_core = 0;
            if (core_group_1.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                TT_FATAL(false, "Core not in specified core ranges");
            }

            uint32_t num_blocks = num_tiles_per_core / num_reqs_at_a_time;

            auto write_size = num_reqs_at_a_time * 512;
            auto sliced_input = slice_vec(input_vec, input_offset, input_offset + write_size - 1);
            for (int block = 0; block < num_blocks; ++block) {
                for (int req = 0; req < num_reqs_at_a_time * 512; ++req) {
                    auto index = input_offset + (block * (num_reqs_at_a_time * 512)) + req;
                    if (result_vec[index] != sliced_input[req]) {
                        return false;
                    }
                }
            }
            input_offset += (num_tiles_per_core) * 512;
        }
    }
    return true;
}

uint32_t get_dram_bandwidth(tt::ARCH arch) {
    constexpr uint32_t GS_DRAM_BANDWIDTH_GB_PER_SEC = 100;
    constexpr uint32_t WH_DRAM_BANDWIDTH_GB_PER_SEC = 384;

    uint32_t dram_bandwidth_gb_per_sec = 0;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        dram_bandwidth_gb_per_sec = WH_DRAM_BANDWIDTH_GB_PER_SEC;
    } else if (arch == tt::ARCH::GRAYSKULL) {
        dram_bandwidth_gb_per_sec = GS_DRAM_BANDWIDTH_GB_PER_SEC;
    }
    return dram_bandwidth_gb_per_sec;
}
