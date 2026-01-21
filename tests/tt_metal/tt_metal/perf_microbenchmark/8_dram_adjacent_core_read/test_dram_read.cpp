// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
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
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/test_utils/bfloat_utils.hpp"

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
//     --k
//     --n
//     --num-blocks
//     --k
//     --k
//     --num-tests <count of tests>
//     --data-type
//     --num-banks
//     --bank-start-id
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::vector<T> slice_vec(const std::vector<T>& v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

// Returns max_page_size (for non-last pages), num_pages, and last_page_size
std::tuple<uint32_t, uint32_t, uint32_t> get_max_page_size_and_num_pages(
    uint32_t num_tiles, uint32_t tile_size, tt::ARCH arch) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * tile_size;

    // NOC_MAX_BURST_SIZE depends on architecture:
    // - Wormhole B0: NOC_MAX_BURST_WORDS (256) * NOC_WORD_BYTES (256/8 = 32) = 8192 bytes
    // - Blackhole: NOC_MAX_BURST_WORDS (256) * NOC_WORD_BYTES (512/8 = 64) = 16384 bytes
    uint32_t max_noc_burst_size;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        max_noc_burst_size = 8192;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        max_noc_burst_size = 16384;
    } else {
        TT_THROW("unknown architecture: {}", arch);
    }

    // Align max_noc_burst_size down to tile_size boundaries
    uint32_t max_page_size = (max_noc_burst_size / tile_size) * tile_size;

    if (total_size <= max_page_size) {
        // If total size fits in one page, use it
        return {total_size, 1, total_size};
    }  // Use max_page_size for all pages except the last one
    uint32_t num_pages = (total_size + max_page_size - 1) / max_page_size;
    uint32_t last_page_size = total_size - ((num_pages - 1) * max_page_size);
    return {max_page_size, num_pages, last_page_size};
}

std::tuple<tt_metal::Program, tt_metal::KernelHandle, uint32_t> create_program(
    tt_metal::distributed::MeshDevice* device,
    const CoreRangeSet& all_cores,
    const uint32_t& single_tile_size,
    const tt::DataFormat& tile_format,
    uint32_t /*num_tiles_cb*/,
    uint32_t /*num_tiles_per_core*/,
    uint32_t k,
    uint32_t n,
    uint32_t num_blocks,
    uint32_t num_banks,
    std::vector<CoreCoord> all_cores_list,
    uint32_t bank_start_id,
    const uint32_t& input_buffer_addr) {
    tt_metal::Program program = tt_metal::Program();

    uint32_t start_tile_id = 0;
    uint32_t kt = k / 32;
    uint32_t nt = n / 32;
    uint32_t block_h = kt / num_blocks;
    uint32_t block_w = nt / num_banks;
    uint32_t block_num_tiles = block_h * block_w;

    uint32_t cb_index = 0;
    uint32_t cb_size = block_h * block_w * single_tile_size;
    auto [page_size, num_pages, last_page_size] =
        get_max_page_size_and_num_pages(block_num_tiles, single_tile_size, device->arch());

    log_info(tt::LogTest, "page_size: {}, num_pages: {}, last_page_size: {}", page_size, num_pages, last_page_size);

    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    tt_metal::CircularBufferConfig cb_config =
        tt_metal::CircularBufferConfig(cb_size, {{cb_index, tile_format}}).set_page_size(cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)input_buffer_addr,
        (std::uint32_t)start_tile_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)num_pages,
        (std::uint32_t)block_num_tiles,
        (std::uint32_t)page_size,
        (std::uint32_t)last_page_size};

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/8_dram_adjacent_core_read/kernels/reader_dram.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> bank_ids;
    for (int i = 0; i < all_cores_list.size(); i++) {
        auto core = all_cores_list[i];
        uint32_t bank_id = i + bank_start_id;
        uint32_t vc = bank_id & 0x3;

        bank_ids.push_back(bank_id);

        for (int j = 0; j < i; ++j) {
            auto core_ = all_cores_list[j];

            if (core_.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {  // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        const std::array rt_args = {(std::uint32_t)bank_id, (std::uint32_t)vc};

        log_info(tt::LogTest, "core: {}, vc: {}", core, vc);

        tt_metal::SetRuntimeArgs(program, reader_kernel, core, rt_args);
    }
    return {std::move(program), reader_kernel, cb_addr};
}

bool validation(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    std::vector<uint32_t>& input_vec,
    const uint32_t& /*num_cores*/,
    std::vector<CoreCoord>& all_cores,
    const uint32_t& /*num_tiles_per_core*/,
    const uint32_t& cb_addr,
    const uint32_t& single_tile_size,
    uint32_t num_tiles_cb,
    uint32_t df,
    uint32_t num_banks,
    uint32_t num_blocks,
    uint32_t block_h,
    uint32_t block_w,
    uint32_t num_datum_per_slice) {
    uint32_t core_id = 0;
    for (auto core : all_cores) {
        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromDeviceL1(
            device->get_devices()[0], core, cb_addr, num_tiles_cb * single_tile_size, result_vec);

        uint32_t num_datum_per_block = block_h * block_w * num_datum_per_slice;
        uint32_t tensor_slice_stride = core_id * num_datum_per_slice;
        uint32_t last_block_offset = (num_blocks - 1) * num_datum_per_block * num_banks;
        uint32_t start_index = tensor_slice_stride + last_block_offset;
        uint32_t num_slices = block_h * block_w;

        if (df == 0) {
            auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result_vec, true, true);
            auto input_bfp8 = unpack_bfp8_tiles_into_float_vec(input_vec, true, true);

            for (uint32_t i = 0; i < num_slices; ++i) {
                uint32_t input_step = start_index + (i * num_datum_per_slice * num_banks);
                std::vector<float> input_slice(
                    input_bfp8.begin() + input_step, input_bfp8.begin() + input_step + num_datum_per_slice);
                uint32_t result_step = i * num_datum_per_slice;
                std::vector<float> result_slice(
                    result_bfp8.begin() + result_step, result_bfp8.begin() + result_step + num_datum_per_slice);

                if (input_slice != result_slice) {
                    return false;
                }
            }

        } else if (df == 2) {
            auto result_bfp4 = unpack_bfp4_tiles_into_float_vec(result_vec, true, true);
            auto input_bfp4 = unpack_bfp4_tiles_into_float_vec(input_vec, true, true);

            for (uint32_t i = 0; i < num_slices; ++i) {
                uint32_t input_step = start_index + (i * num_datum_per_slice * num_banks);
                std::vector<float> input_slice(
                    input_bfp4.begin() + input_step, input_bfp4.begin() + input_step + num_datum_per_slice);
                uint32_t result_step = i * num_datum_per_slice;
                std::vector<float> result_slice(
                    result_bfp4.begin() + result_step, result_bfp4.begin() + result_step + num_datum_per_slice);

                if (input_slice != result_slice) {
                    return false;
                }
            }

        } else {
            auto result_bf16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
            auto input_bf16 = unpack_uint32_vec_into_bfloat16_vec(input_vec);

            for (uint32_t i = 0; i < num_slices; ++i) {
                uint32_t input_step = start_index + (i * num_datum_per_slice * num_banks);
                std::vector<bfloat16> input_slice(
                    input_bf16.begin() + input_step, input_bf16.begin() + input_step + num_datum_per_slice);
                uint32_t result_step = i * num_datum_per_slice;
                std::vector<bfloat16> result_slice(
                    result_bf16.begin() + result_step, result_bf16.begin() + result_step + num_datum_per_slice);

                if (input_slice != result_slice) {
                    return false;
                }
            }
        }
        core_id++;
    }
    return true;
}

uint32_t get_dram_bandwidth(tt::ARCH arch) {
    constexpr uint32_t GS_DRAM_BANDWIDTH_GB_PER_SEC = 100;
    constexpr uint32_t WH_DRAM_BANDWIDTH_GB_PER_SEC = 384;
    constexpr uint32_t BH_DRAM_BANDWIDTH_GB_PER_SEC = 512;

    uint32_t dram_bandwidth_gb_per_sec = 0;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        dram_bandwidth_gb_per_sec = WH_DRAM_BANDWIDTH_GB_PER_SEC;
    } else if (arch == tt::ARCH::GRAYSKULL) {
        dram_bandwidth_gb_per_sec = GS_DRAM_BANDWIDTH_GB_PER_SEC;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        dram_bandwidth_gb_per_sec = BH_DRAM_BANDWIDTH_GB_PER_SEC;
    }
    return dram_bandwidth_gb_per_sec;
}

void get_optimal_dram_bank_to_reader_assignment(
    tt_metal::distributed::MeshDevice* device,
    std::vector<CoreCoord>& all_worker_cores_ordered,
    CoreRangeSet& all_worker_cores,
    tt_metal::NOC noc) {
    all_worker_cores_ordered = device->get_optimal_dram_bank_to_logical_worker_assignment(noc);
    std::set<CoreRange> all_cores_set;
    for (const auto& worker_core : all_worker_cores_ordered) {
        all_cores_set.insert(CoreRange(worker_core));
    }
    all_worker_cores = CoreRangeSet(all_cores_set);
}

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error(tt::LogTest, "Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    bool use_device_profiler = false;
    bool bypass_check = false;
    uint32_t df = 0;
    std::vector<double> dram_bandwidth;
    uint32_t num_tests = 1;
    uint32_t num_blocks = 8;
    uint64_t k = 8192, n = 128;
    uint32_t dram_bandwidth_spec = 0;
    uint32_t num_banks = 1;
    uint32_t bank_start_id = 1;

    log_info(tt::LogTest, "start DRAM benchmark");

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        try {
            std::tie(k, input_args) = test_args::get_command_option_uint64_and_remaining_args(input_args, "--k", 8192);

            std::tie(n, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--n", 12 * 128);

            std::tie(num_blocks, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--num-blocks", 8);

            std::tie(num_tests, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 1);

            std::tie(use_device_profiler, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--use-device-profiler");

            std::tie(bypass_check, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

            std::tie(df, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--data-type", 0);

            std::tie(num_banks, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-banks", 12);

            std::tie(bank_start_id, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--bank-start-id", 0);

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
            TT_FATAL(false, "Command line arguments found exception: {}", e.what());
        }

        if (use_device_profiler) {
            bool device_profiler = tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled();
            TT_FATAL(
                device_profiler,
                "Before running the program, do one of the following in a shell: "
                "either export the environment variable by executing export TT_METAL_DEVICE_PROFILER=1, "
                "or run the program with TT_METAL_DEVICE_PROFILER=1 prefixed to the command");
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Parameters Setup
        ////////////////////////////////////////////////////////////////////////////
        uint32_t input_size = 0;
        tt::DataFormat tile_format = tt::DataFormat::Bfp8_b;
        if (df == 0) {
            input_size = k * n * 1088 / 1024;
            tile_format = tt::DataFormat::Bfp8_b;
        } else if (df == 1) {
            input_size = k * n * 2;
            tile_format = tt::DataFormat::Float16_b;
        } else if (df == 2) {
            input_size = k * n * 576 / 1024;
            tile_format = tt::DataFormat::Bfp4_b;
        } else {
            TT_THROW("Input data format {} is invalid. Please change.", df);
        }
        uint32_t kt = k / 32;
        uint32_t nt = n / 32;
        uint32_t block_h = kt / num_blocks;
        uint32_t block_w = nt / num_banks;
        uint32_t num_datum_per_slice = 32 * 32;

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

        TT_FATAL(
            device->arch() == ARCH::WORMHOLE_B0 or device->arch() == ARCH::BLACKHOLE, "device must be wh_b0 or bh");

        uint32_t num_tiles = static_cast<uint32_t>((input_size + single_tile_size - 1) / single_tile_size);
        uint32_t num_cores = num_banks;  // number of DRAM banks
        // uint32_t num_banks_all = 12;

        CoreRangeSet all_cores;
        std::vector<CoreCoord> all_cores_list;
        get_optimal_dram_bank_to_reader_assignment(device.get(), all_cores_list, all_cores, tt_metal::NOC::NOC_0);

        // Slice all_cores_list to only use the first num_banks cores
        all_cores_list = slice_vec(all_cores_list, bank_start_id, bank_start_id + num_banks - 1);

        // Rebuild all_cores CoreRangeSet with only the selected cores
        std::set<CoreRange> selected_cores_set;
        for (const auto& core : all_cores_list) {
            selected_cores_set.insert(CoreRange(core));
        }
        all_cores = CoreRangeSet(selected_cores_set);

        uint32_t num_tiles_per_core = num_tiles / num_cores;
        uint32_t num_tiles_cb = num_tiles_per_core / num_blocks;

        for (auto core : all_cores_list) {
            auto virtual_core = device->worker_core_from_logical_core(core);
            log_info(tt::LogTest, "logical core: {}, virtual core: {}", core, virtual_core);
        }

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
        std::vector<uint32_t> input_vec;
        if (tile_format == tt::DataFormat::Bfp8_b) {
            // input_vec = create_constant_vector_of_bfp8(
            //     input_size, 100, true);
            input_vec = test_utils::create_random_vector_of_bfp8(input_size, true, 100, 1234);
        } else if (tile_format == tt::DataFormat::Bfp4_b) {
            input_vec = test_utils::create_random_vector_of_bfp4(input_size, true, 100, 1234);
        } else {
            // input_vec = create_constant_vector_of_bfloat16(
            //     input_size * total_banks / num_banks, 100);
            input_vec = create_random_vector_of_bfloat16(input_size, 100, 1234);
        }

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
        auto [program, kernel, cb_addr] = create_program(
            device.get(),
            all_cores,
            single_tile_size,
            tile_format,
            num_tiles_cb,
            num_tiles_per_core,
            k,
            n,
            num_blocks,
            num_banks,
            all_cores_list,
            bank_start_id,
            input_buffer->address());

        ////////////////////////////////////////////////////////////////////////////
        //                      Copy Input To DRAM or L1
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), input_buffer, input_vec, false);
        tt_metal::distributed::Finish(device->mesh_command_queue());

        ////////////////////////////////////////////////////////////////////////////
        //                      Execution Application
        ////////////////////////////////////////////////////////////////////////////
        auto mesh_workload = tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));

        log_info(LogTest, "Num tests {}", num_tests);
        for (uint32_t i = 0; i < num_tests; ++i) {
            auto t_begin = std::chrono::steady_clock::now();
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
            tt_metal::ReadMeshDeviceProfilerResults(*device);
            auto t_end = std::chrono::steady_clock::now();
            auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            dram_bandwidth.push_back((input_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0));
            log_info(
                LogTest,
                "Time elapsed for DRAM accesses: {:.3f}ms ({:.3f}GB/s)",
                elapsed_us / 1000.0,
                dram_bandwidth[i]);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass = validation(
            device,
            input_vec,
            num_cores,
            all_cores_list,
            num_tiles_per_core,
            cb_addr,
            single_tile_size,
            num_tiles_cb,
            df,
            num_banks,
            num_blocks,
            block_h,
            block_w,
            num_datum_per_slice);

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

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    }
    log_error(LogTest, "Test Failed");
    return 1;
}
