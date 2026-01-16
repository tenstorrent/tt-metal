// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
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
#include <tt-metalium/dispatch_core_common.hpp>
#include "impl/context/metal_context.hpp"
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
// A tensix core that's next to a DRAM bank reads from the bank, and writes to
// the neighbour receiver tensix core. It creates a bfloat16/bfloat8_b format
// DRAM buffer of a given input size, and write it to the DRAM banks in the round
// robin style.
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
std::vector<T> slice_vec(std::vector<T> const& v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

void get_max_page_size_and_num_pages(
    uint32_t num_tiles_w,
    uint32_t num_tiles_h,
    uint32_t tile_size,
    uint32_t& page_size,
    uint32_t& num_pages,
    uint32_t& num_pages_w_per_receiver) {
    uint64_t half_row_bytes = static_cast<uint64_t>(num_tiles_w / 2) * tile_size;
    TT_FATAL(num_tiles_w % 2 == 0, "num_tiles_w {} must be divisible by 2", num_tiles_w);

    page_size = (8192 / tile_size) * tile_size;
    // Each receiver core receives half the data, so each receiver cores's block size is half of the total block size
    while (half_row_bytes % page_size != 0 && page_size > tile_size) {
        page_size -= tile_size;
    }
    TT_FATAL(page_size % tile_size == 0, "page_size must be a multiple of tile_size!");
    num_pages = num_tiles_w * num_tiles_h * tile_size / page_size;
    num_pages_w_per_receiver = half_row_bytes / page_size;
}

std::tuple<tt_metal::Program, tt_metal::KernelHandle, uint32_t> create_program(
    tt_metal::distributed::MeshDevice* device,
    const CoreRangeSet& all_dram_reader_cores,
    const CoreRangeSet& /*all_l1_receiver_cores*/,
    const uint32_t& single_tile_size,
    const tt::DataFormat& tile_format,
    uint32_t /*num_tiles_cb*/,
    uint32_t /*num_tiles_per_core*/,
    uint32_t k,
    uint32_t n,
    uint32_t num_blocks,
    uint32_t num_banks,
    std::vector<CoreCoord> all_dram_reader_cores_ordered,
    std::vector<CoreCoord> all_l1_writer_cores_ordered,
    uint32_t bank_start_id,
    const uint32_t& input_buffer_addr) {
    tt_metal::Program program = tt_metal::Program();

    uint32_t start_tile_id = 0;
    uint32_t kt = k / 32;
    uint32_t nt = n / 32;
    uint32_t block_h = kt / num_blocks;
    uint32_t block_w = nt / num_banks;
    uint32_t block_num_tiles = block_h * block_w;

    // DRAM reader CB
    uint32_t reader_cb_index = 0;
    uint32_t reader_cb_size = block_h * block_w * single_tile_size * 3;
    uint32_t page_size, num_pages, num_pages_w_per_receiver;
    get_max_page_size_and_num_pages(block_w, block_h, single_tile_size, page_size, num_pages, num_pages_w_per_receiver);

    log_info(tt::LogTest, "Input block size: {}x{}, num_blocks: {}", block_h, block_w, num_blocks);
    log_info(
        tt::LogTest,
        "Pages set up as page_size: {}, num_pages: {}, num_pages_w_per_receiver: {}",
        page_size,
        num_pages,
        num_pages_w_per_receiver);

    uint32_t reader_cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    tt_metal::CircularBufferConfig reader_cb_config =
        tt_metal::CircularBufferConfig(reader_cb_size, {{reader_cb_index, tile_format}})
            .set_page_size(reader_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_dram_reader_cores, reader_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_buffer_addr,
        (std::uint32_t)start_tile_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)num_pages,
        (std::uint32_t)block_num_tiles,
        (std::uint32_t)page_size,
        (std::uint32_t)tt_metal::NOC::RISCV_0_default};

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/9_dram_adjacent_read_remote_l1_write/kernels/reader_dram.cpp",
        all_dram_reader_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)num_blocks,
        (std::uint32_t)num_pages_w_per_receiver,
        (std::uint32_t)block_h,
        (std::uint32_t)block_num_tiles,
        (std::uint32_t)page_size,
        (std::uint32_t)tt_metal::NOC::RISCV_0_default};

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/9_dram_adjacent_read_remote_l1_write/kernels/writer_l1.cpp",
        all_dram_reader_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = writer_compile_time_args});

    std::vector<uint32_t> bank_ids;
    for (int i = 0; i < all_dram_reader_cores_ordered.size(); i++) {
        auto core = all_dram_reader_cores_ordered[i];
        uint32_t bank_id = i + bank_start_id;
        uint32_t vc = bank_id & 0x1;

        bank_ids.push_back(bank_id);

        for (int j = 0; j < i; ++j) {
            auto core_ = all_dram_reader_cores_ordered[j];

            if (core_.y == core.y and ((bank_id & 0x1) == (bank_ids[j] & 0x1))) {  // same vc and same row
                vc = (vc + 1) & 0x1;
                break;
            }
        }

        const std::array reader_rt_args = {(std::uint32_t)bank_id, (std::uint32_t)vc};

        log_info(tt::LogTest, "core: {}, vc: {}", core, vc);

        tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);

        auto writer_core1 = all_l1_writer_cores_ordered[i * 2];
        auto writer_core_phy1 = device->worker_core_from_logical_core(writer_core1);
        auto writer_core2 = all_l1_writer_cores_ordered[(i * 2) + 1];
        auto writer_core_phy2 = device->worker_core_from_logical_core(writer_core2);

        log_info(tt::LogTest, "writer_core_phy1: {}", writer_core_phy1);
        log_info(tt::LogTest, "writer_core_phy2: {}", writer_core_phy2);

        const std::array writer_rt_args = {
            (std::uint32_t)(vc + 2) & 0x3,
            // First L1 receiver core coordinates
            (std::uint32_t)writer_core_phy1.x,
            (std::uint32_t)writer_core_phy1.y,
            // Second L1 receiver core coordinates
            (std::uint32_t)writer_core_phy2.x,
            (std::uint32_t)writer_core_phy2.y};

        tt_metal::SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);
    }
    return {std::move(program), reader_kernel, reader_cb_addr};
}

template <typename T>
bool validate_data(
    const std::vector<T>& result_data,
    const std::vector<T>& input_data,
    uint32_t block_h,
    uint32_t block_w_per_receiver,
    uint32_t block_w,
    uint32_t datums_per_tile,
    uint32_t num_banks,
    uint32_t input_start_index_for_core) {
    for (uint32_t r = 0; r < block_h; ++r) {
        for (uint32_t c = 0; c < block_w_per_receiver; ++c) {
            uint32_t one_row_bytes = block_w * datums_per_tile * num_banks;
            uint32_t input_step = input_start_index_for_core + (r * one_row_bytes) + (c * datums_per_tile * num_banks);
            auto input_begin = input_data.begin() + input_step;
            auto input_end = input_begin + datums_per_tile;
            std::vector<T> input_slice(input_begin, input_end);

            uint32_t result_step = (r * (datums_per_tile * block_w_per_receiver)) + (c * datums_per_tile);
            auto result_begin = result_data.begin() + result_step;
            auto result_end = result_begin + datums_per_tile;
            std::vector<T> result_slice(result_begin, result_end);

            if (input_slice != result_slice) {
                return false;
            }
        }
    }
    return true;
}

bool validation(
    tt_metal::distributed::MeshDevice* device,
    std::vector<uint32_t>& input_vec,
    uint32_t num_cores,
    std::vector<CoreCoord>& all_cores,
    uint32_t /*num_tiles_per_core*/,
    uint32_t cb_addr,
    uint32_t single_tile_size,
    uint32_t num_tiles_cb,
    uint32_t df,
    uint32_t num_banks,
    uint32_t num_blocks,
    uint32_t block_h,  // block_h per core
    uint32_t block_w,  // block_w per core
    uint32_t block_w_per_receiver,
    uint32_t datums_per_tile) {  // 32x32

    uint32_t core_id = 0;
    uint32_t num_datum_per_block = block_h * block_w * num_cores * datums_per_tile;
    uint32_t last_block_offset = (num_blocks - 1) * num_datum_per_block;
    for (auto core : all_cores | std::views::take(num_cores * 2)) {
        uint32_t dram_bank_id = core_id / 2;  // A pair of two cores share a dram bank
        uint32_t tile_stride_over_dram_banks = dram_bank_id * datums_per_tile;
        uint32_t is_second_core = core_id % 2;
        // Second core in a dram bank pair has an offset of half a block from that dram bank
        uint32_t receiver_core_pair_offset = is_second_core * datums_per_tile * block_w_per_receiver * num_banks;
        uint32_t input_start_index_for_core =
            last_block_offset + tile_stride_over_dram_banks + receiver_core_pair_offset;

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromDeviceL1(
            device->get_devices()[0], core, cb_addr, num_tiles_cb / 2 * single_tile_size, result_vec);

        if (df == 0) {  // BFP4
            auto result_bfp4 = unpack_bfp4_tiles_into_float_vec(result_vec, true, true);
            auto input_bfp4 = unpack_bfp4_tiles_into_float_vec(input_vec, true, true);
            if (!validate_data<float>(
                    result_bfp4,
                    input_bfp4,
                    block_h,
                    block_w_per_receiver,
                    block_w,
                    datums_per_tile,
                    num_banks,
                    input_start_index_for_core)) {
                return false;
            }
        } else if (df == 1) {  // BFP8
            auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result_vec, true, true);
            auto input_bfp8 = unpack_bfp8_tiles_into_float_vec(input_vec, true, true);
            if (!validate_data<float>(
                    result_bfp8,
                    input_bfp8,
                    block_h,
                    block_w_per_receiver,
                    block_w,
                    datums_per_tile,
                    num_banks,
                    input_start_index_for_core)) {
                return false;
            }
        } else if (df == 2) {  // BFLOAT16
            auto result_bf16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
            auto input_bf16 = unpack_uint32_vec_into_bfloat16_vec(input_vec);
            if (!validate_data<bfloat16>(
                    result_bf16,
                    input_bf16,
                    block_h,
                    block_w_per_receiver,
                    block_w,
                    datums_per_tile,
                    num_banks,
                    input_start_index_for_core)) {
                return false;
            }
        }
        core_id++;
    }
    log_info(tt::LogTest, "Validation passed.");
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

void get_l1_writer_core_coords_wormhole_b0(
    std::vector<CoreCoord>& all_dram_reader_cores, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {
    // Place writers horizontally next to DRAM readers in logical space (no column harvesting for WH)
    for (auto dram_reader_core : all_dram_reader_cores) {
        all_cores_ordered.push_back(CoreCoord(dram_reader_core.x + 1, dram_reader_core.y));
        all_cores_ordered.push_back(CoreCoord(dram_reader_core.x + 2, dram_reader_core.y));
    }
    std::set<CoreRange> all_cores_set;
    for (auto core : all_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    all_cores = CoreRangeSet(all_cores_set);
}

void get_l1_writer_core_coords_blackhole(
    std::vector<CoreCoord>& all_dram_reader_cores, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {
    // Place writers horizontally next to DRAM readers in logical space (column harvesting enabled for BH incrementing
    // in logical space can lead to physical physical columns being skipped when placing writers next to readers)
    for (auto dram_reader_core : all_dram_reader_cores) {
        all_cores_ordered.push_back(CoreCoord(dram_reader_core.x + 1, dram_reader_core.y));
        all_cores_ordered.push_back(CoreCoord(dram_reader_core.x + 2, dram_reader_core.y));
    }
    std::set<CoreRange> all_cores_set;
    for (auto core : all_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    all_cores = CoreRangeSet(all_cores_set);
}

void get_l1_writer_core_coords_grayskull(
    std::vector<CoreCoord>& all_dram_reader_cores, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {
    for (auto dram_reader_core : all_dram_reader_cores) {
        all_cores_ordered.push_back(CoreCoord(dram_reader_core.x, dram_reader_core.y + 1));
        all_cores_ordered.push_back(CoreCoord(dram_reader_core.x + 1, dram_reader_core.y + 1));
    }
    std::set<CoreRange> all_cores_set;
    for (auto core : all_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    all_cores = CoreRangeSet(all_cores_set);
}

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error(tt::LogTest, "Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    bool use_device_profiler = false;
    bool bypass_check = false;
    uint32_t df = 2;
    std::vector<double> dram_bandwidth;
    uint32_t num_tests = 1;
    uint32_t num_blocks = 8;
    uint64_t k = 8192, n = 128;
    uint32_t dram_bandwidth_spec = 0;
    uint32_t num_banks = 1;
    uint32_t bank_start_id = 1;

    log_info(tt::LogTest, "start DRAM benchmark");

    // try {
    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    try {
        std::tie(k, input_args) = test_args::get_command_option_uint64_and_remaining_args(input_args, "--k", 8192);

        std::tie(n, input_args) = test_args::get_command_option_uint64_and_remaining_args(input_args, "--n", 12 * 128);

        std::tie(num_blocks, input_args) =
            test_args::get_command_option_uint64_and_remaining_args(input_args, "--num-blocks", 8);

        std::tie(num_tests, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 1);

        std::tie(use_device_profiler, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--use-device-profiler");

        std::tie(bypass_check, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

        std::tie(df, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--data-type", 2);

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
        if (df == 0) {  // BFP4
            input_size = k * n * (512 + 64) / 1024;
            tile_format = tt::DataFormat::Bfp4_b;
        } else if (df == 1) {  // BFP8
            input_size = k * n * 1088 / 1024;
            tile_format = tt::DataFormat::Bfp8_b;
        } else if (df == 2) {  // BFLOAT16
            input_size = k * n * 2;
            tile_format = tt::DataFormat::Float16_b;
        } else {
            TT_THROW("Input data format {} is invalid. Please change.", df);
        }
        uint32_t kt = k / 32;
        uint32_t nt = n / 32;
        uint32_t block_h = kt / num_blocks;
        uint32_t block_w = nt / num_banks;
        uint32_t block_w_per_receiver = block_w / 2;
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
        tt_metal::DispatchCoreConfig dispatch_core_config;
        dispatch_core_config =
            tt_metal::DispatchCoreConfig{tt_metal::DispatchCoreType::WORKER, tt_metal::DispatchCoreAxis::ROW};
        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            {device_id}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
        auto device = reserved_devices[device_id];
        dram_bandwidth_spec = get_dram_bandwidth(device->arch());

        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        [[maybe_unused]] uint32_t num_cores_x = compute_with_storage_grid_size.x;
        [[maybe_unused]] uint32_t num_cores_y = compute_with_storage_grid_size.y;
        log_debug(tt::LogTest, "device x : {}", num_cores_x);
        log_debug(tt::LogTest, "device y : {}", num_cores_y);

        uint32_t num_tiles = static_cast<uint32_t>((input_size + single_tile_size - 1) / single_tile_size);
        uint32_t num_cores = num_banks;  // number of DRAM banks

        CoreRangeSet all_dram_reader_cores;
        std::vector<CoreCoord> all_dram_reader_cores_ordered;
        CoreRangeSet all_l1_receiver_cores;
        std::vector<CoreCoord> all_l1_writer_cores_ordered;
        get_optimal_dram_bank_to_reader_assignment(
            device.get(), all_dram_reader_cores_ordered, all_dram_reader_cores, tt_metal::NOC::NOC_0);

        if (device->arch() == tt::ARCH::BLACKHOLE) {
            get_l1_writer_core_coords_blackhole(
                all_dram_reader_cores_ordered, all_l1_receiver_cores, all_l1_writer_cores_ordered);
        } else if (device->arch() == tt::ARCH::WORMHOLE_B0) {
            get_l1_writer_core_coords_wormhole_b0(
                all_dram_reader_cores_ordered, all_l1_receiver_cores, all_l1_writer_cores_ordered);
        } else {
            get_l1_writer_core_coords_grayskull(
                all_dram_reader_cores_ordered, all_l1_receiver_cores, all_l1_writer_cores_ordered);
        }

        uint32_t num_tiles_per_core = num_tiles / num_cores;
        uint32_t num_tiles_cb = num_tiles_per_core / num_blocks;

        log_info(tt::LogTest, "all_dram_reader_cores");
        for (auto core : all_dram_reader_cores_ordered) {
            auto phys_core = device->worker_core_from_logical_core(core);
            log_info(tt::LogTest, "logical core: {}, virtual core: {}", core, phys_core);
        }
        log_info(tt::LogTest, "all_l1_writer_cores");
        for (auto core : all_l1_writer_cores_ordered) {
            auto phys_core = device->worker_core_from_logical_core(core);
            log_info(tt::LogTest, "logical core: {}, virtual core: {}", core, phys_core);
        }

        log_info(
            LogTest,
            "Measuring DRAM bandwidth for input_size = {} bytes ({:.3f} MB, "
            "{} tiles), using {} DRAM reading cores",
            input_size,
            static_cast<double>(input_size) / 1024 / 1024,
            num_tiles,
            num_cores);

        ////////////////////////////////////////////////////////////////////////////
        //                      Input Setup
        ////////////////////////////////////////////////////////////////////////////
        // DEBUGGING: Create a vector of bfloat16s where each element contains the tile number

        std::vector<uint32_t> input_vec;
        if (tile_format == tt::DataFormat::Bfp4_b) {
            input_vec = test_utils::create_random_vector_of_bfp4(input_size, false, 100, 1234);
        } else if (tile_format == tt::DataFormat::Bfp8_b) {
            input_vec = test_utils::create_random_vector_of_bfp8(input_size, false, 100, 1234);
        } else {
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
        auto [program, kernel, output_cb_addr] = create_program(
            device.get(),
            all_dram_reader_cores,
            all_l1_receiver_cores,
            single_tile_size,
            tile_format,
            num_tiles_cb,
            num_tiles_per_core,
            k,
            n,
            num_blocks,
            num_banks,
            all_dram_reader_cores_ordered,
            all_l1_writer_cores_ordered,
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
            device.get(),
            input_vec,
            num_cores,
            all_l1_writer_cores_ordered,
            num_tiles_per_core,
            output_cb_addr,
            single_tile_size,
            num_tiles_cb,
            df,
            num_banks,
            num_blocks,
            block_h,
            block_w,
            block_w_per_receiver,
            num_datum_per_slice);

        if (!pass) {
            log_info(LogTest, "Validation failed");
        }

        pass &= device->close();
        // } catch (const std::exception& e) {
        //     pass = false;
        //     // Capture the exception error message
        //     log_error(LogTest, "{}", e.what());
        //     // Capture system call errors that may have returned from driver/kernel
        //     log_error(LogTest, "System error message: {}", std::strerror(errno));
        // }

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
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
