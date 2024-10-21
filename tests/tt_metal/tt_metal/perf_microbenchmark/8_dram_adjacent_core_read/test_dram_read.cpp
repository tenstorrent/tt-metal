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
#include <vector>

#include "common/bfloat8.hpp"
#include "common/bfloat16.hpp"
#include "common/tt_backend_api_types.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include "tt_metal/common/work_split.hpp"
#include <yaml-cpp/yaml.h>

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
std::vector<T> slice_vec(std::vector<T> const &v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

void get_max_page_size_and_num_pages(uint32_t num_tiles, uint32_t tile_size, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * tile_size;

    page_size = (8192 / tile_size) * tile_size;
    while (total_size % page_size != 0 && page_size >= tile_size) {
        page_size -= tile_size;
    }
    num_pages = total_size / page_size;
}

std::tuple<tt_metal::Program, tt_metal::KernelHandle, uint32_t> create_program(
    tt_metal::Device *device,
    const CoreRangeSet &all_cores,
    const uint32_t &single_tile_size,
    const tt::DataFormat &tile_format,
    uint32_t num_tiles_cb,
    uint32_t num_tiles_per_core,
    uint32_t k,
    uint32_t n,
    uint32_t num_blocks,
    uint32_t num_banks,
    std::vector<CoreCoord>all_cores_list,
    uint32_t bank_start_id,
    const uint32_t &input_buffer_addr) {
    tt_metal::Program program = tt_metal::Program();

    uint32_t start_tile_id = 0;
    uint32_t kt = k / 32;
    uint32_t nt = n / 32;
    uint32_t block_h = kt / num_blocks;
    uint32_t block_w = nt / num_banks;
    uint32_t block_num_tiles = block_h * block_w;

    uint32_t cb_index = 0;
    uint32_t cb_size = block_h * block_w * single_tile_size;
    uint32_t page_size, num_pages;
    get_max_page_size_and_num_pages(block_num_tiles, single_tile_size, page_size, num_pages);

    uint32_t cb_addr = device->get_base_allocator_addr(HalMemType::L1);
    tt_metal::CircularBufferConfig cb_config =
        tt_metal::CircularBufferConfig(cb_size, {{cb_index, tile_format}})
            .set_page_size(cb_index, single_tile_size);
    auto cb = tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t) input_buffer_addr,
        (std::uint32_t) start_tile_id,
        (std::uint32_t) num_blocks,
        (std::uint32_t) num_pages,
        (std::uint32_t) block_num_tiles,
        (std::uint32_t) page_size
    };

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/8_dram_adjacent_core_read/kernels/reader_dram.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> bank_ids;
    for (int i=0; i < all_cores_list.size(); i++) {
        auto core = all_cores_list[i];
        uint32_t bank_id = i + bank_start_id;
        uint32_t vc = bank_id & 0x3;

        bank_ids.push_back(bank_id);

        for (int j=0; j<i; ++j) {
            auto core_ = all_cores_list[j];

            if (core_.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) { // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        const std::array rt_args = {
            (std::uint32_t) bank_id,
            (std::uint32_t) vc
        };

        log_info("core: {}, vc: {}", core, vc);

        tt_metal::SetRuntimeArgs(program, reader_kernel, core, rt_args);
    }
    return {std::move(program), reader_kernel, cb_addr};
}


bool validation(
    tt_metal::Device *device,
    tt_metal::Buffer &input_buffer,
    std::vector<uint32_t> &input_vec,
    const uint32_t &num_cores,
    std::vector<CoreCoord> &all_cores,
    const uint32_t &num_tiles_per_core,
    const uint32_t &cb_addr,
    const uint32_t &single_tile_size,
    uint32_t num_tiles_cb,
    uint32_t df,
    uint32_t num_banks,
    uint32_t num_blocks,
    uint32_t block_h,
    uint32_t block_w,
    uint32_t num_datum_per_slice) {

    uint32_t core_id = 0;
    for (auto core: all_cores) {
        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromDeviceL1(
            device, core, cb_addr, num_tiles_cb * single_tile_size, result_vec);

        uint32_t num_datum_per_block = block_h * block_w * num_datum_per_slice;
        uint32_t tensor_slice_stride = core_id * num_datum_per_slice;
        uint32_t last_block_offset = (num_blocks - 1) * num_datum_per_block * num_banks;
        uint32_t start_index = tensor_slice_stride + last_block_offset;
        uint32_t num_slices = block_h * block_w;

        if (df == 0) {
            auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result_vec, true, true);
            auto input_bfp8 = unpack_bfp8_tiles_into_float_vec(input_vec, true, true);

            for (uint32_t i=0; i < num_slices; ++i) {
                uint32_t input_step = start_index + i * num_datum_per_slice * num_banks;
                std::vector<float> input_slice(input_bfp8.begin() + input_step, input_bfp8.begin() + input_step + num_datum_per_slice);
                uint32_t result_step = i * num_datum_per_slice;
                std::vector<float> result_slice(result_bfp8.begin() + result_step, result_bfp8.begin() + result_step + num_datum_per_slice);

                if (input_slice != result_slice) {
                    return false;
                }
            }

        } else {
            auto result_bf16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
            auto input_bf16 = unpack_uint32_vec_into_bfloat16_vec(input_vec);

            for (uint32_t i=0; i < num_slices; ++i) {
                uint32_t input_step = start_index + i * num_datum_per_slice * num_banks;
                std::vector<bfloat16> input_slice(input_bf16.begin() + input_step, input_bf16.begin() + input_step + num_datum_per_slice);
                uint32_t result_step = i * num_datum_per_slice;
                std::vector<bfloat16> result_slice(result_bf16.begin() + result_step, result_bf16.begin() + result_step + num_datum_per_slice);

                if (input_slice != result_slice) {
                    return false;
                }
            }
        }
        core_id ++;
    }
    return true;
}

uint32_t get_dram_bandwidth(tt::ARCH arch) {
    constexpr uint32_t GS_DRAM_BANDWIDTH_GB_PER_SEC = 100;
    constexpr uint32_t WH_DRAM_BANDWIDTH_GB_PER_SEC = 384;

    uint32_t dram_bandwidth_gb_per_sec = 0;
    if (arch == tt::ARCH::WORMHOLE || arch == tt::ARCH::WORMHOLE_B0) {
        dram_bandwidth_gb_per_sec = WH_DRAM_BANDWIDTH_GB_PER_SEC;
    } else if (arch == tt::ARCH::GRAYSKULL) {
        dram_bandwidth_gb_per_sec = GS_DRAM_BANDWIDTH_GB_PER_SEC;
    }
    return dram_bandwidth_gb_per_sec;
}


void get_dram_reader_core_coords_grayskull(
    tt_metal::Device* device, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {

    // hardcoded for grayskull
    uint32_t full_grid_size_y = 12;

    // get all the logical coord
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // get dram banks and coords
    uint32_t num_banks = device->num_dram_channels();
    uint32_t max_bank_id = num_banks - 1;
    std::vector<CoreCoord> dram_coord_phy;
    for (int i = 0; i < num_banks; ++i) {
        dram_coord_phy.push_back(device->dram_core_from_dram_channel(i));
    }

    // get worker logical coords
    std::vector<CoreCoord> all_worker_cores_logical;
    for (int i = 0; i < num_cores_x; ++i) {
        for (int j = 0; j < num_cores_y; ++j) {
            all_worker_cores_logical.push_back(CoreCoord(i, j));
        }
    }

    // get y coords of the workers
    std::vector<uint32_t> all_worker_cores_y_physical;
    uint32_t max_worker_y_physical = 0;
    uint32_t min_worker_y_physical = 10000;
    for (int i = 0; i < num_cores_y; ++i) {
        auto core_phy = device->worker_core_from_logical_core(CoreCoord(0, i));
        all_worker_cores_y_physical.push_back(core_phy.y);
        if (core_phy.y > max_worker_y_physical) {
            max_worker_y_physical = core_phy.y;
        }
        if (core_phy.y < min_worker_y_physical) {
            min_worker_y_physical = core_phy.y;
        }
    }

    // get the harvested rows, we treat dram and eth cores as harvested as well
    std::vector<uint32_t> harvested_rows;
    for (int i = 0; i < full_grid_size_y; ++i) {
        auto y = i;

        if (std::find(all_worker_cores_y_physical.begin(), all_worker_cores_y_physical.end(), y) ==
            all_worker_cores_y_physical.end()) {
            harvested_rows.push_back(y);
        }
    }

    // get the ajacent cores of DRAM banks
    std::vector<CoreCoord> adj_core_physical;
    for (int i = 0; i < num_banks; ++i) {
        auto dram_core = dram_coord_phy[i];
        uint32_t adj_core_x = dram_core.x;
        uint32_t adj_core_y = dram_core.y + 1;
        adj_core_physical.push_back(CoreCoord(adj_core_x, adj_core_y));
    }

    // move worker if they are in the harvested rows
    for (auto& coord : adj_core_physical) {
        auto y = coord.y;

        // if row is harvested, move core down by 1
        while (std::find(harvested_rows.begin(), harvested_rows.end(), y) != harvested_rows.end() and y < (full_grid_size_y - 1)) {
            y += 1;
        }

        coord.y = y;
    }

    // find the logical coord from physical coord
    std::vector<CoreCoord> adj_core_logical_realloc;
    for (int i = 0; i < adj_core_physical.size(); ++i) {
        for (int j = 0; j < all_worker_cores_logical.size(); ++j) {
            auto core = device->worker_core_from_logical_core(all_worker_cores_logical[j]);
            if (adj_core_physical[i] == core) {
                adj_core_logical_realloc.push_back(all_worker_cores_logical[j]);
            }
        }
    }

    // create sets
    std::set<CoreRange> all_cores_set;
    for (int i = 0; i < num_banks; ++i) {
        all_cores_set.insert(CoreRange(adj_core_logical_realloc[i]));
    }
    all_cores = CoreRangeSet(all_cores_set);
    all_cores_ordered = adj_core_logical_realloc;
}

void get_dram_reader_core_coords_wormhole_b0(
    tt_metal::Device* device, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {
    // hardcoded for wh_b0
    uint32_t full_grid_size_y = 12;
    uint32_t x_step = 3;

    // get all the logical coord
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // get dram banks and coords
    uint32_t num_banks = device->num_dram_channels();
    uint32_t max_bank_id = num_banks - 1;
    std::vector<CoreCoord> dram_coord_phy; dram_coord_phy.reserve(num_banks);
    for (int i = 0; i < num_banks; ++i) {
        dram_coord_phy.push_back(device->dram_core_from_dram_channel(i));
    }

    // get worker logical coords
    std::vector<CoreCoord> all_worker_cores_logical; all_worker_cores_logical.reserve(num_cores_x * num_cores_y);
    for (int i = 0; i < num_cores_x; ++i) {
        for (int j = 0; j < num_cores_y; ++j) {
            all_worker_cores_logical.push_back(CoreCoord(i, j));
        }
    }

    // get y coords of the workers
    std::vector<uint32_t> all_worker_cores_y_physical; all_worker_cores_y_physical.reserve(num_cores_y);
    uint32_t max_worker_y_physical = 0;
    uint32_t min_worker_y_physical = 10000;
    for (int i = 0; i < num_cores_y; ++i) {
        auto core_phy = device->worker_core_from_logical_core(CoreCoord(0, i));
        all_worker_cores_y_physical.push_back(core_phy.y);
        if (core_phy.y > max_worker_y_physical) {
            max_worker_y_physical = core_phy.y;
        }
        if (core_phy.y < min_worker_y_physical) {
            min_worker_y_physical = core_phy.y;
        }
    }

    // get the harvested rows, we treat dram and eth cores as harvested as well
    std::vector<uint32_t> harvested_rows;
    for (int i = 0; i < full_grid_size_y; ++i) {
        auto y = i;

        if (std::find(all_worker_cores_y_physical.begin(), all_worker_cores_y_physical.end(), y) ==
            all_worker_cores_y_physical.end()) {
            harvested_rows.push_back(y);
        }
    }

    // get the ajacent cores of DRAM banks
    std::vector<CoreCoord> adj_core_physical; adj_core_physical.reserve(num_banks);
    for (int i = 0; i < num_banks; ++i) {
        auto dram_core = dram_coord_phy[i];
        uint32_t adj_core_x = dram_core.x + 1;
        uint32_t adj_core_y = dram_core.y;
        adj_core_physical.push_back(CoreCoord(adj_core_x, adj_core_y));
    }

    // split the adjacent coords into two groups, because DRAM banks has two cols
    std::vector<CoreCoord> adj_core_physical_g1; adj_core_physical_g1.reserve(num_banks);
    std::vector<size_t> adj_core_physical_y_g1; adj_core_physical_y_g1.reserve(num_banks);
    std::vector<CoreCoord> adj_core_physical_g2; adj_core_physical_g2.reserve(num_banks);
    std::vector<size_t> adj_core_physical_y_g2; adj_core_physical_y_g2.reserve(num_banks);
    for (auto core : adj_core_physical) {
        if (core.x == adj_core_physical.front().x) {
            adj_core_physical_g1.push_back(core);
        } else {
            adj_core_physical_g2.push_back(core);
        }
    }
    std::vector<int> indices_g1(adj_core_physical_g1.size());
    std::vector<int> indices_g2(adj_core_physical_g2.size());
    std::iota(indices_g1.begin(), indices_g1.end(), 0);
    std::iota(indices_g2.begin(), indices_g2.end(), 0);
    std::sort(indices_g1.begin(), indices_g1.end(), [&adj_core_physical_g1](int i1, int i2) {
        return adj_core_physical_g1[i1].y < adj_core_physical_g1[i2].y;
    });
    std::sort(indices_g2.begin(), indices_g2.end(), [&adj_core_physical_g2](int i1, int i2) {
        return adj_core_physical_g2[i1].y < adj_core_physical_g2[i2].y;
    });
    std::rotate(indices_g1.begin(), indices_g1.end() - 1, indices_g1.end());
    std::rotate(indices_g2.begin(), indices_g2.end() - 1, indices_g2.end());

    std::vector<int> indices_g1_realloc(adj_core_physical_g1.size());
    std::vector<int> indices_g2_realloc(adj_core_physical_g2.size());
    for (int new_index = 0; new_index < indices_g1.size(); ++new_index) {
        indices_g1_realloc[indices_g1[new_index]] = new_index;
    }
    for (int new_index = 0; new_index < indices_g2.size(); ++new_index) {
        indices_g2_realloc[indices_g2[new_index]] = new_index;
    }

    std::sort(adj_core_physical_g1.begin(), adj_core_physical_g1.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    std::sort(adj_core_physical_g2.begin(), adj_core_physical_g2.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    std::rotate(adj_core_physical_g1.begin(), adj_core_physical_g1.end() - 1, adj_core_physical_g1.end());
    std::rotate(adj_core_physical_g2.begin(), adj_core_physical_g2.end() - 1, adj_core_physical_g2.end());

    for (auto core : adj_core_physical_g1) {
        adj_core_physical_y_g1.push_back(core.y);
    }
    for (auto core : adj_core_physical_g2) {
        adj_core_physical_y_g2.push_back(core.y);
    }

    // move the workers, if they are on harvested rows
    auto process_group = [&](std::vector<CoreCoord>& group, std::vector<size_t>& group_y, uint32_t x_step) {
        for (auto& coord : group) {
            auto y = coord.y;

            if (std::find(harvested_rows.begin(), harvested_rows.end(), y) != harvested_rows.end() ||
                std::count(group_y.begin(), group_y.end(), y) >= 2) {
                auto adjust_coord = [&](int start, int end, int step) {
                    bool found_new_row = false;
                    for (int j = start; step > 0 ? j <= end : j >= end; j += step) {
                        if (std::find(harvested_rows.begin(), harvested_rows.end(), j) == harvested_rows.end() &&
                            std::count(group_y.begin(), group_y.end(), j) == 0) {
                            coord.y = j;
                            coord.x += x_step;
                            x_step--;
                            found_new_row = true;
                            break;
                        }
                    }
                    if (not found_new_row) {
                        for (int j = start; step > 0 ? j <= end : j >= end; j += step) {
                            if (std::find(harvested_rows.begin(), harvested_rows.end(), j) == harvested_rows.end()) {
                                coord.y = j;
                                coord.x += x_step;
                                x_step--;
                                found_new_row = true;
                                break;
                            }
                        }
                    }
                };

                if (y >= max_bank_id) {
                    adjust_coord(max_worker_y_physical, min_worker_y_physical, -1);
                } else {
                    adjust_coord(min_worker_y_physical, max_worker_y_physical, 1);
                }
            }
        }
    };
    // move the workers, if they are on harvested rows
    process_group(adj_core_physical_g1, adj_core_physical_y_g1, x_step);
    process_group(adj_core_physical_g2, adj_core_physical_y_g2, x_step);

    // merge two group into one
    std::vector<CoreCoord> adj_core_physical_realloc; adj_core_physical_realloc.reserve(num_banks);
    for (int i = 0; i < indices_g1_realloc.size(); ++i) {
        adj_core_physical_realloc.push_back(adj_core_physical_g1[indices_g1_realloc[i]]);
    }
    for (int i = 0; i < indices_g2_realloc.size(); ++i) {
        adj_core_physical_realloc.push_back(adj_core_physical_g2[indices_g2_realloc[i]]);
    }

    // find the logical coord from physical coord
    std::vector<CoreCoord> adj_core_logical_realloc; adj_core_logical_realloc.reserve(num_banks);
    for (int i = 0; i < adj_core_physical_realloc.size(); ++i) {
        for (int j = 0; j < all_worker_cores_logical.size(); ++j) {
            auto core = device->worker_core_from_logical_core(all_worker_cores_logical[j]);
            if (adj_core_physical_realloc[i] == core) {
                adj_core_logical_realloc.push_back(all_worker_cores_logical[j]);
            }
        }
    }

    // create sets
    std::set<CoreRange> all_cores_set;
    for (int i = 0; i < num_banks; ++i) {
        all_cores_set.insert(CoreRange(adj_core_logical_realloc[i]));
    }
    all_cores = CoreRangeSet(all_cores_set);
    all_cores_ordered = adj_core_logical_realloc;
}

int main(int argc, char **argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error("Test not supported w/ slow dispatch, exiting");
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

    log_info("start DRAM benchmark");

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        try {
            std::tie(k, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--k", 8192);

            std::tie(n, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--n", 12*128);

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
        } catch (const std::exception &e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
            TT_ASSERT(false);
        }

        if (use_device_profiler) {
#if !defined(TRACY_ENABLE)
            log_error(
                LogTest,
                "Metal library and test code should be build with "
                "profiler option using ./build_metal.sh --enable-profiler");
#endif
            auto device_profiler = getenv("TT_METAL_DEVICE_PROFILER");
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
        uint32_t total_banks = 12;
        if (df == 0) {
            input_size = k * n * 1088 / 1024;
            tile_format = tt::DataFormat::Bfp8_b;
        } else if (df == 1) {
            input_size = k * n * 2;
            tile_format = tt::DataFormat::Float16_b;
        } else {
            TT_THROW("Input data format {} is invalid. Please change.", df);
        }
        uint32_t kt = k / 32;
        uint32_t nt = n / 32;
        uint32_t block_h = kt / num_blocks;
        uint32_t block_w = nt / num_banks;
        uint32_t num_datum_per_slice = 32 * 32;
        uint32_t eth_coord_y_phy = 6;

        uint32_t single_tile_size = tt_metal::detail::TileSize(tile_format);
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
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        dram_bandwidth_spec = get_dram_bandwidth(device->arch());

        TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "device must be wh_b0");

        int clock_freq_mhz = get_tt_npu_clock(device);

        uint32_t num_tiles = static_cast<uint32_t>((input_size + single_tile_size - 1) / single_tile_size);
        uint32_t num_cores = num_banks; // number of DRAM banks
        // uint32_t num_banks_all = 12;

        CoreRangeSet all_cores;
        std::vector<CoreCoord> all_cores_list;
        if (device->arch() == tt::ARCH::WORMHOLE_B0) {
            get_dram_reader_core_coords_wormhole_b0(device, all_cores, all_cores_list);
        } else {
            get_dram_reader_core_coords_grayskull(device, all_cores, all_cores_list);
        }

        uint32_t num_tiles_per_core = num_tiles / num_cores;
        uint32_t num_tiles_cb = num_tiles_per_core / num_blocks;

        for (auto core: all_cores_list) {
            auto phys_core = device->worker_core_from_logical_core(core);
            log_info("logical core: {}, physical coer: {}", core, phys_core);
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
            input_vec = create_random_vector_of_bfp8(
                input_size, true, 100, 1234);
        } else {
            // input_vec = create_constant_vector_of_bfloat16(
            //     input_size * total_banks / num_banks, 100);
            input_vec = create_random_vector_of_bfloat16(
                input_size, 100, 1234);
        }

        tt_metal::Buffer input_buffer(
            device, input_vec.size() * sizeof(uint32_t), single_tile_size, tt_metal::BufferType::DRAM);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [program, kernel, cb_addr] = create_program(device, all_cores, single_tile_size, tile_format, num_tiles_cb, num_tiles_per_core, k, n, num_blocks, num_banks, all_cores_list, bank_start_id, input_buffer.address());

        ////////////////////////////////////////////////////////////////////////////
        //                      Copy Input To DRAM or L1
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::detail::WriteToBuffer(input_buffer, input_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execution Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::detail::CompileProgram(device, program);

        log_info(LogTest, "Num tests {}", num_tests);
        for (uint32_t i = 0; i < num_tests; ++i) {
            auto t_begin = std::chrono::steady_clock::now();
            EnqueueProgram(device->command_queue(), program, false);
            Finish(device->command_queue());
            tt_metal::DumpDeviceProfileResults(device, program);
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
            input_buffer,
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

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    auto avg_dram_bandwidth = calculate_average(dram_bandwidth);
    if (pass && bypass_check == false) {
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
