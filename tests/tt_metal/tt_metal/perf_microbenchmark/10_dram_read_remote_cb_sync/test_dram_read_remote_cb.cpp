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
#include "tests/tt_metal/test_utils/tilization.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/compute/matmul/matmul_utils.hpp"
#include <yaml-cpp/yaml.h>

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
std::vector<T> slice_vec(std::vector<T> const &v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

void get_max_page_size_and_num_pages(uint32_t num_tiles, uint32_t num_datums_per_tile, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * num_datums_per_tile;

    page_size = (8192 / num_datums_per_tile) * num_datums_per_tile;
    while (total_size % page_size != 0 && page_size >= num_datums_per_tile) {
        page_size -= num_datums_per_tile;
    }
    num_pages = total_size / page_size;
}

std::tuple<tt_metal::Program, tt_metal::KernelHandle, uint32_t> create_program(
    tt_metal::Device *device,
    const CoreRangeSet &dram_reader_core,
    const CoreRangeSet &l1_receiver_cores,
    const uint32_t &single_tile_size,
    const tt::DataFormat &tile_format,
    uint32_t k,
    uint32_t n,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t num_receivers,
    uint32_t num_mixed_df_layers,
    uint32_t cb_padding,
    std::shared_ptr<tt::tt_metal::Buffer> input_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> output_buffer
    ) {

    log_info("created program");

    tt_metal::Program program = tt_metal::Program();

    auto all_cores = dram_reader_core.merge(l1_receiver_cores);

    uint32_t start_tile_id = 0;
    uint32_t kt = k / 32;
    uint32_t nt = n / 32;
    uint32_t block_h = kt / num_blocks;
    uint32_t num_tile_rows_write = block_h;
    uint32_t block_w = nt;
    uint32_t block_num_tiles = block_h * block_w;

    // DRAM reader CB
    uint32_t reader_cb_index = 0;
    uint32_t reader_cb_size = block_h * block_w * single_tile_size * 3;
    uint32_t reader_page_size, reader_num_pages;
    get_max_page_size_and_num_pages(block_num_tiles, single_tile_size, reader_page_size, reader_num_pages);

    uint32_t receiver_block_num_tile = block_h * block_w / num_receivers;
    uint32_t writer_page_size, writer_num_pages;
    get_max_page_size_and_num_pages(block_w / num_receivers, single_tile_size, writer_page_size, writer_num_pages);

    log_info("writer_page_size: {}", writer_page_size);
    log_info("writer_num_pages: {}", writer_num_pages);

    uint32_t reader_cb_addr = device->get_base_allocator_addr(HalMemType::L1);
    tt_metal::CircularBufferConfig reader_cb_config =
        tt_metal::CircularBufferConfig(reader_cb_size, {{reader_cb_index, tile_format}})
            .set_page_size(reader_cb_index, single_tile_size);
    auto reader_cb = tt_metal::CreateCircularBuffer(program, dram_reader_core, reader_cb_config);

    // mixed cb dataformat
    uint32_t next_layer_num_blocks = num_blocks * 1;
    uint32_t next_layer_block_h = kt / next_layer_num_blocks;
    uint32_t next_layer_block_num_tiles = next_layer_block_h * block_w;
    uint32_t next_layer_num_tile_rows_write = next_layer_block_h;
    uint32_t next_layer_receiver_block_num_tile = next_layer_block_num_tiles / num_receivers;

    uint32_t next_layer_single_tile_size = single_tile_size;
    if (tile_format == tt::DataFormat::Float16_b) {
        next_layer_single_tile_size = 1088;
    } else {
        next_layer_single_tile_size = 2048;
    }
    uint32_t next_layer_reader_page_size, next_layer_reader_num_pages;
    get_max_page_size_and_num_pages(block_num_tiles, next_layer_single_tile_size, next_layer_reader_page_size, next_layer_reader_num_pages);

    uint32_t next_layer_writer_page_size, next_layer_writer_num_pages;
    get_max_page_size_and_num_pages(block_w / num_receivers, next_layer_single_tile_size, next_layer_writer_page_size, next_layer_writer_num_pages);

    // L1 receiver CB
    uint32_t receiver_cb_index = 0;
    uint32_t receiver_cb_size = block_h * block_w * single_tile_size * cb_num_blocks / num_receivers + cb_padding;
    uint32_t receiver_page_size = 32;
    uint32_t receiver_cb_addr = output_buffer->address();
    tt_metal::CircularBufferConfig receiver_cb_config =
        tt_metal::CircularBufferConfig(receiver_cb_size, {{receiver_cb_index, tile_format}})
            .set_page_size(receiver_cb_index, receiver_page_size).set_globally_allocated_address(*output_buffer);
    auto receiver_cb = tt_metal::CreateCircularBuffer(program, l1_receiver_cores, receiver_cb_config);

    log_info("reader_cb_size: {}", reader_cb_size);
    log_info("receiver_cb_size: {}", receiver_cb_size);

    // semaphore
    std::vector<uint32_t> pages_acked_semaphore_ids(num_receivers);
    std::vector<uint32_t> pages_sent_semaphore_ids(num_receivers);
    for (uint32_t i=0; i < num_receivers; ++i) {
        pages_acked_semaphore_ids[i] = tt_metal::CreateSemaphore(program, all_cores, INVALID);
        pages_sent_semaphore_ids[i] = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    }

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) input_buffer->address(),
        (std::uint32_t) start_tile_id,
        (std::uint32_t) tt_metal::NOC::RISCV_0_default,
        (std::uint32_t) num_mixed_df_layers
    };

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/10_dram_read_remote_cb_sync/kernels/reader_dram.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) tt_metal::NOC::RISCV_0_default,
        (std::uint32_t) receiver_cb_addr,
        (std::uint32_t) receiver_cb_size,
        (std::uint32_t) num_receivers,
        (std::uint32_t) num_mixed_df_layers
    };

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/10_dram_read_remote_cb_sync/kernels/writer_l1.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = writer_compile_time_args});

    std::vector<uint32_t> receiver_compile_time_args = {
        (std::uint32_t) reader_cb_addr,
        (std::uint32_t) receiver_cb_size,
        (std::uint32_t) num_mixed_df_layers,
    };

    auto receiver_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/10_dram_read_remote_cb_sync/kernels/receiver_l1.cpp",
        l1_receiver_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = receiver_compile_time_args});

    // reader rt
    auto dram_reader_core_coord = dram_reader_core.ranges().begin()->start_coord;
    log_info("dram_reader_core_coord: {}", dram_reader_core_coord);
    auto dram_reader_core_coord_physical = device->worker_core_from_logical_core(dram_reader_core_coord);
    uint32_t bank_id = 0;
    uint32_t vc = bank_id & 0x1;
    std::vector<uint32_t> reader_rt_args = {
        (std::uint32_t) bank_id,
        (std::uint32_t) vc
    };
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i%2 == 0 ? reader_page_size : next_layer_reader_page_size);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i%2 == 0 ? reader_num_pages : next_layer_reader_num_pages);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i%2 == 0 ? num_blocks : next_layer_num_blocks);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i%2 == 0 ? block_num_tiles : next_layer_block_num_tiles);
    }
    tt_metal::SetRuntimeArgs(program, reader_kernel, dram_reader_core_coord, reader_rt_args);

    // writer rt
    std::vector<CoreCoord> l1_receiver_core_coords;
    for (auto l1_receiver_core_coord : *l1_receiver_cores.ranges().begin()) {
        l1_receiver_core_coords.push_back(l1_receiver_core_coord);
    }
    std::vector<uint32_t> writer_rt_args;
    for (uint32_t i=0; i < num_receivers; ++i) {
        auto l1_receiver_core_coord_physical = device->worker_core_from_logical_core(l1_receiver_core_coords[i]);
        writer_rt_args.push_back(l1_receiver_core_coord_physical.x);
    }
    for (uint32_t i=0; i < num_receivers; ++i) {
        auto l1_receiver_core_coord_physical = device->worker_core_from_logical_core(l1_receiver_core_coords[i]);
        writer_rt_args.push_back(l1_receiver_core_coord_physical.y);
    }
    for (uint32_t i=0; i < num_receivers; ++i) {
        writer_rt_args.push_back(pages_acked_semaphore_ids[i]);
    }
    for (uint32_t i=0; i < num_receivers; ++i) {
        writer_rt_args.push_back(pages_sent_semaphore_ids[i]);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i%2 == 0 ? writer_page_size : next_layer_writer_page_size);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i%2 == 0 ? writer_num_pages : next_layer_writer_num_pages);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i%2 == 0 ? num_blocks : next_layer_num_blocks);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i%2 == 0 ? block_num_tiles : next_layer_block_num_tiles);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i%2 == 0 ? single_tile_size : next_layer_single_tile_size);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i%2 == 0 ? num_tile_rows_write : next_layer_num_tile_rows_write);
    }
    tt_metal::SetRuntimeArgs(program, writer_kernel, dram_reader_core_coord, writer_rt_args);

    // reciever rt
    for (uint32_t i=0; i < num_receivers; ++i) {
        std::vector<uint32_t> receiver_rt_args = {
            (std::uint32_t) vc & 0x3,
            (std::uint32_t) dram_reader_core_coord_physical.x,
            (std::uint32_t) dram_reader_core_coord_physical.y
        };
        vc ++;

        receiver_rt_args.push_back(pages_acked_semaphore_ids[i]);
        receiver_rt_args.push_back(pages_sent_semaphore_ids[i]);

        for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
            receiver_rt_args.push_back(i%2 == 0 ? single_tile_size : next_layer_single_tile_size);
        }
        for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
            receiver_rt_args.push_back(i%2 == 0 ? num_blocks : next_layer_num_blocks);
        }
        for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
            receiver_rt_args.push_back(i%2 == 0 ? receiver_block_num_tile : next_layer_receiver_block_num_tile);
        }

        log_info("l1_receiver_core_coords: {}", l1_receiver_core_coords[i]);

        tt_metal::SetRuntimeArgs(program, receiver_kernel, l1_receiver_core_coords[i], receiver_rt_args);
    }

    return {std::move(program), reader_kernel, reader_cb_addr};
}

float to_float(bfloat16 bfloat16_num) {
    return bfloat16_num.to_float();
}

float pcc(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }

    int n = x.size();
    float mean_x = 0, mean_y = 0;
    for (int i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    float numerator = 0, sum_sq_x = 0, sum_sq_y = 0;
    for (int i = 0; i < n; ++i) {
        float diff_x = x[i] - mean_x;
        float diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    float denominator = std::sqrt(sum_sq_x * sum_sq_y);
    if (denominator == 0) {
        return 0;
    }

    return numerator / denominator;
}

bool validation_bfp8_b(
    tt::deprecated::Tensor<float> input_tensor,
    const tt::DataFormat &data_format,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t kt,
    uint32_t nt,
    std::shared_ptr<tt::tt_metal::Buffer> out_buffer
) {
    bool pass = true;
    std::vector<float> golden_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0); // Initialize with zeros
    std::vector<float> result_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0);
    auto num_datums_per_cb = kt * nt * 32 * 32 / num_blocks * cb_num_blocks;

    std::vector<float> result_untilized;
    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromBuffer(out_buffer, result);
    auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result, true, false);
    result_untilized = tt::test_utils::untilize(result_bfp8, kt*32 / num_blocks * cb_num_blocks, nt*32);

    const auto& values = input_tensor.get_values();

    int index = 0;
    for (int i = 0; i < kt * nt * 32 * 32; ++i) {
        golden_vec[index] = float(values[i]);
        index++;

        if (index == num_datums_per_cb) {
            index = 0;
        }
    }

    for (int i=0; i<result_untilized.size(); ++i) {
        result_vec[i] = result_untilized[i];
    }

    pass &= pcc(golden_vec, result_vec) >= 0.9999;
    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}


bool validation_fp16(
    tt::deprecated::Tensor<bfloat16> input_tensor,
    const tt::DataFormat &data_format,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t kt,
    uint32_t nt,
    std::shared_ptr<tt::tt_metal::Buffer> out_buffer
) {
    bool pass = true;
    std::vector<float> golden_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0); // Initialize with zeros
    std::vector<float> result_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0);
    auto num_datums_per_cb = kt * nt * 32 * 32 / num_blocks * cb_num_blocks;

    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromBuffer(out_buffer, result);
    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result);
    auto result_flat_layout = convert_to_flat_layout(result_bfp16);
    auto result_untilized = tt::test_utils::untilize(result_flat_layout, kt*32 / num_blocks * cb_num_blocks, nt*32);

    const auto& values = input_tensor.get_values();

    int index = 0;
    for (int i = 0; i < kt * nt * 32 * 32; ++i) {
        golden_vec[index] = to_float(values[i]);
        index++;

        if (index == num_datums_per_cb) {
            index = 0;
        }
    }

    for (int i=0; i<result_untilized.size(); ++i) {
        result_vec[i] = to_float(static_cast<bfloat16>(result_untilized[i]));
    }

    // for (uint32_t i=0; i < golden_vec.size(); ++i ) {
    //     std::cout << golden_vec[i] << " ";

    //     if ((i+1) %32 == 0) {
    //         std::cout << std::endl;
    //     }
    // }

    // std::cout << std::endl;
    // std::cout << std::endl;

    // for (uint32_t i=0; i < result_vec.size(); ++i ) {
    //     std::cout << result_vec[i] << " ";

    //     if ((i+1) %32 == 0) {
    //         std::cout << std::endl;
    //     }
    // }

    pass &= (golden_vec == result_vec);
    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}

bool validation_mixed_df(
    tt::deprecated::Tensor<bfloat16> input_tensor_fp16,
    tt::deprecated::Tensor<float> input_tensor_fp8,
    const tt::DataFormat &data_format,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t kt,
    uint32_t nt,
    std::shared_ptr<tt::tt_metal::Buffer> out_buffer,
    uint32_t num_mixed_df_layers
) {
    bool pass = true;


    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromBuffer(out_buffer, result);

    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result);
    auto result_flat_layout_bfp16 = convert_to_flat_layout(result_bfp16);
    auto result_untilized_fp16 = tt::test_utils::untilize(result_flat_layout_bfp16, kt*32 / num_blocks * cb_num_blocks, nt*32);

    auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result, true, false);
    std::vector<float> result_untilized_fp8 = tt::test_utils::untilize(result_bfp8, result_bfp8.size() / (nt*32), nt*32);

    log_info("num_cb_elements: {}", result_untilized_fp16.size());
    log_info("num_cb_elements: {}", result_untilized_fp8.size());

    int max_cb_num_datums = result_untilized_fp8.size();
    int min_cb_num_datums = result_untilized_fp16.size();

    std::vector<float> golden_vec(max_cb_num_datums);
    std::vector<float> result_vec_fp16(min_cb_num_datums);
    std::vector<float> result_vec_fp8(max_cb_num_datums);
    std::vector<float> result_vec(max_cb_num_datums);

    const auto& values_fp16 = input_tensor_fp16.get_values();
    const auto& values_fp8 = input_tensor_fp8.get_values();

    int index = 0;
    int index_fp16 = 0;
    int index_fp8 = 0;
    for (int l = 0; l < num_mixed_df_layers; ++l) {
        if (l % 2 == 0) {
            index_fp16 = index;
            for (int i = 0; i < values_fp16.size(); ++i) {
                golden_vec[index] = to_float(values_fp16[i]);
                index++;

                if (index == max_cb_num_datums) {
                    index = 0;
                }
            }
        } else {
            index_fp8 = index;
            for (int i = 0; i < values_fp8.size(); ++i) {
                golden_vec[index] = (float)values_fp8[i];
                index++;

                if (index == max_cb_num_datums) {
                    index = 0;
                }
            }
        }

    }

    log_info("start index of fp16: {}", index_fp16);
    log_info("start index of fp8: {}", index_fp8);

    for (int i=0; i<result_untilized_fp16.size(); ++i) {
        result_vec_fp16[i] = to_float(static_cast<bfloat16>(result_untilized_fp16[i]));
    }
    for (int i=0; i<result_untilized_fp8.size(); ++i) {
        result_vec_fp8[i] = result_untilized_fp8[i];
    }

    // if (num_mixed_df_layers % 2 == 0) { // end layer is fp8

    // } else {
    //     int index = 0;
    //     for (int i=index_fp16; i < max_cb_num_datums; ++i) {
    //         result_vec[i] = result_vec_fp16[index];
    //         index++;
    //     }
    // }


    // if (index_fp16 > index_fp8) {
    //     log_info("index_fp16 > index_fp8");

    //     int len_fp8 = (index_fp16 - index_fp8);
    //     for (int i=index_fp8; i < len_fp8; ++i) {
    //         result_vec[i] = result_vec_fp8[i];
    //     }
    //     for (int i=0; i < index_fp8; ++i) {
    //         result_vec[i] = result_vec_fp16[i];
    //     }
    //     for (int i=index_fp16; i < len; ++i) {
    //         result_vec[i] = result_vec_fp16[i];
    //     }

    // } else if (index_fp16 < index_fp8) {
    //     log_info("index_fp16 < index_fp8");

    //     int len_fp16 = (index_fp8 - index_fp16);
    //     for (int i=index_fp16; i < len_fp16; ++i) {
    //         result_vec[i] = result_vec_fp16[i];
    //     }
    //     for (int i=0; i < index_fp8; ++i) {
    //         result_vec[i] = result_vec_fp8[i];
    //     }
    //     for (int i=index_fp8; i < len; ++i) {
    //         result_vec[i] = result_vec_fp8[i];
    //     }

    // } else {
    //     log_info("index_fp16 = index_fp8");

    //     if (num_mixed_df_layers % 2 == 0) { // end layer is fp8
    //         log_info("last layer is fp8");
    //         for (int i=0; i < len; ++i) {
    //             result_vec[i] = result_vec_fp8[i];
    //         }
    //     } else {
    //         log_info("last layer is fp16");
    //         for (int i=0; i < len; ++i) {
    //             result_vec[i] = result_vec_fp16[i];
    //         }
    //     }
    // }

    // for (uint32_t i=0; i < golden_vec.size(); ++i ) {
    //     std::cout << golden_vec[i] << " ";

    //     if ((i+1) %32 == 0) {
    //         std::cout << std::endl;
    //     }
    // }

    // std::cout << std::endl;
    // std::cout << std::endl;

    // for (uint32_t i=0; i < result_vec.size(); ++i ) {
    //     std::cout << result_vec[i] << " ";

    //     if ((i+1) %32 == 0) {
    //         std::cout << std::endl;
    //     }
    // }

    // std::cout << std::endl;
    // std::cout << std::endl;

    for (uint32_t i=0; i < result_vec_fp16.size(); ++i ) {
        std::cout << result_vec_fp16[i] << " ";

        if ((i+1) %32 == 0) {
            std::cout << std::endl;
        }
    }

    // std::cout << std::endl;
    // std::cout << std::endl;

    // for (uint32_t i=0; i < result_vec_fp8.size(); ++i ) {
    //     std::cout << result_vec_fp8[i] << " ";

    //     if ((i+1) %32 == 0) {
    //         std::cout << std::endl;
    //     }
    // }

    pass &= pcc(golden_vec, result_vec) >= 0.9999;
    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}

std::shared_ptr<tt::tt_metal::Buffer> create_and_transfer_data_sharded_cb(
    tt_metal::Device* device,
    vector<uint32_t> input_vec,
    uint32_t ht,
    uint32_t wt,
    BufferType buffer_type,
    tt::DataFormat data_format,
    CoreRangeSet cores,
    uint32_t num_receivers
) {

    uint32_t size_bytes;
    uint32_t page_size_bytes;
    if (data_format == tt::DataFormat::Bfp8_b) {
        size_bytes = ht * wt * 1088;
        page_size_bytes = 1088;
    } else {
        size_bytes = ht * tt::constants::TILE_HEIGHT * wt * tt::constants::TILE_WIDTH * 2;
        page_size_bytes = tt::constants::TILE_HW * 2;
    }

    ShardSpecBuffer shard_spec = ShardSpecBuffer(
                cores,
                {ht * tt::constants::TILE_HEIGHT, wt * tt::constants::TILE_WIDTH / num_receivers},
                ShardOrientation::ROW_MAJOR,
                false,
                {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
                {ht, wt});

    log_info("cores: {}", cores);
    log_info("size_bytes: {}", size_bytes);
    log_info("page_size_bytes: {}", page_size_bytes);

    auto input_buffer = CreateBuffer(tt::tt_metal::ShardedBufferConfig{
                                        .device = device,
                                        .size = size_bytes,
                                        .page_size = page_size_bytes,
                                        .buffer_type = buffer_type,
                                        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
                                        .shard_parameters = shard_spec});
    tt::tt_metal::detail::WriteToBuffer(input_buffer, input_vec);

    log_info("created sharded tensor");

    return input_buffer;
}

int main(int argc, char **argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    bool use_device_profiler = false;
    uint32_t df = 0;
    std::vector<double> dram_bandwidth;
    uint32_t num_tests = 1;
    uint32_t num_blocks = 8;
    uint32_t cb_num_blocks = 8;
    uint32_t cb_padding = 16;
    uint32_t num_receivers = 1;
    uint32_t num_mixed_df_layers = 1;
    uint64_t k = 8192, n = 128;

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
            std::tie(cb_num_blocks, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--cb-num-blocks", 8);
            std::tie(cb_padding, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--cb-padding", 16);
            std::tie(num_tests, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 1);
            std::tie(use_device_profiler, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--use-device-profiler");
            std::tie(df, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--data-type", 0);
            std::tie(num_receivers, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--num-receivers", 1);
            std::tie(num_mixed_df_layers, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--num-mixed-df-layers", 1);


            test_args::validate_remaining_args(input_args);
        } catch (const std::exception &e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
            TT_ASSERT(false);
        }

        log_info("num_mixed_df_layers: {} ", num_mixed_df_layers);
        log_info("num_receivers: {} ", num_receivers);

        if (num_mixed_df_layers > 1) {
            TT_FATAL(df == 1, "must start with bfloat16 format for mix_df test");
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

        ////////////////////////////////////////////////////////////////////////////
        //                      Parameters Setup
        ////////////////////////////////////////////////////////////////////////////
        uint32_t num_banks = 1;
        uint32_t input_size = 0;
        tt::DataFormat tile_format = tt::DataFormat::Bfp8_b;
        if (df == 0) {
            input_size = k * n * 1088 / 1024;
            tile_format = tt::DataFormat::Bfp8_b;
        } else if (df == 1) {
            input_size = k * n * 2;
            tile_format = tt::DataFormat::Float16_b;
        } else {
            TT_THROW("Input data format {} is invalid. Please change.", df);
        }
        uint32_t output_size = input_size / num_blocks * cb_num_blocks;
        uint32_t kt = k / 32;
        uint32_t nt = n / 32;
        uint32_t block_h = kt / num_blocks;
        uint32_t block_w = nt;
        uint32_t num_datums_per_tile = 32 * 32;

        uint32_t single_tile_size = tt_metal::detail::TileSize(tile_format);

        TT_FATAL(input_size % single_tile_size == 0, "input size is not aligned to tile size");
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CoreCoord dram_bank_coord = CoreCoord{0, 0};
        CoreCoord dram_reader_core_coord = CoreCoord{0, 0};
        CoreRange dram_reader_core_coord_range = CoreRange(dram_reader_core_coord);
        CoreRangeSet dram_reader_core{std::set<CoreRange>{CoreRange{dram_reader_core_coord}}};
        CoreRange l1_receiver_core_coord_range = CoreRange(CoreCoord{0, 0});
        if (device->arch() == tt::ARCH::GRAYSKULL) {
            l1_receiver_core_coord_range = CoreRange{CoreCoord{0, 1}, CoreCoord{0, num_receivers}};
        } else {
            l1_receiver_core_coord_range = CoreRange{CoreCoord{1, 0}, CoreCoord{num_receivers, 0}};
        }
        CoreRangeSet l1_receiver_core{std::set<CoreRange>{l1_receiver_core_coord_range}};

        ////////////////////////////////////////////////////////////////////////////
        //                      Input Setup
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::shared_ptr<tt::tt_metal::Buffer> > input_buffers(num_mixed_df_layers);
        std::shared_ptr<tt::tt_metal::Buffer> output_buffer;
        auto input_shape = SHAPE{1, 1, k, n};
        tt::deprecated::Tensor<bfloat16> tensor_fp16 = tt::deprecated::initialize_tensor<bfloat16>(input_shape, tt::deprecated::Initialize::INCREMENT, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt::deprecated::Tensor<float> tensor_fp8 = tt::deprecated::initialize_tensor<float>(input_shape, tt::deprecated::Initialize::INCREMENT, 100, std::chrono::system_clock::now().time_since_epoch().count());
        if (tile_format == tt::DataFormat::Bfp8_b) {
            for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
                if (i%2 == 0) { // even layers
                    auto input_vec_tilized = tt::test_utils::tilize(tensor_fp8.get_values(), k, n);
                    std::vector<uint32_t> packed_input_vec_tile_layout = pack_fp32_vec_as_bfp8_tiles(input_vec_tilized, true, false);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(device, packed_input_vec_tile_layout, kt, nt, tt_metal::BufferType::DRAM, tt::DataFormat::Bfp8_b, dram_reader_core, num_banks);
                } else { // odd layers
                    auto input_vec_tilized = tt::test_utils::tilize(tensor_fp16.get_values(), k, n);
                    auto input_vec_tile_layout = convert_to_tile_layout(input_vec_tilized);
                    vector<uint32_t> packed_input_vec_tile_layout = pack_bfloat16_vec_into_uint32_vec(input_vec_tile_layout);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(device, packed_input_vec_tile_layout, kt, nt, tt_metal::BufferType::DRAM, tt::DataFormat::Float16_b, dram_reader_core, num_banks);
                }
            }

            // output
            vector<uint32_t> outputs = create_constant_vector_of_bfp8(output_size, 0, true);
            output_buffer = create_and_transfer_data_sharded_cb(device, outputs, kt / num_blocks * cb_num_blocks, nt, tt_metal::BufferType::L1, tt::DataFormat::Bfp8_b, l1_receiver_core, num_receivers);

        } else {
            for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
                if (i%2 == 0) { // even layers
                    auto input_vec_tilized = tt::test_utils::tilize(tensor_fp16.get_values(), k, n);
                    auto input_vec_tile_layout = convert_to_tile_layout(input_vec_tilized);
                    vector<uint32_t> packed_input_vec_tile_layout = pack_bfloat16_vec_into_uint32_vec(input_vec_tile_layout);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(device, packed_input_vec_tile_layout, kt, nt, tt_metal::BufferType::DRAM, tt::DataFormat::Float16_b, dram_reader_core, num_banks);
                } else {
                    auto input_vec_tilized = tt::test_utils::tilize(tensor_fp8.get_values(), k, n);
                    std::vector<uint32_t> packed_input_vec_tile_layout = pack_fp32_vec_as_bfp8_tiles(input_vec_tilized, true, false);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(device, packed_input_vec_tile_layout, kt, nt, tt_metal::BufferType::DRAM, tt::DataFormat::Bfp8_b, dram_reader_core, num_banks);
                }
            }

            // output
            vector<uint32_t> outputs = create_constant_vector_of_bfloat16(output_size, 0);
            output_buffer = create_and_transfer_data_sharded_cb(device, outputs, kt / num_blocks * cb_num_blocks, nt, tt_metal::BufferType::L1, tt::DataFormat::Float16_b, l1_receiver_core, num_receivers);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [program, kernel, output_cb_addr] = create_program(device, dram_reader_core, l1_receiver_core, single_tile_size, tile_format, k, n, num_blocks, cb_num_blocks, num_receivers, num_mixed_df_layers, cb_padding, input_buffers[0], output_buffer);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execution Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::detail::CompileProgram(device, program);

        log_info(LogTest, "Num tests {}", num_tests);
        for (uint32_t i = 0; i < num_tests; ++i) {
            EnqueueProgram(device->command_queue(), program, false);
            Finish(device->command_queue());
            tt_metal::DumpDeviceProfileResults(device, program);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        if (num_mixed_df_layers == 1) {
            if (tile_format == tt::DataFormat::Bfp8_b) {
                pass = validation_bfp8_b(
                    tensor_fp8,
                    tile_format,
                    num_blocks,
                    cb_num_blocks,
                    kt,
                    nt,
                    output_buffer);
            } else {
                pass = validation_fp16(
                    tensor_fp16,
                    tile_format,
                    num_blocks,
                    cb_num_blocks,
                    kt,
                    nt,
                    output_buffer);
            }
        } else {
            pass = validation_mixed_df(
                    tensor_fp16,
                    tensor_fp8,
                    tile_format,
                    num_blocks,
                    cb_num_blocks,
                    kt,
                    nt,
                    output_buffer,
                    num_mixed_df_layers);
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception &e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
