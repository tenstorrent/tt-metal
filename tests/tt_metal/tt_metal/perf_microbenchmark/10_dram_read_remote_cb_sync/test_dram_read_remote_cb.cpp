// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
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
#include <tt-metalium/sub_device_types.hpp>
#include "test_common.hpp"
#include <tt-metalium/tilize_utils.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "tt_metal/test_utils/bfloat_utils.hpp"

using std::vector;
using namespace tt;
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
    uint32_t num_tiles, uint32_t num_datums_per_tile, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * num_datums_per_tile;

    page_size = (8192 / num_datums_per_tile) * num_datums_per_tile;
    while (total_size % page_size != 0 && page_size >= num_datums_per_tile) {
        page_size -= num_datums_per_tile;
    }
    num_pages = total_size / page_size;
}

std::tuple<std::vector<tt_metal::distributed::MeshWorkload>, tt_metal::experimental::GlobalCircularBuffer>
create_mesh_workloads(
    tt_metal::distributed::MeshDevice* device,
    const CoreRangeSet& dram_reader_core,
    const CoreRangeSet& l1_receiver_cores,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    const uint32_t& single_tile_size,
    const tt::DataFormat& tile_format,
    uint32_t k,
    uint32_t n,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t num_receivers,
    uint32_t num_mixed_df_layers,
    uint32_t cb_padding,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& input_buffer,
    bool use_sub_devices) {
    log_info(tt::LogTest, "created program");
    std::vector<tt_metal::Program> programs;
    programs.push_back(tt_metal::Program());

    if (use_sub_devices) {
        programs.push_back(tt_metal::Program());
    }
    auto& sender_program = programs[0];
    auto& receiver_program = use_sub_devices ? programs[1] : programs[0];

    auto all_cores = dram_reader_core.merge(l1_receiver_cores);

    uint32_t start_tile_id = 0;
    uint32_t kt = k / 32;
    uint32_t nt = n / 32;
    uint32_t block_h = kt / num_blocks;
    uint32_t num_tile_rows_write = block_h;
    uint32_t block_w = nt;
    uint32_t block_num_tiles = block_h * block_w;

    uint32_t receiver_cb_size = block_h * block_w * single_tile_size * cb_num_blocks / num_receivers;
    uint32_t padded_global_cb_size = receiver_cb_size + cb_padding;

    // DRAM reader CB
    uint32_t reader_cb_index = 0;
    uint32_t reader_cb_size = block_h * block_w * single_tile_size * 3;
    uint32_t writer_cb_index = 31;
    // For debug purpose
    // uint32_t reader_cb_size = block_h * block_w * single_tile_size;
    uint32_t reader_page_size, reader_num_pages;
    get_max_page_size_and_num_pages(block_num_tiles, single_tile_size, reader_page_size, reader_num_pages);

    uint32_t receiver_block_num_tile = block_h * block_w / num_receivers;
    uint32_t writer_page_size, writer_num_pages;
    get_max_page_size_and_num_pages(block_w / num_receivers, single_tile_size, writer_page_size, writer_num_pages);

    log_info(tt::LogTest, "writer_page_size: {}", writer_page_size);
    log_info(tt::LogTest, "writer_num_pages: {}", writer_num_pages);

    tt_metal::CircularBufferConfig reader_cb_config =
        tt_metal::CircularBufferConfig(reader_cb_size, {{reader_cb_index, tile_format}})
            .set_page_size(reader_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(sender_program, dram_reader_core, reader_cb_config);

    auto global_cb = tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, padded_global_cb_size, tt_metal::BufferType::L1);
    tt_metal::CircularBufferConfig writer_cb_config = tt_metal::CircularBufferConfig(receiver_cb_size);
    writer_cb_config.remote_index(writer_cb_index).set_page_size(single_tile_size).set_data_format(tile_format);
    tt_metal::experimental::CreateCircularBuffer(sender_program, dram_reader_core, writer_cb_config, global_cb);

    // mixed cb dataformat
    uint32_t next_layer_num_blocks = num_blocks * 2;
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
    get_max_page_size_and_num_pages(
        next_layer_block_num_tiles,
        next_layer_single_tile_size,
        next_layer_reader_page_size,
        next_layer_reader_num_pages);

    uint32_t next_layer_writer_page_size, next_layer_writer_num_pages;
    get_max_page_size_and_num_pages(
        block_w / num_receivers, next_layer_single_tile_size, next_layer_writer_page_size, next_layer_writer_num_pages);

    // L1 receiver CB
    uint32_t receiver_cb_index = 31;
    tt_metal::CircularBufferConfig receiver_cb_config = tt_metal::CircularBufferConfig(receiver_cb_size);
    receiver_cb_config.remote_index(receiver_cb_index).set_page_size(single_tile_size).set_data_format(tile_format);
    tt_metal::experimental::CreateCircularBuffer(
        receiver_program, l1_receiver_cores, receiver_cb_config, global_cb);

    log_info(tt::LogTest, "reader_cb_size: {}", reader_cb_size);
    log_info(tt::LogTest, "receiver_cb_size: {}", receiver_cb_size);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_buffer->address(),
        (std::uint32_t)start_tile_id,
        (std::uint32_t)tt_metal::NOC::RISCV_0_default,
        (std::uint32_t)num_mixed_df_layers};

    auto reader_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernels/reader_dram.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)tt_metal::NOC::RISCV_0_default,
        (std::uint32_t)num_receivers,
        (std::uint32_t)num_mixed_df_layers,
        (std::uint32_t)writer_cb_index};

    auto writer_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernels/writer_l1.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = writer_compile_time_args});

    std::vector<uint32_t> receiver_compile_time_args = {
        (std::uint32_t)num_mixed_df_layers, (std::uint32_t)receiver_cb_index};

    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/10_dram_read_remote_cb_sync/kernels/receiver_l1.cpp",
        l1_receiver_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = receiver_compile_time_args});

    // reader rt
    auto dram_reader_core_coord = dram_reader_core.ranges().begin()->start_coord;
    log_info(tt::LogTest, "dram_reader_core_coord: {}", dram_reader_core_coord);
    auto dram_reader_core_coord_physical = device->worker_core_from_logical_core(dram_reader_core_coord);
    uint32_t bank_id = 0;
    uint32_t vc = bank_id & 0x1;
    std::vector<uint32_t> reader_rt_args = {(std::uint32_t)bank_id, (std::uint32_t)vc};
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i % 2 == 0 ? reader_page_size : next_layer_reader_page_size);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i % 2 == 0 ? reader_num_pages : next_layer_reader_num_pages);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i % 2 == 0 ? num_blocks : next_layer_num_blocks);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        reader_rt_args.push_back(i % 2 == 0 ? block_num_tiles : next_layer_block_num_tiles);
    }
    tt_metal::SetRuntimeArgs(sender_program, reader_kernel, dram_reader_core_coord, reader_rt_args);

    // writer rt
    std::vector<CoreCoord> l1_receiver_core_coords;
    for (auto l1_receiver_core_coord : *l1_receiver_cores.ranges().begin()) {
        l1_receiver_core_coords.push_back(l1_receiver_core_coord);
    }
    std::vector<uint32_t> writer_rt_args;
    for (uint32_t i = 0; i < num_receivers; ++i) {
        auto l1_receiver_core_coord_physical = device->worker_core_from_logical_core(l1_receiver_core_coords[i]);
        writer_rt_args.push_back(l1_receiver_core_coord_physical.x);
    }
    for (uint32_t i = 0; i < num_receivers; ++i) {
        auto l1_receiver_core_coord_physical = device->worker_core_from_logical_core(l1_receiver_core_coords[i]);
        writer_rt_args.push_back(l1_receiver_core_coord_physical.y);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i % 2 == 0 ? writer_page_size : next_layer_writer_page_size);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i % 2 == 0 ? writer_num_pages : next_layer_writer_num_pages);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i % 2 == 0 ? num_blocks : next_layer_num_blocks);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i % 2 == 0 ? block_num_tiles : next_layer_block_num_tiles);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i % 2 == 0 ? single_tile_size : next_layer_single_tile_size);
    }
    for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
        writer_rt_args.push_back(i % 2 == 0 ? num_tile_rows_write : next_layer_num_tile_rows_write);
    }
    tt_metal::SetRuntimeArgs(sender_program, writer_kernel, dram_reader_core_coord, writer_rt_args);

    // receiver rt
    for (uint32_t i = 0; i < num_receivers; ++i) {
        std::vector<uint32_t> receiver_rt_args = {
            (std::uint32_t)vc & 0x3,
            (std::uint32_t)dram_reader_core_coord_physical.x,
            (std::uint32_t)dram_reader_core_coord_physical.y,
            (std::uint32_t)i};
        vc++;

        for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
            receiver_rt_args.push_back(i % 2 == 0 ? single_tile_size : next_layer_single_tile_size);
        }
        for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
            receiver_rt_args.push_back(i % 2 == 0 ? num_blocks : next_layer_num_blocks);
        }
        for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
            receiver_rt_args.push_back(i % 2 == 0 ? receiver_block_num_tile : next_layer_receiver_block_num_tile);
        }

        log_info(tt::LogTest, "l1_receiver_core_coords: {}", l1_receiver_core_coords[i]);

        tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, l1_receiver_core_coords[i], receiver_rt_args);
    }

    std::vector<tt_metal::distributed::MeshWorkload> mesh_workloads;
    for (auto& program : programs) {
        auto mesh_workload = tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(
            tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        mesh_workloads.push_back(std::move(mesh_workload));
    }

    return {std::move(mesh_workloads), std::move(global_cb)};
}

float to_float(bfloat16 bfloat16_num) { return static_cast<float>(bfloat16_num); }

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
    const tt::deprecated::Tensor<float>& input_tensor,
    const tt::DataFormat&  /*data_format*/,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t kt,
    uint32_t nt,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& out_buffer,
    tt_metal::distributed::MeshDevice* device) {
    bool pass = true;
    std::vector<float> golden_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0);  // Initialize with zeros
    std::vector<float> result_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0);
    auto num_datums_per_cb = kt * nt * 32 * 32 / num_blocks * cb_num_blocks;

    std::vector<float> result_untilized;
    std::vector<uint32_t> result;
    tt_metal::distributed::ReadShard(device->mesh_command_queue(), result, out_buffer, tt_metal::distributed::MeshCoordinate(0, 0), true);
    auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result, true, false);
    result_untilized = untilize_swizzled(result_bfp8, kt * 32 / num_blocks * cb_num_blocks, nt * 32);

    const auto& values = input_tensor.get_values();

    int index = 0;
    for (int i = 0; i < kt * nt * 32 * 32; ++i) {
        golden_vec[index] = float(values[i]);
        index++;

        if (index == num_datums_per_cb) {
            index = 0;
        }
    }

    for (int i = 0; i < result_untilized.size(); ++i) {
        result_vec[i] = result_untilized[i];
    }

    pass &= pcc(golden_vec, result_vec) >= 0.9999;
    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}

bool validation_fp16(
    const tt::deprecated::Tensor<bfloat16>& input_tensor,
    const tt::DataFormat&  /*data_format*/,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t kt,
    uint32_t nt,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& out_buffer,
    tt_metal::distributed::MeshDevice* device) {
    bool pass = true;
    std::vector<float> golden_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0);  // Initialize with zeros
    std::vector<float> result_vec(kt * nt * 32 * 32 / num_blocks * cb_num_blocks, 0);
    auto num_datums_per_cb = kt * nt * 32 * 32 / num_blocks * cb_num_blocks;

    std::vector<uint32_t> result;
    tt_metal::distributed::ReadShard(device->mesh_command_queue(), result, out_buffer, tt_metal::distributed::MeshCoordinate(0, 0), true);
    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result);
    auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
    auto result_untilized = untilize_swizzled(result_flat_layout, kt * 32 / num_blocks * cb_num_blocks, nt * 32);

    const auto& values = input_tensor.get_values();

    int index = 0;
    for (int i = 0; i < kt * nt * 32 * 32; ++i) {
        golden_vec[index] = to_float(values[i]);
        index++;

        if (index == num_datums_per_cb) {
            index = 0;
        }
    }

    for (int i = 0; i < result_untilized.size(); ++i) {
        result_vec[i] = to_float(static_cast<bfloat16>(result_untilized[i]));
    }

    pass &= (golden_vec == result_vec);
    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}

bool validation_mixed_df(
    const tt::deprecated::Tensor<bfloat16>& input_tensor_fp16,
    const tt::deprecated::Tensor<float>&  /*input_tensor_fp8*/,
    const tt::DataFormat&  /*data_format*/,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t kt,
    uint32_t nt,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& out_buffer,
    uint32_t num_mixed_df_layers,
    uint32_t num_receivers,
    tt_metal::distributed::MeshDevice* device) {
    bool pass = true;

    std::vector<uint32_t> result;
    tt_metal::distributed::ReadShard(device->mesh_command_queue(), result, out_buffer, tt_metal::distributed::MeshCoordinate(0, 0), true);

    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result);
    auto result_untilized_fp16 = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));

    std::vector<float> golden_vec(kt * 32 / num_blocks * cb_num_blocks * nt * 32);
    std::vector<float> result_vec_fp16(kt * 32 / num_blocks * cb_num_blocks * nt * 32);

    // compare with the result tilized with tilized
    auto values_fp16 = tilize_swizzled(input_tensor_fp16.get_values(), kt * 32, nt * 32);

    uint32_t block_h = kt / num_blocks;
    uint32_t block_w = nt;
    uint32_t block_num_tiles = block_h * block_w;

    auto num_datums_per_cb = kt * nt * 32 * 32 / num_blocks * cb_num_blocks / num_receivers;
    int start_index = 0;
    int fifo_size = kt * 32 / num_blocks * cb_num_blocks * nt * 32 * 2 / num_receivers;
    int page_size, layer_transfer_size, fifo_wr_ptr = 0;
    for (int l = 0; l < num_mixed_df_layers; ++l) {
        if (l % 2 == 0) {  // fp16
            page_size = 2048;
        } else {
            page_size = 1088;
        }
        layer_transfer_size = page_size * kt * nt / num_receivers;

        uint32_t block_size = block_num_tiles * tt::constants::TILE_HW * datum_size(tt::DataFormat::Float16_b);  // fp16
        uint32_t num_blocks = fifo_size / block_size;
        uint32_t cb_size_block_aligned = num_blocks * block_size;

        bool fifo_wr_ptr_exceed_fifo_limit = fifo_wr_ptr > cb_size_block_aligned;
        uint32_t num_blocks_till_fifo_limit = (cb_size_block_aligned - fifo_wr_ptr) / block_size;
        // start pointer addr of current layer
        fifo_wr_ptr =
            fifo_wr_ptr_exceed_fifo_limit ? 0 : cb_size_block_aligned - (num_blocks_till_fifo_limit * block_size);
        // start index to read, fifo_wr_ptr / 2 because fp16 format
        start_index = fifo_wr_ptr == cb_size_block_aligned ? 0 : fifo_wr_ptr / 2;
        // end pointer addr of current layer
        fifo_wr_ptr = (fifo_wr_ptr + layer_transfer_size) % cb_size_block_aligned;
    }

    std::vector<std::vector<float>> values_fp16_split(
        num_receivers, std::vector<float>(values_fp16.size() / num_receivers));

    int index = 0;
    for (int k = 0; k < kt; ++k) {
        for (int n = 0; n < num_receivers; ++n) {
            for (int i = 0; i < nt * 32 * 32 / num_receivers; ++i) {
                values_fp16_split[n][i + (k * nt * 32 * 32 / num_receivers)] = to_float(values_fp16[index]);
                index++;
            }
        }
    }

    std::vector<std::vector<float>> golden_vec_split(
        num_receivers, std::vector<float>(golden_vec.size() / num_receivers));

    for (int n = 0; n < num_receivers; ++n) {
        index = start_index;
        for (int i = 0; i < kt * nt * 32 * 32 / num_receivers; ++i) {
            golden_vec_split[n][index] = values_fp16_split[n][i];
            index++;

            if (index == num_datums_per_cb) {
                index = 0;
            }
        }
    }

    index = 0;
    for (int k = 0; k < kt / num_blocks * cb_num_blocks; ++k) {
        for (int n = 0; n < num_receivers; ++n) {
            for (int i = 0; i < nt * 32 * 32 / num_receivers; ++i) {
                golden_vec[index] = golden_vec_split[n][i + (k * nt * 32 * 32 / num_receivers)];
                index++;
            }
        }
    }

    for (int i = 0; i < result_untilized_fp16.size(); ++i) {
        result_vec_fp16[i] = to_float(static_cast<bfloat16>(result_untilized_fp16[i]));
    }

    // For debug purpose
    // for (int i = 0; i < golden_vec.size(); ++i) {
    //     std::cout << golden_vec[i] << " ";
    //     if ((i+1) % 32 == 0) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;
    // for (int i = 0; i < result_vec_fp16.size(); ++i) {
    //     std::cout << result_vec_fp16[i] << " ";
    //     if ((i+1) % 32 == 0) {
    //         std::cout << std::endl;
    //     }
    // }

    pass &= pcc(golden_vec, result_vec_fp16) == 1.0;

    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}

std::shared_ptr<tt_metal::distributed::MeshBuffer> create_and_transfer_data_sharded_cb(
    tt_metal::distributed::MeshDevice* device,
    const vector<uint32_t>& input_vec,
    uint32_t ht,
    uint32_t wt,
    BufferType buffer_type,
    tt::DataFormat data_format,
    CoreRangeSet cores,
    uint32_t num_receivers,
    std::optional<DeviceAddr> address = std::nullopt) {
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
        {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        {ht, wt});

    BufferShardingArgs sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::WIDTH_SHARDED);
    log_info(tt::LogTest, "cores: {}", cores);
    log_info(tt::LogTest, "size_bytes: {}", size_bytes);
    log_info(tt::LogTest, "page_size_bytes: {}", page_size_bytes);

    auto device_local_config = tt_metal::distributed::DeviceLocalBufferConfig{
        .page_size = page_size_bytes,
        .buffer_type = buffer_type,
        .sharding_args = sharding_args};

    tt_metal::distributed::ReplicatedBufferConfig global_buf{.size = size_bytes};

    std::shared_ptr<tt_metal::distributed::MeshBuffer> input_buffer;
    if (address.has_value()) {
        input_buffer = tt_metal::distributed::MeshBuffer::create(global_buf, device_local_config, device, address);
    } else {
        input_buffer = tt_metal::distributed::MeshBuffer::create(global_buf, device_local_config, device);
    }
    tt_metal::distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), input_buffer, input_vec, false);
    tt_metal::distributed::Finish(device->mesh_command_queue());

    log_info(tt::LogTest, "created sharded tensor");

    return input_buffer;
}

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error(tt::LogTest, "Test not supported w/ slow dispatch, exiting");
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
    bool use_sub_devices = false;

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
            std::tie(use_sub_devices, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--use-sub-devices");

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
            TT_FATAL(false, "Command line arguments found exception");
        }

        log_info(tt::LogTest, "num_mixed_df_layers: {} ", num_mixed_df_layers);
        log_info(tt::LogTest, "num_receivers: {} ", num_receivers);

        TT_FATAL(
            num_mixed_df_layers % 2 == 1,
            "currently only support odd number of layers testing, due to issue with validation");
        if (num_mixed_df_layers > 1) {
            TT_FATAL(df == 1, "must start with bfloat16 format for mix_df test");
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

        uint32_t single_tile_size = tt::tile_size(tile_format);

        TT_FATAL(input_size % single_tile_size == 0, "input size is not aligned to tile size");
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

        [[maybe_unused]] CoreCoord dram_bank_coord = CoreCoord{0, 0};
        CoreCoord dram_reader_core_coord = CoreCoord{0, 0};
        CoreRangeSet dram_reader_core{std::set<CoreRange>{CoreRange{dram_reader_core_coord}}};
        CoreRange l1_receiver_core_coord_range = CoreRange(CoreCoord{0, 0});
        if (device->arch() == tt::ARCH::GRAYSKULL) {
            l1_receiver_core_coord_range = CoreRange{CoreCoord{0, 1}, CoreCoord{0, num_receivers}};
        } else {
            l1_receiver_core_coord_range = CoreRange{CoreCoord{1, 0}, CoreCoord{num_receivers, 0}};
        }
        CoreRangeSet l1_receiver_core{std::set<CoreRange>{l1_receiver_core_coord_range}};
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {
            { dram_reader_core_coord, l1_receiver_core }
        };
        std::vector<SubDeviceId> receiver_sub_device_ids = {};
        if (use_sub_devices) {
            SubDevice sender_sub_device = SubDevice(std::array{dram_reader_core});
            SubDevice receiver_sub_device = SubDevice(std::array{l1_receiver_core});
            SubDeviceManagerId sdm_id = device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, 0);
            device->load_sub_device_manager(sdm_id);
            receiver_sub_device_ids.push_back(SubDeviceId{1});
        }
        ////////////////////////////////////////////////////////////////////////////
        //                      Input Setup
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::shared_ptr<tt_metal::distributed::MeshBuffer>> input_buffers(num_mixed_df_layers);
        std::shared_ptr<tt_metal::distributed::MeshBuffer> output_buffer;
        auto input_shape = SHAPE{1, 1, k, n};
        tt::deprecated::Tensor<bfloat16> tensor_fp16 = tt::deprecated::initialize_tensor<bfloat16>(
            input_shape,
            tt::deprecated::Initialize::INCREMENT,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        tt::deprecated::Tensor<float> tensor_fp8 = tt::deprecated::initialize_tensor<float>(
            input_shape,
            tt::deprecated::Initialize::INCREMENT,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        if (tile_format == tt::DataFormat::Bfp8_b) {
            for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
                if (i % 2 == 0) {  // even layers
                    auto input_vec_tilized = tilize_swizzled(tensor_fp8.get_values(), k, n);
                    std::vector<uint32_t> packed_input_vec_tile_layout =
                        pack_as_bfp8_tiles(tt::stl::make_const_span(input_vec_tilized), true, false);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(
                        device.get(),
                        packed_input_vec_tile_layout,
                        kt,
                        nt,
                        tt_metal::BufferType::DRAM,
                        tt::DataFormat::Bfp8_b,
                        dram_reader_core,
                        num_banks);
                } else {  // odd layers
                    auto input_vec_tilized = tilize_swizzled(tensor_fp16.get_values(), k, n);
                    auto input_vec_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(input_vec_tilized));
                    vector<uint32_t> packed_input_vec_tile_layout =
                        pack_bfloat16_vec_into_uint32_vec(input_vec_tile_layout);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(
                        device.get(),
                        packed_input_vec_tile_layout,
                        kt,
                        nt,
                        tt_metal::BufferType::DRAM,
                        tt::DataFormat::Float16_b,
                        dram_reader_core,
                        num_banks);
                }
            }
        } else {
            for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
                if (i % 2 == 0) {  // even layers
                    auto input_vec_tilized = tilize_swizzled(tensor_fp16.get_values(), k, n);
                    auto input_vec_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(input_vec_tilized));
                    vector<uint32_t> packed_input_vec_tile_layout =
                        pack_bfloat16_vec_into_uint32_vec(input_vec_tile_layout);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(
                        device.get(),
                        packed_input_vec_tile_layout,
                        kt,
                        nt,
                        tt_metal::BufferType::DRAM,
                        tt::DataFormat::Float16_b,
                        dram_reader_core,
                        num_banks);
                } else {
                    auto input_vec_tilized = tilize_swizzled(tensor_fp8.get_values(), k, n);
                    std::vector<uint32_t> packed_input_vec_tile_layout =
                        pack_as_bfp8_tiles(tt::stl::make_const_span(input_vec_tilized), true, false);
                    input_buffers[i] = create_and_transfer_data_sharded_cb(
                        device.get(),
                        packed_input_vec_tile_layout,
                        kt,
                        nt,
                        tt_metal::BufferType::DRAM,
                        tt::DataFormat::Bfp8_b,
                        dram_reader_core,
                        num_banks);
                }
            }
        }

        for (uint32_t i = 0; i < num_mixed_df_layers; ++i) {
            log_info(tt::LogTest, "input_buffers addr: {}", input_buffers[i]->address());
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [mesh_workloads, global_cb] = create_mesh_workloads(
            device.get(),
            dram_reader_core,
            l1_receiver_core,
            sender_receiver_core_mapping,
            single_tile_size,
            tile_format,
            k,
            n,
            num_blocks,
            cb_num_blocks,
            num_receivers,
            num_mixed_df_layers,
            cb_padding,
            input_buffers[0],
            use_sub_devices);
        if (tile_format == tt::DataFormat::Bfp8_b) {
            // output
            vector<uint32_t> outputs = test_utils::create_constant_vector_of_bfp8(output_size, 0, true);
            output_buffer = create_and_transfer_data_sharded_cb(
                device.get(),
                outputs,
                kt / num_blocks * cb_num_blocks,
                nt,
                tt_metal::BufferType::L1,
                tt::DataFormat::Bfp8_b,
                l1_receiver_core,
                num_receivers,
                global_cb.buffer_address());

        } else {
            // output
            vector<uint32_t> outputs = create_constant_vector_of_bfloat16(output_size, 0);
            output_buffer = create_and_transfer_data_sharded_cb(
                device.get(),
                outputs,
                kt / num_blocks * cb_num_blocks,
                nt,
                tt_metal::BufferType::L1,
                tt::DataFormat::Float16_b,
                l1_receiver_core,
                num_receivers,
                global_cb.buffer_address());
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execution Application
        ////////////////////////////////////////////////////////////////////////////
        log_info(LogTest, "Num tests {}", num_tests);
        for (uint32_t i = 0; i < num_tests; ++i) {
            if (use_sub_devices) {
                // Enqueue the sender program
                tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workloads[0], false);
                device->set_sub_device_stall_group(receiver_sub_device_ids);
                for (uint32_t j = 1; j < mesh_workloads.size(); ++j) {
                    tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workloads[j], false);
                }
                device->reset_sub_device_stall_group();
            } else {
                for (auto& mesh_workload : mesh_workloads) {
                    tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
                }
            }
            tt_metal::distributed::Finish(device->mesh_command_queue());
            for ([[maybe_unused]] auto& mesh_workload : mesh_workloads) {
                tt_metal::ReadMeshDeviceProfilerResults(*device);
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        if (num_mixed_df_layers == 1) {
            if (tile_format == tt::DataFormat::Bfp8_b) {
                pass = validation_bfp8_b(tensor_fp8, tile_format, num_blocks, cb_num_blocks, kt, nt, output_buffer, device.get());
            } else {
                pass = validation_fp16(tensor_fp16, tile_format, num_blocks, cb_num_blocks, kt, nt, output_buffer, device.get());
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
                num_mixed_df_layers,
                num_receivers,
                device.get());
        }

        pass &= device->close();
    } catch (const std::exception& e) {
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
