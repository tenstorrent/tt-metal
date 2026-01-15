// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
#include <tt-metalium/base_types.hpp>
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
// This test runs a DRAM prefetcher and matmul. DRAM prefetcher (core (0, 0))
// syncs and writes weights to the two neighbour matmul cores. activations are
// sharded on the matmul cores.
//
// Usage example:
//   ./build/test/tt_metal/perf_microbenchmark/11_remote_cb_sync_matmul_single_core/test_remote_cb_sync_matmul
//     --m
//     --k
//     --n
//     --num-blocks
//     --cb-num-blocks
//     --cb-padding
//     --num-tests <count of tests>
//     --data-type
//     --num-receivers
//     --num-layers
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::vector<T> slice_vec(const std::vector<T>& v, int m, int n) {
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

std::tuple<uint32_t, uint32_t> get_out_subblock_params(
    uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t choice = 0) {
    constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
        {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
        {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
    }};

    uint32_t index = 0;
    for (const auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        auto subblock_h = std::get<0>(subblock_hw);
        auto subblock_w = std::get<1>(subblock_hw);
        if (per_core_Mt % subblock_h == 0 and per_core_Nt % subblock_w == 0) {
            if (index >= choice) {
                return {subblock_h, subblock_w};
            }
            index++;
        }
    }

    return {1, 1};
}

std::tuple<std::vector<tt_metal::Program>, ::tt_metal::experimental::GlobalCircularBuffer> create_programs(
    tt_metal::distributed::MeshDevice* device,
    const CoreRangeSet& dram_reader_core,
    const CoreRangeSet& l1_receiver_cores,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    const uint32_t& single_tile_size,
    const tt::DataFormat& tile_format,
    uint32_t m,
    uint32_t k,
    uint32_t n,
    uint32_t num_blocks,
    uint32_t cb_num_blocks,
    uint32_t num_receivers,
    uint32_t num_layers,
    uint32_t cb_padding,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& in0_buffer,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& in1_buffer,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& output_buffer,
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
    uint32_t mt = m / 32;
    uint32_t kt = k / 32;
    uint32_t nt = n / 32;
    uint32_t in0_block_h = mt;
    uint32_t in0_block_w = kt / num_blocks;
    uint32_t in0_block_num_tiles = in0_block_h * in0_block_w;
    uint32_t in1_block_h = kt / num_blocks;
    uint32_t in1_num_tile_rows_write = in1_block_h;
    uint32_t in1_block_w = nt;
    uint32_t in1_block_num_tiles = in1_block_h * in1_block_w;

    // DRAM reader CB
    uint32_t in1_reader_cb_index = 0;
    uint32_t in1_reader_cb_size = in1_block_h * in1_block_w * single_tile_size * 3;
    uint32_t in1_reader_page_size, in1_reader_num_pages;
    get_max_page_size_and_num_pages(in1_block_num_tiles, single_tile_size, in1_reader_page_size, in1_reader_num_pages);

    uint32_t in1_receiver_block_num_tile = in1_block_h * in1_block_w / num_receivers;
    uint32_t in1_writer_page_size, in1_writer_num_pages;
    get_max_page_size_and_num_pages(
        in1_block_w / num_receivers, single_tile_size, in1_writer_page_size, in1_writer_num_pages);

    log_info(tt::LogTest, "in1_writer_page_size: {}", in1_writer_page_size);
    log_info(tt::LogTest, "in1_writer_num_pages: {}", in1_writer_num_pages);

    tt_metal::CircularBufferConfig in1_reader_cb_config =
        tt_metal::CircularBufferConfig(in1_reader_cb_size, {{in1_reader_cb_index, tile_format}})
            .set_page_size(in1_reader_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(sender_program, dram_reader_core, in1_reader_cb_config);

    uint32_t in1_receiver_cb_size = in1_block_h * in1_block_w * single_tile_size * cb_num_blocks / num_receivers;
    uint32_t padded_global_cb_size = in1_receiver_cb_size + cb_padding;

    auto global_cb = tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, padded_global_cb_size, tt_metal::BufferType::L1);

    uint32_t in1_writer_cb_index = 31;
    tt_metal::CircularBufferConfig in1_writer_cb_config = tt_metal::CircularBufferConfig(in1_receiver_cb_size);
    in1_writer_cb_config.remote_index(in1_writer_cb_index).set_page_size(single_tile_size).set_data_format(tile_format);
    tt_metal::experimental::CreateCircularBuffer(sender_program, dram_reader_core, in1_writer_cb_config, global_cb);

    // in0 reader CB
    uint32_t in0_reader_cb_index = 0;
    uint32_t in0_reader_cb_size = in0_block_h * in0_block_w * single_tile_size * num_blocks;
    tt_metal::CircularBufferConfig in0_reader_cb_config =
        tt_metal::CircularBufferConfig(in0_reader_cb_size, {{in0_reader_cb_index, tile_format}})
            .set_page_size(in0_reader_cb_index, single_tile_size)
            .set_globally_allocated_address(*in0_buffer->get_backing_buffer());
    tt_metal::CreateCircularBuffer(receiver_program, l1_receiver_cores, in0_reader_cb_config);

    // in1 receiver CB
    uint32_t in1_receiver_cb_index = 31;
    uint32_t in1_pusher_cb_index = 1;
    tt_metal::CircularBufferConfig in1_receiver_cb_config = tt_metal::CircularBufferConfig(in1_receiver_cb_size);
    in1_receiver_cb_config.remote_index(in1_receiver_cb_index)
        .set_page_size(single_tile_size)
        .set_data_format(tile_format);
    in1_receiver_cb_config.index(in1_pusher_cb_index).set_page_size(single_tile_size).set_data_format(tile_format);
    tt_metal::experimental::CreateCircularBuffer(
        receiver_program, l1_receiver_cores, in1_receiver_cb_config, global_cb);

    // output CB
    uint32_t output_cb_index = 16;
    uint32_t output_cb_size = in0_block_h * in1_block_w * single_tile_size / num_receivers;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(output_cb_size, {{output_cb_index, tile_format}})
            .set_page_size(output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output_buffer->get_backing_buffer());
    tt_metal::CreateCircularBuffer(receiver_program, l1_receiver_cores, output_cb_config);

    // sync CB
    uint32_t sync_cb_index = 2;
    uint32_t sync_cb_size = 32;
    tt_metal::CircularBufferConfig sync_cb_config =
        tt_metal::CircularBufferConfig(sync_cb_size, {{sync_cb_index, tile_format}})
            .set_page_size(sync_cb_index, sync_cb_size);
    tt_metal::CreateCircularBuffer(receiver_program, l1_receiver_cores, sync_cb_config);

    log_info(tt::LogTest, "in1_reader_cb_size: {}", in1_reader_cb_size);
    log_info(tt::LogTest, "in1_receiver_cb_size: {}", in1_receiver_cb_size);

    // in1 reader
    std::vector<uint32_t> in1_reader_compile_time_args = {
        (std::uint32_t)in1_buffer->address(),
        (std::uint32_t)start_tile_id,
        (std::uint32_t)tt_metal::NOC::RISCV_0_default,
        (std::uint32_t)num_layers};

    auto in1_reader_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernels/reader_dram.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = in1_reader_compile_time_args});

    // in1 writer
    std::vector<uint32_t> in1_writer_compile_time_args = {
        (std::uint32_t)tt_metal::NOC::RISCV_0_default,
        (std::uint32_t)num_receivers,
        (std::uint32_t)num_layers,
        (std::uint32_t)in1_writer_cb_index};

    auto in1_writer_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernels/writer_l1.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = in1_writer_compile_time_args});

    // in0 reader
    vector<uint32_t> in0_reader_compile_time_args = {(std::uint32_t)num_layers};

    auto in0_reader_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/11_remote_cb_sync_matmul_single_core/kernels/in0_reader.cpp",
        l1_receiver_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = in0_reader_compile_time_args});

    // in1 receiver
    std::vector<uint32_t> in1_receiver_compile_time_args = {
        (std::uint32_t)num_layers, (std::uint32_t)in1_receiver_cb_index};

    auto in1_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/11_remote_cb_sync_matmul_single_core/kernels/receiver_l1.cpp",
        l1_receiver_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = in1_receiver_compile_time_args});

    // compute
    uint32_t in1_per_core_w = nt / num_receivers;
    uint32_t out_block_w = nt / num_receivers;
    uint32_t out_block_num_tiles = mt * nt / num_receivers;
    vector<uint32_t> compute_kernel_compile_time_args = {
        in0_block_w,  // in0_block_w
        in0_block_num_tiles,
        in1_block_num_tiles / num_receivers,  // in1_block_num_tiles
        in1_per_core_w,                       // in1_per_core_w
        num_blocks,                           // num_blocks
        mt,                                   // out_subblock_h
        out_block_w,                          // out_block_w
        out_block_num_tiles,                  // out_block_num_tiles
        num_layers};

    tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/11_remote_cb_sync_matmul_single_core/kernels/"
        "bmm_large_block_zm_fused_bias_activation_copy.cpp",
        l1_receiver_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = tile_format == tt::DataFormat::Float16_b ? MathFidelity::HiFi2 : MathFidelity::LoFi,
            .fp32_dest_acc_en = true,
            .math_approx_mode = true,
            .compile_args = compute_kernel_compile_time_args});

    // reader rt
    auto dram_reader_core_coord = dram_reader_core.ranges().begin()->start_coord;
    log_info(tt::LogTest, "dram_reader_core_coord: {}", dram_reader_core_coord);
    auto dram_reader_core_coord_physical = device->worker_core_from_logical_core(dram_reader_core_coord);
    uint32_t bank_id = 0;
    uint32_t vc = bank_id & 0x1;
    std::vector<uint32_t> reader_rt_args = {(std::uint32_t)bank_id, (std::uint32_t)vc};
    for (uint32_t i = 0; i < num_layers; ++i) {
        reader_rt_args.push_back(in1_reader_page_size);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        reader_rt_args.push_back(in1_reader_num_pages);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        reader_rt_args.push_back(num_blocks);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        reader_rt_args.push_back(in1_block_num_tiles);
    }
    tt_metal::SetRuntimeArgs(sender_program, in1_reader_kernel, dram_reader_core_coord, reader_rt_args);

    // in1 writer rt
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
    for (uint32_t i = 0; i < num_layers; ++i) {
        writer_rt_args.push_back(in1_writer_page_size);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        writer_rt_args.push_back(in1_writer_num_pages);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        writer_rt_args.push_back(num_blocks);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        writer_rt_args.push_back(in1_block_num_tiles);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        writer_rt_args.push_back(single_tile_size);
    }
    for (uint32_t i = 0; i < num_layers; ++i) {
        writer_rt_args.push_back(in1_num_tile_rows_write);
    }
    tt_metal::SetRuntimeArgs(sender_program, in1_writer_kernel, dram_reader_core_coord, writer_rt_args);

    // in1 receiver rt
    for (uint32_t i = 0; i < num_receivers; ++i) {
        std::vector<uint32_t> receiver_rt_args = {
            (std::uint32_t)vc & 0x3,
            (std::uint32_t)dram_reader_core_coord_physical.x,
            (std::uint32_t)dram_reader_core_coord_physical.y,
            (std::uint32_t)i};
        vc++;

        for (uint32_t i = 0; i < num_layers; ++i) {
            receiver_rt_args.push_back(single_tile_size);
        }
        for (uint32_t i = 0; i < num_layers; ++i) {
            receiver_rt_args.push_back(num_blocks);
        }
        for (uint32_t i = 0; i < num_layers; ++i) {
            receiver_rt_args.push_back(in1_receiver_block_num_tile);
        }

        log_info(tt::LogTest, "l1_receiver_core_coords: {}", l1_receiver_core_coords[i]);

        tt_metal::SetRuntimeArgs(receiver_program, in1_receiver_kernel, l1_receiver_core_coords[i], receiver_rt_args);
    }

    // in0 reader
    for (uint32_t i = 0; i < num_receivers; ++i) {
        std::vector<uint32_t> in0_reader_rt_args;
        in0_reader_rt_args.reserve(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i) {
            in0_reader_rt_args.push_back(num_blocks);
        }
        for (uint32_t i = 0; i < num_layers; ++i) {
            in0_reader_rt_args.push_back(in0_block_num_tiles);
        }
        for (uint32_t i = 0; i < num_layers; ++i) {
            in0_reader_rt_args.push_back(out_block_num_tiles);
        }
        tt_metal::SetRuntimeArgs(receiver_program, in0_reader_kernel, l1_receiver_core_coords[i], in0_reader_rt_args);
    }

    return {std::move(programs), std::move(global_cb)};
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
    const tt::deprecated::Tensor<float>& in0_tensor,
    const tt::deprecated::Tensor<float>& in1_tensor,
    const tt::DataFormat& /*data_format*/,
    uint32_t /*num_blocks*/,
    uint32_t /*cb_num_blocks*/,
    uint32_t mt,
    uint32_t kt,
    uint32_t nt,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& out_buffer,
    uint32_t num_receivers,
    tt_metal::distributed::MeshDevice* device) {
    bool pass = true;
    std::vector<float> golden_vec(mt * nt * 32 * 32, 0);  // Initialize with zeros
    std::vector<float> result_vec(mt * nt * 32 * 32, 0);

    std::vector<float> result_untilized;
    std::vector<uint32_t> result;
    tt_metal::distributed::ReadShard(
        device->mesh_command_queue(), result, out_buffer, tt_metal::distributed::MeshCoordinate(0, 0), true);
    auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result, true, false);
    result_untilized = untilize_swizzled(result_bfp8, mt * 32, nt * 32);

    const auto& in0_values = in0_tensor.get_values();
    const auto& in1_values = in1_tensor.get_values();

    auto per_core_n = nt * 32 / num_receivers;
    for (size_t i = 0; i < mt * 32; ++i) {
        for (size_t n = 0; n < num_receivers; ++n) {
            for (size_t j = 0; j < per_core_n; ++j) {
                float sum = 0;
                for (size_t k = 0; k < kt * 32; ++k) {
                    sum += to_float(in0_values[(n * kt * 32) + (i * num_receivers * kt * 32) + k]) *
                           to_float(in1_values[(n * per_core_n) + (k * nt * 32) + j]);
                }
                golden_vec[(i * nt * 32) + (n * per_core_n) + j] = sum;
            }
        }
    }

    for (int i = 0; i < result_untilized.size(); ++i) {
        result_vec[i] = result_untilized[i];
    }

    float res = pcc(golden_vec, result_vec);
    pass &= res >= 0.999;
    if (!pass) {
        TT_FATAL(pass, "validation failed, pcc: {}", res);
    }
    return pass;
}

bool validation_fp16(
    const tt::deprecated::Tensor<bfloat16>& in0_tensor,
    const tt::deprecated::Tensor<bfloat16>& in1_tensor,
    const tt::DataFormat& /*data_format*/,
    uint32_t /*num_blocks*/,
    uint32_t /*cb_num_blocks*/,
    uint32_t mt,
    uint32_t kt,
    uint32_t nt,
    const std::shared_ptr<tt_metal::distributed::MeshBuffer>& out_buffer,
    uint32_t num_receivers,
    tt_metal::distributed::MeshDevice* device) {
    bool pass = true;
    std::vector<float> golden_vec(mt * nt * 32 * 32, 0);  // Initialize with zeros
    std::vector<float> result_vec(mt * nt * 32 * 32, 0);

    std::vector<uint32_t> result;
    tt_metal::distributed::ReadShard(
        device->mesh_command_queue(), result, out_buffer, tt_metal::distributed::MeshCoordinate(0, 0), true);
    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result);
    auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
    auto result_untilized = untilize_swizzled(result_flat_layout, mt * 32, nt * 32);

    const auto& in0_values = in0_tensor.get_values();
    const auto& in1_values = in1_tensor.get_values();

    auto per_core_n = nt * 32 / num_receivers;
    for (size_t i = 0; i < mt * 32; ++i) {
        for (size_t n = 0; n < num_receivers; ++n) {
            for (size_t j = 0; j < per_core_n; ++j) {
                float sum = 0;
                for (size_t k = 0; k < kt * 32; ++k) {
                    sum += to_float(in0_values[(n * kt * 32) + (i * num_receivers * kt * 32) + k]) *
                           to_float(in1_values[(n * per_core_n) + (k * nt * 32) + j]);
                }
                golden_vec[(i * nt * 32) + (n * per_core_n) + j] = sum;
            }
        }
    }

    for (int i = 0; i < result_untilized.size(); ++i) {
        result_vec[i] = to_float(static_cast<bfloat16>(result_untilized[i]));
    }

    float res = pcc(golden_vec, result_vec);
    pass &= res >= 0.999;
    if (!pass) {
        TT_FATAL(pass, "validation failed, pcc: {}", res);
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
    log_info(tt::LogTest, "num_receivers: {}", num_receivers);

    auto device_local_config = tt_metal::distributed::DeviceLocalBufferConfig{
        .page_size = page_size_bytes, .buffer_type = buffer_type, .sharding_args = sharding_args};

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
    uint32_t num_layers = 1;
    uint64_t m = 32, k = 8192, n = 128;
    bool use_sub_devices = false;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        try {
            std::tie(m, input_args) = test_args::get_command_option_uint64_and_remaining_args(input_args, "--m", 32);
            std::tie(k, input_args) = test_args::get_command_option_uint64_and_remaining_args(input_args, "--k", 8192);
            std::tie(n, input_args) = test_args::get_command_option_uint64_and_remaining_args(input_args, "--n", 128);
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
            std::tie(num_layers, input_args) =
                test_args::get_command_option_uint64_and_remaining_args(input_args, "--num-layers", 1);
            std::tie(use_sub_devices, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--use-sub-devices");

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
            TT_FATAL(false, "Command line arguments found exception");
        }

        log_info(tt::LogTest, "num_layers: {} ", num_layers);
        log_info(tt::LogTest, "num_receivers: {} ", num_receivers);

        TT_FATAL(cb_num_blocks >= num_blocks, "Global CB must contain more (or equal) blocks than a single layer");

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
        uint32_t in1_size = 0;
        tt::DataFormat tile_format = tt::DataFormat::Bfp8_b;
        if (df == 0) {
            in1_size = k * n * 1088 / 1024;
            tile_format = tt::DataFormat::Bfp8_b;
        } else if (df == 1) {
            in1_size = k * n * 2;
            tile_format = tt::DataFormat::Float16_b;
        } else {
            TT_THROW("Input data format {} is invalid. Please change.", df);
        }
        uint32_t mt = m / 32;
        uint32_t kt = k / 32;
        uint32_t nt = n / 32;

        uint32_t single_tile_size = tt::tile_size(tile_format);

        TT_FATAL(in1_size % single_tile_size == 0, "input size is not aligned to tile size");
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
            {dram_reader_core_coord, l1_receiver_core}};
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
        std::shared_ptr<tt_metal::distributed::MeshBuffer> in0_buffer;
        std::vector<std::shared_ptr<tt_metal::distributed::MeshBuffer>> in1_buffers(num_layers);
        std::shared_ptr<tt_metal::distributed::MeshBuffer> in1_l1_buffer;
        std::shared_ptr<tt_metal::distributed::MeshBuffer> output_buffer;
        SHAPE in0_shape = SHAPE{1, 1, m, k * num_receivers};
        tt::deprecated::Tensor<bfloat16> in0_tensor_fp16 = tt::deprecated::initialize_tensor<bfloat16>(
            in0_shape,
            tt::deprecated::Initialize::RANDOM,
            -1,
            1,
            std::chrono::system_clock::now().time_since_epoch().count());
        tt::deprecated::Tensor<float> in0_tensor_fp8 = tt::deprecated::initialize_tensor<float>(
            in0_shape,
            tt::deprecated::Initialize::RANDOM,
            -1,
            1,
            std::chrono::system_clock::now().time_since_epoch().count());
        auto in1_shape = SHAPE{1, 1, k, n};
        tt::deprecated::Tensor<bfloat16> in1_tensor_fp16 = tt::deprecated::initialize_tensor<bfloat16>(
            in1_shape,
            tt::deprecated::Initialize::RANDOM,
            -1,
            1,
            std::chrono::system_clock::now().time_since_epoch().count());
        tt::deprecated::Tensor<float> in1_tensor_fp8 = tt::deprecated::initialize_tensor<float>(
            in1_shape,
            tt::deprecated::Initialize::RANDOM,
            -1,
            1,
            std::chrono::system_clock::now().time_since_epoch().count());
        if (tile_format == tt::DataFormat::Bfp8_b) {
            // in1 DRAM
            for (uint32_t i = 0; i < num_layers; ++i) {
                auto input_vec_tilized = tilize_swizzled(in1_tensor_fp8.get_values(), k, n);
                std::vector<uint32_t> packed_input_vec_tile_layout =
                    pack_as_bfp8_tiles(tt::stl::make_const_span(input_vec_tilized), true, false);
                in1_buffers[i] = create_and_transfer_data_sharded_cb(
                    device.get(),
                    packed_input_vec_tile_layout,
                    kt,
                    nt,
                    tt_metal::BufferType::DRAM,
                    tt::DataFormat::Bfp8_b,
                    dram_reader_core,
                    num_banks);
            }

            // in0
            auto activations_tilized = tilize_swizzled(in0_tensor_fp8.get_values(), m, k * num_receivers);
            std::vector<uint32_t> activations =
                pack_as_bfp8_tiles(tt::stl::make_const_span(activations_tilized), true, false);
            in0_buffer = create_and_transfer_data_sharded_cb(
                device.get(),
                activations,
                mt,
                kt * num_receivers,
                tt_metal::BufferType::L1,
                tt::DataFormat::Bfp8_b,
                l1_receiver_core,
                num_receivers);

            // output
            vector<uint32_t> outputs = test_utils::create_constant_vector_of_bfp8(mt * nt * single_tile_size, 0, false);
            output_buffer = create_and_transfer_data_sharded_cb(
                device.get(),
                outputs,
                mt,
                nt,
                tt_metal::BufferType::L1,
                tt::DataFormat::Bfp8_b,
                l1_receiver_core,
                num_receivers);

        } else {
            // in1
            for (uint32_t i = 0; i < num_layers; ++i) {
                auto input_vec_tilized = tilize_swizzled(in1_tensor_fp16.get_values(), k, n);
                auto input_vec_tile_layout =
                    convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(input_vec_tilized));
                vector<uint32_t> packed_input_vec_tile_layout =
                    pack_bfloat16_vec_into_uint32_vec(input_vec_tile_layout);
                in1_buffers[i] = create_and_transfer_data_sharded_cb(
                    device.get(),
                    packed_input_vec_tile_layout,
                    kt,
                    nt,
                    tt_metal::BufferType::DRAM,
                    tt::DataFormat::Float16_b,
                    dram_reader_core,
                    num_banks);
            }

            // in0
            auto activations_tilized = tilize_swizzled(in0_tensor_fp16.get_values(), m, k * num_receivers);
            auto activations_tile_layout =
                convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(activations_tilized));
            vector<uint32_t> activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
            in0_buffer = create_and_transfer_data_sharded_cb(
                device.get(),
                activations,
                mt,
                kt * num_receivers,
                tt_metal::BufferType::L1,
                tt::DataFormat::Float16_b,
                l1_receiver_core,
                num_receivers);

            // output
            vector<uint32_t> outputs = create_constant_vector_of_bfloat16(mt * nt * single_tile_size, 0);
            output_buffer = create_and_transfer_data_sharded_cb(
                device.get(),
                outputs,
                mt,
                nt,
                tt_metal::BufferType::L1,
                tt::DataFormat::Float16_b,
                l1_receiver_core,
                num_receivers);
        }

        for (uint32_t i = 0; i < num_layers; ++i) {
            log_info(tt::LogTest, "in1_buffers addr: {}", in1_buffers[i]->address());
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [programs, global_cb] = create_programs(
            device.get(),
            dram_reader_core,
            l1_receiver_core,
            sender_receiver_core_mapping,
            single_tile_size,
            tile_format,
            m,
            k,
            n,
            num_blocks,
            cb_num_blocks,
            num_receivers,
            num_layers,
            cb_padding,
            in0_buffer,
            in1_buffers[0],
            output_buffer,
            use_sub_devices);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execution Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<tt_metal::distributed::MeshWorkload> mesh_workloads;
        for (auto& program : programs) {
            auto mesh_workload = tt_metal::distributed::MeshWorkload();
            mesh_workload.add_program(
                tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
            mesh_workloads.push_back(std::move(mesh_workload));
        }

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
        if (tile_format == tt::DataFormat::Bfp8_b) {
            pass = validation_bfp8_b(
                in0_tensor_fp8,
                in1_tensor_fp8,
                tile_format,
                num_blocks,
                cb_num_blocks,
                mt,
                kt,
                nt,
                output_buffer,
                num_receivers,
                device.get());
        } else {
            pass = validation_fp16(
                in0_tensor_fp16,
                in1_tensor_fp16,
                tile_format,
                num_blocks,
                cb_num_blocks,
                mt,
                kt,
                nt,
                output_buffer,
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
