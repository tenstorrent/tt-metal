// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_kv_reduce_program_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_kv_reduce_reader_kernel.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_kv_reduce_writer_kernel.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_kv_reduce_compute_kernel.cpp";

constexpr auto kPartialsCbIndex = tt::CBIndex::c_0;  // G fp32 tiles per output tile
constexpr auto kOutputCbIndex = tt::CBIndex::c_1;    // one bf16 tile

// reader runtime args
constexpr uint32_t kReaderKeyPartialIdx = 0;
constexpr uint32_t kReaderValuePartialIdx = 1U;
// writer runtime args
constexpr uint32_t kWriterKeyIdx = 0;
constexpr uint32_t kWriterValueIdx = 1U;

}  // namespace

namespace ttml::metal::ops::sdpa_bw::device {

/**
 *   Work items are the output tiles: first every tile of grad_K (page order), then every tile of grad_V.
 *   Item p of a segment with `row_tiles` tiles per (batch, kv head) reads the G partial tiles
 *   ((p / row_tiles) * G + h) * row_tiles + p % row_tiles, h < G, sums them in fp32 and packs bf16.
 */
SDPABackwardKVReduceProgramFactory::cached_program_t SDPABackwardKVReduceProgramFactory::create(
    const kv_reduce::operation_attributes_t& args,
    const kv_reduce::tensor_args_t& tensor_args,
    kv_reduce::tensor_return_value_t& output) {
    const auto& gk_partial = tensor_args.grad_key_partial;
    const auto& gv_partial = tensor_args.grad_value_partial;
    auto& [grad_key, grad_value] = output;
    auto* device = gk_partial.device();
    tt::tt_metal::Program program{};

    const uint32_t G = args.heads_per_group;
    const auto [kB, kNH, kS, kE] = grad_key.padded_shape().to_array_4D();
    const auto [vB, vNH, vS, vE] = grad_value.padded_shape().to_array_4D();
    const uint32_t St = kS / tt::constants::TILE_HEIGHT;
    const uint32_t kWt = kE / tt::constants::TILE_WIDTH;
    const uint32_t vWt = vE / tt::constants::TILE_WIDTH;
    const uint32_t k_row_tiles = St * kWt;  // tiles per (batch, kv head) of grad_K
    const uint32_t v_row_tiles = St * vWt;
    const uint32_t k_tiles = kB * kNH * k_row_tiles;
    const uint32_t v_tiles = vB * vNH * v_row_tiles;
    const uint32_t total_items = k_tiles + v_tiles;

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, items_group_1, items_group_2] =
        tt::tt_metal::split_work_to_cores(grid, total_items);

    const uint32_t fp32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    const auto out_format = datatype_to_dataformat_converter(grad_key.dtype());
    const uint32_t out_tile_bytes = tt::tile_size(out_format);
    create_circular_buffer(program, all_cores, kPartialsCbIndex, tt::DataFormat::Float32, fp32_tile_bytes, 2 * G);
    create_circular_buffer(program, all_cores, kOutputCbIndex, out_format, out_tile_bytes, 2);

    std::vector<uint32_t> reader_args = {G, k_tiles, k_row_tiles, v_row_tiles};
    tt::tt_metal::TensorAccessorArgs(gk_partial.buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(gv_partial.buffer()).append_to(reader_args);
    auto reader = create_reader_kernel(program, all_cores, reader_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_args = {k_tiles};
    tt::tt_metal::TensorAccessorArgs(grad_key.buffer()).append_to(writer_args);
    tt::tt_metal::TensorAccessorArgs(grad_value.buffer()).append_to(writer_args);
    auto writer = create_writer_kernel(program, all_cores, writer_args, {}, kWriterKernelPath);

    auto compute = create_compute_kernel(program, all_cores, {G}, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    for_each_core_with_work(
        num_cores, num_cores_y, core_group_1, core_group_2, items_group_1, items_group_2, [&](const CoreWork& work) {
            const auto& [core, core_index, count, start, in_group_1] = work;
            SetRuntimeArgs(
                program, reader, core, {gk_partial.buffer()->address(), gv_partial.buffer()->address(), start, count});
            SetRuntimeArgs(
                program, writer, core, {grad_key.buffer()->address(), grad_value.buffer()->address(), start, count});
            SetRuntimeArgs(program, compute, core, {count});
        });

    return cached_program_t{
        std::move(program),
        {.reader_kernel = reader, .writer_kernel = writer, .num_cores = num_cores, .num_cores_y = num_cores_y}};
}

void SDPABackwardKVReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const kv_reduce::operation_attributes_t&,
    const kv_reduce::tensor_args_t& tensor_args,
    kv_reduce::tensor_return_value_t& output) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;
    auto& [grad_key, grad_value] = output;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel);
    for_each_core(shared.num_cores, shared.num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderKeyPartialIdx] = tensor_args.grad_key_partial.buffer()->address();
        r[kReaderValuePartialIdx] = tensor_args.grad_value_partial.buffer()->address();
        auto& w = writer_args[core.x][core.y];
        w[kWriterKeyIdx] = grad_key.buffer()->address();
        w[kWriterValueIdx] = grad_value.buffer()->address();
    });
}

}  // namespace ttml::metal::ops::sdpa_bw::device
