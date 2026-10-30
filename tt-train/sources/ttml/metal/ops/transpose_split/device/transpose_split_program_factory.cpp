// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_split_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "transpose_split_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/transpose_split/device/kernels/dataflow/reader_transpose_split.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/transpose_split/device/kernels/dataflow/writer_transpose_split.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/transpose_split/device/kernels/compute/transpose_split_kernel.cpp";

constexpr uint32_t kInCb = 0;
constexpr uint32_t kOutCb = 1;

constexpr uint32_t kReaderInIdx = 0U;
constexpr uint32_t kWriterDxIdx = 0U;
constexpr uint32_t kWriterTailIdx = 1U;

}  // namespace

namespace ttml::metal::ops::transpose_split::device {

// Items are the output tiles: first dx's Tt x rows_t tiles row-major (item -> (t, r), input tile (r, t)), then
// the Tt tail tiles (input tile row rows_t). One tile per item: read, transpose, (scale), write.
TransposeSplitProgramFactory::cached_program_t TransposeSplitProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    auto& [dx, tail] = output;
    auto* device = input.device();
    tt::tt_metal::Program program{};

    const uint32_t Tt = input.padded_shape()[3] / tt::constants::TILE_WIDTH;
    const uint32_t rows_t = args.rows / tt::constants::TILE_HEIGHT;
    const uint32_t dx_items = Tt * rows_t;
    const uint32_t total_items = dx_items + Tt;
    const auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, total_items);

    const auto bf16 = tt::DataFormat::Float16_b;
    const uint32_t tile = tt::tile_size(bf16);
    create_circular_buffer(program, all_cores, kInCb, bf16, tile, 2U);
    create_circular_buffer(program, all_cores, kOutCb, bf16, tile, 2U);

    const uint32_t scale_bits = std::bit_cast<uint32_t>(args.tail_scale);
    std::map<std::string, std::string> defines;

    std::vector<uint32_t> reader_ct_args{Tt, rows_t, dx_items};
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{Tt, rows_t, dx_items};
    tt::tt_metal::TensorAccessorArgs(dx.buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(tail.buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        return create_compute_kernel(
            program,
            cores,
            {work_count, Tt, rows_t, dx_items, scale_bits},
            defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true);
    };
    const auto compute_1 = make_compute(core_group_1, work_per_core_1);
    const auto compute_2 = core_group_2.ranges().empty() ? compute_1 : make_compute(core_group_2, work_per_core_2);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0, work_start = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / grid.y, i % grid.y};
        const bool in_group_1 = core_group_1.contains(core);
        const uint32_t work_count = in_group_1 ? work_per_core_1 : work_per_core_2;
        SetRuntimeArgs(program, in_group_1 ? compute_1 : compute_2, core, {work_start});
        SetRuntimeArgs(program, reader, core, {input.buffer()->address(), work_start, work_count});
        SetRuntimeArgs(
            program, writer, core, {dx.buffer()->address(), tail.buffer()->address(), work_start, work_count});
        cores.push_back(core);
        work_start += work_count;
    }
    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void TransposeSplitProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& [dx, tail] = output;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        reader_args[core.x][core.y][kReaderInIdx] = tensor_args.input.buffer()->address();
        auto& w = writer_args[core.x][core.y];
        w[kWriterDxIdx] = dx.buffer()->address();
        w[kWriterTailIdx] = tail.buffer()->address();
    }
}

}  // namespace ttml::metal::ops::transpose_split::device
