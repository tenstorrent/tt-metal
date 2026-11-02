// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "skinny_matmul_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/skinny_matmul_cbs.hpp"
#include "metal/common/program_utils.hpp"
#include "skinny_matmul_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/dataflow/reader_skinny_matmul.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/dataflow/writer_skinny_matmul.cpp";
constexpr auto kPartialComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/compute/skinny_matmul_partial_kernel.cpp";
constexpr auto kReduceComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/compute/skinny_matmul_reduce_kernel.cpp";

constexpr uint32_t kBlockTiles = 4U;  // x tiles per CB push

constexpr uint32_t kReaderIn0Idx = 0U;
constexpr uint32_t kReaderIn1Idx = 1U;
constexpr uint32_t kWriterOutIdx = 0U;
constexpr uint32_t kWriterTailIdx = 1U;

struct CoreSplit {
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

CoreSplit split_items(ttnn::IDevice* device, uint32_t total_work) {
    const auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, total_work);
    return CoreSplit{
        .num_cores = num_cores,
        .num_cores_y = grid.y,
        .all_cores = all_cores,
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2};
}

template <typename ComputeArgsFn>
std::vector<tt::tt_metal::CoreCoord> create_compute_and_assign_args(
    tt::tt_metal::Program& program,
    const CoreSplit& split,
    const std::map<std::string, std::string>& defines,
    const char* compute_path,
    ComputeArgsFn&& compute_args_for,
    tt::tt_metal::KernelHandle reader,
    tt::tt_metal::KernelHandle writer,
    std::vector<uint32_t> reader_addrs,
    std::vector<uint32_t> writer_addrs) {
    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        return create_compute_kernel(
            program, cores, compute_args_for(work_count), defines, compute_path, /*fp32_dest_acc_en=*/true);
    };
    const auto compute_1 = make_compute(split.core_group_1, split.work_per_core_1);
    const auto compute_2 =
        split.core_group_2.ranges().empty() ? compute_1 : make_compute(split.core_group_2, split.work_per_core_2);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(split.num_cores);
    for (uint32_t i = 0, work_start = 0; i < split.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / split.num_cores_y, i % split.num_cores_y};
        const bool in_group_1 = split.core_group_1.contains(core);
        const uint32_t work_count = in_group_1 ? split.work_per_core_1 : split.work_per_core_2;
        SetRuntimeArgs(program, in_group_1 ? compute_1 : compute_2, core, {work_start});
        auto r_args = reader_addrs;
        r_args.push_back(work_start);
        r_args.push_back(work_count);
        SetRuntimeArgs(program, reader, core, r_args);
        auto w_args = writer_addrs;
        w_args.push_back(work_start);
        w_args.push_back(work_count);
        SetRuntimeArgs(program, writer, core, w_args);
        cores.push_back(core);
        work_start += work_count;
    }
    return cores;
}

}  // namespace

namespace ttml::metal::ops::skinny_matmul::device {

// ---------------------------------------------------------------------------------------------------------------
// Phase A: partial products per (row group, K slice)
// ---------------------------------------------------------------------------------------------------------------

SkinnyMatmulPartialProgramFactory::cached_program_t SkinnyMatmulPartialProgramFactory::create(
    const partial::operation_attributes_t& args, const partial::tensor_args_t& tensor_args, ttnn::Tensor& output) {
    namespace cb = skinny_matmul_cb;
    const auto& x = tensor_args.x;
    auto* device = x.device();
    tt::tt_metal::Program program{};

    const auto geo = skinny_geometry(x, tensor_args.w, args.transpose_b, args.x_transposed);
    const uint32_t R = args.rows_per_item;
    const uint32_t S = args.num_slices;
    const uint32_t St = args.slice_tiles;
    const uint32_t groups = (geo.rows + R - 1U) / R;
    // x^T's tiles of one row sit a row stride apart (one DRAM page each): the whole slice in one batch per row
    const uint32_t block = args.x_transposed ? St : std::min(kBlockTiles, St);
    const CoreSplit split = split_items(device, groups * S);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t f32_tile = tt::tile_size(f32);

    create_circular_buffer(program, split.all_cores, cb::x, bf16, bf16_tile, 2U * block);
    create_circular_buffer(program, split.all_cores, cb::w, bf16, bf16_tile, St * geo.Nt);
    create_circular_buffer(program, split.all_cores, cb::partial_out, f32, f32_tile, 2U * geo.Nt);

    std::map<std::string, std::string> defines;
    if (args.transpose_b) {
        defines["TRANSPOSE_B"] = "1";
    }
    if (args.x_transposed) {
        defines["X_TRANSPOSED"] = "1";  // x tile (r, k) at page k * Kx + r (Kx = rows)
    }

    std::vector<uint32_t> reader_ct_args{geo.rows, geo.Kt, geo.Nt, R, S, St, block, geo.Kx};
    tt::tt_metal::TensorAccessorArgs(x.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.w.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, split.all_cores, reader_ct_args, defines, kReaderKernelPath);

    // Writer ct args 7..10 are the REDUCE phase's tail target (unused here); the second accessor is a stand-in.
    std::vector<uint32_t> writer_ct_args{geo.rows, geo.Kt, geo.Nt, R, S, St, block, 0U, 0U, 0U, 0U};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, split.all_cores, writer_ct_args, defines, kWriterKernelPath);

    auto cores = create_compute_and_assign_args(
        program,
        split,
        defines,
        kPartialComputeKernelPath,
        [&](uint32_t work_count) {
            return std::vector<uint32_t>{work_count, geo.rows, geo.Kt, geo.Nt, R, S, St, block};
        },
        reader,
        writer,
        {x.buffer()->address(), tensor_args.w.buffer()->address()},
        {output.buffer()->address(), 0U});

    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void SkinnyMatmulPartialProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const partial::operation_attributes_t&,
    const partial::tensor_args_t& tensor_args,
    ttnn::Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderIn0Idx] = tensor_args.x.buffer()->address();
        r[kReaderIn1Idx] = tensor_args.w.buffer()->address();
        writer_args[core.x][core.y][kWriterOutIdx] = output.buffer()->address();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Phase B: sum of the slices per output tile
// ---------------------------------------------------------------------------------------------------------------

SkinnyMatmulReduceProgramFactory::cached_program_t SkinnyMatmulReduceProgramFactory::create(
    const reduce::operation_attributes_t& args, const reduce::tensor_args_t& tensor_args, ttnn::Tensor& output) {
    namespace cb = skinny_matmul_cb;
    const auto& x = tensor_args.x;
    auto* device = x.device();
    tt::tt_metal::Program program{};

    const auto geo = skinny_geometry(x, tensor_args.w, args.transpose_b, args.x_transposed);
    const uint32_t S = args.num_slices;
    const CoreSplit split = split_items(device, geo.rows * geo.Nt);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    create_circular_buffer(program, split.all_cores, cb::partials, f32, tt::tile_size(f32), 2U * S);
    create_circular_buffer(program, split.all_cores, cb::out, bf16, tt::tile_size(bf16), 2U);
    create_circular_buffer(program, split.all_cores, cb::zero, bf16, tt::tile_size(bf16), 1U);

    std::map<std::string, std::string> defines{{"REDUCE", "1"}};
    if (args.transpose_out) {
        defines["TRANSPOSE_OUT"] = "1";  // the summed tile transposed in DEST, written as tile (n, r)
    }
    if (args.x_transposed) {
        defines["X_TRANSPOSED"] = "1";  // the partials are the transposed tiles already
    }

    std::vector<uint32_t> reader_ct_args{geo.rows, geo.Kt, geo.Nt, args.rows_per_item, S, args.slice_tiles, 1U, geo.Kx};
    tt::tt_metal::TensorAccessorArgs(tensor_args.partials.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, split.all_cores, reader_ct_args, defines, kReaderKernelPath);

    const bool tail = tensor_args.tail_out.has_value();
    const uint32_t tail_stride = tail ? tensor_args.tail_out->padded_shape()[3] / tt::constants::TILE_WIDTH : 0U;
    // (transposed output: the target's row stride in tiles, the tiles land in its rows tail_col ..)
    std::vector<uint32_t> writer_ct_args{
        geo.rows,
        geo.Kt,
        geo.Nt,
        args.rows_per_item,
        S,
        args.slice_tiles,
        1U,
        tail ? 1U : 0U,
        tail_stride,
        args.tail_col,
        args.tail_tiles};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(tail ? tensor_args.tail_out->buffer() : output.buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, split.all_cores, writer_ct_args, defines, kWriterKernelPath);

    auto cores = create_compute_and_assign_args(
        program,
        split,
        defines,
        kReduceComputeKernelPath,
        [&](uint32_t work_count) {
            return std::vector<uint32_t>{
                work_count, geo.rows, geo.Kt, geo.Nt, args.rows_per_item, S, args.slice_tiles, 1U};
        },
        reader,
        writer,
        {tensor_args.partials.buffer()->address(), /*in1*/ 0U},
        {output.buffer()->address(), tail ? tensor_args.tail_out->buffer()->address() : 0U});

    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void SkinnyMatmulReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const reduce::operation_attributes_t&,
    const reduce::tensor_args_t& tensor_args,
    ttnn::Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        reader_args[core.x][core.y][kReaderIn0Idx] = tensor_args.partials.buffer()->address();
        writer_args[core.x][core.y][kWriterOutIdx] = output.buffer()->address();
        writer_args[core.x][core.y][kWriterTailIdx] =
            tensor_args.tail_out.has_value() ? tensor_args.tail_out->buffer()->address() : 0U;
    }
}

}  // namespace ttml::metal::ops::skinny_matmul::device
