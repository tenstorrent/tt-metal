// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_rmsnorm_fw_allcore_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/add_rmsnorm_fw_allcore_cbs.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_allcore/device/kernels/dataflow/reader_add_rmsnorm_fw_allcore.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_allcore/device/kernels/dataflow/writer_add_rmsnorm_fw_allcore.cpp";
constexpr auto kPartialComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_allcore/device/kernels/compute/"
    "add_rmsnorm_fw_allcore_partial_kernel.cpp";
constexpr auto kApplyComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_allcore/device/kernels/compute/add_rmsnorm_fw_allcore_kernel.cpp";

// Phase A keeps the running sum of squares in one fp32 DEST register next to the block's sum tiles and a scratch
// register; with fp32 accumulation DEST holds four tiles, so blocks of two.
constexpr uint32_t kPartialBlockTiles = 2U;
// Phase B streams one tile at a time through DEST (gamma multiply, then the scale); four tiles per CB push.
constexpr uint32_t kApplyBlockTiles = 4U;

// Reader / writer runtime-arg slots holding buffer addresses (refreshed on program-cache hits).
constexpr uint32_t kReaderIn0Idx = 0U;
constexpr uint32_t kReaderIn1Idx = 1U;
constexpr uint32_t kReaderIn2Idx = 2U;
constexpr uint32_t kWriterOut0Idx = 0U;
constexpr uint32_t kWriterOut1Idx = 1U;

struct Geometry {
    uint32_t rows = 0;  // B * N * Ht
    uint32_t Wt = 0;
    uint32_t num_inner = 0;
};

Geometry get_geometry(const ttnn::Tensor& t) {
    const auto& padded = t.padded_shape();
    return Geometry{
        .rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT),
        .Wt = padded[3] / tt::constants::TILE_WIDTH,
        .num_inner = t.logical_shape()[-1]};
}

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
        // The compute kernel derives each item's slice geometry from its absolute work index.
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

namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device {

// ---------------------------------------------------------------------------------------------------------------
// Phase A: sum and partial sums of squares
// ---------------------------------------------------------------------------------------------------------------

AddRMSNormForwardPartialProgramFactory::cached_program_t AddRMSNormForwardPartialProgramFactory::create(
    const partial::operation_attributes_t& args,
    const partial::tensor_args_t& tensor_args,
    partial::tensor_return_value_t& output) {
    namespace cb = add_rmsnorm_fw_cb;
    const auto& residual = tensor_args.residual;
    auto* device = residual.device();
    tt::tt_metal::Program program{};

    const Geometry geo = get_geometry(residual);
    const uint32_t S = args.num_slices;
    const uint32_t St = args.slice_tiles;
    const uint32_t block = std::min(kPartialBlockTiles, St);
    const CoreSplit split = split_items(device, geo.rows * S);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t f32_tile = tt::tile_size(f32);

    create_circular_buffer(program, split.all_cores, cb::residual, bf16, bf16_tile, 2U * block);
    create_circular_buffer(program, split.all_cores, cb::delta, bf16, bf16_tile, 2U * block);
    create_circular_buffer(program, split.all_cores, cb::sum_out, bf16, bf16_tile, 2U * block);
    create_circular_buffer(program, split.all_cores, cb::acc, f32, f32_tile, 1U);
    create_circular_buffer(program, split.all_cores, cb::partial_out, f32, f32_tile, 2U);
    create_circular_buffer(program, split.all_cores, cb::zero, bf16, bf16_tile, 1U);

    std::map<std::string, std::string> defines;

    std::vector<uint32_t> reader_ct_args{geo.Wt, S, St, block};
    tt::tt_metal::TensorAccessorArgs(residual.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.delta.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, split.all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{geo.Wt, S, St, block};
    tt::tt_metal::TensorAccessorArgs(output[0].buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(output[1].buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, split.all_cores, writer_ct_args, defines, kWriterKernelPath);

    auto cores = create_compute_and_assign_args(
        program,
        split,
        defines,
        kPartialComputeKernelPath,
        [&](uint32_t work_count) { return std::vector<uint32_t>{work_count, geo.Wt, S, St, block, 0U, 0U}; },
        reader,
        writer,
        {residual.buffer()->address(), tensor_args.delta.buffer()->address(), /*in2*/ 0U},
        {output[0].buffer()->address(), output[1].buffer()->address()});

    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void AddRMSNormForwardPartialProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const partial::operation_attributes_t&,
    const partial::tensor_args_t& tensor_args,
    partial::tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderIn0Idx] = tensor_args.residual.buffer()->address();
        r[kReaderIn1Idx] = tensor_args.delta.buffer()->address();
        auto& w = writer_args[core.x][core.y];
        w[kWriterOut0Idx] = output[0].buffer()->address();
        w[kWriterOut1Idx] = output[1].buffer()->address();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Phase B: rms and output
// ---------------------------------------------------------------------------------------------------------------

AddRMSNormForwardProgramFactory::cached_program_t AddRMSNormForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    namespace cb = add_rmsnorm_fw_cb;
    const auto& sum = tensor_args.sum;
    auto* device = sum.device();
    tt::tt_metal::Program program{};

    const Geometry geo = get_geometry(sum);
    const uint32_t S = args.num_slices;
    const uint32_t St = args.slice_tiles;
    const uint32_t block = std::min(kApplyBlockTiles, St);
    const CoreSplit split = split_items(device, geo.rows * S);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t f32_tile = tt::tile_size(f32);

    create_circular_buffer(program, split.all_cores, cb::sum, bf16, bf16_tile, 2U * block);
    create_circular_buffer(program, split.all_cores, cb::gamma, bf16, bf16_tile, 2U * block);
    create_circular_buffer(program, split.all_cores, cb::partials, f32, f32_tile, 2U * S);
    create_circular_buffer(program, split.all_cores, cb::ones, bf16, bf16_tile, 1U);
    create_circular_buffer(program, split.all_cores, cb::zero, bf16, bf16_tile, 1U);
    create_circular_buffer(program, split.all_cores, cb::row_acc, f32, f32_tile, 1U);
    create_circular_buffer(program, split.all_cores, cb::rms_tmp, f32, f32_tile, 1U);
    create_circular_buffer(program, split.all_cores, cb::rms_out, bf16, bf16_tile, 2U);
    create_circular_buffer(program, split.all_cores, cb::inv, f32, f32_tile, 1U);
    create_circular_buffer(program, split.all_cores, cb::out, bf16, bf16_tile, 2U * block);

    std::map<std::string, std::string> defines{{"APPLY", "1"}};

    std::vector<uint32_t> reader_ct_args{geo.Wt, S, St, block};
    tt::tt_metal::TensorAccessorArgs(sum.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.gamma.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.partials.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, split.all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{geo.Wt, S, St, block};
    tt::tt_metal::TensorAccessorArgs(output[0].buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(output[1].buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, split.all_cores, writer_ct_args, defines, kWriterKernelPath);

    const uint32_t inv_c_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(geo.num_inner));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(args.epsilon);
    auto cores = create_compute_and_assign_args(
        program,
        split,
        defines,
        kApplyComputeKernelPath,
        [&](uint32_t work_count) {
            return std::vector<uint32_t>{work_count, geo.Wt, S, St, block, inv_c_bits, eps_bits};
        },
        reader,
        writer,
        {sum.buffer()->address(), tensor_args.gamma.buffer()->address(), tensor_args.partials.buffer()->address()},
        {output[0].buffer()->address(), output[1].buffer()->address()});

    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void AddRMSNormForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderIn0Idx] = tensor_args.sum.buffer()->address();
        r[kReaderIn1Idx] = tensor_args.gamma.buffer()->address();
        r[kReaderIn2Idx] = tensor_args.partials.buffer()->address();
        auto& w = writer_args[core.x][core.y];
        w[kWriterOut0Idx] = output[0].buffer()->address();
        w[kWriterOut1Idx] = output[1].buffer()->address();
    }
}

}  // namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device
