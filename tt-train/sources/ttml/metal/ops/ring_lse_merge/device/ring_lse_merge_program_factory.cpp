// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_lse_merge_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "ring_lse_merge_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/ring_lse_merge/device/kernels/dataflow/reader_ring_lse_merge.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/ring_lse_merge/device/kernels/dataflow/writer_ring_lse_merge.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/ring_lse_merge/device/kernels/compute/ring_lse_merge_compute.cpp";

constexpr auto kCbRunningLse = tt::CBIndex::c_0;
constexpr auto kCbStepLse = tt::CBIndex::c_1;
constexpr auto kCbAccum = tt::CBIndex::c_2;
constexpr auto kCbStepOut = tt::CBIndex::c_3;
constexpr auto kCbCol0Mask = tt::CBIndex::c_4;
constexpr auto kCbNewLse = tt::CBIndex::c_5;
constexpr auto kCbNewLseScratch = tt::CBIndex::c_6;
constexpr auto kCbWOld = tt::CBIndex::c_7;
constexpr auto kCbWNew = tt::CBIndex::c_8;
constexpr auto kCbOut = tt::CBIndex::c_9;

// Runtime arg slots patched on a program-cache hit.
constexpr uint32_t kReaderAccumAddrIdx = 0;
constexpr uint32_t kReaderRunningLseAddrIdx = 1;
constexpr uint32_t kReaderStepOutputAddrIdx = 2;
constexpr uint32_t kReaderStepIntermediatesAddrIdx = 3;
constexpr uint32_t kWriterAccumAddrIdx = 0;
constexpr uint32_t kWriterRunningLseAddrIdx = 1;

// Accumulator tiles moved per CB push; the CBs hold two blocks so reads overlap compute.
constexpr uint32_t kMaxBlockSize = 4U;

}  // namespace

namespace ttml::metal::ops::ring_lse_merge::device {

RingLseMergeProgramFactory::cached_program_t RingLseMergeProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    const auto& accum = tensor_args.output_accum;
    const auto& running_lse = tensor_args.running_lse;
    const auto& step_output = tensor_args.step_output;
    const auto& step_intermediates = tensor_args.step_intermediates;
    auto* device = accum.device();

    tt::tt_metal::Program program{};

    // -------------------------------------------------------------------------
    // 1) Work split: one unit = one tile row (b, h, 32 sequence positions)
    // -------------------------------------------------------------------------
    const auto& padded_shape = accum.padded_shape();
    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t total_rows = padded_shape[0] * padded_shape[1] * Ht;
    const uint32_t block_size = get_block_size(Wt, kMaxBlockSize);

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2] =
        tt::tt_metal::split_work_to_cores(grid, total_rows);

    // -------------------------------------------------------------------------
    // 2) Circular buffers
    // -------------------------------------------------------------------------
    const tt::DataFormat fp32 = tt::DataFormat::Float32;
    const tt::DataFormat bf16 = tt::DataFormat::Float16_b;
    const uint32_t fp32_tile = tt::tile_size(fp32);
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t two_blocks = 2U * block_size;

    create_circular_buffer(program, all_cores, kCbRunningLse, fp32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, kCbStepLse, fp32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, kCbAccum, fp32, fp32_tile, two_blocks);
    create_circular_buffer(program, all_cores, kCbStepOut, bf16, bf16_tile, two_blocks);
    create_circular_buffer(program, all_cores, kCbCol0Mask, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, kCbNewLse, fp32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, kCbNewLseScratch, fp32, fp32_tile, 1U);
    create_circular_buffer(program, all_cores, kCbWOld, fp32, fp32_tile, 1U);
    create_circular_buffer(program, all_cores, kCbWNew, fp32, fp32_tile, 1U);
    create_circular_buffer(program, all_cores, kCbOut, fp32, fp32_tile, two_blocks);

    // -------------------------------------------------------------------------
    // 3) Kernels
    // -------------------------------------------------------------------------
    auto* accum_buffer = accum.buffer();
    auto* running_lse_buffer = running_lse.buffer();
    auto* step_output_buffer = step_output.buffer();
    auto* step_intermediates_buffer = step_intermediates.buffer();

    std::vector<uint32_t> reader_ct_args{Wt, block_size};
    tt::tt_metal::TensorAccessorArgs(accum_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(running_lse_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(step_output_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(step_intermediates_buffer).append_to(reader_ct_args);
    const auto reader = create_reader_kernel(program, all_cores, reader_ct_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{Wt, block_size};
    tt::tt_metal::TensorAccessorArgs(accum_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(running_lse_buffer).append_to(writer_ct_args);
    const auto writer = create_writer_kernel(program, all_cores, writer_ct_args, {}, kWriterKernelPath);

    // FP32 CBs that compute copies or broadcasts into DST bypass the 19-bit SrcA/SrcB registers.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (const auto cb : {kCbRunningLse, kCbStepLse, kCbAccum, kCbNewLseScratch, kCbWOld, kCbWNew}) {
        unpack_to_dest[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    // One compute kernel for all cores; the row count is a runtime arg so the two split groups share it.
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest,
            .math_approx_mode = false,
            .compile_args = {Wt, block_size}});

    // -------------------------------------------------------------------------
    // 4) Runtime args
    // -------------------------------------------------------------------------
    for_each_core_with_work(
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
        [&](const CoreWork& work) {
            SetRuntimeArgs(
                program,
                reader,
                work.core,
                {accum_buffer->address(),
                 running_lse_buffer->address(),
                 step_output_buffer->address(),
                 step_intermediates_buffer->address(),
                 work.num_units,
                 work.start});
            SetRuntimeArgs(
                program,
                writer,
                work.core,
                {accum_buffer->address(), running_lse_buffer->address(), work.num_units, work.start});
            SetRuntimeArgs(program, compute, work.core, {work.num_units});
        });

    return cached_program_t{std::move(program), {reader, writer, num_cores, num_cores_y}};
}

void RingLseMergeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    const uint32_t accum_addr = tensor_args.output_accum.buffer()->address();
    const uint32_t running_lse_addr = tensor_args.running_lse.buffer()->address();
    const uint32_t step_output_addr = tensor_args.step_output.buffer()->address();
    const uint32_t step_intermediates_addr = tensor_args.step_intermediates.buffer()->address();

    auto& reader_rt = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_rt = GetRuntimeArgs(program, shared.writer_kernel_id);
    for_each_core(shared.num_cores, shared.num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
        auto& reader_args = reader_rt[core.x][core.y];
        reader_args[kReaderAccumAddrIdx] = accum_addr;
        reader_args[kReaderRunningLseAddrIdx] = running_lse_addr;
        reader_args[kReaderStepOutputAddrIdx] = step_output_addr;
        reader_args[kReaderStepIntermediatesAddrIdx] = step_intermediates_addr;
        auto& writer_args = writer_rt[core.x][core.y];
        writer_args[kWriterAccumAddrIdx] = accum_addr;
        writer_args[kWriterRunningLseAddrIdx] = running_lse_addr;
    });
}

}  // namespace ttml::metal::ops::ring_lse_merge::device
