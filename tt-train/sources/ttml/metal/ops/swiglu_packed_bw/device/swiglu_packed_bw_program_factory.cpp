// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_bw_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_packed_bw/device/kernels/dataflow/reader_swiglu_packed_bw_interleaved.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_packed_bw/device/kernels/dataflow/writer_swiglu_packed_bw_interleaved.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_packed_bw/device/kernels/compute/swiglu_packed_bw_kernel.cpp";

constexpr uint32_t kPackedBufferIdx = 0U;
constexpr uint32_t kDhBufferIdx = 1U;
constexpr uint32_t kDPackedBufferIdx = 0U;

constexpr auto kGateCbIndex = tt::CBIndex::c_0;   // gate branch (silu'd)
constexpr auto kUpCbIndex = tt::CBIndex::c_1;     // up branch (plain)
constexpr auto kDhCbIndex = tt::CBIndex::c_2;     // upstream grad dL/dh
constexpr auto kDGateCbIndex = tt::CBIndex::c_3;  // output: grad wrt gate branch
constexpr auto kDUpCbIndex = tt::CBIndex::c_4;    // output: grad wrt up branch
constexpr auto kSigmoidCbIndex = tt::CBIndex::c_5;
constexpr auto kScratchCbIndex = tt::CBIndex::c_6;
constexpr auto kSiluGradCbIndex = tt::CBIndex::c_7;

}  // namespace

namespace ttml::metal::ops::swiglu_packed_bw::device {

struct SwigluPackedBwKernels {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle writer{};
    tt::tt_metal::KernelHandle compute_group_1{};
    tt::tt_metal::KernelHandle compute_group_2{};
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const SwigluPackedBwKernels& kernels,
    const tt::tt_metal::Buffer* packed_buffer,
    const tt::tt_metal::Buffer* dh_buffer,
    const tt::tt_metal::Buffer* dpacked_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_blocks_per_core_group_1,
    uint32_t num_blocks_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {packed_buffer->address(), dh_buffer->address(), num_blocks_per_core, num_blocks_written});
        SetRuntimeArgs(
            program, kernels.writer, core, {dpacked_buffer->address(), num_blocks_per_core, num_blocks_written});

        num_blocks_written += num_blocks_per_core;
    }
}

SwigluPackedBwProgramFactory::cached_program_t SwigluPackedBwProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& packed = tensor_args.packed;
    const auto& dh = tensor_args.dL_dh;

    auto* const device = packed.device();
    tt::tt_metal::Program program{};

    const tt::DataFormat data_format = datatype_to_dataformat_converter(packed.dtype());
    const uint32_t tile_size_bytes = tt::tile_size(data_format);

    const auto padded_shape = packed.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "Input tensor must be 4D");
    // Wt = per-branch width in tiles; packed/dpacked are 2*Wt wide, dh is Wt.
    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH / 2U;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    const uint32_t total_rows = NC * Ht;

    const auto grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid_size.y;
    // Largest block <=4 dividing Wt (=4 for real MLP widths, where I/32 is a multiple of 8).
    const uint32_t block_size = get_block_size(Wt, 4U);

    const uint32_t blocks_per_row = Wt / block_size;
    const uint32_t total_blocks = total_rows * blocks_per_row;

    const auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_g1, num_blocks_g2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_blocks);

    const uint32_t twice_block = 2U * block_size;
    create_circular_buffer(program, all_cores, kGateCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kUpCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDhCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDGateCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDUpCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kSigmoidCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kScratchCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kSiluGradCbIndex, data_format, tile_size_bytes, twice_block);

    auto* const packed_buf = packed.buffer();
    auto* const dh_buf = dh.buffer();
    auto* const dpacked_buf = output.buffer();

    SwigluPackedBwKernels kernels;

    std::vector<uint32_t> reader_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(packed_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(dh_buf).append_to(reader_ct_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_ct_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dpacked_buf).append_to(writer_ct_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_ct_args, {}, kWriterKernelPath);

    std::vector<uint32_t> compute_g1_args = {num_blocks_g1, block_size};
    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_g1_args, {}, kComputeKernelPath, true);

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_g2_args = {num_blocks_g2, block_size};
        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_g2_args, {}, kComputeKernelPath, true);
    }

    assign_per_core_runtime_args(
        program,
        kernels,
        packed_buf,
        dh_buf,
        dpacked_buf,
        num_cores,
        num_cores_y,
        num_blocks_g1,
        num_blocks_g2,
        core_group_1,
        core_group_2);

    return cached_program_t{
        std::move(program),
        {kernels.reader,
         kernels.writer,
         kernels.compute_group_1,
         kernels.compute_group_2,
         core_group_1,
         core_group_2,
         num_cores,
         num_cores_y}};
}

void SwigluPackedBwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    auto& reader_rt = GetRuntimeArgs(program, sv.reader_kernel_id);
    auto& writer_rt = GetRuntimeArgs(program, sv.writer_kernel_id);

    for (uint32_t i = 0; i < sv.num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / sv.num_cores_y, i % sv.num_cores_y};
        {
            auto& args = reader_rt[core.x][core.y];
            args[kPackedBufferIdx] = tensor_args.packed.buffer()->address();
            args[kDhBufferIdx] = tensor_args.dL_dh.buffer()->address();
        }
        writer_rt[core.x][core.y][kDPackedBufferIdx] = output.buffer()->address();
    }
}

}  // namespace ttml::metal::ops::swiglu_packed_bw::device
