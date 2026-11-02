// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_bw_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "metal/ops/common/swiglu_packed_common.hpp"

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
constexpr auto kUpCbIndex = tt::CBIndex::c_1;     // up branch
constexpr auto kDhCbIndex = tt::CBIndex::c_2;     // upstream grad dL/dh
constexpr auto kDGateCbIndex = tt::CBIndex::c_3;  // output: grad wrt gate branch
constexpr auto kDUpCbIndex = tt::CBIndex::c_4;    // output: grad wrt up branch
constexpr auto kSigmoidCbIndex = tt::CBIndex::c_5;
constexpr auto kScratchCbIndex = tt::CBIndex::c_6;
constexpr auto kSiluGradCbIndex = tt::CBIndex::c_7;
constexpr auto kDhTCbIndex = tt::CBIndex::c_8;    // TRANSPOSED_GRAD / TRANSPOSED_IO: dh^T tiles as read
constexpr auto kACbIndex = tt::CBIndex::c_9;      // lora_a tiles: the block's columns (GRAD) / the row's one tile (IO)
constexpr auto kTailCbIndex = tt::CBIndex::c_10;  // s_dh^T tiles: the row's one tile (GRAD) / the column block's (IO)
constexpr auto kSdhCbIndex = tt::CBIndex::c_11;   // the transposed, scaled operand: s_dh (GRAD) / lora_a^T (IO)
constexpr uint32_t kReaderABufferIdx = 2U;        // reader runtime arg: lora_a address

}  // namespace

namespace ttml::metal::ops::swiglu_packed_bw::device {

struct SwigluPackedBwKernels {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle writer{};
    tt::tt_metal::KernelHandle compute_group_1{};
    tt::tt_metal::KernelHandle compute_group_2{};
};

SwigluPackedBwProgramFactory::cached_program_t SwigluPackedBwProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& packed = tensor_args.packed;
    const auto& dh = tensor_args.dL_dh;

    auto* const device = packed.device();
    tt::tt_metal::Program program{};

    const tt::DataFormat data_format = datatype_to_dataformat_converter(packed.dtype());
    const uint32_t tile_size_bytes = tt::tile_size(data_format);

    // Transposed IO (iteration 39): the blocks run over the I-rows of packed^T [2I, T] (column-major, so a column
    // block's s_dh^T tiles are read once); the same elementwise kernels with the transposed layout's strides.
    const auto geometry = swiglu_packed::block_geometry(packed.padded_shape(), args.transposed_io);
    const uint32_t Wt = geometry.Wt;
    const uint32_t block_size = geometry.block_size;
    const auto grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid_size.y;

    const auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_g1, num_blocks_g2] =
        tt::tt_metal::split_work_to_cores(grid_size, geometry.total_blocks);

    const uint32_t twice_block = 2U * block_size;
    create_circular_buffer(program, all_cores, kGateCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kUpCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDhCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDGateCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDUpCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kSigmoidCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kScratchCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kSiluGradCbIndex, data_format, tile_size_bytes, twice_block);
    std::map<std::string, std::string> defines;
    if (args.transposed_grad) {
        defines["TRANSPOSED_GRAD"] = "1";
        create_circular_buffer(program, all_cores, kDhTCbIndex, data_format, tile_size_bytes, twice_block);
        create_circular_buffer(program, all_cores, kACbIndex, data_format, tile_size_bytes, twice_block);
        create_circular_buffer(program, all_cores, kTailCbIndex, data_format, tile_size_bytes, 2U);
        create_circular_buffer(program, all_cores, kSdhCbIndex, data_format, tile_size_bytes, 1U);
    } else if (args.transposed_io) {
        defines["TRANSPOSED_IO"] = "1";
        create_circular_buffer(program, all_cores, kDhTCbIndex, data_format, tile_size_bytes, twice_block);
        create_circular_buffer(program, all_cores, kACbIndex, data_format, tile_size_bytes, 2U);
        create_circular_buffer(program, all_cores, kTailCbIndex, data_format, tile_size_bytes, twice_block);
        create_circular_buffer(program, all_cores, kSdhCbIndex, data_format, tile_size_bytes, 1U);
    }
    const bool lowrank = args.transposed_grad || args.transposed_io;

    auto* const packed_buf = packed.buffer();
    auto* const dh_buf = dh.buffer();
    auto* const dpacked_buf = output.buffer();

    SwigluPackedBwKernels kernels;

    // dL_dh row stride in tiles; transposed, its tile (c, r) sits at page c * Ht + r (Ht = token tile rows). The
    // column-major block orders run the rows modulo col_major_rows (token rows under TRANSPOSED_GRAD, I-rows under
    // TRANSPOSED_IO); the LoRA tail row block of dh^T starts at tile Wt * Ht under both.
    const uint32_t Ht = geometry.Ht;
    const uint32_t Wt_dh = lowrank ? Ht : tensor_args.dL_dh.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t col_major_rows = args.transposed_io ? Wt : Ht;
    const uint32_t tail_start = Wt * Ht;
    std::vector<uint32_t> reader_ct_args{
        block_size,
        geometry.blocks_per_row,
        geometry.row_stride,
        geometry.up_offset,
        Wt_dh,
        col_major_rows,
        tail_start};
    tt::tt_metal::TensorAccessorArgs(packed_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(dh_buf).append_to(reader_ct_args);
    if (lowrank) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.lora_a->buffer()).append_to(reader_ct_args);
    }
    kernels.reader = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);
    const uint32_t scale_bits = std::bit_cast<uint32_t>(args.lowrank_scale);

    std::vector<uint32_t> writer_ct_args{
        block_size, geometry.blocks_per_row, geometry.row_stride, geometry.up_offset, col_major_rows};
    tt::tt_metal::TensorAccessorArgs(dpacked_buf).append_to(writer_ct_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    std::vector<uint32_t> compute_g1_args = {num_blocks_g1, block_size, scale_bits, col_major_rows};
    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_g1_args, defines, kComputeKernelPath, true);

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_g2_args = {num_blocks_g2, block_size, scale_bits, col_major_rows};
        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_g2_args, defines, kComputeKernelPath, true);
    }

    for_each_core_with_work(
        num_cores, num_cores_y, core_group_1, core_group_2, num_blocks_g1, num_blocks_g2, [&](const CoreWork& work) {
            const auto& [core, core_index, num_blocks, start_block, in_group_1] = work;
            SetRuntimeArgs(
                program,
                kernels.reader,
                core,
                {packed_buf->address(),
                 dh_buf->address(),
                 lowrank ? tensor_args.lora_a->buffer()->address() : 0U,
                 num_blocks,
                 start_block});
            SetRuntimeArgs(program, kernels.writer, core, {dpacked_buf->address(), num_blocks, start_block});
            SetRuntimeArgs(
                program, in_group_1 ? kernels.compute_group_1 : kernels.compute_group_2, core, {start_block});
        });

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

    for_each_core(sv.num_cores, sv.num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
        {
            auto& args = reader_rt[core.x][core.y];
            args[kPackedBufferIdx] = tensor_args.packed.buffer()->address();
            args[kDhBufferIdx] = tensor_args.dL_dh.buffer()->address();
            if (tensor_args.lora_a.has_value()) {
                args[kReaderABufferIdx] = tensor_args.lora_a->buffer()->address();
            }
        }
        writer_rt[core.x][core.y][kDPackedBufferIdx] = output.buffer()->address();
    });
}

}  // namespace ttml::metal::ops::swiglu_packed_bw::device
