// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_q_rope_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "mla_q_rope_device_operation.hpp"
#include "mla_q_rope_device_operation_types.hpp"

// L1 circular buffer sizes (tiles). Tr <= 4 (asserted below), max_dst_tiles = 4
// (fp32_dest_acc_en=true), nope_chunk_tiles = 4.
//   cb_nope=2*nope_batch_tiles; scales with heads batched per DST pass and round_up(Tn, 4)
//   cb_q_pe=dst_batch?max_chunk:2*Tr; worst 8
//   cb_rope_out=2*Tr; worst 8
//   cb_cos=2*Tr; worst 8
//   cb_sin=2*Tr; worst 8
//   cb_trans=1; worst 1
//   rotated_in=dst_batch?max_chunk:Tr; worst 4
//   cos_interm=Tr; worst 4
//   sin_interm=Tr; worst 4

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_q_rope/device/kernels/dataflow/reader_mla_q_rope.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_q_rope/device/kernels/dataflow/writer_mla_q_rope.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_q_rope/device/kernels/compute/mla_q_rope_kernel.cpp";

// Kernel push order: reader, writer, then the single compute kernel on all cores
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;
constexpr uint32_t kComputeKernelIdx = 2U;

// reader runtime args
constexpr uint32_t kQInAddrIdx = 0U;
constexpr uint32_t kCosAddrIdx = 1U;
constexpr uint32_t kSinAddrIdx = 2U;
constexpr uint32_t kTransAddrIdx = 3U;
// writer runtime args
constexpr uint32_t kQOutAddrIdx = 0U;

constexpr auto kQPeCbIndex = tt::CBIndex::c_0;
constexpr auto kCosCbIndex = tt::CBIndex::c_1;
constexpr auto kSinCbIndex = tt::CBIndex::c_2;
constexpr auto kTransCbIndex = tt::CBIndex::c_3;
constexpr auto kNopeCbIndex = tt::CBIndex::c_4;
constexpr auto kRotatedInIntermCbIndex = tt::CBIndex::c_24;
constexpr auto kCosIntermCbIndex = tt::CBIndex::c_25;
constexpr auto kSinIntermCbIndex = tt::CBIndex::c_26;
constexpr auto kRopeOutCbIndex = tt::CBIndex::c_16;

constexpr uint32_t kCbDoubleBuffer = 2U;
constexpr uint32_t kNopeChunkTiles = 4U;

}  // namespace

namespace ttml::metal::ops::mla_q_rope::device {

using namespace tt::constants;

namespace {

// q_in's (batch, sequence tile row) blocks over the device's compute grid. create_descriptor() pins it against
// split_work_to_cores, and override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& q_in) {
    const auto& shape = q_in.logical_shape();
    return compute_work_split(q_in.device()->compute_with_storage_grid_size(), shape[0] * (shape[2] / TILE_HEIGHT));
}

}  // namespace

tt::tt_metal::ProgramDescriptor MlaQRopeProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q_in = tensor_args.q_in;
    auto& q_out = output;

    auto* device = q_in.device();
    tt::tt_metal::ProgramDescriptor desc;

    const auto q_shape = q_in.logical_shape();
    const uint32_t B = q_shape[0];
    const uint32_t S = q_shape[2];
    const uint32_t qk_head = args.qk_nope_dim + args.qk_rope_dim;
    const uint32_t H = args.packed_input ? (q_shape[3] / qk_head) : q_shape[1];

    const uint32_t Tn = args.qk_nope_dim / TILE_WIDTH;
    const uint32_t Tr = args.qk_rope_dim / TILE_WIDTH;
    const uint32_t Th = Tn + Tr;
    const uint32_t Ts = S / TILE_HEIGHT;
    const uint32_t tiles_per_head = Ts * Th;

    const uint32_t num_blocks = B * Ts;
    const uint32_t packed_input_u32 = args.packed_input ? 1U : 0U;

    TT_FATAL(
        Tr <= 4U,
        "MlaQRope: qk_rope_dim ({}) / TILE_W = {} exceeds max rope tiles (4, head_dim 128).",
        args.qk_rope_dim,
        Tr);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    const WorkSplit split = make_work_split(q_in);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2);

    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(q_in.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    // fp32_dest_acc_en=true -> DST holds 4 tiles.
    constexpr uint32_t max_dst_tiles = 4U;
    // Heads per dst-batch matmul: largest count with dst_batch_heads * Tr <= max_dst_tiles.
    const uint32_t dst_batch_heads = std::max(1U, max_dst_tiles / Tr);
    // q_pe / rotated_in CB depth for one dst batch (fills DST: dst_batch_heads * Tr).
    const uint32_t max_chunk_tiles = dst_batch_heads * Tr;
    // True when compute batches multiple heads; drives single-buffer q_pe and larger cb_nope.
    const bool dst_batch = dst_batch_heads > 1U;

    // Stream width for read/write_full_row_tiles on q_nope (4 tiles with fp32 dest acc).
    constexpr uint32_t nope_chunk_tiles = kNopeChunkTiles;
    // CB slots one head consumes (tail chunk pads up to a full nope_chunk_tiles reserve/push).
    const uint32_t nope_slots_per_head = tt::round_up(Tn, nope_chunk_tiles);
    // Heads whose nope may be in flight while writer waits on rope (capped by actual H).
    const uint32_t heads_per_nope_batch = dst_batch ? std::min(dst_batch_heads, H) : 1U;
    // cb_nope half-buffer: one nope_chunk per-head; dst_batch holds in-flight streamed rows for the batch.
    const uint32_t nope_batch_tiles = dst_batch ? heads_per_nope_batch * nope_slots_per_head : nope_chunk_tiles;

    const uint32_t nope_cb_num_tiles = kCbDoubleBuffer * nope_batch_tiles;
    const uint32_t q_pe_cb_num_tiles = dst_batch ? max_chunk_tiles : (kCbDoubleBuffer * Tr);
    const uint32_t rope_cb_num_tiles = kCbDoubleBuffer * Tr;
    const uint32_t trig_cb_num_tiles = kCbDoubleBuffer * Tr;
    const uint32_t rotated_interm_cb_num_tiles = dst_batch ? max_chunk_tiles : Tr;

    desc.cbs.push_back(make_cb_descriptor(all_cores, kQPeCbIndex, data_format, single_tile_size, q_pe_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kCosCbIndex, data_format, single_tile_size, trig_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kSinCbIndex, data_format, single_tile_size, trig_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kTransCbIndex, data_format, single_tile_size, 1U));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kNopeCbIndex, data_format, single_tile_size, nope_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kRotatedInIntermCbIndex, data_format, single_tile_size, rotated_interm_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kCosIntermCbIndex, data_format, single_tile_size, Tr));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kSinIntermCbIndex, data_format, single_tile_size, Tr));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kRopeOutCbIndex, data_format, single_tile_size, rope_cb_num_tiles));

    auto* q_in_buffer = q_in.buffer();
    auto* cos_buffer = tensor_args.cos_cache.buffer();
    auto* sin_buffer = tensor_args.sin_cache.buffer();
    auto* trans_buffer = tensor_args.trans_mat.buffer();
    auto* q_out_buffer = q_out.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        Tn, Tr, H, Ts, tiles_per_head, nope_chunk_tiles, packed_input_u32};
    tt::tt_metal::TensorAccessorArgs(q_in_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(cos_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(sin_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(trans_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        Tn, Tr, H, Ts, tiles_per_head, nope_chunk_tiles, packed_input_u32};
    tt::tt_metal::TensorAccessorArgs(q_out_buffer).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)kQPeCbIndex,
        (std::uint32_t)kCosCbIndex,
        (std::uint32_t)kSinCbIndex,
        (std::uint32_t)kTransCbIndex,
        (std::uint32_t)kRotatedInIntermCbIndex,
        (std::uint32_t)kCosIntermCbIndex,
        (std::uint32_t)kSinIntermCbIndex,
        (std::uint32_t)kRopeOutCbIndex,
        Tr,
        H,
        dst_batch_heads,
    };

    desc.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, {}, kReaderKernelPath));
    desc.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, {}, kWriterKernelPath));
    desc.kernels.push_back(  // kComputeKernelIdx
        make_compute_kernel_descriptor(
            all_cores, compute_compile_time_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true));

    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // The block counts and tile bases follow from the hashed q_in shape and attributes.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    auto& compute = desc.kernels[kComputeKernelIdx];
    const uint32_t packed_block_stride = H * Th;
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_blocks, start_block, in_group_1] = work;
        const uint32_t b_start = start_block / Ts;
        const uint32_t sb_start = start_block % Ts;
        // Packed: tile id of (b, sb, h=0) = block_index * H * Th
        const uint32_t packed_base = start_block * packed_block_stride;
        // Head-major: tile id of (b, h=0, sb) = b * H * tiles_per_head + sb * Th
        const uint32_t head_major_base = b_start * H * tiles_per_head + sb_start * Th;
        const uint32_t q_in_tile_base = args.packed_input ? packed_base : head_major_base;
        const uint32_t q_out_tile_base = args.packed_input ? head_major_base : packed_base;
        reader.emplace_runtime_args(
            core, {q_in_buffer, cos_buffer, sin_buffer, trans_buffer, num_blocks, sb_start, q_in_tile_base});
        writer.emplace_runtime_args(core, {q_out_buffer, num_blocks, sb_start, q_out_tile_base});
        compute.emplace_runtime_args(core, {num_blocks});
    });

    return desc;
}

void MlaQRopeProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.q_in);

    const uint32_t q_in_addr = tensor_args.q_in.buffer()->address();
    const uint32_t cos_addr = tensor_args.cos_cache.buffer()->address();
    const uint32_t sin_addr = tensor_args.sin_cache.buffer()->address();
    const uint32_t trans_addr = tensor_args.trans_mat.buffer()->address();
    const uint32_t q_out_addr = output.buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kQInAddrIdx] = q_in_addr;
        reader_args[kCosAddrIdx] = cos_addr;
        reader_args[kSinAddrIdx] = sin_addr;
        reader_args[kTransAddrIdx] = trans_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kQOutAddrIdx] = q_out_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(MlaQRopeDeviceOperation, MlaQRopeProgramFactory);

}  // namespace ttml::metal::ops::mla_q_rope::device
