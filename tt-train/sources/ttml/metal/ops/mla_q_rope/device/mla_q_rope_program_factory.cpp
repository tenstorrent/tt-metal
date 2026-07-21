// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_q_rope_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "mla_q_rope_device_operation_types.hpp"

// L1 circular buffer sizes (tiles). Tr <= 8, max_dst <= 8, nope_chunk_tiles <= 8.
//   cb_nope=2*nope_batch_tiles; worst 16
//   cb_q_pe=dst_batch?max_chunk:2*Tr; worst 16
//   cb_rope_out=2*Tr; worst 16
//   cb_cos=2*Tr; worst 16
//   cb_sin=2*Tr; worst 16
//   cb_trans=1; worst 1
//   rotated_in=dst_batch?max_chunk:Tr; worst 8
//   cos_interm=Tr; worst 8
//   sin_interm=Tr; worst 8
//   total worst 105 tiles at Tr=8 (~210 KiB at 2 KiB/tile)

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_q_rope/device/kernels/dataflow/reader_mla_q_rope.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_q_rope/device/kernels/dataflow/writer_mla_q_rope.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_q_rope/device/kernels/compute/mla_q_rope_kernel.cpp";

constexpr uint32_t kReaderArgQInAddr = 0;
constexpr uint32_t kReaderArgCosAddr = 1;
constexpr uint32_t kReaderArgSinAddr = 2;
constexpr uint32_t kReaderArgTransAddr = 3;

constexpr uint32_t kWriterArgQOutAddr = 0;

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

struct MlaQRopeKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute;
};

static void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const MlaQRopeKernels& kernels,
    const tt::tt_metal::Buffer* q_in_buffer,
    const tt::tt_metal::Buffer* cos_buffer,
    const tt::tt_metal::Buffer* sin_buffer,
    const tt::tt_metal::Buffer* trans_buffer,
    const tt::tt_metal::Buffer* q_out_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_blocks_per_core_group_1,
    uint32_t num_blocks_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t Ts,
    uint32_t Th,
    uint32_t n_heads,
    bool packed_input) {
    const uint32_t tiles_per_head = Ts * Th;
    const uint32_t packed_block_stride = n_heads * Th;

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_FATAL(false, "MlaQRope: core not in any work group");
        }

        const uint32_t b_start = num_blocks_written / Ts;
        const uint32_t sb_start = num_blocks_written % Ts;
        const uint32_t packed_base = num_blocks_written * packed_block_stride;
        const uint32_t head_major_base = b_start * n_heads * tiles_per_head + sb_start * Th;

        const uint32_t q_in_tile_base = packed_input ? packed_base : head_major_base;
        const uint32_t q_out_tile_base = packed_input ? head_major_base : packed_base;

        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {q_in_buffer->address(),
             cos_buffer->address(),
             sin_buffer->address(),
             trans_buffer->address(),
             num_blocks_per_core,
             sb_start,
             q_in_tile_base});

        SetRuntimeArgs(
            program, kernels.writer, core, {q_out_buffer->address(), num_blocks_per_core, sb_start, q_out_tile_base});

        SetRuntimeArgs(program, kernels.compute, core, {num_blocks_per_core});

        num_blocks_written += num_blocks_per_core;
    }
}

MlaQRopeProgramFactory::cached_program_t MlaQRopeProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q_in = tensor_args.q_in;
    auto& q_out = output;

    auto* device = q_in.device();
    tt::tt_metal::Program program{};

    const auto q_shape = q_in.padded_shape();
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
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

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

    [[maybe_unused]] auto cb_q_pe =
        create_circular_buffer(program, all_cores, kQPeCbIndex, data_format, single_tile_size, q_pe_cb_num_tiles);
    [[maybe_unused]] auto cb_cos =
        create_circular_buffer(program, all_cores, kCosCbIndex, data_format, single_tile_size, trig_cb_num_tiles);
    [[maybe_unused]] auto cb_sin =
        create_circular_buffer(program, all_cores, kSinCbIndex, data_format, single_tile_size, trig_cb_num_tiles);
    [[maybe_unused]] auto cb_trans =
        create_circular_buffer(program, all_cores, kTransCbIndex, data_format, single_tile_size, 1U);
    [[maybe_unused]] auto cb_nope =
        create_circular_buffer(program, all_cores, kNopeCbIndex, data_format, single_tile_size, nope_cb_num_tiles);
    [[maybe_unused]] auto cb_rotated_in = create_circular_buffer(
        program, all_cores, kRotatedInIntermCbIndex, data_format, single_tile_size, rotated_interm_cb_num_tiles);
    [[maybe_unused]] auto cb_cos_interm =
        create_circular_buffer(program, all_cores, kCosIntermCbIndex, data_format, single_tile_size, Tr);
    [[maybe_unused]] auto cb_sin_interm =
        create_circular_buffer(program, all_cores, kSinIntermCbIndex, data_format, single_tile_size, Tr);
    [[maybe_unused]] auto cb_rope_out =
        create_circular_buffer(program, all_cores, kRopeOutCbIndex, data_format, single_tile_size, rope_cb_num_tiles);

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

    MlaQRopeKernels kernels;
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, {}, kReaderKernelPath);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, {}, kWriterKernelPath);
    kernels.compute = create_compute_kernel(
        program, all_cores, compute_compile_time_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    assign_per_core_runtime_args(
        program,
        kernels,
        q_in_buffer,
        cos_buffer,
        sin_buffer,
        trans_buffer,
        q_out_buffer,
        num_cores,
        num_cores_y,
        num_blocks_per_core_group_1,
        num_blocks_per_core_group_2,
        core_group_1,
        core_group_2,
        Ts,
        Th,
        H,
        args.packed_input);

    return cached_program_t{
        std::move(program), {kernels.reader, kernels.writer, kernels.compute, num_cores, num_cores_y}};
}

void MlaQRopeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*args*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* q_in_buffer = tensor_args.q_in.buffer();
    auto* cos_buffer = tensor_args.cos_cache.buffer();
    auto* sin_buffer = tensor_args.sin_cache.buffer();
    auto* trans_buffer = tensor_args.trans_mat.buffer();
    auto* q_out_buffer = output.buffer();

    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / shared.num_cores_y, i % shared.num_cores_y};
        {
            auto& ra = reader_runtime_args[core.x][core.y];
            ra[kReaderArgQInAddr] = q_in_buffer->address();
            ra[kReaderArgCosAddr] = cos_buffer->address();
            ra[kReaderArgSinAddr] = sin_buffer->address();
            ra[kReaderArgTransAddr] = trans_buffer->address();
        }
        {
            auto& ra = writer_runtime_args[core.x][core.y];
            ra[kWriterArgQOutAddr] = q_out_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::mla_q_rope::device
