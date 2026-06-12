// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "q_rope_fw_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "q_rope_fw_device_operation_types.hpp"

// L1 budget: all circular buffers comfortably fit in the ~1.5 MB per-core L1.
// Per core the op allocates (in tiles, with kCbDoubleBuffer = 2):
//   cb_q   = 2*Th   cb_out = 2*Th        (whole head rows, double-buffered)
//   cb_cos = 2*Tr   cb_sin = 2*Tr        (one sequence-tile of trig, double-buffered)
//   cb_trans = 1                          (constant rotation matrix)
//   c_24 + c_25 + c_26 = 3*Tr             (rope intermediates)
//   => total = 4*Th + 7*Tr + 1 tiles.
//
// DST strategy: Tr <= 8 always (qk_rope_dim <= 256). When Tn <= 8, batch all nope tiles in DST;
// when Tn > 8, compile with Q_ROPE_CHUNKED_NOPE and copy nope in fp32-dependent chunks
// (4 tiles when fp32 dest acc is on, 8 when off).

namespace {

// Max nope tiles per copy/pack acquire block. Matches usable half-sync Dst capacity with
// dst_full_sync_en=false (default): 8 tiles (bf16 dest) or 4 tiles (fp32 dest acc). Batching
// more copy_tile + pack_tile ops in one acquire crosses the half-sync boundary and corrupts output.
constexpr uint32_t kMaxNopeDstTilesBf16 = 8U;
constexpr uint32_t kMaxNopeDstTilesFp32 = 4U;

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/q_rope_fw/device/kernels/dataflow/reader_q_rope_fw.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/q_rope_fw/device/kernels/dataflow/writer_q_rope_fw.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/q_rope_fw/device/kernels/compute/q_rope_fw_kernel.cpp";

constexpr uint32_t kReaderArgQInAddr = 0;
constexpr uint32_t kReaderArgCosAddr = 1;
constexpr uint32_t kReaderArgSinAddr = 2;
constexpr uint32_t kReaderArgTransAddr = 3;

constexpr uint32_t kWriterArgQOutAddr = 0;

constexpr auto kQCbIndex = tt::CBIndex::c_0;
constexpr auto kCosCbIndex = tt::CBIndex::c_1;
constexpr auto kSinCbIndex = tt::CBIndex::c_2;
constexpr auto kTransCbIndex = tt::CBIndex::c_3;
constexpr auto kOutCbIndex = tt::CBIndex::c_16;

constexpr uint32_t kCbDoubleBuffer = 2U;

}  // namespace

namespace ttml::metal::ops::q_rope_fw::device {

using namespace tt::constants;

struct QRopeFwKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute;
};

static void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const QRopeFwKernels& kernels,
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
    uint32_t Tr,
    uint32_t n_heads) {
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_FATAL(false, "QRopeFw: core not in any work group");
        }

        // Q [B, H, S, D] head-major pages: tile_id(b, h, sb, w) = b*H*tiles_per_head + h*tiles_per_head + sb*Th + w.
        const uint32_t b_start = num_blocks_written / Ts;
        const uint32_t sb_start = num_blocks_written % Ts;
        const uint32_t tiles_per_head = Ts * Th;
        const uint32_t q_block_base_start = b_start * n_heads * tiles_per_head + sb_start * Th;

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
             q_block_base_start});

        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {q_out_buffer->address(), num_blocks_per_core, sb_start, q_block_base_start});

        SetRuntimeArgs(program, kernels.compute, core, {num_blocks_per_core});

        num_blocks_written += num_blocks_per_core;
    }
}

QRopeFwProgramFactory::cached_program_t QRopeFwProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q_in = tensor_args.q_in;
    auto& q_out = output;

    auto* device = q_in.device();
    tt::tt_metal::Program program{};

    const auto q_shape = q_in.padded_shape();
    const uint32_t B = q_shape[0];
    const uint32_t S = q_shape[2];
    const uint32_t H = q_shape[1];

    const uint32_t Tn = args.qk_nope_dim / TILE_WIDTH;
    const uint32_t Tr = args.qk_rope_dim / TILE_WIDTH;
    const uint32_t Th = Tn + Tr;
    const uint32_t Ts = S / TILE_HEIGHT;
    const uint32_t tiles_per_head = Ts * Th;

    const uint32_t num_blocks = B * Ts;

    const bool fp32_dest_acc_en = args.fp32_dest_acc_en;
    const uint32_t max_nope_dst_tiles = fp32_dest_acc_en ? kMaxNopeDstTilesFp32 : kMaxNopeDstTilesBf16;
    const bool use_chunked_nope = (Tn > max_nope_dst_tiles);
    TT_FATAL(
        Tr <= 8U,
        "QRopeFw: qk_rope_dim ({}) / TILE_W = {} exceeds max rope tiles (8, head_dim 256).",
        args.qk_rope_dim,
        Tr);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(q_in.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    // Q in/out CBs hold whole head rows (Th tiles), double-buffered. The reader pushes Th and the
    // compute/writer consume Th, so sizing by Th (not an arbitrary chunk) keeps the counts matched.
    const uint32_t q_cb_num_tiles = kCbDoubleBuffer * Th;
    const uint32_t trig_cb_num_tiles = kCbDoubleBuffer * Tr;

    [[maybe_unused]] auto cb_q =
        create_circular_buffer(program, all_cores, kQCbIndex, data_format, single_tile_size, q_cb_num_tiles);
    [[maybe_unused]] auto cb_cos =
        create_circular_buffer(program, all_cores, kCosCbIndex, data_format, single_tile_size, trig_cb_num_tiles);
    [[maybe_unused]] auto cb_sin =
        create_circular_buffer(program, all_cores, kSinCbIndex, data_format, single_tile_size, trig_cb_num_tiles);
    create_circular_buffer(program, all_cores, kTransCbIndex, data_format, single_tile_size, 1U);

    create_circular_buffer(program, all_cores, tt::CBIndex::c_24, data_format, single_tile_size, Tr);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_25, data_format, single_tile_size, Tr);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_26, data_format, single_tile_size, Tr);
    create_circular_buffer(program, all_cores, kOutCbIndex, data_format, single_tile_size, q_cb_num_tiles);

    auto* q_in_buffer = q_in.buffer();
    auto* cos_buffer = tensor_args.cos_cache.buffer();
    auto* sin_buffer = tensor_args.sin_cache.buffer();
    auto* trans_buffer = tensor_args.trans_mat.buffer();
    auto* q_out_buffer = q_out.buffer();

    std::vector<uint32_t> reader_compile_time_args = {Th, Tr, H, Ts, tiles_per_head};
    tt::tt_metal::TensorAccessorArgs(q_in_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(cos_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(sin_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(trans_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {Th, H, Ts, tiles_per_head};
    tt::tt_metal::TensorAccessorArgs(q_out_buffer).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)kQCbIndex,
        (std::uint32_t)kCosCbIndex,
        (std::uint32_t)kSinCbIndex,
        (std::uint32_t)kTransCbIndex,
        (std::uint32_t)tt::CBIndex::c_24,
        (std::uint32_t)tt::CBIndex::c_25,
        (std::uint32_t)tt::CBIndex::c_26,
        (std::uint32_t)kOutCbIndex,
        Tn,
        Tr,
        Th,
        H,
    };

    std::map<std::string, std::string> defines;
    if (use_chunked_nope) {
        defines["Q_ROPE_CHUNKED_NOPE"] = "1";
        defines["Q_ROPE_NOPE_CHUNK_TILES"] = std::to_string(max_nope_dst_tiles);
    }

    QRopeFwKernels kernels;
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);
    kernels.compute = create_compute_kernel(
        program, all_cores, compute_compile_time_args, defines, kComputeKernelPath, fp32_dest_acc_en);

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
        Tr,
        H);

    return cached_program_t{
        std::move(program), {kernels.reader, kernels.writer, kernels.compute, num_cores, num_cores_y}};
}

void QRopeFwProgramFactory::override_runtime_arguments(
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

}  // namespace ttml::metal::ops::q_rope_fw::device
