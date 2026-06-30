// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_qkv_assemble_fw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "mla_qkv_assemble_fw_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_qkv_assemble_fw/device/kernels/dataflow/"
    "reader_mla_qkv_assemble_fw.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_qkv_assemble_fw/device/kernels/dataflow/"
    "writer_mla_qkv_assemble_fw.cpp";

// reader runtime arg indices (for override_runtime_arguments)
constexpr uint32_t kReaderArgQPreAddr = 0;
constexpr uint32_t kReaderArgKvUpAddr = 1;
constexpr uint32_t kReaderArgKpeAddr = 2;

// writer runtime arg indices
constexpr uint32_t kWriterArgQAddr = 0;
constexpr uint32_t kWriterArgKAddr = 1;
constexpr uint32_t kWriterArgVAddr = 2;

// CB indices. k_nope and v are demuxed into separate CBs so each output stream can be chunked and
// double-buffered independently; the writer reinserts the broadcast k_pe between them.
constexpr auto kQCbIndex = tt::CBIndex::c_0;
constexpr auto kKnopeCbIndex = tt::CBIndex::c_1;
constexpr auto kVCbIndex = tt::CBIndex::c_2;
constexpr auto kKpeCbIndex = tt::CBIndex::c_3;

// Each Q / k_nope / v CB is double-buffered (2 * block_size) so the reader can fetch the next chunk on
// its NoC while the writer drains the current one on the other NoC.
constexpr uint32_t kCbDoubleBuffer = 2U;

// block_size is the per-stream chunk: tiles moved per NoC transaction. A sweep over
// {1,2,4,6,8,12,16,24,32,48,64} on N300 shows a clear penalty below 4, a small step up to ~6-8, then a
// flat plateau. Larger values are inert — the per-head stream length (Th, Tn, Tv, each <=
// head_dim / TILE_W) caps the useful chunk, and head-major outputs prevent coalescing writes across
// heads — so 8 is the most robust/fastest plateau point. Unlike fused-math kernels (silu, swiglu,
// polynorm) where block_size is pinned to 4 by the DEST register limit, this is a pure data-movement op
// with NO compute kernel: the choice is not tied to any register count, only NoC/L1 throughput. L1
// cost is bounded by block_size (not head_dim) and trivial (~100 KB of the ~1.5 MB budget).
constexpr uint32_t block_size = 8U;

}  // namespace

namespace ttml::metal::ops::mla_qkv_assemble_fw::device {

using namespace tt::constants;

struct MLAQKVAssembleFwKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

static void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const MLAQKVAssembleFwKernels& kernels,
    const tt::tt_metal::Buffer* q_pre_buffer,
    const tt::tt_metal::Buffer* kv_up_buffer,
    const tt::tt_metal::Buffer* k_pe_buffer,
    const tt::tt_metal::Buffer* q_buffer,
    const tt::tt_metal::Buffer* k_buffer,
    const tt::tt_metal::Buffer* v_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_blocks_per_core_group_1,
    uint32_t num_blocks_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t Ts,
    uint32_t q_tiles_per_block,
    uint32_t kv_up_tiles_per_block,
    uint32_t kpe_tiles_per_block,
    uint32_t Th,
    uint32_t Tv,
    uint32_t kq_HtWt,
    uint32_t v_HtWt,
    uint32_t n_heads) {
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_FATAL(false, "MLAQKVAssembleFw: core not in any work group");
        }

        // Block index = b * Ts + sb. Inputs q_pre, kv_up, k_pe are flat across blocks.
        const uint32_t q_pre_tile_id_start = num_blocks_written * q_tiles_per_block;
        const uint32_t kv_up_tile_id_start = num_blocks_written * kv_up_tiles_per_block;
        const uint32_t k_pe_tile_id_start = num_blocks_written * kpe_tiles_per_block;

        // Outputs are head-major: tile_id(b, h, sb, w) = b*H*HtWt + h*HtWt + sb*Wt + w.
        const uint32_t b_start = num_blocks_written / Ts;
        const uint32_t sb_start = num_blocks_written % Ts;
        const uint32_t q_tile_id_start = b_start * n_heads * kq_HtWt + sb_start * Th;
        const uint32_t k_tile_id_start = q_tile_id_start;  // q and k share head_dim → same head-stride
        const uint32_t v_tile_id_start = b_start * n_heads * v_HtWt + sb_start * Tv;

        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {q_pre_buffer->address(),
             kv_up_buffer->address(),
             k_pe_buffer->address(),
             num_blocks_per_core,
             q_pre_tile_id_start,
             kv_up_tile_id_start,
             k_pe_tile_id_start});

        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {q_buffer->address(),
             k_buffer->address(),
             v_buffer->address(),
             num_blocks_per_core,
             sb_start,
             q_tile_id_start,
             k_tile_id_start,
             v_tile_id_start});

        num_blocks_written += num_blocks_per_core;
    }
}

MLAQKVAssembleFwProgramFactory::cached_program_t MLAQKVAssembleFwProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q_pre = tensor_args.q_pre;
    const auto& kv_up = tensor_args.kv_up;
    const auto& k_pe = tensor_args.k_pe;
    auto& q_out = output[0];
    auto& k = output[1];
    auto& v = output[2];

    auto* device = kv_up.device();
    tt::tt_metal::Program program{};

    // ── Tile geometry ──
    const auto kv_up_shape = kv_up.padded_shape();
    const uint32_t B = kv_up_shape[0];
    const uint32_t S = kv_up_shape[2];
    const uint32_t H = args.n_heads;

    const uint32_t Tn = args.qk_nope_dim / TILE_WIDTH;
    const uint32_t Tr = args.qk_rope_dim / TILE_WIDTH;
    const uint32_t Tv = args.v_dim / TILE_WIDTH;
    const uint32_t Th = Tn + Tr;
    const uint32_t Ts = S / TILE_HEIGHT;

    // q and k share per-head width (qk_head). v has its own width.
    const uint32_t kq_HtWt = Ts * Th;
    const uint32_t v_HtWt = Ts * Tv;

    const uint32_t q_tiles_per_block = H * Th;
    const uint32_t kv_up_tiles_per_block = H * (Tn + Tv);
    const uint32_t kpe_tiles_per_block = Tr;

    const uint32_t num_blocks = B * Ts;

    // ── Work split ──
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    // ── CBs ──
    // Validation guarantees q_pre / kv_up / k_pe share dtype, so any of them yields the same
    // data_format. kv_up is picked because it's also the source-of-truth for B and S above.
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(kv_up.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    // cb_q / cb_knope / cb_v are sized 2 * block_size (see block_size above). cb_kpe instead holds all
    // Tr tiles for the whole block because the writer broadcasts them to every head; Tr is small and
    // inherent to the broadcast. Even worst case these CBs are a small fraction of L1, so none is too big.
    const uint32_t io_cb_num_tiles = kCbDoubleBuffer * block_size;
    const uint32_t kpe_cb_num_tiles = kCbDoubleBuffer * Tr;

    [[maybe_unused]] auto cb_q =
        create_circular_buffer(program, all_cores, kQCbIndex, data_format, single_tile_size, io_cb_num_tiles);

    [[maybe_unused]] auto cb_knope =
        create_circular_buffer(program, all_cores, kKnopeCbIndex, data_format, single_tile_size, io_cb_num_tiles);

    [[maybe_unused]] auto cb_v =
        create_circular_buffer(program, all_cores, kVCbIndex, data_format, single_tile_size, io_cb_num_tiles);

    [[maybe_unused]] auto cb_kpe =
        create_circular_buffer(program, all_cores, kKpeCbIndex, data_format, single_tile_size, kpe_cb_num_tiles);

    // ── Kernels ──
    auto* q_pre_buffer = q_pre.buffer();
    auto* kv_up_buffer = kv_up.buffer();
    auto* k_pe_buffer = k_pe.buffer();
    auto* q_buffer = q_out.buffer();
    auto* k_buffer = k.buffer();
    auto* v_buffer = v.buffer();

    std::vector<uint32_t> reader_compile_time_args = {Tn, Tv, Tr, H, block_size};
    tt::tt_metal::TensorAccessorArgs(q_pre_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(kv_up_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_pe_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {Tn, Tr, Tv, H, kq_HtWt, v_HtWt, Ts, block_size};
    tt::tt_metal::TensorAccessorArgs(q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(writer_compile_time_args);

    const std::map<std::string, std::string> defines;

    MLAQKVAssembleFwKernels kernels;
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    // ── Runtime args ──
    assign_per_core_runtime_args(
        program,
        kernels,
        q_pre_buffer,
        kv_up_buffer,
        k_pe_buffer,
        q_buffer,
        k_buffer,
        v_buffer,
        num_cores,
        num_cores_y,
        num_blocks_per_core_group_1,
        num_blocks_per_core_group_2,
        core_group_1,
        core_group_2,
        Ts,
        q_tiles_per_block,
        kv_up_tiles_per_block,
        kpe_tiles_per_block,
        Th,
        Tv,
        kq_HtWt,
        v_HtWt,
        H);

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id = */ kernels.reader,
         /* writer_kernel_id = */ kernels.writer,
         /* num_cores        = */ num_cores,
         /* num_cores_y      = */ num_cores_y}};
}

void MLAQKVAssembleFwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*args*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* q_pre_buffer = tensor_args.q_pre.buffer();
    auto* kv_up_buffer = tensor_args.kv_up.buffer();
    auto* k_pe_buffer = tensor_args.k_pe.buffer();
    auto* q_buffer = output[0].buffer();
    auto* k_buffer = output[1].buffer();
    auto* v_buffer = output[2].buffer();

    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / shared.num_cores_y, i % shared.num_cores_y};
        {
            auto& ra = reader_runtime_args[core.x][core.y];
            ra[kReaderArgQPreAddr] = q_pre_buffer->address();
            ra[kReaderArgKvUpAddr] = kv_up_buffer->address();
            ra[kReaderArgKpeAddr] = k_pe_buffer->address();
        }
        {
            auto& ra = writer_runtime_args[core.x][core.y];
            ra[kWriterArgQAddr] = q_buffer->address();
            ra[kWriterArgKAddr] = k_buffer->address();
            ra[kWriterArgVAddr] = v_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::mla_qkv_assemble_fw::device
