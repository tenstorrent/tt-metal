// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_kv_assemble_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "mla_kv_assemble_bw_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_kv_assemble_bw/device/kernels/dataflow/"
    "reader_mla_kv_assemble_bw.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_kv_assemble_bw/device/kernels/dataflow/"
    "writer_mla_kv_assemble_bw.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_kv_assemble_bw/device/kernels/compute/"
    "mla_kv_assemble_bw_kernel.cpp";

// reader runtime arg indices (for override_runtime_arguments)
constexpr uint32_t kReaderArgDKAddr = 0;
constexpr uint32_t kReaderArgDVAddr = 1;

// writer runtime arg indices
constexpr uint32_t kWriterArgDKVUpAddr = 0;
constexpr uint32_t kWriterArgDKPeAddr = 1;

// CB indices
constexpr auto kDKnopeCbIndex = tt::CBIndex::c_1;
constexpr auto kDVCbIndex = tt::CBIndex::c_2;
constexpr auto kDKpeInCbIndex = tt::CBIndex::c_3;
constexpr auto kDKpeOutCbIndex = tt::CBIndex::c_4;

// CB sizing. dK_nope / dV stream in block_size chunks, double-buffered (2 * block_size) for
// reader/writer NoC overlap, so L1 use is bounded by block_size rather than head_dim. block_size = 8
// matches the forward op (knee of the N300 throughput sweep); it is not tied to the DST register limit
// because the only compute here is the small dk_pe head-axis reduction. See mla_kv_assemble_fw.
constexpr uint32_t kCbDoubleBuffer = 2U;
// On the throughput plateau (N300 sweep over {4,8,16} is flat within noise); not tied to DST registers
// since the only compute is the small dk_pe head-axis reduction. Matches the forward op.
constexpr uint32_t block_size = 8U;

}  // namespace

namespace ttml::metal::ops::mla_kv_assemble_bw::device {

using namespace tt::constants;

struct MLAKVAssembleBwKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute;
};

static void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const MLAKVAssembleBwKernels& kernels,
    const tt::tt_metal::Buffer* dK_buffer,
    const tt::tt_metal::Buffer* dV_buffer,
    const tt::tt_metal::Buffer* dkv_up_buffer,
    const tt::tt_metal::Buffer* dk_pe_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_blocks_per_core_group_1,
    uint32_t num_blocks_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t Ts,
    uint32_t Th,
    uint32_t Tn,
    uint32_t Tr,
    uint32_t Tv,
    uint32_t k_HtWt,
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
            TT_FATAL(false, "MLAKVAssembleBw: core not in any work group");
        }

        // Inputs are head-major: tile_id(b, h, sb, w) = b*H*HtWt + h*HtWt + sb*Wt + w.
        const uint32_t b_start = num_blocks_written / Ts;
        const uint32_t sb_start = num_blocks_written % Ts;
        const uint32_t dK_block_base_start = b_start * n_heads * k_HtWt + sb_start * Th;
        const uint32_t dV_block_base_start = b_start * n_heads * v_HtWt + sb_start * Tv;

        // Outputs are flat across blocks: head 0 of (b, sb) at (b*Ts + sb) * (per_block_tiles).
        const uint32_t flat_block_idx = num_blocks_written;
        const uint32_t dkv_up_block_base_start = flat_block_idx * n_heads * (Tn + Tv);
        const uint32_t dk_pe_block_base_start = flat_block_idx * Tr;

        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {dK_buffer->address(),
             dV_buffer->address(),
             num_blocks_per_core,
             sb_start,
             dK_block_base_start,
             dV_block_base_start});

        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {dkv_up_buffer->address(),
             dk_pe_buffer->address(),
             num_blocks_per_core,
             dkv_up_block_base_start,
             dk_pe_block_base_start});

        SetRuntimeArgs(program, kernels.compute, core, {num_blocks_per_core});

        num_blocks_written += num_blocks_per_core;
    }
}

MLAKVAssembleBwProgramFactory::cached_program_t MLAKVAssembleBwProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& dK = tensor_args.dK;
    const auto& dV = tensor_args.dV;
    auto& dkv_up = output[0];
    auto& dk_pe = output[1];

    auto* device = dK.device();
    tt::tt_metal::Program program{};

    // ── Tile geometry ──
    const auto dK_shape = dK.padded_shape();
    const uint32_t B = dK_shape[0];
    const uint32_t S = dK_shape[2];
    const uint32_t H = args.n_heads;

    const uint32_t Tn = args.qk_nope_dim / TILE_WIDTH;
    const uint32_t Tr = args.qk_rope_dim / TILE_WIDTH;
    const uint32_t Tv = args.v_dim / TILE_WIDTH;
    const uint32_t Th = Tn + Tr;
    const uint32_t Ts = S / TILE_HEIGHT;

    const uint32_t k_HtWt = Ts * Th;
    const uint32_t v_HtWt = Ts * Tv;

    const uint32_t num_blocks = B * Ts;

    // Compute kernel uses Tr persistent dst-register slots as the head-axis accumulator, plus 1 temp
    // slot. With fp32 dest accumulation the DST holds 8 fp32 tiles, so the bound is Tr + 1 ≤ 8.
    TT_FATAL(
        Tr + 1U <= 8U,
        "MLAKVAssembleBw: qk_rope_dim ({}) / TILE_W = {} too large for in-register accumulator (max 7).",
        args.qk_rope_dim,
        Tr);

    // ── Work split ──
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    // ── CBs ──
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dK.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    const uint32_t io_cb_num_tiles = kCbDoubleBuffer * block_size;
    const uint32_t kpe_cb_num_tiles = kCbDoubleBuffer * Tr;

    [[maybe_unused]] auto cb_dknope =
        create_circular_buffer(program, all_cores, kDKnopeCbIndex, data_format, single_tile_size, io_cb_num_tiles);
    [[maybe_unused]] auto cb_dv =
        create_circular_buffer(program, all_cores, kDVCbIndex, data_format, single_tile_size, io_cb_num_tiles);
    [[maybe_unused]] auto cb_dkpe_in =
        create_circular_buffer(program, all_cores, kDKpeInCbIndex, data_format, single_tile_size, kpe_cb_num_tiles);
    [[maybe_unused]] auto cb_dkpe_out =
        create_circular_buffer(program, all_cores, kDKpeOutCbIndex, data_format, single_tile_size, kpe_cb_num_tiles);

    // ── Kernels ──
    auto* dK_buffer = dK.buffer();
    auto* dV_buffer = dV.buffer();
    auto* dkv_up_buffer = dkv_up.buffer();
    auto* dk_pe_buffer = dk_pe.buffer();

    std::vector<uint32_t> reader_compile_time_args = {Th, Tn, Tr, Tv, H, k_HtWt, v_HtWt, Ts, block_size};
    tt::tt_metal::TensorAccessorArgs(dK_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dV_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {Tn, Tr, Tv, H, block_size};
    tt::tt_metal::TensorAccessorArgs(dkv_up_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dk_pe_buffer).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {Tr, H};

    const std::map<std::string, std::string> defines;

    MLAKVAssembleBwKernels kernels;
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);
    // fp32 dest accumulation: the dk_pe head-axis sum adds up to n_heads BF16 tiles, so accumulating in
    // fp32 (instead of BF16 dest) keeps the gradient accurate for large head counts.
    kernels.compute = create_compute_kernel(
        program, all_cores, compute_compile_time_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // ── Runtime args ──
    assign_per_core_runtime_args(
        program,
        kernels,
        dK_buffer,
        dV_buffer,
        dkv_up_buffer,
        dk_pe_buffer,
        num_cores,
        num_cores_y,
        num_blocks_per_core_group_1,
        num_blocks_per_core_group_2,
        core_group_1,
        core_group_2,
        Ts,
        Th,
        Tn,
        Tr,
        Tv,
        k_HtWt,
        v_HtWt,
        H);

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id  = */ kernels.reader,
         /* writer_kernel_id  = */ kernels.writer,
         /* compute_kernel_id = */ kernels.compute,
         /* num_cores         = */ num_cores,
         /* num_cores_y       = */ num_cores_y}};
}

void MLAKVAssembleBwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*args*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* dK_buffer = tensor_args.dK.buffer();
    auto* dV_buffer = tensor_args.dV.buffer();
    auto* dkv_up_buffer = output[0].buffer();
    auto* dk_pe_buffer = output[1].buffer();

    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / shared.num_cores_y, i % shared.num_cores_y};
        {
            auto& ra = reader_runtime_args[core.x][core.y];
            ra[kReaderArgDKAddr] = dK_buffer->address();
            ra[kReaderArgDVAddr] = dV_buffer->address();
        }
        {
            auto& ra = writer_runtime_args[core.x][core.y];
            ra[kWriterArgDKVUpAddr] = dkv_up_buffer->address();
            ra[kWriterArgDKPeAddr] = dk_pe_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::mla_kv_assemble_bw::device
