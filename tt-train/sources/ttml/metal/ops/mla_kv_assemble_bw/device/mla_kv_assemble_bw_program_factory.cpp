// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_kv_assemble_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "mla_kv_assemble_bw_device_operation.hpp"
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

// Kernel push order into ProgramDescriptor::kernels: reader, writer, then the single compute kernel (it covers all
// cores, since its per-core block count is a runtime arg).
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;
constexpr uint32_t kComputeKernelIdx = 2U;

// reader runtime args
constexpr uint32_t kDKAddrIdx = 0U;
constexpr uint32_t kDVAddrIdx = 1U;
// writer runtime args
constexpr uint32_t kDKvUpAddrIdx = 0U;
constexpr uint32_t kDKpeAddrIdx = 1U;

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

namespace {

// dK's (batch, sequence tile row) blocks over the device's compute grid. create_descriptor() pins it against
// split_work_to_cores, and override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& dK) {
    const auto& shape = dK.padded_shape();
    return compute_work_split(dK.device()->compute_with_storage_grid_size(), shape[0] * (shape[2] / TILE_HEIGHT));
}

}  // namespace

tt::tt_metal::ProgramDescriptor MLAKVAssembleBwProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& dK = tensor_args.dK;
    const auto& dV = tensor_args.dV;
    auto& dkv_up = output[0];
    auto& dk_pe = output[1];

    auto* device = dK.device();
    tt::tt_metal::ProgramDescriptor desc;

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
    // slot. With fp32 dest accumulation the DST holds max_dst_tiles fp32 tiles, so Tr + 1 ≤ max_dst_tiles.
    constexpr uint32_t max_dst_tiles = 4U;
    TT_FATAL(
        Tr + 1U <= max_dst_tiles,
        "MLAKVAssembleBw: qk_rope_dim ({}) / TILE_W = {} too large for in-register accumulator (max {}).",
        args.qk_rope_dim,
        Tr,
        max_dst_tiles - 1U);

    // ── Work split ──
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    const WorkSplit split = make_work_split(dK);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2);

    // ── CBs ──
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dK.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    // dK_nope / dV are sized 2 * block_size (see block_size above). cb_dkpe_in instead holds all
    // Tr tiles for the whole block because the writer broadcasts them to every head; Tr is small
    // and inherent to the broadcast.
    const uint32_t io_cb_num_tiles = kCbDoubleBuffer * block_size;
    const uint32_t kpe_cb_num_tiles = kCbDoubleBuffer * Tr;

    desc.cbs.push_back(make_cb_descriptor(all_cores, kDKnopeCbIndex, data_format, single_tile_size, io_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kDVCbIndex, data_format, single_tile_size, io_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kDKpeInCbIndex, data_format, single_tile_size, kpe_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kDKpeOutCbIndex, data_format, single_tile_size, kpe_cb_num_tiles));

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

    desc.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));
    desc.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));
    // fp32 dest accumulation: the dk_pe head-axis sum adds up to n_heads BF16 tiles, so accumulating in
    // fp32 (instead of BF16 dest) keeps the gradient accurate for large head counts.
    desc.kernels.push_back(  // kComputeKernelIdx
        make_compute_kernel_descriptor(
            all_cores, compute_compile_time_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));

    // ── Runtime args ──
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // Block counts and tile offsets follow from the hashed shapes and attributes.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    auto& compute = desc.kernels[kComputeKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_blocks, start_block, in_group_1] = work;
        const uint32_t b_start = start_block / Ts;
        const uint32_t sb_start = start_block % Ts;
        const uint32_t dK_block_base_start = b_start * H * k_HtWt + sb_start * Th;
        const uint32_t dV_block_base_start = b_start * H * v_HtWt + sb_start * Tv;
        const uint32_t dkv_up_block_base_start = start_block * H * (Tn + Tv);
        const uint32_t dk_pe_block_base_start = start_block * Tr;
        reader.emplace_runtime_args(
            core, {dK_buffer, dV_buffer, num_blocks, sb_start, dK_block_base_start, dV_block_base_start});
        writer.emplace_runtime_args(
            core, {dkv_up_buffer, dk_pe_buffer, num_blocks, dkv_up_block_base_start, dk_pe_block_base_start});
        compute.emplace_runtime_args(core, {num_blocks});
    });

    return desc;
}

void MLAKVAssembleBwProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.dK);

    const uint32_t dK_addr = tensor_args.dK.buffer()->address();
    const uint32_t dV_addr = tensor_args.dV.buffer()->address();
    const uint32_t dkv_up_addr = output[0].buffer()->address();
    const uint32_t dk_pe_addr = output[1].buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kDKAddrIdx] = dK_addr;
        reader_args[kDVAddrIdx] = dV_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kDKvUpAddrIdx] = dkv_up_addr;
        writer_args[kDKpeAddrIdx] = dk_pe_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(MLAKVAssembleBwDeviceOperation, MLAKVAssembleBwProgramFactory);

}  // namespace ttml::metal::ops::mla_kv_assemble_bw::device
