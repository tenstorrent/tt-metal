// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_kv_assemble_fw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "mla_kv_assemble_fw_device_operation.hpp"
#include "mla_kv_assemble_fw_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_kv_assemble_fw/device/kernels/dataflow/"
    "reader_mla_kv_assemble_fw.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/mla_kv_assemble_fw/device/kernels/dataflow/"
    "writer_mla_kv_assemble_fw.cpp";

// Kernel push order: reader, then writer (there is no compute kernel)
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kKvUpAddrIdx = 0U;
constexpr uint32_t kKpeAddrIdx = 1U;
// writer runtime args
constexpr uint32_t kKAddrIdx = 0U;
constexpr uint32_t kVAddrIdx = 1U;

// CB indices. k_nope and v are demuxed into separate CBs so each output stream can be chunked and
// double-buffered independently; the writer reinserts the broadcast k_pe between them.
constexpr auto kKnopeCbIndex = tt::CBIndex::c_0;
constexpr auto kVCbIndex = tt::CBIndex::c_1;
constexpr auto kKpeCbIndex = tt::CBIndex::c_2;

// Each k_nope / v CB is double-buffered (2 * block_size) so the reader can fetch the next chunk on
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

namespace ttml::metal::ops::mla_kv_assemble_fw::device {

using namespace tt::constants;

namespace {

// kv_up's (batch, sequence tile row) blocks over the device's compute grid. create_descriptor() pins it against
// split_work_to_cores, and override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& kv_up) {
    const auto& shape = kv_up.padded_shape();
    return compute_work_split(
        kv_up.device()->compute_with_storage_grid_size(), shape[0] * (shape[2] / tt::constants::TILE_HEIGHT));
}

}  // namespace

tt::tt_metal::ProgramDescriptor MLAKVAssembleFwProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& kv_up = tensor_args.kv_up;
    const auto& k_pe = tensor_args.k_pe;
    auto& k = output[0];
    auto& v = output[1];

    auto* device = kv_up.device();
    tt::tt_metal::ProgramDescriptor desc;

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

    const uint32_t k_HtWt = Ts * Th;
    const uint32_t v_HtWt = Ts * Tv;

    const uint32_t kv_up_tiles_per_block = H * (Tn + Tv);
    const uint32_t kpe_tiles_per_block = Tr;

    const uint32_t num_blocks = B * Ts;

    // ── Work split ──
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    const WorkSplit split = make_work_split(kv_up);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2);

    // ── CBs ──
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(kv_up.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    // knope / v are sized 2 * block_size (see block_size above). cb_kpe instead holds all
    // Tr tiles for the whole block because the writer broadcasts them to every head; Tr is small
    // and inherent to the broadcast.
    const uint32_t io_cb_num_tiles = kCbDoubleBuffer * block_size;
    const uint32_t kpe_cb_num_tiles = kCbDoubleBuffer * Tr;

    desc.cbs.push_back(make_cb_descriptor(all_cores, kKnopeCbIndex, data_format, single_tile_size, io_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kVCbIndex, data_format, single_tile_size, io_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kKpeCbIndex, data_format, single_tile_size, kpe_cb_num_tiles));

    // ── Kernels ──
    auto* kv_up_buffer = kv_up.buffer();
    auto* k_pe_buffer = k_pe.buffer();
    auto* k_buffer = k.buffer();
    auto* v_buffer = v.buffer();

    std::vector<uint32_t> reader_compile_time_args = {Tn, Tv, Tr, H, block_size};
    tt::tt_metal::TensorAccessorArgs(kv_up_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_pe_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {Tn, Tr, Tv, H, k_HtWt, v_HtWt, Ts, block_size};
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(writer_compile_time_args);

    const std::map<std::string, std::string> defines;

    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));
    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));

    // ── Runtime args ──
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // num_blocks and the tile offsets follow from the hashed shapes and head dims.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_blocks, start_block, in_group_1] = work;
        // Block index = b * Ts + sb. Inputs kv_up, k_pe are flat across blocks.
        const uint32_t kv_up_tile_id_start = start_block * kv_up_tiles_per_block;
        const uint32_t k_pe_tile_id_start = start_block * kpe_tiles_per_block;
        // Outputs are head-major: tile_id(b, h, sb, w) = b*H*HtWt + h*HtWt + sb*Wt + w.
        const uint32_t b_start = start_block / Ts;
        const uint32_t sb_start = start_block % Ts;
        const uint32_t k_tile_id_start = b_start * H * k_HtWt + sb_start * Th;
        const uint32_t v_tile_id_start = b_start * H * v_HtWt + sb_start * Tv;
        reader.emplace_runtime_args(
            core, {kv_up_buffer, k_pe_buffer, num_blocks, kv_up_tile_id_start, k_pe_tile_id_start});
        writer.emplace_runtime_args(core, {k_buffer, v_buffer, num_blocks, sb_start, k_tile_id_start, v_tile_id_start});
    });

    return desc;
}

void MLAKVAssembleFwProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.kv_up);

    const uint32_t kv_up_addr = tensor_args.kv_up.buffer()->address();
    const uint32_t k_pe_addr = tensor_args.k_pe.buffer()->address();
    const uint32_t k_addr = output[0].buffer()->address();
    const uint32_t v_addr = output[1].buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kKvUpAddrIdx] = kv_up_addr;
        reader_args[kKpeAddrIdx] = k_pe_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kKAddrIdx] = k_addr;
        writer_args[kVAddrIdx] = v_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(MLAKVAssembleFwDeviceOperation, MLAKVAssembleFwProgramFactory);

}  // namespace ttml::metal::ops::mla_kv_assemble_fw::device
