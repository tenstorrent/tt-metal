// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "select_target_logit_program_factory.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "select_target_logit_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/select_target_logit/device/kernels/dataflow/select_target_logit_reader.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/select_target_logit/device/kernels/dataflow/select_target_logit_writer.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// Reader runtime arg indices
constexpr uint32_t kLogitBufferIdx = 0U;
constexpr uint32_t kTargetBufferIdx = 1U;
constexpr uint32_t kFirstVIdx = 4U;
constexpr uint32_t kLastVIdx = 5U;

// Writer runtime arg indices
constexpr uint32_t kOutputBufferIdx = 0U;

// CB indices
constexpr auto kTargetCbIndex = tt::CBIndex::c_0;        // scratch: target page (uint32)
constexpr auto kLogitScratchCbIndex = tt::CBIndex::c_1;  // scratch: one logit tile (bfloat16)
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;        // output tiles (bfloat16)

constexpr uint32_t kNumOutputTiles = 2U;  // double-buffered

// Units of the work split: tile rows (NC * Ht) of the logit's padded shape. create_descriptor() and the override split
// them over the same grid, so the override patches exactly the cores that got runtime args.
uint32_t num_tile_rows(const ttnn::Shape& padded_shape) {
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    return NC * Ht;
}

}  // namespace

namespace ttml::metal::ops::select_target_logit::device {

namespace {

// Per-device shard window [first_v, last_v) derived from the operation attributes and the
// device's mesh coordinate.
//
//   - cluster_axis set: tp_rank = mesh_coord[*cluster_axis] (the natural mesh-derived index).
//   - cluster_axis unset: tp_rank = row-major flat mesh index.
//
// create_descriptor() writes it into the reader's runtime args and override_runtime_arguments() re-applies it on every
// cache hit.
struct ShardWindow {
    uint32_t first_v;
    uint32_t last_v;
};

ShardWindow compute_shard_window(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinate& mesh_coord,
    const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    const uint32_t tp_rank = attrs.cluster_axis.has_value()
                                 ? mesh_coord[*attrs.cluster_axis]
                                 : static_cast<uint32_t>(mesh_coord.to_linear_index(mesh_shape));
    const uint32_t first_v = attrs.first_v + tp_rank * attrs.local_V;
    const uint32_t last_v = first_v + attrs.local_V;
    return {first_v, last_v};
}

}  // namespace

// Builds the program of the device at `mesh_coordinate`, with that device's shard window. The framework calls it once
// per device coordinate; with cluster_axis set, the devices that share a tp_rank get identical programs.
tt::tt_metal::ProgramDescriptor SelectTargetLogitProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    TT_FATAL(mesh_coordinate.has_value(), "select_target_logit: create_descriptor needs the device's mesh coordinate");
    const auto& logit = tensor_args.logit;
    const auto& target = tensor_args.target;

    auto* device = logit.device();
    TT_FATAL(device != nullptr, "select_target_logit: logit must be on a (mesh) device");
    const auto window = compute_shard_window(operation_attributes, *mesh_coordinate, device->shape());

    tt::tt_metal::ProgramDescriptor program;

    TT_FATAL(
        datatype_to_dataformat_converter(logit.dtype()) == tt::DataFormat::Float16_b,
        "select_target_logit: logit must be BFLOAT16");

    const auto logit_data_format = tt::DataFormat::Float16_b;
    const uint32_t logit_tile_bytes = tt::tile_size(logit_data_format);
    const auto output_data_format = logit_data_format;
    const uint32_t output_tile_bytes = logit_tile_bytes;

    const auto padded_shape = logit.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "select_target_logit: logit must be rank 4");

    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t total_rows = num_tile_rows(padded_shape);

    const uint32_t target_page_size = target.logical_shape()[-1] * target.element_size();
    const uint32_t target_read_page_size = tt::datum_size(tt::DataFormat::UInt32) * tt::constants::TILE_HEIGHT;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows);
    const auto split = compute_work_split(compute_with_storage_grid_size, total_rows);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kTargetCbIndex, tt::DataFormat::UInt32, target_read_page_size, /*num_tiles=*/1U));

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kLogitScratchCbIndex, logit_data_format, logit_tile_bytes, /*num_tiles=*/1U));

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kOutputCbIndex, output_data_format, output_tile_bytes, kNumOutputTiles));

    auto* logit_buffer = logit.buffer();
    TT_FATAL(
        logit_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "select_target_logit: logit buffer must be DRAM, got {}",
        enchantum::to_string(logit_buffer->buffer_type()));

    auto* target_buffer = target.buffer();
    TT_FATAL(
        target_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "select_target_logit: target buffer must be DRAM, got {}",
        enchantum::to_string(target_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "select_target_logit: output buffer must be DRAM, got {}",
        enchantum::to_string(output_buffer->buffer_type()));

    std::vector<uint32_t> reader_ct_args{Wt, Ht, target_page_size, target_read_page_size};
    tt::tt_metal::TensorAccessorArgs(logit_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(target_buffer).append_to(reader_ct_args);

    program.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(all_cores, reader_ct_args, {}, kReaderKernelPath));

    std::vector<uint32_t> writer_ct_args{};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    program.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(all_cores, writer_ct_args, {}, kWriterKernelPath));

    // Buffers are passed as Buffer*, never as raw addresses. On a cache hit override_runtime_arguments() rewrites the
    // address slots and the shard window; row counts and start rows follow from the hashed logit shape.
    auto& reader = program.kernels[kReaderKernelIdx];
    auto& writer = program.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        reader.emplace_runtime_args(
            core, {logit_buffer, target_buffer, num_rows, start_row, window.first_v, window.last_v});
        writer.emplace_runtime_args(core, {output_buffer, num_rows, start_row});
    });

    return program;
}

void SelectTargetLogitProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    // The program belongs to one device; the adapter passes its coordinate on every hit.
    TT_FATAL(
        mesh_coordinate.has_value(),
        "select_target_logit: override_runtime_arguments needs the device's mesh coordinate");
    const auto& logit = tensor_args.logit;
    auto* device = logit.device();
    const auto window = compute_shard_window(operation_attributes, *mesh_coordinate, device->shape());

    const uint32_t logit_addr = logit.buffer()->address();
    const uint32_t target_addr = tensor_args.target.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    const auto split =
        compute_work_split(device->compute_with_storage_grid_size(), num_tile_rows(logit.padded_shape()));

    auto& reader_rt = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_rt = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        auto& reader_args = reader_rt[work.core.x][work.core.y];
        reader_args[kLogitBufferIdx] = logit_addr;
        reader_args[kTargetBufferIdx] = target_addr;
        reader_args[kFirstVIdx] = window.first_v;
        reader_args[kLastVIdx] = window.last_v;

        auto& writer_args = writer_rt[work.core.x][work.core.y];
        writer_args[kOutputBufferIdx] = output_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SelectTargetLogitDeviceOperation, SelectTargetLogitProgramFactory);

}  // namespace ttml::metal::ops::select_target_logit::device
