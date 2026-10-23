// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subtract_at_target_program_factory.hpp"

#include <cstring>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "subtract_at_target_device_operation.hpp"
#include "subtract_at_target_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/subtract_at_target/device/kernels/dataflow/subtract_at_target_reader.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/subtract_at_target/device/kernels/dataflow/subtract_at_target_writer.cpp";

// Kernel push order: reader, then writer
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// Reader runtime arg indices
constexpr uint32_t kInputBufferIdx = 0U;
constexpr uint32_t kTargetBufferIdx = 1U;
constexpr uint32_t kFirstVIdx = 4U;
constexpr uint32_t kLastVIdx = 5U;
constexpr uint32_t kSubtractValueIdx = 6U;

// Writer runtime arg indices
constexpr uint32_t kOutputBufferIdx = 0U;

// CB indices.  c_1 is intentionally unused: the reader streams input tiles
// directly into the output CB and patches them in place before push_back.
constexpr auto kTargetCbIndex = tt::CBIndex::c_0;  // scratch: target page (uint32)
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;  // output tiles (bfloat16)

constexpr uint32_t kNumOutputTiles = 2U;  // double-buffered

uint32_t float_to_uint32_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(uint32_t));
    return bits;
}

// Rows of tiles (NC * Ht) of the padded input shape: the unit of the work split. create_descriptor() and the override
// both split them over the device grid with compute_work_split, so the override patches exactly the cores that got
// runtime args.
uint32_t num_tile_rows(const ttnn::Shape& padded_shape) {
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    return NC * Ht;
}

}  // namespace

namespace ttml::metal::ops::subtract_at_target::device {

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SubtractAtTargetDeviceOperation, SubtractAtTargetProgramFactory);

namespace {

// Per-device shard window [first_v, last_v) derived from the operation attributes and the
// device's mesh coordinate.
//
//   - cluster_axis set: tp_rank = mesh_coord[*cluster_axis] (the natural mesh-derived index).
//   - cluster_axis unset: tp_rank = row-major flat mesh index (matches the previous host-side
//     `tp_rank_from_device_idx` fallback when no cluster_axis was provided).
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

// The reader runtime args left out of the program hash: the device's shard window and the bits of subtract_value.
// create_descriptor() and override_runtime_arguments() both take them from here, so a cache hit writes exactly what a
// miss would.
struct RuntimeScalars {
    ShardWindow window;
    uint32_t subtract_bits;
};

RuntimeScalars compute_runtime_scalars(
    const operation_attributes_t& attrs,
    const ttnn::Tensor& input,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    TT_FATAL(mesh_coordinate.has_value(), "subtract_at_target: the shard window needs the device's mesh coordinate");
    return RuntimeScalars{
        compute_shard_window(attrs, *mesh_coordinate, input.device()->shape()),
        float_to_uint32_bits(attrs.subtract_value)};
}

}  // namespace

// Builds the program of one device, with the shard window of its mesh coordinate.
tt::tt_metal::ProgramDescriptor SubtractAtTargetProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    const auto& input = tensor_args.input;
    const auto& target = tensor_args.target;

    auto* device = input.device();
    TT_FATAL(device != nullptr, "subtract_at_target: input must be on a (mesh) device");
    tt::tt_metal::ProgramDescriptor desc;

    TT_FATAL(
        datatype_to_dataformat_converter(input.dtype()) == tt::DataFormat::Float16_b,
        "subtract_at_target: input must be BFLOAT16");

    const uint32_t bfloat16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    const auto padded_shape = input.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "subtract_at_target: input must be rank 4");

    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t total_rows = num_tile_rows(padded_shape);

    const uint32_t target_page_size = target.logical_shape()[-1] * target.element_size();
    const uint32_t target_read_page_size = tt::datum_size(tt::DataFormat::UInt32) * tt::constants::TILE_HEIGHT;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows);
    const WorkSplit split = compute_work_split(compute_with_storage_grid_size, total_rows);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kTargetCbIndex, tt::DataFormat::UInt32, target_read_page_size, /*num_tiles=*/1U));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kOutputCbIndex, tt::DataFormat::Float16_b, bfloat16_tile_bytes, kNumOutputTiles));

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "subtract_at_target: input buffer must be DRAM, got {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* target_buffer = target.buffer();
    TT_FATAL(
        target_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "subtract_at_target: target buffer must be DRAM, got {}",
        enchantum::to_string(target_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "subtract_at_target: output buffer must be DRAM, got {}",
        enchantum::to_string(output_buffer->buffer_type()));

    std::vector<uint32_t> reader_ct_args{Wt, Ht, target_page_size, target_read_page_size};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(target_buffer).append_to(reader_ct_args);

    desc.kernels.push_back(make_reader_kernel_descriptor(all_cores, reader_ct_args, {}, kReaderKernelPath));

    std::vector<uint32_t> writer_ct_args{Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    desc.kernels.push_back(make_writer_kernel_descriptor(all_cores, writer_ct_args, {}, kWriterKernelPath));

    // Buffers are passed as Buffer*; override_runtime_arguments() rewrites them, the window and the subtract bits on
    // every cache hit. Row counts and start rows follow from the hashed input shape.
    const RuntimeScalars scalars = compute_runtime_scalars(operation_attributes, input, mesh_coordinate);
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        reader.emplace_runtime_args(
            core,
            {input_buffer,
             target_buffer,
             num_rows,
             start_row,
             scalars.window.first_v,
             scalars.window.last_v,
             scalars.subtract_bits});
        writer.emplace_runtime_args(core, {output_buffer, num_rows, start_row});
    });

    return desc;
}

void SubtractAtTargetProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    // The output address comes from the return value, which is the input itself on the in-place production call.
    // Row counts and start rows follow from the hashed input shape and stay as create_descriptor() set them.
    const auto& input = tensor_args.input;
    const uint32_t input_addr = input.buffer()->address();
    const uint32_t target_addr = tensor_args.target.buffer()->address();
    const uint32_t output_addr = tensor_return_value.buffer()->address();
    const RuntimeScalars scalars = compute_runtime_scalars(operation_attributes, input, mesh_coordinate);
    const WorkSplit split =
        compute_work_split(input.device()->compute_with_storage_grid_size(), num_tile_rows(input.padded_shape()));

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_addr;
            runtime_args[kTargetBufferIdx] = target_addr;
            runtime_args[kFirstVIdx] = scalars.window.first_v;
            runtime_args[kLastVIdx] = scalars.window.last_v;
            runtime_args[kSubtractValueIdx] = scalars.subtract_bits;
        }
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_addr;
        }
    });
}

}  // namespace ttml::metal::ops::subtract_at_target::device
