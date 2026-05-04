// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subtract_at_target_program_factory.hpp"

#include <cstring>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "subtract_at_target_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/subtract_at_target/device/kernels/dataflow/subtract_at_target_reader.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/subtract_at_target/device/kernels/dataflow/subtract_at_target_writer.cpp";

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

constexpr uint32_t kPageElementsNumber = 32U;
constexpr uint32_t kNumOutputTiles = 2U;  // double-buffered

uint32_t float_to_uint32_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(uint32_t));
    return bits;
}

}  // namespace

namespace ttml::metal::ops::subtract_at_target::device {

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

uint32_t flat_index(const ttnn::MeshCoordinate& mesh_coord, const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    uint32_t linear_index = 0U;
    uint32_t stride = 1U;
    for (int d = static_cast<int>(mesh_shape.dims()) - 1; d >= 0; --d) {
        linear_index += mesh_coord[d] * stride;
        stride *= mesh_shape[d];
    }
    return linear_index;
}

ShardWindow compute_shard_window(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinate& mesh_coord,
    const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    const uint32_t tp_rank =
        attrs.cluster_axis.has_value() ? mesh_coord[*attrs.cluster_axis] : flat_index(mesh_coord, mesh_shape);
    const uint32_t first_v = attrs.first_v + tp_rank * attrs.local_V;
    const uint32_t last_v = first_v + attrs.local_V;
    return {first_v, last_v};
}

struct CreatedProgram {
    tt::tt_metal::Program program;
    SubtractAtTargetProgramFactory::shared_variables_t shared_variables;
};

// Builds a single-device program with explicit shard window [first_v, last_v).
// Mirrors the pre-mesh-workload `create` body; factored here so create_mesh_workload
// can call it once per mesh coordinate.
CreatedProgram create_program_for_device(
    const ShardWindow& window, float subtract_value, const tensor_args_t& tensor_args, const ttnn::Tensor& output) {
    const auto& input = tensor_args.input;
    const auto& target = tensor_args.target;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    TT_FATAL(
        datatype_to_dataformat_converter(input.dtype()) == tt::DataFormat::Float16_b,
        "subtract_at_target: input must be BFLOAT16");

    const uint32_t bfloat16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    const auto padded_shape = input.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "subtract_at_target: input must be rank 4");

    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    const uint32_t total_rows = NC * Ht;

    const uint32_t target_page_size = target.logical_shape()[-1] * target.element_size();
    const uint32_t target_read_page_size = tt::datum_size(tt::DataFormat::UInt32) * kPageElementsNumber;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows);

    create_circular_buffer(
        program, all_cores, kTargetCbIndex, tt::DataFormat::UInt32, target_read_page_size, /*num_tiles=*/1U);

    create_circular_buffer(
        program, all_cores, kOutputCbIndex, tt::DataFormat::Float16_b, bfloat16_tile_bytes, kNumOutputTiles);

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

    auto reader_kernel = create_reader_kernel(program, all_cores, reader_ct_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    auto writer_kernel = create_writer_kernel(program, all_cores, writer_ct_args, {}, kWriterKernelPath);

    const uint32_t subtract_bits = float_to_uint32_bits(subtract_value);

    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        const uint32_t num_rows_this_core = core_group_1.contains(core)   ? num_rows_per_core_group_1
                                            : core_group_2.contains(core) ? num_rows_per_core_group_2
                                                                          : 0U;
        TT_FATAL(num_rows_this_core > 0U, "subtract_at_target: core not in any group");

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {input_buffer->address(),
             target_buffer->address(),
             num_rows_this_core,
             num_rows_written,
             window.first_v,
             window.last_v,
             subtract_bits});

        SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address(), num_rows_this_core, num_rows_written});

        num_rows_written += num_rows_this_core;
    }

    return CreatedProgram{
        std::move(program),
        SubtractAtTargetProgramFactory::shared_variables_t{
            reader_kernel, writer_kernel, core_group_1, core_group_2, num_cores, num_cores_y}};
}

}  // namespace

SubtractAtTargetProgramFactory::cached_mesh_workload_t SubtractAtTargetProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto* mesh_device = tensor_args.input.device();
    TT_FATAL(mesh_device != nullptr, "subtract_at_target: input must be on a (mesh) device");
    const auto mesh_shape = mesh_device->shape();

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    // Iterate every coord in the input tensor's range set so each device gets a program with
    // its own shard-window runtime args.  We register a `single_coord_range` per coord — same
    // pattern as RingSDPABwKVProgramFactory and SelectTargetLogitProgramFactory.  The kernel
    // binary is identical across coords (the program hash only depends on shape/dtype), so the
    // framework's program cache will dedupe the compile; only the per-program runtime args differ.
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : range) {
            auto window = compute_shard_window(operation_attributes, mesh_coord, mesh_shape);
            auto created = create_program_for_device(window, operation_attributes.subtract_value, tensor_args, output);

            ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
            mesh_workload.add_program(single_coord_range, std::move(created.program));
            shared_vars[single_coord_range] = std::move(created.shared_variables);
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

void SubtractAtTargetProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto* input_buffer = tensor_args.input.buffer();
    auto* target_buffer = tensor_args.target.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    auto* mesh_device = tensor_args.input.device();
    const auto mesh_shape = mesh_device->shape();

    const uint32_t subtract_bits = float_to_uint32_bits(operation_attributes.subtract_value);

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(coord_range);
        const auto& start_coord = coord_range.start_coord();

        const auto window = compute_shard_window(operation_attributes, start_coord, mesh_shape);

        auto& reader_args = GetRuntimeArgs(program, sv.reader_kernel_id);
        auto& writer_args = GetRuntimeArgs(program, sv.writer_kernel_id);

        for (uint32_t i = 0; i < sv.num_cores; ++i) {
            tt::tt_metal::CoreCoord core = {i / sv.num_cores_y, i % sv.num_cores_y};

            {
                auto& args = reader_args[core.x][core.y];
                args[kInputBufferIdx] = input_buffer->address();
                args[kTargetBufferIdx] = target_buffer->address();
                args[kFirstVIdx] = window.first_v;
                args[kLastVIdx] = window.last_v;
                args[kSubtractValueIdx] = subtract_bits;
            }
            {
                auto& args = writer_args[core.x][core.y];
                args[kOutputBufferIdx] = output_buffer->address();
            }
        }
    }
}

}  // namespace ttml::metal::ops::subtract_at_target::device
