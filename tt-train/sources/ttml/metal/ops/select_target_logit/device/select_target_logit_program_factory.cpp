// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "select_target_logit_program_factory.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "select_target_logit_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/select_target_logit/device/kernels/dataflow/select_target_logit_reader.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/select_target_logit/device/kernels/dataflow/select_target_logit_writer.cpp";

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

}  // namespace

namespace ttml::metal::ops::select_target_logit::device {

namespace {

// Per-device shard window [first_v, last_v) derived from the operation attributes and the
// device's mesh coordinate.
//
//   - cluster_axis set: tp_rank = mesh_coord[*cluster_axis] (the natural mesh-derived index).
//   - cluster_axis unset: tp_rank = row-major flat mesh index.
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

struct CreatedProgram {
    tt::tt_metal::Program program;
    SelectTargetLogitProgramFactory::shared_variables_t shared_variables;
};

// Builds a single-device program with explicit shard window [first_v, last_v).
// Mirrors the pre-mesh-workload `create` body; factored here so create_mesh_workload
// can call it once per mesh coordinate.
CreatedProgram create_program_for_device(
    const ShardWindow& window, const tensor_args_t& tensor_args, const ttnn::Tensor& output) {
    const auto& logit = tensor_args.logit;
    const auto& target = tensor_args.target;

    auto* device = logit.device();
    tt::tt_metal::Program program{};

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
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    const uint32_t total_rows = NC * Ht;

    const uint32_t target_page_size = target.logical_shape()[-1] * target.element_size();
    const uint32_t target_read_page_size = tt::datum_size(tt::DataFormat::UInt32) * tt::constants::TILE_HEIGHT;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows);

    create_circular_buffer(
        program, all_cores, kTargetCbIndex, tt::DataFormat::UInt32, target_read_page_size, /*num_tiles=*/1U);

    create_circular_buffer(
        program, all_cores, kLogitScratchCbIndex, logit_data_format, logit_tile_bytes, /*num_tiles=*/1U);

    create_circular_buffer(program, all_cores, kOutputCbIndex, output_data_format, output_tile_bytes, kNumOutputTiles);

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

    auto reader_kernel = create_reader_kernel(program, all_cores, reader_ct_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    auto writer_kernel = create_writer_kernel(program, all_cores, writer_ct_args, {}, kWriterKernelPath);

    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        const uint32_t num_rows_this_core = core_group_1.contains(core)   ? num_rows_per_core_group_1
                                            : core_group_2.contains(core) ? num_rows_per_core_group_2
                                                                          : 0U;
        TT_FATAL(num_rows_this_core > 0U, "select_target_logit: core not in any group");

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {logit_buffer->address(),
             target_buffer->address(),
             num_rows_this_core,
             num_rows_written,
             window.first_v,
             window.last_v});

        SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address(), num_rows_this_core, num_rows_written});

        num_rows_written += num_rows_this_core;
    }

    return CreatedProgram{
        std::move(program),
        SelectTargetLogitProgramFactory::shared_variables_t{
            reader_kernel, writer_kernel, core_group_1, core_group_2, num_cores, num_cores_y}};
}

}  // namespace

SelectTargetLogitProgramFactory::cached_mesh_workload_t SelectTargetLogitProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto* mesh_device = tensor_args.logit.device();
    TT_FATAL(mesh_device != nullptr, "select_target_logit: logit must be on a (mesh) device");
    const auto mesh_shape = mesh_device->shape();

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    if (operation_attributes.cluster_axis.has_value()) {
        // All devices sharing the same tp_rank form a contiguous slab (the hyperplane fixing
        // the TP axis), so we register exactly one program per tp_rank.
        const uint32_t tp_axis = *operation_attributes.cluster_axis;
        const uint32_t tp_size = mesh_shape[tp_axis];

        auto slab_start = ttnn::MeshCoordinate::zero_coordinate(mesh_shape.dims());
        auto slab_end = ttnn::MeshCoordinate::zero_coordinate(mesh_shape.dims());
        for (size_t d = 0; d < mesh_shape.dims(); ++d) {
            slab_end[static_cast<int32_t>(d)] = mesh_shape[d] - 1U;
        }

        for (uint32_t rank = 0; rank < tp_size; ++rank) {
            slab_start[static_cast<int32_t>(tp_axis)] = rank;
            slab_end[static_cast<int32_t>(tp_axis)] = rank;
            ttnn::MeshCoordinateRange slab{slab_start, slab_end};

            const ShardWindow window{
                operation_attributes.first_v + rank * operation_attributes.local_V,
                operation_attributes.first_v + (rank + 1U) * operation_attributes.local_V};

            // Restrict the slab to the part that's actually backed by the input tensor.
            for (const auto& range : tensor_coords.ranges()) {
                auto inter = slab.intersection(range);
                if (!inter.has_value()) {
                    continue;
                }

                auto created = create_program_for_device(window, tensor_args, output);
                mesh_workload.add_program(*inter, std::move(created.program));
                shared_vars[*inter] = std::move(created.shared_variables);
            }
        }
    } else {
        // No cluster_axis: every coord may have a distinct flat-index tp_rank, so each device
        // needs its own program with its own runtime args.
        for (const auto& range : tensor_coords.ranges()) {
            for (const auto& mesh_coord : range) {
                auto window = compute_shard_window(operation_attributes, mesh_coord, mesh_shape);
                auto created = create_program_for_device(window, tensor_args, output);

                ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
                mesh_workload.add_program(single_coord_range, std::move(created.program));
                shared_vars[single_coord_range] = std::move(created.shared_variables);
            }
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

void SelectTargetLogitProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto* logit_buffer = tensor_args.logit.buffer();
    auto* target_buffer = tensor_args.target.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    auto* mesh_device = tensor_args.logit.device();
    const auto mesh_shape = mesh_device->shape();

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
                args[kLogitBufferIdx] = logit_buffer->address();
                args[kTargetBufferIdx] = target_buffer->address();
                args[kFirstVIdx] = window.first_v;
                args[kLastVIdx] = window.last_v;
            }
            {
                auto& args = writer_args[core.x][core.y];
                args[kOutputBufferIdx] = output_buffer->address();
            }
        }
    }
}

}  // namespace ttml::metal::ops::select_target_logit::device
