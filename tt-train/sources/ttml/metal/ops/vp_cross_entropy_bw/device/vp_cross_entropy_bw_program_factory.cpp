// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vp_cross_entropy_bw_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/vp_cross_entropy_bw/device/kernels/dataflow/reader_vp_cross_entropy_bw.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/vp_cross_entropy_bw/device/kernels/dataflow/writer_vp_cross_entropy_bw.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/vp_cross_entropy_bw/device/kernels/compute/vp_cross_entropy_bw_kernel.cpp";

// circular buffers
constexpr uint32_t kInCb = 0;      // bf16 logits tiles, `block` at a time
constexpr uint32_t kLogZCb = 1;    // fp32 log-normalizer tile of the item's row
constexpr uint32_t kGradCb = 2;    // the upstream gradient tile (compute side)
constexpr uint32_t kOutCb = 3;     // bf16 output tiles
constexpr uint32_t kTargetCb = 4;  // uint32 target page of the item's row (writer side)
constexpr uint32_t kGradWCb = 5;   // the upstream gradient tile (writer side)

constexpr uint32_t kBlock = 2U;

constexpr uint32_t kReaderLogitsIdx = 0U;
constexpr uint32_t kReaderLogZIdx = 1U;
constexpr uint32_t kReaderGradIdx = 2U;
constexpr uint32_t kWriterOutIdx = 0U;
constexpr uint32_t kWriterTargetIdx = 1U;
constexpr uint32_t kWriterGradIdx = 2U;
constexpr uint32_t kWriterFirstVIdx = 3U;

struct ShardWindow {
    uint32_t first_v;
    uint32_t last_v;
};

}  // namespace

namespace ttml::metal::ops::vp_cross_entropy_bw::device {

namespace {

ShardWindow shard_window(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinate& mesh_coord,
    const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    const uint32_t tp_rank = attrs.cluster_axis.has_value()
                                 ? mesh_coord[*attrs.cluster_axis]
                                 : static_cast<uint32_t>(mesh_coord.to_linear_index(mesh_shape));
    const uint32_t first_v = attrs.first_v + tp_rank * attrs.local_V;
    return {first_v, first_v + attrs.local_V};
}

struct Created {
    tt::tt_metal::Program program;
    VpCrossEntropyBwProgramFactory::shared_variables_t shared;
};

// Items = (row tile, column slice): the row tiles times `slices` slices of the Wt columns, enough items for
// two or three per core; a slice's tiles stream through L1 `kBlock` at a time.
Created create_program(
    const ShardWindow& window,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    const ttnn::Tensor& output) {
    const auto& logits = tensor_args.logits;
    auto* device = logits.device();
    tt::tt_metal::Program program{};

    const auto padded = logits.padded_shape();
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded[2] / tt::constants::TILE_HEIGHT;
    const uint32_t rows = padded[0] * padded[1] * Ht;
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_total = grid.x * grid.y;
    uint32_t slices = std::min<uint32_t>(Wt, std::max<uint32_t>(1U, (2U * num_cores_total + rows - 1U) / rows));
    uint32_t St = (Wt + slices - 1U) / slices;
    if (St % kBlock != 0U) {
        St += kBlock - (St % kBlock);
    }
    slices = (Wt + St - 1U) / St;
    const uint32_t total_items = rows * slices;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_1, work_2] =
        tt::tt_metal::split_work_to_cores(grid, total_items);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const auto grad_format = datatype_to_dataformat_converter(tensor_args.grad.dtype());
    const uint32_t target_page_bytes = tensor_args.target.logical_shape()[-1] * tensor_args.target.element_size();
    const uint32_t target_read_bytes = tt::datum_size(tt::DataFormat::UInt32) * tt::constants::TILE_HEIGHT;
    create_circular_buffer(program, all_cores, kInCb, bf16, tt::tile_size(bf16), 2U * kBlock);
    create_circular_buffer(program, all_cores, kLogZCb, f32, tt::tile_size(f32), 2U);
    create_circular_buffer(program, all_cores, kGradCb, grad_format, tt::tile_size(grad_format), 1U);
    create_circular_buffer(program, all_cores, kOutCb, bf16, tt::tile_size(bf16), 2U * kBlock);
    create_circular_buffer(program, all_cores, kTargetCb, tt::DataFormat::UInt32, target_read_bytes, 1U);
    create_circular_buffer(program, all_cores, kGradWCb, grad_format, tt::tile_size(grad_format), 1U);

    const uint32_t scale_bits = std::bit_cast<uint32_t>(args.scale);
    std::map<std::string, std::string> defines;
    if (tensor_args.grad.dtype() == tt::tt_metal::DataType::FLOAT32) {
        defines["GRAD_FP32"] = "1";
    }

    std::vector<uint32_t> reader_ct{Wt, Ht, slices, St, kBlock};
    tt::tt_metal::TensorAccessorArgs(logits.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(tensor_args.log_normalizer.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(tensor_args.grad.buffer()).append_to(reader_ct);
    auto reader = create_reader_kernel(program, all_cores, reader_ct, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct{Wt, Ht, slices, St, kBlock, target_page_bytes, target_read_bytes, scale_bits};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    tt::tt_metal::TensorAccessorArgs(tensor_args.target.buffer()).append_to(writer_ct);
    tt::tt_metal::TensorAccessorArgs(tensor_args.grad.buffer()).append_to(writer_ct);
    auto writer = create_writer_kernel(program, all_cores, writer_ct, defines, kWriterKernelPath);

    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        return create_compute_kernel(
            program,
            cores,
            {work_count, Wt, slices, St, kBlock, scale_bits},
            defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true);
    };
    const auto compute_1 = make_compute(core_group_1, work_1);
    const auto compute_2 = core_group_2.ranges().empty() ? compute_1 : make_compute(core_group_2, work_2);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0, work_start = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / grid.y, i % grid.y};
        const bool in_group_1 = core_group_1.contains(core);
        const uint32_t work_count = in_group_1 ? work_1 : work_2;
        SetRuntimeArgs(program, in_group_1 ? compute_1 : compute_2, core, {work_start});
        SetRuntimeArgs(
            program,
            reader,
            core,
            {logits.buffer()->address(),
             tensor_args.log_normalizer.buffer()->address(),
             tensor_args.grad.buffer()->address(),
             work_start,
             work_count});
        SetRuntimeArgs(
            program,
            writer,
            core,
            {output.buffer()->address(),
             tensor_args.target.buffer()->address(),
             tensor_args.grad.buffer()->address(),
             window.first_v,
             window.last_v,
             work_start,
             work_count});
        cores.push_back(core);
        work_start += work_count;
    }
    return Created{std::move(program), {reader, writer, std::move(cores)}};
}

}  // namespace

VpCrossEntropyBwProgramFactory::cached_mesh_workload_t VpCrossEntropyBwProgramFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto* mesh_device = tensor_args.logits.device();
    const auto mesh_shape = mesh_device->shape();
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            auto created = create_program(shard_window(args, coord, mesh_shape), args, tensor_args, output);
            ttnn::MeshCoordinateRange single{coord};
            workload.add_program(single, std::move(created.program));
            shared[single] = std::move(created.shared);
        }
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void VpCrossEntropyBwProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto mesh_shape = tensor_args.logits.device()->shape();
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(coord_range);
        const auto window = shard_window(args, coord_range.start_coord(), mesh_shape);
        auto& reader_args = GetRuntimeArgs(program, sv.reader_kernel_id);
        auto& writer_args = GetRuntimeArgs(program, sv.writer_kernel_id);
        for (const auto& core : sv.cores) {
            auto& r = reader_args[core.x][core.y];
            r[kReaderLogitsIdx] = tensor_args.logits.buffer()->address();
            r[kReaderLogZIdx] = tensor_args.log_normalizer.buffer()->address();
            r[kReaderGradIdx] = tensor_args.grad.buffer()->address();
            auto& w = writer_args[core.x][core.y];
            w[kWriterOutIdx] = output.buffer()->address();
            w[kWriterTargetIdx] = tensor_args.target.buffer()->address();
            w[kWriterGradIdx] = tensor_args.grad.buffer()->address();
            w[kWriterFirstVIdx] = window.first_v;
            w[kWriterFirstVIdx + 1U] = window.last_v;
        }
    }
}

}  // namespace ttml::metal::ops::vp_cross_entropy_bw::device
