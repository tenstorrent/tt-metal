// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_softmax_merge_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "metal/common/program_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_softmax_merge {

namespace {

constexpr auto kReaderPath =
    "tt-train/sources/ttml/metal/ops/ring_softmax_merge/device/kernels/dataflow/reader_ring_softmax_merge.cpp";
constexpr auto kWriterPath =
    "tt-train/sources/ttml/metal/ops/ring_softmax_merge/device/kernels/dataflow/writer_ring_softmax_merge.cpp";
constexpr auto kComputePath =
    "tt-train/sources/ttml/metal/ops/ring_softmax_merge/device/kernels/compute/ring_softmax_merge_kernel.cpp";

// Whether this chip runs at this step.
bool chip_runs(const operation_attributes_t& attrs, uint32_t device_ring_id) {
    if (attrs.zigzag) {
        return ops::zigzag_visitor_runs(
            device_ring_id, attrs.step, attrs.ring_size, attrs.ring_direction, attrs.visitor);
    }
    return ops::get_device_execution_info(
               device_ring_id, attrs.step, attrs.ring_size, attrs.mask_type, attrs.ring_direction)
        .first;
}

// One chip's program: the row tiles dealt over the grid, each core taking
// its rows one after another; three kernels, the addresses as runtime args.
std::pair<tt::tt_metal::Program, RingSoftmaxMergeSharedVariables> build_program(
    const tensor_args_t& t, tt::tt_metal::IDevice* device) {
    using namespace tt::tt_metal;
    Program program{};
    const auto shape = t.out_acc.padded_shape();
    const uint32_t Wt = shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t rows = shape[0] * shape[1] * Ht;  // row tiles over batch, heads, sequence

    const auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group_1, group_2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(grid, rows);

    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    // Inputs: the two lse tiles, the running output row, the step's output row.
    create_circular_buffer(program, all_cores, tt::CBIndex::c_0, tt::DataFormat::Float32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_1, tt::DataFormat::Float32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_2, tt::DataFormat::Float32, fp32_tile, 2U * Wt);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_3, tt::DataFormat::Float16_b, bf16_tile, 2U * Wt);
    // The weights (column 0) and the outputs: the new lse tile, the new output row.
    create_circular_buffer(program, all_cores, tt::CBIndex::c_4, tt::DataFormat::Float32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_5, tt::DataFormat::Float32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_6, tt::DataFormat::Float32, fp32_tile, 2U);
    create_circular_buffer(program, all_cores, tt::CBIndex::c_7, tt::DataFormat::Float32, fp32_tile, 2U * Wt);

    std::vector<uint32_t> reader_args = {Wt};
    for (const auto* tensor : {&t.lse_acc, &t.step_lse, &t.out_acc, &t.step_out}) {
        TensorAccessorArgs(*tensor->buffer()).append_to(reader_args);
    }
    std::vector<uint32_t> writer_args = {Wt};
    for (const auto* tensor : {&t.lse_acc, &t.out_acc}) {
        TensorAccessorArgs(*tensor->buffer()).append_to(writer_args);
    }
    RingSoftmaxMergeSharedVariables vars;
    vars.reader = create_reader_kernel(program, all_cores, reader_args, {}, kReaderPath);
    vars.writer = create_writer_kernel(program, all_cores, writer_args, {}, kWriterPath);
    // Every Float32 tile the compute kernel reads -- the two lse tiles (also
    // for their column broadcast: the 32-bit broadcast path requires it), the
    // running output, the two weights -- is unpacked straight to dest; through
    // the matmul source registers it would keep 19 of its 32 bits at every
    // merge, which cost the backward a factor of six on dQ.
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (const auto cb : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_4, tt::CBIndex::c_5}) {
        unpack_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    vars.compute = CreateKernel(
        program,
        kComputePath,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            // The statistics use five Float32 DST tiles; in half-sync mode only four exist.
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = unpack_mode,
            .math_approx_mode = false,
            .compile_args = {Wt}});

    uint32_t row = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core = {i / grid.y, i % grid.y};
        const uint32_t rows_here = group_1.contains(core) ? rows_per_core_1 : rows_per_core_2;
        SetRuntimeArgs(
            program,
            vars.reader,
            core,
            {t.lse_acc.buffer()->address(),
             t.step_lse.buffer()->address(),
             t.out_acc.buffer()->address(),
             t.step_out.buffer()->address(),
             rows_here,
             row});
        SetRuntimeArgs(
            program, vars.writer, core, {t.lse_acc.buffer()->address(), t.out_acc.buffer()->address(), rows_here, row});
        SetRuntimeArgs(program, vars.compute, core, {rows_here});
        vars.cores.push_back(core);
        row += rows_here;
    }
    return {std::move(program), std::move(vars)};
}

}  // namespace

RingSoftmaxMergeProgramFactory::cached_mesh_workload_t RingSoftmaxMergeProgramFactory::create_mesh_workload(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    auto* mesh_device = tensor_args.out_acc.device();
    TT_FATAL(mesh_device != nullptr, "The accumulators must be on a mesh device");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(attrs.ring_axis < mesh_shape.dims(), "Ring axis {} must be < mesh dimensions {}", attrs.ring_axis, mesh_shape.dims());

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;
    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        if (!chip_runs(attrs, mesh_coord[attrs.ring_axis])) {
            continue;
        }
        auto [program, vars] = build_program(tensor_args, mesh_device);
        ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
        mesh_workload.add_program(single_coord_range, std::move(program));
        shared_vars[single_coord_range] = std::move(vars);
    }
    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingSoftmaxMergeProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& t,
    tensor_return_value_t& /*tensor_return_value*/) {
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        const auto& vars = cached_workload.shared_variables.at(coord_range);
        auto& reader_args = GetRuntimeArgs(program, vars.reader);
        auto& writer_args = GetRuntimeArgs(program, vars.writer);
        for (const auto& core : vars.cores) {
            auto& r = reader_args[core.x][core.y];
            r[0] = t.lse_acc.buffer()->address();
            r[1] = t.step_lse.buffer()->address();
            r[2] = t.out_acc.buffer()->address();
            r[3] = t.step_out.buffer()->address();
            auto& w = writer_args[core.x][core.y];
            w[0] = t.lse_acc.buffer()->address();
            w[1] = t.out_acc.buffer()->address();
        }
    }
}

}  // namespace ttml::metal::ops::ring_softmax_merge
