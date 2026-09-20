// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_cyclic_sdpa_fw_program_factory.hpp"

#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ops/cyclic_sdpa_fw/device/cyclic_sdpa_fw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_fw {

namespace {

namespace cyclic = ttml::metal::ops::cyclic_sdpa_fw::device;

// Whether this chip runs at this step, and with which mask: the same
// decisions ring_ttnn_sdpa_fw makes, from the same helpers.
std::pair<bool, AttentionMaskType> chip_plan(const operation_attributes_t& attrs, uint32_t device_ring_id) {
    if (attrs.zigzag) {
        return {
            ops::zigzag_visitor_runs(device_ring_id, attrs.step, attrs.ring_size, attrs.ring_direction, attrs.visitor),
            attrs.mask_type};
    }
    return ops::get_device_execution_info(
        device_ring_id, attrs.step, attrs.ring_size, attrs.mask_type, attrs.ring_direction);
}

cyclic::operation_attributes_t cyclic_attrs(const operation_attributes_t& attrs, AttentionMaskType mask) {
    return cyclic::operation_attributes_t{.rows_per_block_tiles = attrs.rows_per_block_tiles, .mask_type = mask};
}

cyclic::tensor_args_t cyclic_tensors(const tensor_args_t& t, tensor_return_value_t& out) {
    return cyclic::tensor_args_t{
        .query = t.query,
        .key = t.key,
        .value = t.value,
        .preallocated_output = out[0],
        .preallocated_intermediates = out[1]};
}

}  // namespace

RingCyclicSdpaFwProgramFactory::cached_mesh_workload_t RingCyclicSdpaFwProgramFactory::create_mesh_workload(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto* mesh_device = tensor_args.query.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(
        attrs.ring_axis < mesh_shape.dims(), "Ring axis {} must be < mesh dimensions {}", attrs.ring_axis,
        mesh_shape.dims());

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;
    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        const auto [execute, mask] = chip_plan(attrs, mesh_coord[attrs.ring_axis]);
        if (!execute) {
            // No program at all: the driver pre-fills the intermediates with
            // -inf so this chip contributes nothing to the merge.
            continue;
        }
        auto c_attrs = cyclic_attrs(attrs, mask);
        auto c_tensors = cyclic_tensors(tensor_args, tensor_return_value);
        auto cached = cyclic::CyclicSDPAForwardProgramFactory::create(c_attrs, c_tensors, tensor_return_value);
        ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
        shared_vars[single_coord_range] = std::move(cached.shared_variables);
        mesh_workload.add_program(single_coord_range, std::move(cached.program));
    }
    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingCyclicSdpaFwProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared = cached_workload.shared_variables.at(coord_range);
        const auto [execute, mask] = chip_plan(attrs, coord_range.start_coord()[attrs.ring_axis]);
        (void)execute;  // only executing chips have a program
        auto c_attrs = cyclic_attrs(attrs, mask);
        auto c_tensors = cyclic_tensors(tensor_args, tensor_return_value);
        auto proxy = cyclic::CyclicSDPAForwardProgramFactory::cached_program_t::proxy(program, shared);
        cyclic::CyclicSDPAForwardProgramFactory::override_runtime_arguments(
            proxy, c_attrs, c_tensors, tensor_return_value);
    }
}

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_fw
