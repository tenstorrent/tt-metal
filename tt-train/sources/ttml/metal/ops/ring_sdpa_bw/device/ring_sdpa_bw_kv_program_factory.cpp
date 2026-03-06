// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_kv_program_factory.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation_types.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.hpp"
#include "ring_sdpa_bw_factory_utils.hpp"
#include "ring_sdpa_bw_kv_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {
// ============== Backward KV Program Factory ==============

RingSDPABwKVProgramFactory::cached_mesh_workload_t RingSDPABwKVProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    namespace sdpa_kv = ttml::metal::ops::sdpa_bw::device::kv;

    const auto& grad_output = tensor_args.grad_output;
    const auto& attn_output = tensor_args.attn_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& intermediates = tensor_args.intermediates;
    auto& grad_key = std::get<0>(tensor_return_value);
    auto& grad_value = std::get<1>(tensor_return_value);

    auto* mesh_device = query.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device");

    const auto mesh_shape = mesh_device->shape();
    const uint32_t ring_axis = operation_attributes.ring_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t step = operation_attributes.step;
    const auto mask_type = operation_attributes.mask_type;
    const auto ring_direction = operation_attributes.ring_direction;

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        uint32_t device_ring_id = mesh_coord[ring_axis];

        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);

        if (!should_execute) {
            continue;
        }

        // Create SDPA backward KV with mask_type (no explicit mask tensor needed)
        sdpa_kv::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_kv::tensor_args_t sdpa_tensor_args{
            .grad_output = grad_output,
            .attn_output = attn_output,
            .query = query,
            .key = key,
            .value = value,
            .attn_mask = std::nullopt,  // No explicit mask - using mask_type
            .intermediates = intermediates,
            .preallocated_grad_key = grad_key,
            .preallocated_grad_value = grad_value};

        sdpa_kv::tensor_return_value_t sdpa_return_value{grad_key, grad_value};

        auto cached_program =
            sdpa_bw::device::SDPABackwardKVProgramFactory::create(sdpa_attrs, sdpa_tensor_args, sdpa_return_value);

        // Store SDPA shared variables for runtime argument override
        const auto& sdpa_vars = cached_program.shared_variables;
        shared_variables_t ring_vars{
            .sdpa_bw_reader_kernel = sdpa_vars.sdpa_bw_reader_kernel,
            .sdpa_bw_writer_kernel = sdpa_vars.sdpa_bw_writer_kernel,
            .sdpa_bw_kernel_group_1 = sdpa_vars.sdpa_bw_kernel_group_1,
            .sdpa_bw_kernel_group_2 = sdpa_vars.sdpa_bw_kernel_group_2,
            .core_group_1 = sdpa_vars.core_group_1,
            .core_group_2 = sdpa_vars.core_group_2,
            .num_cores = sdpa_vars.num_cores,
            .num_cores_y = sdpa_vars.num_cores_y};

        ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
        mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        shared_vars[single_coord_range] = std::move(ring_vars);
    }

    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingSDPABwKVProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    namespace sdpa_kv = ttml::metal::ops::sdpa_bw::device::kv;

    const auto& grad_output = tensor_args.grad_output;
    const auto& attn_output = tensor_args.attn_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& intermediates = tensor_args.intermediates;
    auto& grad_key = std::get<0>(tensor_return_value);
    auto& grad_value = std::get<1>(tensor_return_value);

    auto* mesh_device = query.device();
    const auto mesh_shape = mesh_device->shape();
    const uint32_t ring_axis = operation_attributes.ring_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t step = operation_attributes.step;
    const auto mask_type = operation_attributes.mask_type;
    const auto ring_direction = operation_attributes.ring_direction;

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coord_range);
        const auto& start_coord = coord_range.start_coord();

        // Determine effective mask type for this device
        uint32_t device_ring_id = start_coord[ring_axis];
        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);

        // Create SDPA attributes and tensor args
        sdpa_kv::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_kv::tensor_args_t sdpa_tensor_args{
            .grad_output = grad_output,
            .attn_output = attn_output,
            .query = query,
            .key = key,
            .value = value,
            .attn_mask = std::nullopt,
            .intermediates = intermediates,
            .preallocated_grad_key = grad_key,
            .preallocated_grad_value = grad_value};

        sdpa_kv::tensor_return_value_t sdpa_return_value{grad_key, grad_value};

        // Convert our shared_variables to SDPA's shared_variables type
        sdpa_bw::device::SDPABackwardKVProgramFactory::shared_variables_t sdpa_shared_vars{
            .sdpa_bw_reader_kernel = shared_vars.sdpa_bw_reader_kernel,
            .sdpa_bw_writer_kernel = shared_vars.sdpa_bw_writer_kernel,
            .sdpa_bw_kernel_group_1 = shared_vars.sdpa_bw_kernel_group_1,
            .sdpa_bw_kernel_group_2 = shared_vars.sdpa_bw_kernel_group_2,
            .core_group_1 = shared_vars.core_group_1,
            .core_group_2 = shared_vars.core_group_2,
            .num_cores = shared_vars.num_cores,
            .num_cores_y = shared_vars.num_cores_y};

        // Create a proxy CachedProgram and call SDPA's override_runtime_arguments
        auto cached_program =
            sdpa_bw::device::SDPABackwardKVProgramFactory::cached_program_t::proxy(program, sdpa_shared_vars);

        sdpa_bw::device::SDPABackwardKVProgramFactory::override_runtime_arguments(
            cached_program, sdpa_attrs, sdpa_tensor_args, sdpa_return_value);
    }
}

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv
