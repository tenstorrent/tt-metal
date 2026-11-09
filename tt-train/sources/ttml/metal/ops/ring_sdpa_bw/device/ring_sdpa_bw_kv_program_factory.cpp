// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_kv_program_factory.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation_types.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.hpp"
#include "ring_sdpa_bw_kv_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {

namespace {

// SDPA backward-KV inputs of one device at one ring step that depend on which query it uses:
// the op's primary ones, or (RingSchedule::Striped, src > d) the shifted ones. K/V, u_scaler and
// the dK/dV outputs are shared (dK/dV are in the K/V source's coordinates either way).
struct KVDeviceTensors {
    const ttnn::Tensor& grad_output;
    const ttnn::Tensor& query;
    const ttnn::Tensor& intermediates;
};

KVDeviceTensors select_device_tensors(bool shifted_query, const tensor_args_t& tensor_args) {
    if (!shifted_query) {
        return {tensor_args.grad_output, tensor_args.query, tensor_args.intermediates};
    }
    TT_FATAL(tensor_args.shifted.has_value(), "Device was assigned the shifted query but none was given");
    const auto& shifted = tensor_args.shifted.value();
    return {shifted.grad_output, shifted.query, shifted.intermediates};
}

}  // namespace
// ============== Backward KV Program Factory ==============

RingSDPABwKVProgramFactory::cached_mesh_workload_t RingSDPABwKVProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    namespace sdpa_kv = ttml::metal::ops::sdpa_bw::device::kv;

    const auto& u_scaler = tensor_args.u_scaler;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
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

        const auto work = get_ring_step_work(
            device_ring_id, step, ring_size, mask_type, ring_direction, operation_attributes.schedule);
        const auto effective_mask_type = work.mask_type;

        if (!work.execute) {
            continue;
        }
        const auto io = select_device_tensors(work.shifted_query, tensor_args);

        sdpa_kv::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_kv::tensor_args_t sdpa_tensor_args{
            .grad_output = io.grad_output,
            .query = io.query,
            .key = key,
            .value = value,
            .attn_mask = std::nullopt,
            .intermediates = io.intermediates,
            .u_scaler = u_scaler,
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

    const auto& u_scaler = tensor_args.u_scaler;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
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

        uint32_t device_ring_id = start_coord[ring_axis];
        const auto work = get_ring_step_work(
            device_ring_id, step, ring_size, mask_type, ring_direction, operation_attributes.schedule);
        const auto effective_mask_type = work.mask_type;
        TT_FATAL(work.execute, "Cached ring SDPA backward program found for a device that does not execute");
        const auto io = select_device_tensors(work.shifted_query, tensor_args);

        sdpa_kv::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_kv::tensor_args_t sdpa_tensor_args{
            .grad_output = io.grad_output,
            .query = io.query,
            .key = key,
            .value = value,
            .attn_mask = std::nullopt,
            .intermediates = io.intermediates,
            .u_scaler = u_scaler,
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
