// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_fw_program_factory.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_device_operation_types.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_program_factory.hpp"
#include "ring_sdpa_fw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_fw {

namespace {

// SDPAForwardInputs holds the mask and gate as const references, so the "absent" value they
// bind to must outlive the LaneBinding returned below (a temporary std::nullopt would dangle).
const std::optional<ttnn::Tensor> kNoTensor = std::nullopt;

// SDPA forward inputs and outputs of one device at one ring step: lane 0 is the op's
// primary tensors, lane 1 the second-lane tensors (RingLaneSelection::ZigZag only).
struct LaneBinding {
    ttml::metal::ops::sdpa_fw::device::tensor_args_t tensor_args;
    ttml::metal::ops::sdpa_fw::device::tensor_return_value_t return_value;
};

LaneBinding bind_lane(uint32_t lane, const tensor_args_t& tensor_args, tensor_return_value_t& tensor_return_value) {
    if (lane == 0U) {
        auto& output = std::get<0>(tensor_return_value);
        auto& intermediates = std::get<1>(tensor_return_value);
        return LaneBinding{
            .tensor_args =
                {.query = tensor_args.query,
                 .key = tensor_args.key,
                 .value = tensor_args.value,
                 .mask = kNoTensor,  // No explicit mask - SDPA kernel generates causal mask internally
                 .preallocated_intermediate = intermediates,
                 .preallocated_output = output,
                 .gate = kNoTensor},
            .return_value = {output, intermediates}};
    }
    TT_FATAL(tensor_args.lane1.has_value(), "Device was assigned lane 1 but no second-lane tensors were given");
    const auto& lane1 = tensor_args.lane1.value();
    return LaneBinding{
        .tensor_args =
            {.query = lane1.query,
             .key = lane1.key,
             .value = lane1.value,
             .mask = kNoTensor,
             .preallocated_intermediate = lane1.intermediates,
             .preallocated_output = lane1.output,
             .gate = kNoTensor},
        .return_value = {lane1.output, lane1.intermediates}};
}

}  // namespace

RingSDPAFwProgramFactory::cached_mesh_workload_t RingSDPAFwProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    namespace sdpa_fw = ttml::metal::ops::sdpa_fw;

    auto* mesh_device = tensor_args.query.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device");

    const auto mesh_shape = mesh_device->shape();
    const uint32_t ring_axis = operation_attributes.ring_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t step = operation_attributes.step;
    const auto mask_type = operation_attributes.mask_type;
    const auto ring_direction = operation_attributes.ring_direction;
    const auto lane_selection = operation_attributes.lane_selection;

    TT_FATAL(ring_axis < mesh_shape.dims(), "Ring axis {} must be < mesh dimensions {}", ring_axis, mesh_shape.dims());

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;

    // Iterate over all device coordinates in the mesh
    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        uint32_t device_ring_id = mesh_coord[ring_axis];

        // Check if this device should execute at this step, with which mask and on which lane
        const auto work =
            get_ring_step_work(device_ring_id, step, ring_size, mask_type, ring_direction, lane_selection);

        if (!work.execute) {
            continue;
        }

        sdpa_fw::device::operation_attributes_t sdpa_attrs{
            .return_intermediates = true, .mask_type = work.mask_type, .dropout_probability = 0.0F};
        auto binding = bind_lane(work.lane, tensor_args, tensor_return_value);

        // Create the program
        auto cached_program =
            sdpa_fw::device::SDPAForwardProgramFactory::create(sdpa_attrs, binding.tensor_args, binding.return_value);

        // Store SDPA shared variables for runtime argument override
        const auto& sdpa_vars = cached_program.shared_variables;
        shared_variables_t ring_vars{
            .sdpa_fw_reader_kernel = sdpa_vars.sdpa_fw_reader_kernel,
            .sdpa_fw_writer_kernel = sdpa_vars.sdpa_fw_writer_kernel,
            .sdpa_fw_kernel_group_1 = sdpa_vars.sdpa_fw_kernel_group_1,
            .sdpa_fw_kernel_group_2 = sdpa_vars.sdpa_fw_kernel_group_2,
            .core_group_1 = sdpa_vars.core_group_1,
            .core_group_2 = sdpa_vars.core_group_2,
            .num_cores = sdpa_vars.num_cores,
            .num_cores_y = sdpa_vars.num_cores_y};

        // Add program to mesh workload
        ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
        mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        shared_vars[single_coord_range] = std::move(ring_vars);
    }

    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingSDPAFwProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    namespace sdpa_fw = ttml::metal::ops::sdpa_fw::device;

    auto* mesh_device = tensor_args.query.device();
    const auto mesh_shape = mesh_device->shape();
    const uint32_t ring_axis = operation_attributes.ring_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t step = operation_attributes.step;
    const auto mask_type = operation_attributes.mask_type;
    const auto ring_direction = operation_attributes.ring_direction;
    const auto lane_selection = operation_attributes.lane_selection;

    // Iterate over cached programs and update runtime arguments
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coord_range);

        // Get the mesh coordinate for this program (single coord range)
        const auto& start_coord = coord_range.start_coord();

        // Determine this device's mask and lane (only executing devices have a cached program)
        uint32_t device_ring_id = start_coord[ring_axis];
        const auto work =
            get_ring_step_work(device_ring_id, step, ring_size, mask_type, ring_direction, lane_selection);
        TT_FATAL(work.execute, "Cached ring SDPA forward program found for a device that does not execute");

        sdpa_fw::operation_attributes_t sdpa_attrs{
            .return_intermediates = true, .mask_type = work.mask_type, .dropout_probability = 0.0F};
        auto binding = bind_lane(work.lane, tensor_args, tensor_return_value);

        // Convert our shared_variables to SDPA's shared_variables type
        sdpa_fw::SDPAForwardProgramFactory::shared_variables_t sdpa_shared_vars{
            .sdpa_fw_reader_kernel = shared_vars.sdpa_fw_reader_kernel,
            .sdpa_fw_writer_kernel = shared_vars.sdpa_fw_writer_kernel,
            .sdpa_fw_kernel_group_1 = shared_vars.sdpa_fw_kernel_group_1,
            .sdpa_fw_kernel_group_2 = shared_vars.sdpa_fw_kernel_group_2,
            .core_group_1 = shared_vars.core_group_1,
            .core_group_2 = shared_vars.core_group_2,
            .num_cores = shared_vars.num_cores,
            .num_cores_y = shared_vars.num_cores_y};

        // Create a proxy CachedProgram and call SDPA's override_runtime_arguments
        auto cached_program = sdpa_fw::SDPAForwardProgramFactory::cached_program_t::proxy(program, sdpa_shared_vars);

        sdpa_fw::SDPAForwardProgramFactory::override_runtime_arguments(
            cached_program, sdpa_attrs, binding.tensor_args, binding.return_value);
    }
}
}  // namespace ttml::metal::ops::ring_sdpa_fw
