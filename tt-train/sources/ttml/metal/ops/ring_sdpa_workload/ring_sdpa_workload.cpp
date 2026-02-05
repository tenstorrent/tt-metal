// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_workload.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

#include "metal/ops/sdpa_fw/device/sdpa_fw_device_operation_types.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_program_factory.hpp"
#include "ring_sdpa_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa {

using ttml::metal::AttentionMaskType;

namespace {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// Determine if device should execute at this step and which mask type to use
// Returns: (should_execute, mask_type_to_use)
std::pair<bool, AttentionMaskType> get_device_execution_info(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    AttentionMaskType mask_type,
    RingDirection ring_direction) {
    if (mask_type != AttentionMaskType::Causal) {
        // Non-causal: all devices execute with no mask (full attention)
        return {true, AttentionMaskType::None};
    }

    // Causal masking logic for ring attention:
    // At step s, device d processes K/V from source device based on ring direction
    // - Backward direction: src = (d + s) % ring_size
    // - Forward direction: src = (d - s + ring_size) % ring_size
    // Then apply causal logic:
    // - If src == device_id: diagonal chunk, use causal mask
    // - If src < device_id: earlier chunk, use full attention (no mask)
    // - If src > device_id: later chunk, skip (all positions masked)
    uint32_t src_device;
    if (ring_direction == RingDirection::Backward) {
        src_device = (device_ring_id + step) % ring_size;
    } else {
        src_device = (device_ring_id - step + ring_size) % ring_size;
    }

    if (src_device == device_ring_id) {
        return {true, AttentionMaskType::Causal};  // Diagonal: use causal mask
    } else if (src_device < device_ring_id) {
        return {true, AttentionMaskType::None};  // Earlier: full attention (no mask)
    } else {
        return {false, AttentionMaskType::None};  // Later: skip
    }
}

}  // namespace

RingSDPAProgramFactory::cached_mesh_workload_t RingSDPAProgramFactory::create_mesh_workload(
    const RingSDPAParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const RingSDPAInputs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    using namespace ttml::metal::ops::sdpa_fw::device;

    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    auto& output = std::get<0>(tensor_return_value);
    auto& intermediates = std::get<1>(tensor_return_value);

    auto* mesh_device = query.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device");

    const auto mesh_shape = mesh_device->shape();
    const uint32_t ring_axis = operation_attributes.ring_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t step = operation_attributes.step;
    const auto mask_type = operation_attributes.mask_type;
    const auto ring_direction = operation_attributes.ring_direction;

    TT_FATAL(ring_axis < mesh_shape.dims(), "Ring axis {} must be < mesh dimensions {}", ring_axis, mesh_shape.dims());

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;

    // Get mesh buffers
    auto query_mesh_buffer = query.mesh_buffer();
    auto key_mesh_buffer = key.mesh_buffer();
    auto value_mesh_buffer = value.mesh_buffer();
    auto output_mesh_buffer = output.mesh_buffer();
    auto intermediates_mesh_buffer = intermediates.mesh_buffer();

    // Iterate over all device coordinates in the mesh
    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        uint32_t device_ring_id = mesh_coord[ring_axis];

        // Check if this device should execute at this step and which mask to use
        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);

        if (!should_execute) {
            continue;
        }

        // Create DeviceStorage objects for this coordinate
        std::vector<tt::tt_metal::distributed::MeshCoordinate> single_coord_vec{mesh_coord};
        tt::tt_metal::DeviceStorage query_storage(query_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage key_storage(key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage value_storage(value_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage output_storage(output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage intermediates_storage(intermediates_mesh_buffer, single_coord_vec);

        // Create TensorTopology for single device
        ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements(mesh_shape.dims());
        for (size_t i = 0; i < mesh_shape.dims(); i++) {
            placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
        tt::tt_metal::TensorTopology tensor_topology{mesh_shape, placements, single_coord_vec};

        // Create single-device tensors
        auto query_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(query_storage)), query.tensor_spec(), tensor_topology);
        auto key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(key_storage)), key.tensor_spec(), tensor_topology);
        auto value_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(value_storage)), value.tensor_spec(), tensor_topology);
        auto output_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(output_storage)), output.tensor_spec(), tensor_topology);
        auto intermediates_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(intermediates_storage)), intermediates.tensor_spec(), tensor_topology);

        // Create SDPA forward operation with the effective mask type
        // No explicit mask tensor needed - SDPA kernel generates causal mask internally
        operation_attributes_t sdpa_attrs{
            .return_intermediates = true, .mask_type = effective_mask_type, .dropout_probability = 0.0F};

        tensor_args_t sdpa_tensor_args{
            .query = query_tensor,
            .key = key_tensor,
            .value = value_tensor,
            .mask = std::nullopt,  // No explicit mask - use mask_type
            .preallocated_intermediate = intermediates_tensor,
            .preallocated_output = output_tensor};

        tensor_return_value_t sdpa_return_value{output_tensor, intermediates_tensor};

        // Create the program
        auto cached_program = SDPAForwardProgramFactory::create(sdpa_attrs, sdpa_tensor_args, sdpa_return_value);

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

void RingSDPAProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RingSDPAParams& operation_attributes,
    const RingSDPAInputs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    namespace sdpa_fw = ttml::metal::ops::sdpa_fw::device;

    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    auto& output = std::get<0>(tensor_return_value);
    auto& intermediates = std::get<1>(tensor_return_value);

    auto* mesh_device = query.device();
    const auto mesh_shape = mesh_device->shape();
    const uint32_t ring_axis = operation_attributes.ring_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t step = operation_attributes.step;
    const auto mask_type = operation_attributes.mask_type;
    const auto ring_direction = operation_attributes.ring_direction;

    // Get mesh buffers
    auto query_mesh_buffer = query.mesh_buffer();
    auto key_mesh_buffer = key.mesh_buffer();
    auto value_mesh_buffer = value.mesh_buffer();
    auto output_mesh_buffer = output.mesh_buffer();
    auto intermediates_mesh_buffer = intermediates.mesh_buffer();

    // Iterate over cached programs and update runtime arguments
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coord_range);

        // Get the mesh coordinate for this program (single coord range)
        const auto& start_coord = coord_range.start_coord();

        // Determine effective mask type for this device
        uint32_t device_ring_id = start_coord[ring_axis];
        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);
        (void)should_execute;  // Already filtered in create_mesh_workload

        // Create DeviceStorage objects for this coordinate
        std::vector<tt::tt_metal::distributed::MeshCoordinate> single_coord_vec{start_coord};
        tt::tt_metal::DeviceStorage query_storage(query_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage key_storage(key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage value_storage(value_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage output_storage(output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage intermediates_storage(intermediates_mesh_buffer, single_coord_vec);

        // Create TensorTopology
        ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements(mesh_shape.dims());
        for (size_t i = 0; i < mesh_shape.dims(); i++) {
            placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
        tt::tt_metal::TensorTopology tensor_topology{mesh_shape, placements, single_coord_vec};

        // Create single-device tensors
        auto query_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(query_storage)), query.tensor_spec(), tensor_topology);
        auto key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(key_storage)), key.tensor_spec(), tensor_topology);
        auto value_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(value_storage)), value.tensor_spec(), tensor_topology);
        auto output_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(output_storage)), output.tensor_spec(), tensor_topology);
        auto intermediates_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(intermediates_storage)), intermediates.tensor_spec(), tensor_topology);

        // Create SDPA attributes and tensor args
        std::optional<ttnn::Tensor> mask_opt = std::nullopt;
        sdpa_fw::operation_attributes_t sdpa_attrs{
            .return_intermediates = true, .mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_fw::tensor_args_t sdpa_tensor_args{
            .query = query_tensor,
            .key = key_tensor,
            .value = value_tensor,
            .mask = mask_opt,
            .preallocated_intermediate = intermediates_tensor,
            .preallocated_output = output_tensor};

        sdpa_fw::tensor_return_value_t sdpa_return_value{output_tensor, intermediates_tensor};

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
            cached_program, sdpa_attrs, sdpa_tensor_args, sdpa_return_value);
    }
}

tt::tt_metal::distributed::MeshWorkload create_ring_sdpa_workload(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    ttnn::Tensor& output,
    ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type) {
    RingSDPAParams params{.ring_size = ring_size, .ring_axis = ring_axis, .step = step, .mask_type = mask_type};

    RingSDPAInputs inputs{
        .query = query,
        .key = key,
        .value = value,
        .preallocated_output = output,
        .preallocated_intermediates = intermediates};

    std::tuple<ttnn::Tensor, ttnn::Tensor> return_value{output, intermediates};

    auto* mesh_device = query.device();
    const auto mesh_shape = mesh_device->shape();
    ttnn::MeshCoordinateRange full_range(mesh_shape);
    ttnn::MeshCoordinateRangeSet tensor_coords{full_range};

    auto cached_workload = RingSDPAProgramFactory::create_mesh_workload(params, tensor_coords, inputs, return_value);

    return std::move(cached_workload.workload);
}

}  // namespace ttml::metal::ops::ring_sdpa
