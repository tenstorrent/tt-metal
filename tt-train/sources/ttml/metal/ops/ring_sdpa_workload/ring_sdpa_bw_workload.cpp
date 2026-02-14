// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_workload.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation_types.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_q_device_operation_types.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_q_program_factory.hpp"
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

// ============== Backward Q Program Factory ==============

RingSDPABwQProgramFactory::cached_mesh_workload_t RingSDPABwQProgramFactory::create_mesh_workload(
    const RingSDPABwQParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const RingSDPABwQInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    namespace sdpa_q = ttml::metal::ops::sdpa_bw::device::q;

    const auto& grad_output = tensor_args.grad_output;
    const auto& attn_output = tensor_args.attn_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& intermediates = tensor_args.intermediates;
    auto& grad_query = tensor_return_value;

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

    // Get mesh buffers
    auto grad_output_mesh_buffer = grad_output.mesh_buffer();
    auto attn_output_mesh_buffer = attn_output.mesh_buffer();
    auto query_mesh_buffer = query.mesh_buffer();
    auto key_mesh_buffer = key.mesh_buffer();
    auto value_mesh_buffer = value.mesh_buffer();
    auto intermediates_mesh_buffer = intermediates.mesh_buffer();
    auto grad_query_mesh_buffer = grad_query.mesh_buffer();

    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        uint32_t device_ring_id = mesh_coord[ring_axis];

        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);

        if (!should_execute) {
            continue;
        }

        // Create DeviceStorage objects
        std::vector<tt::tt_metal::distributed::MeshCoordinate> single_coord_vec{mesh_coord};
        tt::tt_metal::DeviceStorage grad_output_storage(grad_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage attn_output_storage(attn_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage query_storage(query_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage key_storage(key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage value_storage(value_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage intermediates_storage(intermediates_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage grad_query_storage(grad_query_mesh_buffer, single_coord_vec);

        // Create TensorTopology
        ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements(mesh_shape.dims());
        for (size_t i = 0; i < mesh_shape.dims(); i++) {
            placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
        tt::tt_metal::TensorTopology tensor_topology{mesh_shape, placements, single_coord_vec};

        // Create single-device tensors
        auto grad_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_output_storage)), grad_output.tensor_spec(), tensor_topology);
        auto attn_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(attn_output_storage)), attn_output.tensor_spec(), tensor_topology);
        auto query_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(query_storage)), query.tensor_spec(), tensor_topology);
        auto key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(key_storage)), key.tensor_spec(), tensor_topology);
        auto value_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(value_storage)), value.tensor_spec(), tensor_topology);
        auto intermediates_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(intermediates_storage)), intermediates.tensor_spec(), tensor_topology);
        auto grad_query_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_query_storage)), grad_query.tensor_spec(), tensor_topology);

        // Create SDPA backward Q with mask_type (no explicit mask tensor needed)
        sdpa_q::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_q::tensor_args_t sdpa_tensor_args{
            .grad_output = grad_output_tensor,
            .attn_output = attn_output_tensor,
            .query = query_tensor,
            .key = key_tensor,
            .value = value_tensor,
            .attn_mask = std::nullopt,  // No explicit mask - using mask_type
            .intermediates = intermediates_tensor,
            .preallocated_grad_query = grad_query_tensor};

        sdpa_q::tensor_return_value_t sdpa_return_value{grad_query_tensor};

        auto cached_program =
            sdpa_bw::device::SDPABackwardQProgramFactory::create(sdpa_attrs, sdpa_tensor_args, sdpa_return_value);

        // Store SDPA shared variables for runtime argument override
        const auto& sdpa_vars = cached_program.shared_variables;
        shared_variables_t ring_vars{
            .sdpa_bw_q_reader_kernel = sdpa_vars.sdpa_bw_q_reader_kernel,
            .sdpa_bw_q_writer_kernel = sdpa_vars.sdpa_bw_q_writer_kernel,
            .sdpa_bw_q_kernel_group_1 = sdpa_vars.sdpa_bw_q_kernel_group_1,
            .sdpa_bw_q_kernel_group_2 = sdpa_vars.sdpa_bw_q_kernel_group_2,
            .core_group_1 = sdpa_vars.core_group_1,
            .core_group_2 = sdpa_vars.core_group_2,
            .num_cores = sdpa_vars.num_cores,
            .num_cores_y = sdpa_vars.num_cores_y};

        ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
        mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        shared_vars[single_coord_range] = std::move(ring_vars);
    }

    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingSDPABwQProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RingSDPABwQParams& operation_attributes,
    const RingSDPABwQInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    namespace sdpa_q = ttml::metal::ops::sdpa_bw::device::q;

    const auto& grad_output = tensor_args.grad_output;
    const auto& attn_output = tensor_args.attn_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& intermediates = tensor_args.intermediates;
    auto& grad_query = tensor_return_value;

    auto* mesh_device = query.device();
    const auto mesh_shape = mesh_device->shape();
    const uint32_t ring_axis = operation_attributes.ring_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t step = operation_attributes.step;
    const auto mask_type = operation_attributes.mask_type;
    const auto ring_direction = operation_attributes.ring_direction;

    // Get mesh buffers
    auto grad_output_mesh_buffer = grad_output.mesh_buffer();
    auto attn_output_mesh_buffer = attn_output.mesh_buffer();
    auto query_mesh_buffer = query.mesh_buffer();
    auto key_mesh_buffer = key.mesh_buffer();
    auto value_mesh_buffer = value.mesh_buffer();
    auto intermediates_mesh_buffer = intermediates.mesh_buffer();
    auto grad_query_mesh_buffer = grad_query.mesh_buffer();

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coord_range);
        const auto& start_coord = coord_range.start_coord();

        // Determine effective mask type for this device
        uint32_t device_ring_id = start_coord[ring_axis];
        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);
        (void)should_execute;  // Already filtered in create_mesh_workload

        // Create DeviceStorage objects for this coordinate
        std::vector<tt::tt_metal::distributed::MeshCoordinate> single_coord_vec{start_coord};
        tt::tt_metal::DeviceStorage grad_output_storage(grad_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage attn_output_storage(attn_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage query_storage(query_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage key_storage(key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage value_storage(value_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage intermediates_storage(intermediates_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage grad_query_storage(grad_query_mesh_buffer, single_coord_vec);

        // Create TensorTopology
        ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements(mesh_shape.dims());
        for (size_t i = 0; i < mesh_shape.dims(); i++) {
            placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
        tt::tt_metal::TensorTopology tensor_topology{mesh_shape, placements, single_coord_vec};

        // Create single-device tensors
        auto grad_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_output_storage)), grad_output.tensor_spec(), tensor_topology);
        auto attn_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(attn_output_storage)), attn_output.tensor_spec(), tensor_topology);
        auto query_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(query_storage)), query.tensor_spec(), tensor_topology);
        auto key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(key_storage)), key.tensor_spec(), tensor_topology);
        auto value_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(value_storage)), value.tensor_spec(), tensor_topology);
        auto intermediates_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(intermediates_storage)), intermediates.tensor_spec(), tensor_topology);
        auto grad_query_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_query_storage)), grad_query.tensor_spec(), tensor_topology);

        // Create SDPA attributes and tensor args
        sdpa_q::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_q::tensor_args_t sdpa_tensor_args{
            .grad_output = grad_output_tensor,
            .attn_output = attn_output_tensor,
            .query = query_tensor,
            .key = key_tensor,
            .value = value_tensor,
            .attn_mask = std::nullopt,
            .intermediates = intermediates_tensor,
            .preallocated_grad_query = grad_query_tensor};

        sdpa_q::tensor_return_value_t sdpa_return_value{grad_query_tensor};

        // Convert our shared_variables to SDPA's shared_variables type
        sdpa_bw::device::SDPABackwardQProgramFactory::shared_variables_t sdpa_shared_vars{
            .sdpa_bw_q_reader_kernel = shared_vars.sdpa_bw_q_reader_kernel,
            .sdpa_bw_q_writer_kernel = shared_vars.sdpa_bw_q_writer_kernel,
            .sdpa_bw_q_kernel_group_1 = shared_vars.sdpa_bw_q_kernel_group_1,
            .sdpa_bw_q_kernel_group_2 = shared_vars.sdpa_bw_q_kernel_group_2,
            .core_group_1 = shared_vars.core_group_1,
            .core_group_2 = shared_vars.core_group_2,
            .num_cores = shared_vars.num_cores,
            .num_cores_y = shared_vars.num_cores_y};

        // Create a proxy CachedProgram and call SDPA's override_runtime_arguments
        auto cached_program =
            sdpa_bw::device::SDPABackwardQProgramFactory::cached_program_t::proxy(program, sdpa_shared_vars);

        sdpa_bw::device::SDPABackwardQProgramFactory::override_runtime_arguments(
            cached_program, sdpa_attrs, sdpa_tensor_args, sdpa_return_value);
    }
}

tt::tt_metal::distributed::MeshWorkload create_ring_sdpa_bw_q_workload(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    ttnn::Tensor& grad_query,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type) {
    RingSDPABwQParams params{.ring_size = ring_size, .ring_axis = ring_axis, .step = step, .mask_type = mask_type};

    RingSDPABwQInputs inputs{
        .grad_output = grad_output,
        .attn_output = attn_output,
        .query = query,
        .key = key,
        .value = value,
        .intermediates = intermediates,
        .preallocated_grad_query = grad_query};

    auto* mesh_device = query.device();
    const auto mesh_shape = mesh_device->shape();
    ttnn::MeshCoordinateRange full_range(mesh_shape);
    ttnn::MeshCoordinateRangeSet tensor_coords{full_range};

    auto cached_workload = RingSDPABwQProgramFactory::create_mesh_workload(params, tensor_coords, inputs, grad_query);

    return std::move(cached_workload.workload);
}

// ============== Backward KV Program Factory ==============

RingSDPABwKVProgramFactory::cached_mesh_workload_t RingSDPABwKVProgramFactory::create_mesh_workload(
    const RingSDPABwKVParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const RingSDPABwKVInputs& tensor_args,
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

    // Get mesh buffers
    auto grad_output_mesh_buffer = grad_output.mesh_buffer();
    auto attn_output_mesh_buffer = attn_output.mesh_buffer();
    auto query_mesh_buffer = query.mesh_buffer();
    auto key_mesh_buffer = key.mesh_buffer();
    auto value_mesh_buffer = value.mesh_buffer();
    auto intermediates_mesh_buffer = intermediates.mesh_buffer();
    auto grad_key_mesh_buffer = grad_key.mesh_buffer();
    auto grad_value_mesh_buffer = grad_value.mesh_buffer();

    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        uint32_t device_ring_id = mesh_coord[ring_axis];

        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);

        if (!should_execute) {
            continue;
        }

        // Create DeviceStorage objects
        std::vector<tt::tt_metal::distributed::MeshCoordinate> single_coord_vec{mesh_coord};
        tt::tt_metal::DeviceStorage grad_output_storage(grad_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage attn_output_storage(attn_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage query_storage(query_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage key_storage(key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage value_storage(value_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage intermediates_storage(intermediates_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage grad_key_storage(grad_key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage grad_value_storage(grad_value_mesh_buffer, single_coord_vec);

        // Create TensorTopology
        ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements(mesh_shape.dims());
        for (size_t i = 0; i < mesh_shape.dims(); i++) {
            placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
        tt::tt_metal::TensorTopology tensor_topology{mesh_shape, placements, single_coord_vec};

        // Create single-device tensors
        auto grad_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_output_storage)), grad_output.tensor_spec(), tensor_topology);
        auto attn_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(attn_output_storage)), attn_output.tensor_spec(), tensor_topology);
        auto query_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(query_storage)), query.tensor_spec(), tensor_topology);
        auto key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(key_storage)), key.tensor_spec(), tensor_topology);
        auto value_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(value_storage)), value.tensor_spec(), tensor_topology);
        auto intermediates_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(intermediates_storage)), intermediates.tensor_spec(), tensor_topology);
        auto grad_key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(grad_key_storage)), grad_key.tensor_spec(), tensor_topology);
        auto grad_value_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_value_storage)), grad_value.tensor_spec(), tensor_topology);

        // Create SDPA backward KV with mask_type (no explicit mask tensor needed)
        sdpa_kv::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_kv::tensor_args_t sdpa_tensor_args{
            .grad_output = grad_output_tensor,
            .attn_output = attn_output_tensor,
            .query = query_tensor,
            .key = key_tensor,
            .value = value_tensor,
            .attn_mask = std::nullopt,  // No explicit mask - using mask_type
            .intermediates = intermediates_tensor,
            .preallocated_grad_key = grad_key_tensor,
            .preallocated_grad_value = grad_value_tensor};

        sdpa_kv::tensor_return_value_t sdpa_return_value{grad_key_tensor, grad_value_tensor};

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
    const RingSDPABwKVParams& operation_attributes,
    const RingSDPABwKVInputs& tensor_args,
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

    // Get mesh buffers
    auto grad_output_mesh_buffer = grad_output.mesh_buffer();
    auto attn_output_mesh_buffer = attn_output.mesh_buffer();
    auto query_mesh_buffer = query.mesh_buffer();
    auto key_mesh_buffer = key.mesh_buffer();
    auto value_mesh_buffer = value.mesh_buffer();
    auto intermediates_mesh_buffer = intermediates.mesh_buffer();
    auto grad_key_mesh_buffer = grad_key.mesh_buffer();
    auto grad_value_mesh_buffer = grad_value.mesh_buffer();

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coord_range);
        const auto& start_coord = coord_range.start_coord();

        // Determine effective mask type for this device
        uint32_t device_ring_id = start_coord[ring_axis];
        auto [should_execute, effective_mask_type] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);

        // Create DeviceStorage objects for this coordinate
        std::vector<tt::tt_metal::distributed::MeshCoordinate> single_coord_vec{start_coord};
        tt::tt_metal::DeviceStorage grad_output_storage(grad_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage attn_output_storage(attn_output_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage query_storage(query_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage key_storage(key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage value_storage(value_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage intermediates_storage(intermediates_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage grad_key_storage(grad_key_mesh_buffer, single_coord_vec);
        tt::tt_metal::DeviceStorage grad_value_storage(grad_value_mesh_buffer, single_coord_vec);

        // Create TensorTopology
        ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements(mesh_shape.dims());
        for (size_t i = 0; i < mesh_shape.dims(); i++) {
            placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
        tt::tt_metal::TensorTopology tensor_topology{mesh_shape, placements, single_coord_vec};

        // Create single-device tensors
        auto grad_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_output_storage)), grad_output.tensor_spec(), tensor_topology);
        auto attn_output_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(attn_output_storage)), attn_output.tensor_spec(), tensor_topology);
        auto query_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(query_storage)), query.tensor_spec(), tensor_topology);
        auto key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(key_storage)), key.tensor_spec(), tensor_topology);
        auto value_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(value_storage)), value.tensor_spec(), tensor_topology);
        auto intermediates_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(intermediates_storage)), intermediates.tensor_spec(), tensor_topology);
        auto grad_key_tensor =
            ttnn::Tensor(tt::tt_metal::Storage(std::move(grad_key_storage)), grad_key.tensor_spec(), tensor_topology);
        auto grad_value_tensor = ttnn::Tensor(
            tt::tt_metal::Storage(std::move(grad_value_storage)), grad_value.tensor_spec(), tensor_topology);

        // Create SDPA attributes and tensor args
        sdpa_kv::operation_attributes_t sdpa_attrs{.mask_type = effective_mask_type, .dropout_probability = 0.0F};

        sdpa_kv::tensor_args_t sdpa_tensor_args{
            .grad_output = grad_output_tensor,
            .attn_output = attn_output_tensor,
            .query = query_tensor,
            .key = key_tensor,
            .value = value_tensor,
            .attn_mask = std::nullopt,
            .intermediates = intermediates_tensor,
            .preallocated_grad_key = grad_key_tensor,
            .preallocated_grad_value = grad_value_tensor};

        sdpa_kv::tensor_return_value_t sdpa_return_value{grad_key_tensor, grad_value_tensor};

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

tt::tt_metal::distributed::MeshWorkload create_ring_sdpa_bw_kv_workload(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    ttnn::Tensor& grad_key,
    ttnn::Tensor& grad_value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type) {
    RingSDPABwKVParams params{.ring_size = ring_size, .ring_axis = ring_axis, .step = step, .mask_type = mask_type};

    RingSDPABwKVInputs inputs{
        .grad_output = grad_output,
        .attn_output = attn_output,
        .query = query,
        .key = key,
        .value = value,
        .intermediates = intermediates,
        .preallocated_grad_key = grad_key,
        .preallocated_grad_value = grad_value};

    std::tuple<ttnn::Tensor, ttnn::Tensor> return_value{grad_key, grad_value};

    auto* mesh_device = query.device();
    const auto mesh_shape = mesh_device->shape();
    ttnn::MeshCoordinateRange full_range(mesh_shape);
    ttnn::MeshCoordinateRangeSet tensor_coords{full_range};

    auto cached_workload =
        RingSDPABwKVProgramFactory::create_mesh_workload(params, tensor_coords, inputs, return_value);

    return std::move(cached_workload.workload);
}

}  // namespace ttml::metal::ops::ring_sdpa
