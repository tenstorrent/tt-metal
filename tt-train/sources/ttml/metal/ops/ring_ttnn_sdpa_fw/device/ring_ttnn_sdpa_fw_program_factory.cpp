// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_ttnn_sdpa_fw_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "ring_ttnn_sdpa_fw_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttml::metal::ops::ring_ttnn_sdpa_fw {

namespace {

// Whether this chip runs at this step, and with which mask.
std::pair<bool, AttentionMaskType> chip_plan(const operation_attributes_t& attrs, uint32_t device_ring_id) {
    if (attrs.zigzag) {
        return {
            ops::zigzag_visitor_runs(
                device_ring_id, attrs.step, attrs.ring_size, attrs.ring_direction, attrs.visitor),
            attrs.mask_type};
    }
    return ops::get_device_execution_info(
        device_ring_id, attrs.step, attrs.ring_size, attrs.mask_type, attrs.ring_direction);
}

// ttnn's SDPA descriptor for this launch: Float32 accumulation (the lse
// needs the standard compute path), the default HiFi2 fidelity, causal on the
// diagonal step, both outputs preallocated by the driver.
tt::tt_metal::ProgramDescriptor sdpa_descriptor(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    AttentionMaskType mask) {
    auto& output = std::get<0>(tensor_return_value);
    auto& intermediates = std::get<1>(tensor_return_value);
    auto* device = tensor_args.query.device();
    const uint32_t rows = static_cast<uint32_t>(tensor_args.query.padded_shape()[2]);
    const uint32_t chunk = std::max(32U, std::min(attrs.chunk_size, rows) / 32U * 32U);
    ttnn::operations::transformer::SDPAProgramConfig program_config{
        .compute_with_storage_grid_size = device->compute_with_storage_grid_size(),
        .sub_core_grids = std::nullopt,
        .q_chunk_size = chunk,
        .k_chunk_size = chunk,
        .exp_approx_mode = std::nullopt};
    const auto kernel_config = ttnn::init_device_compute_kernel_config(
        device->arch(),
        std::nullopt,
        tt::tt_metal::MathFidelity::HiFi2,
        /* math_approx_mode */ true,
        /* fp32_dest_acc_en */ true,
        /* packer_l1_acc */ false);
    ttnn::prim::SDPAParams sdpa_attrs{
        .scale = std::nullopt,
        .output_mem_config = output.memory_config(),
        .program_config = program_config,
        .is_causal = mask == AttentionMaskType::Causal,
        .compute_kernel_config = kernel_config,
        .return_lse = true};
    ttnn::prim::SDPAInputs sdpa_inputs{
        .q = tensor_args.query,
        .k = tensor_args.key,
        .v = tensor_args.value,
        .optional_output_tensor = output,
        .optional_lse_tensor = intermediates};
    std::vector<ttnn::Tensor> outputs{output, intermediates};
    return ttnn::prim::SDPAOperation::SDPAProgramFactory::create_descriptor(sdpa_attrs, sdpa_inputs, outputs);
}

}  // namespace

RingTtnnSdpaFwProgramFactory::cached_mesh_workload_t RingTtnnSdpaFwProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto* mesh_device = tensor_args.query.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(
        operation_attributes.ring_axis < mesh_shape.dims(),
        "Ring axis {} must be < mesh dimensions {}",
        operation_attributes.ring_axis,
        mesh_shape.dims());

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;
    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        const auto [execute, mask] = chip_plan(operation_attributes, mesh_coord[operation_attributes.ring_axis]);
        if (!execute) {
            // No program at all: the driver pre-fills the intermediates with
            // -inf so this chip contributes nothing to the combine.
            continue;
        }
        const auto descriptor = sdpa_descriptor(operation_attributes, tensor_args, tensor_return_value, mask);
        tt::tt_metal::Program program(descriptor);
        ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
        mesh_workload.add_program(single_coord_range, std::move(program));
        shared_vars[single_coord_range] = shared_variables_t{};
    }
    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingTtnnSdpaFwProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        const auto [execute, mask] =
            chip_plan(operation_attributes, coord_range.start_coord()[operation_attributes.ring_axis]);
        (void)execute;  // only executing chips have a program
        const auto descriptor = sdpa_descriptor(operation_attributes, tensor_args, tensor_return_value, mask);
        tt::tt_metal::apply_descriptor_runtime_args(program, descriptor);
    }
}

}  // namespace ttml::metal::ops::ring_ttnn_sdpa_fw
