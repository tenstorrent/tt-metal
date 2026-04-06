// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_fw_device_operation.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttml::metal::ops::ring_sdpa_fw {

using namespace tt::tt_metal;
using namespace ttnn;

void RingSDPAFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.query.device() != nullptr, "Query tensor must be on device");
    TT_FATAL(tensor_args.key.device() != nullptr, "Key tensor must be on device");
    TT_FATAL(tensor_args.value.device() != nullptr, "Value tensor must be on device");

    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step must be < ring_size");
}

RingSDPAFwDeviceOperation::spec_return_value_t RingSDPAFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    // Handle output spec
    ttnn::TensorSpec output_spec =
        tensor_args.preallocated_output.has_value()
            ? tensor_args.preallocated_output->tensor_spec()
            : ttnn::TensorSpec(
                  tensor_args.query.logical_shape(),
                  tt::tt_metal::TensorLayout(
                      tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));

    // Handle intermediates spec - shape is (B, H, S, 64)
    auto query_shape = tensor_args.query.logical_shape();
    auto [batch, heads, seq_len, dim] = query_shape.to_array_4D();
    ttnn::TensorSpec intermediates_spec =
        tensor_args.preallocated_intermediates.has_value()
            ? tensor_args.preallocated_intermediates->tensor_spec()
            : ttnn::TensorSpec(
                  ttnn::Shape{batch, heads, seq_len, 64U},
                  tt::tt_metal::TensorLayout(
                      tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));

    return {output_spec, intermediates_spec};
}

RingSDPAFwDeviceOperation::tensor_return_value_t RingSDPAFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [output_spec, intermediates_spec] = compute_output_specs(attrs, tensor_args);

    // Handle output
    ttnn::Tensor output = tensor_args.preallocated_output.has_value()
                              ? tensor_args.preallocated_output.value()
                              : create_device_tensor(output_spec, tensor_args.query.device());

    // Handle intermediates
    ttnn::Tensor intermediates = tensor_args.preallocated_intermediates.has_value()
                                     ? tensor_args.preallocated_intermediates.value()
                                     : create_device_tensor(intermediates_spec, tensor_args.query.device());

    return {output, intermediates};
}

ttsl::hash::hash_t RingSDPAFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Hash based on operation configuration - buffer addresses are updated via override_runtime_arguments
    return ttsl::hash::hash_objects(
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        attrs.mask_type,
        static_cast<int>(attrs.ring_direction),
        tensor_args.query.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.key.logical_shape());
}

}  // namespace ttml::metal::ops::ring_sdpa_fw

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_fw::RingSDPAFwDeviceOperation::tensor_return_value_t ttml_ring_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa_fw::RingDirection ring_direction,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates) {
    using OperationType = ttml::metal::ops::ring_sdpa_fw::RingSDPAFwDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction};

    auto tensor_args = OperationType::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .preallocated_output = preallocated_output,
        .preallocated_intermediates = preallocated_intermediates};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
