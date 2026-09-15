// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ring_zigzag_sdpa: the ring_sdpa_fw / ring_sdpa_bw mesh wrapper over the
// single-chip SDPA kernels, with the chips that run chosen by where this
// step's visiting chunk comes from (ops::ZigzagVisitor) instead of by the
// contiguous layout's causal rule. The kernels and the reference ring ops
// are untouched; this exists so the zigzag layout can use them on one chunk
// pair per launch.

#include "ring_zigzag_fw_device_operation.hpp"

#include <tt-metalium/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_zigzag_fw {

using namespace tt::tt_metal;
using namespace ttnn;

void RingZigzagFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_ring_attributes(attrs, tensor_args.query);
    validate_ring_qkv(tensor_args.query, tensor_args.key, tensor_args.value);
    if (tensor_args.preallocated_output.has_value()) {
        validate_output_like_tensor(
            tensor_args.preallocated_output.value(), "Preallocated output", tensor_args.query, tensor_args.value);
    }
    if (tensor_args.preallocated_intermediates.has_value()) {
        validate_intermediates_tensor(tensor_args.preallocated_intermediates.value(), tensor_args.query);
    }
}

RingZigzagFwDeviceOperation::spec_return_value_t RingZigzagFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    // Handle output spec
    tt::tt_metal::TensorSpec output_spec =
        tensor_args.preallocated_output.has_value()
            ? tensor_args.preallocated_output->tensor_spec()
            : tt::tt_metal::TensorSpec(
                  tensor_args.query.logical_shape(),
                  tt::tt_metal::TensorLayout(
                      tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));

    // Handle intermediates spec - shape is (B, H, S, 32) = 1 FP32 tile wide (logsumexp)
    auto query_shape = tensor_args.query.logical_shape();
    auto [batch, heads, seq_len, dim] = query_shape.to_array_4D();
    tt::tt_metal::TensorSpec intermediates_spec =
        tensor_args.preallocated_intermediates.has_value()
            ? tensor_args.preallocated_intermediates->tensor_spec()
            : tt::tt_metal::TensorSpec(
                  ttnn::Shape{batch, heads, seq_len, 32U},
                  tt::tt_metal::TensorLayout(
                      ttnn::DataType::FLOAT32, tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));

    return {output_spec, intermediates_spec};
}

RingZigzagFwDeviceOperation::tensor_return_value_t RingZigzagFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [output_spec, intermediates_spec] = compute_output_specs(attrs, tensor_args);

    // Handle output
    ttnn::Tensor output = tensor_args.preallocated_output.has_value()
                              ? tensor_args.preallocated_output.value()
                              : ttnn::create_device_tensor(output_spec, tensor_args.query.device());

    // Handle intermediates
    ttnn::Tensor intermediates = tensor_args.preallocated_intermediates.has_value()
                                     ? tensor_args.preallocated_intermediates.value()
                                     : ttnn::create_device_tensor(intermediates_spec, tensor_args.query.device());

    return {output, intermediates};
}

ttsl::hash::hash_t RingZigzagFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Hash based on operation configuration - buffer addresses are updated via override_runtime_arguments
    return ttsl::hash::hash_objects(
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        attrs.mask_type,
        static_cast<int>(attrs.ring_direction),
        static_cast<int>(attrs.visitor),
        tensor_args.query.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.key.logical_shape());
}

}  // namespace ttml::metal::ops::ring_zigzag_fw

namespace ttnn::prim {

ttml::metal::ops::ring_zigzag_fw::RingZigzagFwDeviceOperation::tensor_return_value_t ttml_ring_zigzag_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_zigzag_fw::RingDirection ring_direction,
    ttml::metal::ops::ZigzagVisitor visitor,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates) {
    using OperationType = ttml::metal::ops::ring_zigzag_fw::RingZigzagFwDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction,
        .visitor = visitor};

    auto tensor_args = OperationType::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .preallocated_output = preallocated_output,
        .preallocated_intermediates = preallocated_intermediates};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
