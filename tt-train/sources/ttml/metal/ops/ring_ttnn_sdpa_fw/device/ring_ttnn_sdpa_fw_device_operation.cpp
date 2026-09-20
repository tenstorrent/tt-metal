// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_ttnn_sdpa_fw_device_operation.hpp"

#include <tt-metalium/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_ttnn_sdpa_fw {

void RingTtnnSdpaFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_ring_attributes(attrs, tensor_args.query);
    validate_ring_qkv(tensor_args.query, tensor_args.key, tensor_args.value);
    TT_FATAL(
        attrs.mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "ring_ttnn_sdpa_fw takes Causal or None; ttnn's kernel makes the causal mask itself");
    TT_FATAL(attrs.chunk_size % 32U == 0U && attrs.chunk_size > 0U, "chunk_size must be a positive multiple of 32");
    if (tensor_args.preallocated_output.has_value()) {
        validate_output_like_tensor(
            tensor_args.preallocated_output.value(), "Preallocated output", tensor_args.query, tensor_args.value);
    }
    if (tensor_args.preallocated_intermediates.has_value()) {
        validate_intermediates_tensor(tensor_args.preallocated_intermediates.value(), tensor_args.query);
        TT_FATAL(
            tensor_args.preallocated_intermediates->logical_shape()[3] == 32U,
            "ring_ttnn_sdpa_fw writes one Float32 tile per query row tile: the intermediates must be "
            "(B, H, S, 32), got {}",
            tensor_args.preallocated_intermediates->logical_shape());
    }
}

RingTtnnSdpaFwDeviceOperation::spec_return_value_t RingTtnnSdpaFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    tt::tt_metal::TensorSpec output_spec =
        tensor_args.preallocated_output.has_value()
            ? tensor_args.preallocated_output->tensor_spec()
            : tt::tt_metal::TensorSpec(
                  tensor_args.query.logical_shape(),
                  tt::tt_metal::TensorLayout(
                      tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    auto [batch, heads, seq_len, dim] = tensor_args.query.logical_shape().to_array_4D();
    tt::tt_metal::TensorSpec intermediates_spec =
        tensor_args.preallocated_intermediates.has_value()
            ? tensor_args.preallocated_intermediates->tensor_spec()
            : tt::tt_metal::TensorSpec(
                  ttnn::Shape{batch, heads, seq_len, 32U},
                  tt::tt_metal::TensorLayout(
                      ttnn::DataType::FLOAT32, tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    return {output_spec, intermediates_spec};
}

RingTtnnSdpaFwDeviceOperation::tensor_return_value_t RingTtnnSdpaFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [output_spec, intermediates_spec] = compute_output_specs(attrs, tensor_args);
    ttnn::Tensor output = tensor_args.preallocated_output.has_value()
                              ? tensor_args.preallocated_output.value()
                              : ttnn::create_device_tensor(output_spec, tensor_args.query.device());
    ttnn::Tensor intermediates = tensor_args.preallocated_intermediates.has_value()
                                     ? tensor_args.preallocated_intermediates.value()
                                     : ttnn::create_device_tensor(intermediates_spec, tensor_args.query.device());
    return {output, intermediates};
}

ttsl::hash::hash_t RingTtnnSdpaFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return ttsl::hash::hash_objects(
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        attrs.mask_type,
        static_cast<int>(attrs.ring_direction),
        attrs.zigzag,
        static_cast<int>(attrs.visitor),
        attrs.chunk_size,
        tensor_args.query.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.key.logical_shape());
}

}  // namespace ttml::metal::ops::ring_ttnn_sdpa_fw

namespace ttnn::prim {

ttml::metal::ops::ring_ttnn_sdpa_fw::RingTtnnSdpaFwDeviceOperation::tensor_return_value_t ttml_ring_ttnn_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_ttnn_sdpa_fw::RingDirection ring_direction,
    bool zigzag,
    ttml::metal::ops::ZigzagVisitor visitor,
    uint32_t chunk_size,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates) {
    using OperationType = ttml::metal::ops::ring_ttnn_sdpa_fw::RingTtnnSdpaFwDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction,
        .zigzag = zigzag,
        .visitor = visitor,
        .chunk_size = chunk_size};
    auto tensor_args = OperationType::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .preallocated_output = preallocated_output,
        .preallocated_intermediates = preallocated_intermediates};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
