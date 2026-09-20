// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_cyclic_sdpa_fw_device_operation.hpp"

#include <tt-metalium/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_fw {

void RingCyclicSdpaFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_ring_attributes(attrs, tensor_args.query);
    validate_ring_qkv(tensor_args.query, tensor_args.key, tensor_args.value);
    TT_FATAL(
        attrs.mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "ring_cyclic_sdpa_fw takes Causal or None; the cyclic kernels make the causal mask themselves");
    if (tensor_args.preallocated_output.has_value()) {
        validate_output_like_tensor(
            tensor_args.preallocated_output.value(), "Preallocated output", tensor_args.query, tensor_args.value);
        TT_FATAL(
            tensor_args.preallocated_output->dtype() == ttnn::DataType::BFLOAT16,
            "ring_cyclic_sdpa_fw writes a bfloat16 output; got {}",
            tensor_args.preallocated_output->dtype());
    }
    if (tensor_args.preallocated_intermediates.has_value()) {
        validate_intermediates_tensor(tensor_args.preallocated_intermediates.value(), tensor_args.query);
        TT_FATAL(
            tensor_args.preallocated_intermediates->logical_shape()[3] == 32U,
            "ring_cyclic_sdpa_fw writes one Float32 tile per query row tile: the intermediates must be "
            "(B, H, S, 32), got {}",
            tensor_args.preallocated_intermediates->logical_shape());
    }
}

RingCyclicSdpaFwDeviceOperation::spec_return_value_t RingCyclicSdpaFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto [batch, heads, seq_len, dim] = tensor_args.query.logical_shape().to_array_4D();
    const auto spec = [&](const ttnn::Shape& shape, ttnn::DataType dt) {
        return tt::tt_metal::TensorSpec(
            shape, tt::tt_metal::TensorLayout(dt, tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    };
    return {
        tensor_args.preallocated_output.has_value() ? tensor_args.preallocated_output->tensor_spec()
                                                    : spec(tensor_args.query.logical_shape(), ttnn::DataType::BFLOAT16),
        tensor_args.preallocated_intermediates.has_value()
            ? tensor_args.preallocated_intermediates->tensor_spec()
            : spec(ttnn::Shape{batch, heads, seq_len, 32U}, ttnn::DataType::FLOAT32),
        spec(tensor_args.query.logical_shape(), ttnn::DataType::FLOAT32),
        spec(ttnn::Shape{batch, heads, seq_len, 64U}, ttnn::DataType::FLOAT32)};
}

RingCyclicSdpaFwDeviceOperation::tensor_return_value_t RingCyclicSdpaFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.query.device();
    const auto take = [&](const std::optional<ttnn::Tensor>& preallocated, size_t i) {
        return preallocated.has_value() ? preallocated.value() : ttnn::create_device_tensor(specs[i], device);
    };
    return {
        take(tensor_args.preallocated_output, 0U),
        take(tensor_args.preallocated_intermediates, 1U),
        ttnn::create_device_tensor(specs[2], device),
        ttnn::create_device_tensor(specs[3], device)};
}

ttsl::hash::hash_t RingCyclicSdpaFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return ttsl::hash::hash_objects(
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        attrs.mask_type,
        static_cast<int>(attrs.ring_direction),
        attrs.zigzag,
        static_cast<int>(attrs.visitor),
        attrs.rows_per_block_tiles,
        tensor_args.query.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.key.logical_shape());
}

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_fw

namespace ttnn::prim {

ttml::metal::ops::ring_cyclic_sdpa_fw::RingCyclicSdpaFwDeviceOperation::tensor_return_value_t
ttml_ring_cyclic_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_cyclic_sdpa_fw::RingDirection ring_direction,
    bool zigzag,
    ttml::metal::ops::ZigzagVisitor visitor,
    uint32_t rows_per_block_tiles,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates) {
    using OperationType = ttml::metal::ops::ring_cyclic_sdpa_fw::RingCyclicSdpaFwDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction,
        .zigzag = zigzag,
        .visitor = visitor,
        .rows_per_block_tiles = rows_per_block_tiles};
    auto tensor_args = OperationType::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .preallocated_output = preallocated_output,
        .preallocated_intermediates = preallocated_intermediates};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
