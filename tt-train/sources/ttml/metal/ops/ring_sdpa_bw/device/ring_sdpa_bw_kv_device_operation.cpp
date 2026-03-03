// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_kv_device_operation.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

namespace ttml::metal::ops::ring_sdpa_bw::kv {

using namespace tt::tt_metal;
using namespace ttnn;

// ============== Backward KV Device Operation ==============

void RingSDPABwKVDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.query.device() != nullptr, "Query tensor must be on device");
    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step must be < ring_size");
}

RingSDPABwKVDeviceOperation::spec_return_value_t RingSDPABwKVDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    // Handle grad_key spec
    ttnn::TensorSpec grad_key_spec =
        tensor_args.preallocated_grad_key.has_value()
            ? tensor_args.preallocated_grad_key->tensor_spec()
            : ttnn::TensorSpec(
                  tensor_args.key.logical_shape(),
                  tt::tt_metal::TensorLayout(
                      tensor_args.key.dtype(), tt::tt_metal::Layout::TILE, tensor_args.key.memory_config()));

    // Handle grad_value spec
    ttnn::TensorSpec grad_value_spec =
        tensor_args.preallocated_grad_value.has_value()
            ? tensor_args.preallocated_grad_value->tensor_spec()
            : ttnn::TensorSpec(
                  tensor_args.value.logical_shape(),
                  tt::tt_metal::TensorLayout(
                      tensor_args.value.dtype(), tt::tt_metal::Layout::TILE, tensor_args.value.memory_config()));

    return {grad_key_spec, grad_value_spec};
}

RingSDPABwKVDeviceOperation::tensor_return_value_t RingSDPABwKVDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [grad_key_spec, grad_value_spec] = compute_output_specs(attrs, tensor_args);

    // Handle grad_key
    ttnn::Tensor grad_key = tensor_args.preallocated_grad_key.has_value()
                                ? tensor_args.preallocated_grad_key.value()
                                : create_device_tensor(grad_key_spec, tensor_args.key.device());

    // Handle grad_value
    ttnn::Tensor grad_value = tensor_args.preallocated_grad_value.has_value()
                                  ? tensor_args.preallocated_grad_value.value()
                                  : create_device_tensor(grad_value_spec, tensor_args.value.device());

    return {grad_key, grad_value};
}

tt::stl::hash::hash_t RingSDPABwKVDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Hash based on operation configuration - buffer addresses are updated via override_runtime_arguments
    return tt::stl::hash::hash_objects(
        1,  // KV marker (different from Q)
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        static_cast<int>(attrs.mask_type),
        static_cast<int>(attrs.ring_direction),
        tensor_args.query.tensor_spec().logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.key.tensor_spec().logical_shape());
}

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVDeviceOperation::tensor_return_value_t ttml_ring_sdpa_bw_kv(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa_bw::RingDirection ring_direction,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value) {
    using OperationType = ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction};

    auto tensor_args = OperationType::tensor_args_t{
        .grad_output = grad_output,
        .attn_output = attn_output,
        .query = query,
        .key = key,
        .value = value,
        .intermediates = intermediates,
        .preallocated_grad_key = preallocated_grad_key,
        .preallocated_grad_value = preallocated_grad_value};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
