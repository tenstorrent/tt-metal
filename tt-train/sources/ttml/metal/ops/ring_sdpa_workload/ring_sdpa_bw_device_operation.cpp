// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_device_operation.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

namespace ttml::metal::ops::ring_sdpa {

// ============== Backward Q Device Operation ==============

RingSDPABwQDeviceOperation::program_factory_t RingSDPABwQDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return RingSDPABwQProgramFactory{};
}

void RingSDPABwQDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
}

void RingSDPABwQDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.query.device() != nullptr, "Query tensor must be on device");
    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step must be < ring_size");
}

RingSDPABwQDeviceOperation::spec_return_value_t RingSDPABwQDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.preallocated_grad_query->tensor_spec();
}

RingSDPABwQDeviceOperation::tensor_return_value_t RingSDPABwQDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return *tensor_args.preallocated_grad_query;
}

tt::stl::hash::hash_t RingSDPABwQDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return tt::stl::hash::hash_objects(
        0,  // Q marker (different from KV)
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        static_cast<int>(attrs.mask_type),
        static_cast<int>(attrs.ring_direction),
        tensor_args.query.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.key.logical_shape());
}

// ============== Backward KV Device Operation ==============

RingSDPABwKVDeviceOperation::program_factory_t RingSDPABwKVDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return RingSDPABwKVProgramFactory{};
}

void RingSDPABwKVDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
}

void RingSDPABwKVDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.query.device() != nullptr, "Query tensor must be on device");
    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step must be < ring_size");
}

RingSDPABwKVDeviceOperation::spec_return_value_t RingSDPABwKVDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return {tensor_args.preallocated_grad_key->tensor_spec(), tensor_args.preallocated_grad_value->tensor_spec()};
}

RingSDPABwKVDeviceOperation::tensor_return_value_t RingSDPABwKVDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return {*tensor_args.preallocated_grad_key, *tensor_args.preallocated_grad_value};
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

}  // namespace ttml::metal::ops::ring_sdpa

namespace ttml::metal::prim {

ttnn::Tensor ring_sdpa_bw_q(
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
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa::RingDirection ring_direction) {
    using OperationType = ttml::metal::ops::ring_sdpa::RingSDPABwQDeviceOperation;

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
        .preallocated_grad_query = grad_query};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_sdpa_bw_kv(
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
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa::RingDirection ring_direction) {
    using OperationType = ttml::metal::ops::ring_sdpa::RingSDPABwKVDeviceOperation;

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
        .preallocated_grad_key = grad_key,
        .preallocated_grad_value = grad_value};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttml::metal::prim
