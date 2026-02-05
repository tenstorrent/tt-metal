// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_device_operation.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttml::metal::ops::ring_sdpa {

RingSDPADeviceOperation::program_factory_t RingSDPADeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return RingSDPAProgramFactory{};
}

void RingSDPADeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    // Validation on cache hit - minimal checks
}

void RingSDPADeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.query.device() != nullptr, "Query tensor must be on device");
    TT_FATAL(tensor_args.key.device() != nullptr, "Key tensor must be on device");
    TT_FATAL(tensor_args.value.device() != nullptr, "Value tensor must be on device");

    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step must be < ring_size");
}

RingSDPADeviceOperation::spec_return_value_t RingSDPADeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    // Output and intermediates specs match what's passed in
    return {tensor_args.preallocated_output->tensor_spec(), tensor_args.preallocated_intermediates->tensor_spec()};
}

RingSDPADeviceOperation::tensor_return_value_t RingSDPADeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    // Return the preallocated output tensors
    return {*tensor_args.preallocated_output, *tensor_args.preallocated_intermediates};
}

tt::stl::hash::hash_t RingSDPADeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Hash based on operation configuration - buffer addresses are updated via override_runtime_arguments
    return tt::stl::hash::hash_objects(
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        attrs.mask_type,
        static_cast<int>(attrs.ring_direction),
        tensor_args.query.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.key.logical_shape(),
        tensor_args.preallocated_intermediates->logical_shape());
}

}  // namespace ttml::metal::ops::ring_sdpa

namespace ttml::metal::prim {

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_sdpa(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    ttnn::Tensor& output,
    ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa::RingDirection ring_direction) {
    using OperationType = ttml::metal::ops::ring_sdpa::RingSDPADeviceOperation;

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
        .preallocated_output = output,
        .preallocated_intermediates = intermediates};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttml::metal::prim
