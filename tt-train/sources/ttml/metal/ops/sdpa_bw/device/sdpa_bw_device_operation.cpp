// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_device_operation.hpp"

#include <cstdint>
#include <ttnn/tensor/tensor_utils.hpp>

namespace ttml::metal::ops::sdpa_bw::device {

using namespace tt::tt_metal;
using namespace ttnn;

SDPABackwardDeviceOperation::program_factory_t SDPABackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SDPABackwardProgramFactory{};
}

void SDPABackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void SDPABackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& grad_output = tensor_args.grad_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& intermediates = tensor_args.intermediates;

    // Validate shapes are compatible
    auto grad_output_shape = grad_output.logical_shape();
    auto query_shape = query.logical_shape();
    auto key_shape = key.logical_shape();
    auto value_shape = value.logical_shape();

    TT_FATAL(
        grad_output_shape == query_shape,
        "Grad output shape {} must match query shape {}",
        grad_output_shape,
        query_shape);

    TT_FATAL(key_shape == value_shape, "Key shape {} must match value shape {}", key_shape, value_shape);

    TT_FATAL(
        query_shape[0] == key_shape[0] && query_shape[2] == key_shape[2],
        "Batch and sequence dimensions must match between query {} and key {}",
        query_shape,
        key_shape);

    // Validate data formats
    TT_FATAL(
        grad_output.dtype() == query.dtype() && query.dtype() == key.dtype() && key.dtype() == value.dtype(),
        "All input tensors must have the same data type");

    // Validate device placement
    TT_FATAL(
        grad_output.device() == query.device() && query.device() == key.device() && key.device() == value.device(),
        "All input tensors must be on the same device");

    // Extract and validate heads from tensor shapes
    auto [qB, qH, qS, qE] = query_shape.to_array_4D();
    auto [kB, kH, kS, kE] = key_shape.to_array_4D();
    auto [vB, vH, vS, vE] = value_shape.to_array_4D();

    TT_FATAL(
        qH > 0 && kH > 0 && vH > 0,
        "Number of heads must be greater than zero. Got q_heads={}, kv_heads={}, v_heads={}",
        qH,
        kH,
        vH);

    TT_FATAL(
        qH % kH == 0,
        "Number of query heads ({}) must be divisible by number of key/value heads ({}) for grouped attention.",
        qH,
        kH);

    TT_FATAL(kH == vH, "Key and Value must have the same number of heads. Got key_heads={}, value_heads={}", kH, vH);
}

SDPABackwardDeviceOperation::spec_return_value_t SDPABackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(3U);  // Always 3 outputs: grad_query, grad_key, grad_value

    // Handle grad_query
    if (tensor_args.preallocated_grad_query.has_value()) {
        output_specs.push_back(tensor_args.preallocated_grad_query->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.query.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    }

    // Handle grad_key
    if (tensor_args.preallocated_grad_key.has_value()) {
        output_specs.push_back(tensor_args.preallocated_grad_key->tensor_spec());
    } else {
        //[DEBUG]: I changed shape of grad_key to write in grad_key some temporary results which I need to caclulate
        // real grad_key
        auto shape = tensor_args.key.logical_shape();
        shape[3] = 1U;  // I put here scaler u need to fix it later
        output_specs.emplace_back(
            shape,
            tt::tt_metal::TensorLayout(
                tensor_args.key.dtype(), tt::tt_metal::Layout::TILE, tensor_args.key.memory_config()));
    }

    // Handle grad_value
    if (tensor_args.preallocated_grad_value.has_value()) {
        output_specs.push_back(tensor_args.preallocated_grad_value->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.value.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.value.dtype(), tt::tt_metal::Layout::TILE, tensor_args.value.memory_config()));
    }

    return output_specs;
}

SDPABackwardDeviceOperation::tensor_return_value_t SDPABackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(3U);  // Always 3 outputs: grad_query, grad_key, grad_value

    spec_return_value_t output_specs = compute_output_specs(operation_attributes, tensor_args);

    // Handle grad_query
    if (tensor_args.preallocated_grad_query.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_grad_query.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.query.device()));
    }

    // Handle grad_key
    if (tensor_args.preallocated_grad_key.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_grad_key.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[1], tensor_args.key.device()));
    }

    // Handle grad_value
    if (tensor_args.preallocated_grad_value.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_grad_value.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[2], tensor_args.value.device()));
    }

    return output_tensors;
}

ttsl::hash::hash_t SDPABackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto hash = tt::tt_metal::operation::hash_operation<SDPABackwardDeviceOperation>(
        operation_attributes,
        tensor_args.query.logical_shape(),
        tensor_args.key.logical_shape(),
        tensor_args.intermediates.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.query.memory_config());

    return hash;
}

std::tuple<SDPABackwardDeviceOperation::operation_attributes_t, SDPABackwardDeviceOperation::tensor_args_t>
SDPABackwardDeviceOperation::invoke(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& key_tensor,
    const ttnn::Tensor& value_tensor,
    const std::optional<ttnn::Tensor>& attn_mask,
    const ttnn::Tensor& intermediates,
    const float dropout_probability,
    const bool fp32_dest_acc_en,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value) {
    operation_attributes_t operation_attributes{
        .fp32_dest_acc_en = fp32_dest_acc_en, .dropout_probability = dropout_probability};

    tensor_args_t tensor_args{
        .grad_output = grad_output,
        .attn_output = attn_output,
        .query = query_tensor,
        .key = key_tensor,
        .value = value_tensor,
        .attn_mask = attn_mask,
        .intermediates = intermediates,
        .preallocated_grad_query = preallocated_grad_query,
        .preallocated_grad_key = preallocated_grad_key,
        .preallocated_grad_value = preallocated_grad_value,
    };

    return {operation_attributes, tensor_args};
}

}  // namespace ttml::metal::ops::sdpa_bw::device
