// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_kv_device_operation.hpp"

#include <cstdint>
#include <ttnn/tensor/tensor_utils.hpp>

namespace ttml::metal::ops::sdpa_bw::device {

using namespace tt::tt_metal;
using namespace ttnn;

SDPABackwardKVDeviceOperation::program_factory_t SDPABackwardKVDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SDPABackwardKVProgramFactory{};
}

void SDPABackwardKVDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void SDPABackwardKVDeviceOperation::validate_on_program_cache_miss(
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

    // Validate embedding dimensions match
    TT_FATAL(
        qE == kE && qE == vE,
        "Embedding dimensions of Q, K, V must be the same. Got qEmbd={}, kEmbd={}, vEmbd={}",
        qE,
        kE,
        vE);

    // Validate physical volumes are tile-aligned
    TT_FATAL(
        grad_output.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            query.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            key.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            value.physical_volume() % tt::constants::TILE_WIDTH == 0,
        "Physical volume of input tensors must be multiple of TILE_WIDTH. Got grad_output={}, query={}, key={}, "
        "value={}",
        grad_output.physical_volume(),
        query.physical_volume(),
        key.physical_volume(),
        value.physical_volume());

    // Validate mask shape if provided - must be (1, 1, S, S)
    if (tensor_args.attn_mask.has_value()) {
        const auto& mask = tensor_args.attn_mask.value();
        auto mask_shape = mask.logical_shape();
        auto [mB, mH, mS1, mS2] = mask_shape.to_array_4D();

        TT_FATAL(
            mB == 1 && mH == 1,
            "Attention mask must have shape (1, 1, S, S) for broadcasting. "
            "Got mask shape ({}, {}, {}, {}). "
            "Full (B, H, S, S) masks will be supported in a future PR.",
            mB,
            mH,
            mS1,
            mS2);

        TT_FATAL(
            mS1 == qS && mS2 == qS,
            "Attention mask sequence dimensions must match query sequence length. "
            "Got mask shape ({}, {}, {}, {}), expected (1, 1, {}, {})",
            mB,
            mH,
            mS1,
            mS2,
            qS,
            qS);
    }
}

SDPABackwardKVDeviceOperation::spec_return_value_t SDPABackwardKVDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(2U);  // 2 outputs: grad_key, grad_value

    // Handle grad_key
    if (tensor_args.preallocated_grad_key.has_value()) {
        output_specs.push_back(tensor_args.preallocated_grad_key->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.key.logical_shape(),
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

SDPABackwardKVDeviceOperation::tensor_return_value_t SDPABackwardKVDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(2U);  // 2 outputs: grad_key, grad_value

    spec_return_value_t output_specs = compute_output_specs(operation_attributes, tensor_args);

    // Handle grad_key
    if (tensor_args.preallocated_grad_key.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_grad_key.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.key.device()));
    }

    // Handle grad_value
    if (tensor_args.preallocated_grad_value.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_grad_value.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[1], tensor_args.value.device()));
    }

    return output_tensors;
}

ttsl::hash::hash_t SDPABackwardKVDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto hash = tt::tt_metal::operation::hash_operation<SDPABackwardKVDeviceOperation>(
        operation_attributes,
        tensor_args.query.logical_shape(),
        tensor_args.key.logical_shape(),
        tensor_args.intermediates.logical_shape(),
        tensor_args.query.dtype(),
        tensor_args.query.memory_config());

    return hash;
}

std::tuple<SDPABackwardKVDeviceOperation::operation_attributes_t, SDPABackwardKVDeviceOperation::tensor_args_t>
SDPABackwardKVDeviceOperation::invoke(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& key_tensor,
    const ttnn::Tensor& value_tensor,
    const std::optional<ttnn::Tensor>& attn_mask,
    const ttnn::Tensor& intermediates,
    const float dropout_probability,
    const bool fp32_dest_acc_en,
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
        .preallocated_grad_key = preallocated_grad_key,
        .preallocated_grad_value = preallocated_grad_value,
    };

    return {operation_attributes, tensor_args};
}

}  // namespace ttml::metal::ops::sdpa_bw::device
