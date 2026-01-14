// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_q_device_operation.hpp"

#include <cstdint>
#include <ttnn/tensor/tensor_utils.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::sdpa_bw::device {

using namespace tt::tt_metal;
using namespace ttnn;

SDPABackwardQDeviceOperation::program_factory_t SDPABackwardQDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SDPABackwardQProgramFactory{};
}

void SDPABackwardQDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void SDPABackwardQDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& grad_output = tensor_args.grad_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;

    // Validate that input and output tensor shapes are compatible
    const auto grad_output_shape = grad_output.logical_shape();
    const auto query_shape = query.logical_shape();
    const auto key_shape = key.logical_shape();
    const auto value_shape = value.logical_shape();

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
    const auto [qB, qH, qS, qE] = query_shape.to_array_4D();
    const auto [kB, kH, kS, kE] = key_shape.to_array_4D();
    const auto [vB, vH, vS, vE] = value_shape.to_array_4D();

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

    // Validate tensors have tile layout
    TT_FATAL(
        grad_output.layout() == tt::tt_metal::Layout::TILE && query.layout() == tt::tt_metal::Layout::TILE &&
            key.layout() == tt::tt_metal::Layout::TILE && value.layout() == tt::tt_metal::Layout::TILE,
        "All input tensors must have TILE layout");

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

SDPABackwardQDeviceOperation::spec_return_value_t SDPABackwardQDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Return single grad_query spec
    if (tensor_args.preallocated_grad_query.has_value()) {
        return tensor_args.preallocated_grad_query->tensor_spec();
    } else {
        return ttnn::TensorSpec(
            tensor_args.query.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    }
}

SDPABackwardQDeviceOperation::tensor_return_value_t SDPABackwardQDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    spec_return_value_t output_spec = compute_output_specs(operation_attributes, tensor_args);

    // Return single grad_query tensor
    if (tensor_args.preallocated_grad_query.has_value()) {
        return tensor_args.preallocated_grad_query.value();
    } else {
        return create_device_tensor(output_spec, tensor_args.query.device());
    }
}

ttsl::hash::hash_t SDPABackwardQDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto hash = tt::tt_metal::operation::hash_operation<SDPABackwardQDeviceOperation>(
        operation_attributes,
        tensor_args.query.logical_shape(),
        tensor_args.key.logical_shape(),
        tensor_args.intermediates.logical_shape(),
        tensor_args.query.dtype());

    return hash;
}

}  // namespace ttml::metal::ops::sdpa_bw::device

namespace ttnn::prim {

ttml::metal::ops::sdpa_bw::device::SDPABackwardQDeviceOperation::tensor_return_value_t ttml_sdpa_q_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& key_tensor,
    const ttnn::Tensor& value_tensor,
    const std::optional<ttnn::Tensor>& attn_mask,
    const ttnn::Tensor& intermediates,
    const float dropout_probability,
    const bool fp32_dest_acc_en,
    const std::optional<ttnn::Tensor>& preallocated_grad_query) {
    using OperationType = ttml::metal::ops::sdpa_bw::device::SDPABackwardQDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .fp32_dest_acc_en = fp32_dest_acc_en, .dropout_probability = dropout_probability};

    auto tensor_args = OperationType::tensor_args_t{
        .grad_output = grad_output,
        .attn_output = attn_output,
        .query = query_tensor,
        .key = key_tensor,
        .value = value_tensor,
        .attn_mask = attn_mask,
        .intermediates = intermediates,
        .preallocated_grad_query = preallocated_grad_query,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
