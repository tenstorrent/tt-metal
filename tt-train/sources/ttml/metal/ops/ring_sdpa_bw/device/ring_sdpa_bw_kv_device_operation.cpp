// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_kv_device_operation.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {

using namespace tt::tt_metal;
using namespace ttnn;

// ============== Backward KV Device Operation ==============

void RingSDPABwKVDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_ring_attributes(attrs, tensor_args.query);
    validate_ring_qkv(tensor_args.query, tensor_args.key, tensor_args.value);
    validate_output_like_tensor(tensor_args.grad_output, "Grad output", tensor_args.query, tensor_args.value);
    validate_intermediates_tensor(tensor_args.intermediates, tensor_args.query);

    // u_scaler comes from the bw_q op: one FP32 tile column per query row.
    validate_sdpa_tensor(tensor_args.u_scaler, "U scaler", tensor_args.query, ttnn::DataType::FLOAT32);
    const auto [B, NH, S, E] = tensor_args.query.padded_shape().to_array_4D();
    const ttnn::Shape expected_u_scaler_shape{1, 1, B * NH * S, tt::constants::TILE_WIDTH};
    TT_FATAL(
        tensor_args.u_scaler.logical_shape() == expected_u_scaler_shape,
        "U scaler shape {} must be {} (one FP32 tile column per query row)",
        tensor_args.u_scaler.logical_shape(),
        expected_u_scaler_shape);

    if (tensor_args.preallocated_grad_key.has_value()) {
        validate_grad_like_tensor(
            tensor_args.preallocated_grad_key.value(), "Preallocated grad key", tensor_args.key, tensor_args.query);
    }
    if (tensor_args.preallocated_grad_value.has_value()) {
        validate_grad_like_tensor(
            tensor_args.preallocated_grad_value.value(),
            "Preallocated grad value",
            tensor_args.value,
            tensor_args.query);
    }
}

RingSDPABwKVDeviceOperation::spec_return_value_t RingSDPABwKVDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    // Handle grad_key spec
    tt::tt_metal::TensorSpec grad_key_spec =
        tensor_args.preallocated_grad_key.has_value()
            ? tensor_args.preallocated_grad_key->tensor_spec()
            : tt::tt_metal::TensorSpec(
                  tensor_args.key.logical_shape(),
                  tt::tt_metal::TensorLayout(
                      tensor_args.key.dtype(), tt::tt_metal::Layout::TILE, tensor_args.key.memory_config()));

    // Handle grad_value spec
    tt::tt_metal::TensorSpec grad_value_spec =
        tensor_args.preallocated_grad_value.has_value()
            ? tensor_args.preallocated_grad_value->tensor_spec()
            : tt::tt_metal::TensorSpec(
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
                                : ttnn::create_device_tensor(grad_key_spec, tensor_args.key.device());

    // Handle grad_value
    ttnn::Tensor grad_value = tensor_args.preallocated_grad_value.has_value()
                                  ? tensor_args.preallocated_grad_value.value()
                                  : ttnn::create_device_tensor(grad_value_spec, tensor_args.value.device());

    return {grad_key, grad_value};
}

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVDeviceOperation::tensor_return_value_t ttml_ring_sdpa_bw_kv(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& u_scaler,
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
        .u_scaler = u_scaler,
        .query = query,
        .key = key,
        .value = value,
        .intermediates = intermediates,
        .preallocated_grad_key = preallocated_grad_key,
        .preallocated_grad_value = preallocated_grad_value};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
