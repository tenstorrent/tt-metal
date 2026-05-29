// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_q_device_operation.hpp"

#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>

namespace ttml::metal::ops::ring_sdpa_bw::q {

using namespace tt::tt_metal;
using namespace ttnn;

// ============== Backward Q Device Operation ==============

void RingSDPABwQDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.query.device() != nullptr, "Query tensor must be on device");
    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step must be < ring_size");
}

RingSDPABwQDeviceOperation::spec_return_value_t RingSDPABwQDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    ttnn::TensorSpec grad_query_spec =
        tensor_args.preallocated_grad_query.has_value()
            ? tensor_args.preallocated_grad_query->tensor_spec()
            : ttnn::TensorSpec(
                  tensor_args.query.logical_shape(),
                  tt::tt_metal::TensorLayout(
                      tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));

    // u_scaler: one FP32 tile per query row, shape = (1, 1, B*NH*S, TILE_WIDTH)
    const auto [B, NH, S, E] = tensor_args.query.padded_shape().to_array_4D();
    auto u_scaler_shape = ttnn::Shape({1, 1, B * NH * S, tt::constants::TILE_WIDTH});
    auto mem_config =
        tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM);
    ttnn::TensorSpec u_scaler_spec(
        u_scaler_shape,
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, mem_config));

    return {grad_query_spec, u_scaler_spec};
}

RingSDPABwQDeviceOperation::tensor_return_value_t RingSDPABwQDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [grad_query_spec, u_scaler_spec] = compute_output_specs(attrs, tensor_args);

    ttnn::Tensor grad_query = tensor_args.preallocated_grad_query.has_value()
                                  ? tensor_args.preallocated_grad_query.value()
                                  : create_device_tensor(grad_query_spec, tensor_args.query.device());

    // TODO: accept preallocated_u_scaler to avoid per-step allocation in the ring loop.
    // The tensor is small (one FP32 tile per Q row), so the overhead is negligible for now.
    ttnn::Tensor u_scaler = create_device_tensor(u_scaler_spec, tensor_args.query.device());

    return {grad_query, u_scaler};
}

ttsl::hash::hash_t RingSDPABwQDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return ttsl::hash::hash_objects(
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

}  // namespace ttml::metal::ops::ring_sdpa_bw::q

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_bw::q::RingSDPABwQDeviceOperation::tensor_return_value_t ttml_ring_sdpa_bw_q(
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
    const std::optional<ttnn::Tensor>& preallocated_grad_query) {
    using OperationType = ttml::metal::ops::ring_sdpa_bw::q::RingSDPABwQDeviceOperation;

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
        .preallocated_grad_query = preallocated_grad_query};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
