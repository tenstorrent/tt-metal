// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_kv_device_operation.hpp"

#include <fmt/core.h>

#include <tt-metalium/workload/legacy/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {

using namespace tt::tt_metal;
using namespace ttnn;

// ============== Backward KV Device Operation ==============

void RingSDPABwKVDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& chunks = tensor_args.chunks;
    validate_ring_load_balancing(attrs, chunks.size());
    validate_ring_attributes(attrs, chunks.front().query);
    for (const auto& chunk : chunks) {
        validate_ring_qkv(chunk.query, chunk.key, chunk.value);
        validate_output_like_tensor(chunk.grad_output, "Grad output", chunk.query, chunk.value);
        validate_intermediates_tensor(chunk.intermediates, chunk.query);
        validate_grad_like_tensor(chunk.grad_key, "Grad key", chunk.key, chunk.query);
        validate_grad_like_tensor(chunk.grad_value, "Grad value", chunk.value, chunk.query);
    }
    for (size_t c = 1; c < chunks.size(); ++c) {
        validate_same_spec(chunks[c].grad_output, chunks.front().grad_output, "grad_output", c);
        validate_same_spec(chunks[c].query, chunks.front().query, "query", c);
        validate_same_spec(chunks[c].key, chunks.front().key, "key", c);
        validate_same_spec(chunks[c].value, chunks.front().value, "value", c);
        validate_same_spec(chunks[c].intermediates, chunks.front().intermediates, "intermediates", c);
        validate_same_spec(chunks[c].grad_key, chunks.front().grad_key, "grad_key", c);
        validate_same_spec(chunks[c].grad_value, chunks.front().grad_value, "grad_value", c);
    }

    // u_scaler comes from the bw_q op: one FP32 tile column per query row.
    const auto& query = chunks.front().query;
    validate_sdpa_tensor(tensor_args.u_scaler, "U scaler", query, ttnn::DataType::FLOAT32);
    const auto [B, NH, S, E] = query.padded_shape().to_array_4D();
    const ttnn::Shape expected_u_scaler_shape{1, 1, B * NH * S, tt::constants::TILE_WIDTH};
    TT_FATAL(
        tensor_args.u_scaler.logical_shape() == expected_u_scaler_shape,
        "U scaler shape {} must be {} (one FP32 tile column per query row)",
        tensor_args.u_scaler.logical_shape(),
        expected_u_scaler_shape);
}

RingSDPABwKVDeviceOperation::spec_return_value_t RingSDPABwKVDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    spec_return_value_t specs;
    specs.reserve(tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        specs.push_back({.grad_key = chunk.grad_key.tensor_spec(), .grad_value = chunk.grad_value.tensor_spec()});
    }
    return specs;
}

RingSDPABwKVDeviceOperation::tensor_return_value_t RingSDPABwKVDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    outputs.reserve(tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        outputs.push_back({.grad_key = chunk.grad_key, .grad_value = chunk.grad_value});
    }
    return outputs;
}

std::vector<tt::tt_metal::TensorTopology> RingSDPABwKVDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    std::vector<tt::tt_metal::TensorTopology> topologies;
    topologies.reserve(2U * tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        topologies.push_back(chunk.grad_key.tensor_topology());
        topologies.push_back(chunk.grad_value.tensor_topology());
    }
    return topologies;
}

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVDeviceOperation::tensor_return_value_t ttml_ring_sdpa_bw_kv(
    std::vector<ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVChunk> chunks,
    const ttnn::Tensor& u_scaler,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa_bw::RingDirection ring_direction,
    ttml::metal::ops::RingLoadBalancing load_balancing) {
    using OperationType = ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVDeviceOperation;

    TT_FATAL(!chunks.empty(), "Ring SDPA needs at least one chunk");

    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction,
        .load_balancing = load_balancing};

    auto tensor_args = OperationType::tensor_args_t{.chunks = std::move(chunks), .u_scaler = u_scaler};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
