// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_q_device_operation.hpp"

#include <fmt/core.h>

#include <tt-metalium/workload/legacy/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::q {

using namespace tt::tt_metal;
using namespace ttnn;

namespace {

// u_scaler: one FP32 tile per query row, shape = (1, 1, B*NH*S, TILE_WIDTH)
tt::tt_metal::TensorSpec u_scaler_spec(const ttnn::Tensor& query) {
    const auto [B, NH, S, E] = query.padded_shape().to_array_4D();
    auto u_scaler_shape = ttnn::Shape({1, 1, B * NH * S, tt::constants::TILE_WIDTH});
    auto mem_config =
        tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM);
    return tt::tt_metal::TensorSpec(
        u_scaler_shape,
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, mem_config));
}

}  // namespace

// ============== Backward Q Device Operation ==============

void RingSDPABwQDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& chunks = tensor_args.chunks;
    validate_ring_load_balancing(attrs, chunks.size());
    validate_ring_attributes(attrs, chunks.front().query);
    for (const auto& chunk : chunks) {
        validate_ring_qkv(chunk.query, chunk.key, chunk.value);
        validate_output_like_tensor(chunk.grad_output, "Grad output", chunk.query, chunk.value);
        validate_output_like_tensor(chunk.attn_output, "Attention output", chunk.query, chunk.value);
        validate_intermediates_tensor(chunk.intermediates, chunk.query);
        validate_grad_like_tensor(chunk.grad_query, "Grad query", chunk.query, chunk.query);
    }
    for (size_t c = 1; c < chunks.size(); ++c) {
        validate_same_spec(chunks[c].grad_output, chunks.front().grad_output, "grad_output", c);
        validate_same_spec(chunks[c].attn_output, chunks.front().attn_output, "attn_output", c);
        validate_same_spec(chunks[c].query, chunks.front().query, "query", c);
        validate_same_spec(chunks[c].key, chunks.front().key, "key", c);
        validate_same_spec(chunks[c].value, chunks.front().value, "value", c);
        validate_same_spec(chunks[c].intermediates, chunks.front().intermediates, "intermediates", c);
        validate_same_spec(chunks[c].grad_query, chunks.front().grad_query, "grad_query", c);
    }
}

RingSDPABwQDeviceOperation::spec_return_value_t RingSDPABwQDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    spec_return_value_t specs{.chunks = {}, .u_scaler = u_scaler_spec(tensor_args.chunks.front().query)};
    specs.chunks.reserve(tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        specs.chunks.push_back({.grad_query = chunk.grad_query.tensor_spec()});
    }
    return specs;
}

RingSDPABwQDeviceOperation::tensor_return_value_t RingSDPABwQDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& query = tensor_args.chunks.front().query;
    // TODO: accept preallocated_u_scaler to avoid per-step allocation in the ring loop.
    // The tensor is small (one FP32 tile per Q row), so the overhead is negligible for now.
    tensor_return_value_t outputs{
        .chunks = {}, .u_scaler = ttnn::create_device_tensor(u_scaler_spec(query), query.device())};
    outputs.chunks.reserve(tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        outputs.chunks.push_back({.grad_query = chunk.grad_query});
    }
    return outputs;
}

// u_scaler holds one row per query row, so it takes the query's label.
std::vector<tt::tt_metal::TensorTopology> RingSDPABwQDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    std::vector<tt::tt_metal::TensorTopology> topologies;
    topologies.reserve(tensor_args.chunks.size() + 1U);
    for (const auto& chunk : tensor_args.chunks) {
        topologies.push_back(chunk.grad_query.tensor_topology());
    }
    topologies.push_back(tensor_args.chunks.front().query.tensor_topology());
    return topologies;
}

}  // namespace ttml::metal::ops::ring_sdpa_bw::q

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_bw::q::RingSDPABwQDeviceOperation::tensor_return_value_t ttml_ring_sdpa_bw_q(
    std::vector<ttml::metal::ops::ring_sdpa_bw::q::RingSDPABwQChunk> chunks,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa_bw::RingDirection ring_direction,
    ttml::metal::ops::RingLoadBalancing load_balancing) {
    using OperationType = ttml::metal::ops::ring_sdpa_bw::q::RingSDPABwQDeviceOperation;

    TT_FATAL(!chunks.empty(), "Ring SDPA needs at least one chunk");

    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction,
        .load_balancing = load_balancing};

    auto tensor_args = OperationType::tensor_args_t{.chunks = std::move(chunks)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
