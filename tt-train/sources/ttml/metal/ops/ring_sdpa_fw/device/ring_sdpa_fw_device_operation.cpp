// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_fw_device_operation.hpp"

#include <tt-metalium/workload/legacy/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_sdpa_fw {

using namespace tt::tt_metal;
using namespace ttnn;

void RingSDPAFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& chunks = tensor_args.chunks;
    validate_ring_load_balancing(attrs, chunks.size());
    validate_ring_attributes(attrs, chunks.front().query);
    for (const auto& chunk : chunks) {
        validate_ring_qkv(chunk.query, chunk.key, chunk.value);
        validate_output_like_tensor(chunk.output, "Output", chunk.query, chunk.value);
        validate_intermediates_tensor(chunk.intermediates, chunk.query);
    }
    for (size_t c = 1; c < chunks.size(); ++c) {
        validate_same_spec(chunks[c].query, chunks.front().query, "query", c);
        validate_same_spec(chunks[c].key, chunks.front().key, "key", c);
        validate_same_spec(chunks[c].value, chunks.front().value, "value", c);
        validate_same_spec(chunks[c].output, chunks.front().output, "output", c);
        validate_same_spec(chunks[c].intermediates, chunks.front().intermediates, "intermediates", c);
    }
}

RingSDPAFwDeviceOperation::spec_return_value_t RingSDPAFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    spec_return_value_t specs;
    specs.reserve(tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        specs.push_back({.output = chunk.output.tensor_spec(), .intermediates = chunk.intermediates.tensor_spec()});
    }
    return specs;
}

RingSDPAFwDeviceOperation::tensor_return_value_t RingSDPAFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    outputs.reserve(tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        outputs.push_back({.output = chunk.output, .intermediates = chunk.intermediates});
    }
    return outputs;
}

std::vector<tt::tt_metal::TensorTopology> RingSDPAFwDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    std::vector<tt::tt_metal::TensorTopology> topologies;
    topologies.reserve(2U * tensor_args.chunks.size());
    for (const auto& chunk : tensor_args.chunks) {
        topologies.push_back(chunk.output.tensor_topology());
        topologies.push_back(chunk.intermediates.tensor_topology());
    }
    return topologies;
}

}  // namespace ttml::metal::ops::ring_sdpa_fw

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_fw::RingSDPAFwDeviceOperation::tensor_return_value_t ttml_ring_sdpa_fw(
    std::vector<ttml::metal::ops::ring_sdpa_fw::RingSDPAFwChunk> chunks,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa_fw::RingDirection ring_direction,
    ttml::metal::ops::RingLoadBalancing load_balancing) {
    using OperationType = ttml::metal::ops::ring_sdpa_fw::RingSDPAFwDeviceOperation;

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
