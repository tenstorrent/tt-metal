// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw.hpp"

#include "device/ring_sdpa_bw_kv_device_operation.hpp"
#include "device/ring_sdpa_bw_q_device_operation.hpp"

namespace ttml::metal {
using RingDirection = ttml::metal::ops::ring_sdpa_bw::RingDirection;

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_sdpa_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    RingDirection ring_direction,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value,
    ops::RingLaneSelection lane_selection,
    const std::optional<RingSDPABwLane>& lane1) {
    std::optional<ops::ring_sdpa_bw::q::RingSDPABwQLane> q_lane1;
    std::optional<ops::ring_sdpa_bw::kv::RingSDPABwKVLane> kv_lane1;
    if (lane1.has_value()) {
        q_lane1 = ops::ring_sdpa_bw::q::RingSDPABwQLane{
            .grad_output = lane1->grad_output,
            .attn_output = lane1->attn_output,
            .query = lane1->query,
            .key = lane1->key,
            .value = lane1->value,
            .intermediates = lane1->intermediates,
            .grad_query = lane1->grad_query};
        kv_lane1 = ops::ring_sdpa_bw::kv::RingSDPABwKVLane{
            .grad_output = lane1->grad_output,
            .query = lane1->query,
            .key = lane1->key,
            .value = lane1->value,
            .intermediates = lane1->intermediates,
            .grad_key = lane1->grad_key,
            .grad_value = lane1->grad_value};
    }

    // Q kernel first: computes grad_Q and u_scaler = rowsum(dO * O)
    auto [grad_Q, u_scaler] = ttnn::prim::ttml_ring_sdpa_bw_q(
        grad_output,
        attn_output,
        query,
        key,
        value,
        intermediates,
        ring_size,
        ring_axis,
        step,
        mask_type,
        ring_direction,
        preallocated_grad_query,
        lane_selection,
        q_lane1);

    // KV kernel: uses precomputed u_scaler (shared by both lanes: each device fills it for the
    // one lane it computes at this step)
    auto [grad_K, grad_V] = ttnn::prim::ttml_ring_sdpa_bw_kv(
        grad_output,
        u_scaler,
        query,
        key,
        value,
        intermediates,
        ring_size,
        ring_axis,
        step,
        mask_type,
        ring_direction,
        preallocated_grad_key,
        preallocated_grad_value,
        lane_selection,
        kv_lane1);

    return {grad_Q, grad_K, grad_V};
}

}  // namespace ttml::metal
