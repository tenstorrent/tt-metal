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
    ops::RingSchedule schedule,
    const std::optional<RingSDPABwShifted>& shifted) {
    std::optional<ops::ring_sdpa_bw::q::RingSDPABwQShifted> q_shifted;
    std::optional<ops::ring_sdpa_bw::kv::RingSDPABwKVShifted> kv_shifted;
    if (shifted.has_value()) {
        q_shifted = ops::ring_sdpa_bw::q::RingSDPABwQShifted{
            .grad_output = shifted->grad_output,
            .attn_output = shifted->attn_output,
            .query = shifted->query,
            .intermediates = shifted->intermediates,
            .grad_query = shifted->grad_query};
        kv_shifted = ops::ring_sdpa_bw::kv::RingSDPABwKVShifted{
            .grad_output = shifted->grad_output, .query = shifted->query, .intermediates = shifted->intermediates};
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
        schedule,
        q_shifted);

    // KV kernel: uses precomputed u_scaler (shared by the primary and shifted query: each device
    // fills it for the query it uses at this step)
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
        schedule,
        kv_shifted);

    return {grad_Q, grad_K, grad_V};
}

}  // namespace ttml::metal
