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
    const std::optional<ttnn::Tensor>& preallocated_grad_value) {
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
        preallocated_grad_query);

    // KV kernel: uses precomputed u_scaler
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
        preallocated_grad_value);

    return {grad_Q, grad_K, grad_V};
}

}  // namespace ttml::metal
