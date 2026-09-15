// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_zigzag_sdpa.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_zigzag_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ops::ZigzagVisitor visitor,
    AttentionMaskType mask_type,
    ops::ring_zigzag_fw::RingDirection ring_direction,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates) {
    return ttnn::prim::ttml_ring_zigzag_fw(
        query,
        key,
        value,
        ring_size,
        ring_axis,
        step,
        mask_type,
        ring_direction,
        visitor,
        preallocated_output,
        preallocated_intermediates);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_zigzag_sdpa_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ops::ZigzagVisitor visitor,
    AttentionMaskType mask_type,
    ops::ring_zigzag_bw::RingDirection ring_direction,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value) {
    auto [grad_Q, u_scaler] = ttnn::prim::ttml_ring_zigzag_bw_q(
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
        visitor,
        preallocated_grad_query);
    auto [grad_K, grad_V] = ttnn::prim::ttml_ring_zigzag_bw_kv(
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
        visitor,
        preallocated_grad_key,
        preallocated_grad_value);
    return {grad_Q, grad_K, grad_V};
}

}  // namespace ttml::metal
