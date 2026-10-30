// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw.hpp"

#include "device/sdpa_bw_kv_device_operation.hpp"
#include "device/sdpa_bw_kv_reduce_device_operation.hpp"
#include "device/sdpa_bw_q_device_operation.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>> sdpa_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    AttentionMaskType mask_type,
    const std::optional<ttnn::Tensor>& attn_mask,
    const float dropout_probability,
    const std::optional<ttnn::Tensor>& gate,
    const std::optional<ttnn::Tensor>& preallocated_grad_gate) {
    // Call Q kernel first to compute grad_Q and u_scaler = rowsum(dO * O)
    auto [grad_Q, u_scaler, grad_gate, gated_grad_output] = ttnn::prim::ttml_sdpa_q_bw(
        grad_output,
        attn_output,
        query,
        key,
        value,
        mask_type,
        attn_mask,
        intermediates,
        dropout_probability,
        /*preallocated_grad_query=*/std::nullopt,
        /*preallocated_u_scaler=*/std::nullopt,
        gate,
        preallocated_grad_gate);

    const auto& kv_grad_output = gated_grad_output.has_value() ? gated_grad_output.value() : grad_output;

    // Causal GQA: one work item per (K/V row, query head) keeps every core busy (64 rows alone would use 64
    // cores, the longest row 32x the shortest); the per-head fp32 partials are summed per group afterwards.
    const uint32_t heads_per_group = query.logical_shape()[1] / key.logical_shape()[1];
    const bool head_split = (mask_type == AttentionMaskType::Causal) && heads_per_group > 1;

    auto [grad_K, grad_V] = ttnn::prim::ttml_sdpa_kv_bw(
        kv_grad_output,
        query,
        key,
        value,
        mask_type,
        attn_mask,
        intermediates,
        u_scaler,
        dropout_probability,
        /*preallocated_grad_key=*/std::nullopt,
        /*preallocated_grad_value=*/std::nullopt,
        head_split);
    if (head_split) {
        std::tie(grad_K, grad_V) = ttnn::prim::ttml_sdpa_kv_bw_reduce(grad_K, grad_V, key, value, heads_per_group);
    }

    return {grad_Q, grad_K, grad_V, grad_gate};
}

}  // namespace ttml::metal
