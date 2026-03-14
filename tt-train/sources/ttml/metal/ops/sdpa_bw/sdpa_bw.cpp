// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw.hpp"

#include "device/sdpa_bw_kv_device_operation.hpp"
#include "device/sdpa_bw_q_device_operation.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> sdpa_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    AttentionMaskType mask_type,
    const std::optional<ttnn::Tensor>& attn_mask,
    const float dropout_probability) {
    // Call Q kernel first to compute grad_Q and u_scaler = rowsum(dO * O)
    auto [grad_Q, u_scaler] = ttnn::prim::ttml_sdpa_q_bw(
        grad_output, attn_output, query, key, value, mask_type, attn_mask, intermediates, dropout_probability);

    // Call KV kernel with precomputed u_scaler (no longer needs attn_output)
    auto [grad_K, grad_V] = ttnn::prim::ttml_sdpa_kv_bw(
        grad_output,
        query,
        key,
        value,
        mask_type,
        attn_mask,
        intermediates,
        u_scaler,
        std::nullopt,  // attn_output not needed when u_scaler is provided
        dropout_probability);

    return {grad_Q, grad_K, grad_V};
}

}  // namespace ttml::metal
