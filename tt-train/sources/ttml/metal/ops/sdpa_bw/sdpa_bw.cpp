// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw.hpp"

#include "device/sdpa_bw_kv_device_operation.hpp"
#include "device/sdpa_bw_q_device_operation.hpp"

namespace ttml::metal::ops::sdpa_bw {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> SDPABackwardOperation::invoke(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const std::optional<ttnn::Tensor>& attn_mask,
    const ttnn::Tensor& intermediates,
    const float dropout_probability,
    const bool fp32_dest_acc_en) {
    // Call KV kernel to compute grad_K and grad_V
    auto [grad_K, grad_V] = ttnn::prim::ttml_sdpa_kv_bw(
        grad_output, attn_output, query, key, value, attn_mask, intermediates, dropout_probability, fp32_dest_acc_en);

    // Call Q kernel to compute grad_Q
    auto grad_Q = ttnn::prim::ttml_sdpa_q_bw(
        grad_output, attn_output, query, key, value, attn_mask, intermediates, dropout_probability, fp32_dest_acc_en);

    return {grad_Q, grad_K, grad_V};
}

}  // namespace ttml::metal::ops::sdpa_bw
