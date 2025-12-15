// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw.hpp"

#include "device/sdpa_bw_kv_device_operation.hpp"
#include "device/sdpa_bw_q_device_operation.hpp"

namespace ttml::metal::ops::sdpa_bw {

std::vector<ttnn::Tensor> SDPABackwardOperation::invoke(
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
    // Returns [grad_Q_placeholder, grad_K, grad_V]
    auto kv_result = ttnn::prim::ttml_sdpa_kv_bw(
        grad_output, attn_output, query, key, value, attn_mask, intermediates, dropout_probability, fp32_dest_acc_en);

    // // Call Q kernel to compute grad_Q
    // // Returns grad_Q as a single tensor
    auto grad_Q = ttnn::prim::ttml_sdpa_q_bw(
        grad_output, attn_output, query, key, value, attn_mask, intermediates, dropout_probability, fp32_dest_acc_en);

    // Combine results: [grad_Q, grad_K, grad_V]
    std::vector<ttnn::Tensor> result;
    result.reserve(3U);
    result.push_back(grad_Q);       // grad_Q from Q kernel
    result.push_back(kv_result[0]); // grad_K from KV kernel
    result.push_back(kv_result[1]); // grad_V from KV kernel

    return result;  // Returns [grad_Q, grad_K, grad_V]
};

}  // namespace ttml::metal::ops::sdpa_bw
