// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_bw::device::kv {

struct SDPABackwardKVParams {
    AttentionMaskType mask_type{AttentionMaskType::Arbitrary};
    float dropout_probability{0.0F};
    // Causal only: one work item per (K/V row, query head of its group); the outputs are then fp32 per-head
    // partials of shape (B, q_heads, S, D) that ttml_sdpa_kv_bw_reduce sums per group into bf16 grad_K / grad_V.
    bool head_split{false};
};

struct SDPABackwardKVInputs {
    const ttnn::Tensor& grad_output;               // Gradient w.r.t. output
    const ttnn::Tensor& query;                     // Input Q (needed for gradients)
    const ttnn::Tensor& key;                       // Input K (needed for gradients)
    const ttnn::Tensor& value;                     // Input V (needed for gradients)
    const std::optional<ttnn::Tensor>& attn_mask;  // attention mask
    const ttnn::Tensor& intermediates;             // From forward pass (FP32 logsumexp per row)
    const ttnn::Tensor& u_scaler;                  // Precomputed rowsum(dO * O) from Q kernel

    // Preallocated gradient tensors (optional)
    std::optional<ttnn::Tensor> preallocated_grad_key;
    std::optional<ttnn::Tensor> preallocated_grad_value;
};

using operation_attributes_t = SDPABackwardKVParams;
using tensor_args_t = SDPABackwardKVInputs;

using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // [grad_K, grad_V]

using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_bw::device::kv

namespace ttml::metal::ops::sdpa_bw::device::kv_reduce {

struct SDPABackwardKVReduceParams {
    uint32_t heads_per_group{1};
};

struct SDPABackwardKVReduceInputs {
    const ttnn::Tensor& grad_key_partial;    // fp32 (B, q_heads, S, kE)
    const ttnn::Tensor& grad_value_partial;  // fp32 (B, q_heads, S, vE)
    const ttnn::Tensor& key;                 // shape, dtype and memory config of grad_K
    const ttnn::Tensor& value;               // shape, dtype and memory config of grad_V
};

using operation_attributes_t = SDPABackwardKVReduceParams;
using tensor_args_t = SDPABackwardKVReduceInputs;
using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // [grad_K, grad_V]
using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_bw::device::kv_reduce
