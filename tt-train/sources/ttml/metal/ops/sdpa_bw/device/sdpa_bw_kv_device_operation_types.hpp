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
    // Key rows to compute and the query rows that attend to them; std::nullopt = whole sequence.
    // q_range must lie within the q_range given to ttml_sdpa_q_bw and the forward that produced u_scaler and the
    // intermediates. dK/dV rows outside k_range are left unwritten.
    std::optional<SeqTileRange> q_range;
    std::optional<SeqTileRange> k_range;
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
