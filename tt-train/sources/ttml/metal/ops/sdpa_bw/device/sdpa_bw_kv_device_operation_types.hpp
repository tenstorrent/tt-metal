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
    // Which outputs to produce. grad_K needs u_scaler; grad_V does not.
    bool compute_grad_key{true};
    bool compute_grad_value{true};
};

struct SDPABackwardKVInputs {
    const ttnn::Tensor& grad_output;               // Gradient w.r.t. output
    const ttnn::Tensor& query;                     // Input Q (needed for gradients)
    const ttnn::Tensor& key;                       // Input K (needed for gradients)
    const ttnn::Tensor& value;                     // Input V (needed for gradients)
    const std::optional<ttnn::Tensor>& attn_mask;  // attention mask
    const ttnn::Tensor& intermediates;             // From forward pass (FP32 logsumexp per row)
    // Precomputed rowsum(dO * O) from Q kernel. Only dK uses it (through dS), so it is absent when only dV is needed.
    std::optional<ttnn::Tensor> u_scaler;

    // Preallocated gradient tensors (optional)
    std::optional<ttnn::Tensor> preallocated_grad_key;
    std::optional<ttnn::Tensor> preallocated_grad_value;
};

using operation_attributes_t = SDPABackwardKVParams;
using tensor_args_t = SDPABackwardKVInputs;

// [grad_K, grad_V]; an output is nullopt when its compute_* flag is false.
using tensor_return_value_t = std::tuple<std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>>;

using spec_return_value_t =
    std::tuple<std::optional<tt::tt_metal::TensorSpec>, std::optional<tt::tt_metal::TensorSpec>>;

}  // namespace ttml::metal::ops::sdpa_bw::device::kv
