// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_bw::device::q {

struct SDPABackwardQParams {
    AttentionMaskType mask_type{AttentionMaskType::Arbitrary};
    float dropout_probability{0.0F};
    bool has_gate{false};
};

struct SDPABackwardQInputs {
    const ttnn::Tensor& grad_output;               // Gradient w.r.t. output
    const ttnn::Tensor& attn_output;               // sdpa forward output (needed for gradients)
    const ttnn::Tensor& query;                     // Input Q (needed for gradients)
    const ttnn::Tensor& key;                       // Input K (needed for gradients)
    const ttnn::Tensor& value;                     // Input V (needed for gradients)
    const std::optional<ttnn::Tensor>& attn_mask;  // attention mask (only for Arbitrary)
    const ttnn::Tensor& intermediates;             // From forward pass (max_val,1/sum_exp values)

    // Preallocated gradient tensor (optional)
    std::optional<ttnn::Tensor> preallocated_grad_query;
    // Preallocated u_scaler tensor for sharing with KV kernel (optional)
    std::optional<ttnn::Tensor> preallocated_u_scaler;

    const std::optional<ttnn::Tensor>& gate;  // Optional gate tensor
    // Preallocated gate gradient tensor (optional)
    std::optional<ttnn::Tensor> preallocated_grad_gate;
    // Preallocated gated gradient output tensor (optional)
    std::optional<ttnn::Tensor> preallocated_gated_grad_output;
};

using operation_attributes_t = SDPABackwardQParams;
using tensor_args_t = SDPABackwardQInputs;

// [grad_Q, u_scaler, grad_gate, gated_grad_output]
using tensor_return_value_t =
    std::tuple<ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>>;

using spec_return_value_t = std::tuple<
    tt::tt_metal::TensorSpec,
    tt::tt_metal::TensorSpec,
    std::optional<tt::tt_metal::TensorSpec>,
    std::optional<tt::tt_metal::TensorSpec>>;

}  // namespace ttml::metal::ops::sdpa_bw::device::q
