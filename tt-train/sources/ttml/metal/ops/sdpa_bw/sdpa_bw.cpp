// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw.hpp"

#include "device/sdpa_bw_device_operation.hpp"

namespace ttml::metal::ops::sdpa_bw {

std::vector<ttnn::Tensor> SDPABackwardOperation::invoke(
    const ttnn::Tensor& grad_output,     
    const ttnn::Tensor& query,           
    const ttnn::Tensor& key,             
    const ttnn::Tensor& value,           
    const std::optional<ttnn::Tensor>& attn_mask,  
    const ttnn::Tensor& intermediates,   
    const float dropout_probability,
    const bool fp32_dest_acc_en) {
    auto result = ttnn::prim::ttml_sdpa_bw(
        grad_output, query, key, value, attn_mask, intermediates, dropout_probability, fp32_dest_acc_en);

    return result;  // Returns [grad_Q, grad_K, grad_V]
};

}  // namespace ttml::metal::ops::sdpa_bw
