// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_bw::device::kv {

struct operation_attributes_t {
    AttentionMaskType mask_type{AttentionMaskType::Arbitrary};
    float dropout_probability{0.0F};

    static constexpr auto attribute_names = std::forward_as_tuple("mask_type", "dropout_probability");
    auto attribute_values() const {
        return std::forward_as_tuple(mask_type, dropout_probability);
    }
};

struct tensor_args_t {
    const ttnn::Tensor& grad_output;               // Gradient w.r.t. output
    const ttnn::Tensor& attn_output;               // sdap forward output (needed for gradients)
    const ttnn::Tensor& query;                     // Input Q (needed for gradients)
    const ttnn::Tensor& key;                       // Input K (needed for gradients)
    const ttnn::Tensor& value;                     // Input V (needed for gradients)
    const std::optional<ttnn::Tensor>& attn_mask;  // attention mask
    const ttnn::Tensor& intermediates;             // From forward pass (max_val,1/sum_exp values)

    // Preallocated gradient tensors (optional)
    std::optional<ttnn::Tensor> preallocated_grad_key;
    std::optional<ttnn::Tensor> preallocated_grad_value;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "grad_output",
        "attn_output",
        "query",
        "key",
        "value",
        "attn_mask",
        "intermediates",
        "preallocated_grad_key",
        "preallocated_grad_value");
    auto attribute_values() const {
        return std::forward_as_tuple(
            grad_output,
            attn_output,
            query,
            key,
            value,
            attn_mask,
            intermediates,
            preallocated_grad_key,
            preallocated_grad_value);
    }
};

using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // [grad_K, grad_V]

using spec_return_value_t = std::tuple<ttnn::TensorSpec, ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_bw::device::kv
