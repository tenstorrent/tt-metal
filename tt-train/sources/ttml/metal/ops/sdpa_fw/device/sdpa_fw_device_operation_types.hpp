// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_fw::device {

struct operation_attributes_t {
    bool return_intermediates{false};
    AttentionMaskType mask_type{AttentionMaskType::Causal};
    float dropout_probability{0.0F};

    static constexpr auto attribute_names =
        std::forward_as_tuple("return_intermediates", "mask_type", "dropout_probability");
    auto attribute_values() const {
        return std::forward_as_tuple(return_intermediates, mask_type, dropout_probability);
    }
};

struct tensor_args_t {
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const std::optional<ttnn::Tensor>& mask;  // attention mask

    std::optional<ttnn::Tensor> preallocated_intermediate;
    std::optional<ttnn::Tensor> preallocated_output;

    static constexpr auto attribute_names =
        std::forward_as_tuple("query", "key", "value", "mask", "preallocated_intermediate", "preallocated_output");
    auto attribute_values() const {
        return std::forward_as_tuple(query, key, value, mask, preallocated_intermediate, preallocated_output);
    }
};

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_fw::device
