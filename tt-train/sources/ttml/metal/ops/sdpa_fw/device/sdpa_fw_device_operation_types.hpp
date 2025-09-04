// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_fw::device {

struct operation_attributes_t {
    uint32_t q_heads{1U};
    uint32_t kv_heads{1U};
    bool return_intermediates{false};
    float dropout_probability{0.8F};  // default value
};

struct tensor_args_t {
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const std::optional<ttnn::Tensor>& mask;  // attention mask

    std::optional<ttnn::Tensor> preallocated_intermediate;
    std::optional<ttnn::Tensor> preallocated_output;
};

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_fw::device
