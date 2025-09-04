// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_fw {

struct SDPAForwardOperation {
    static std::vector<std::optional<ttnn::Tensor>>
    invoke(  // returns a vector of optional tensors due to the possibility of returning intermediates
        const ttnn::Tensor& query,
        const ttnn::Tensor& key,
        const ttnn::Tensor& value,
        const std::optional<ttnn::Tensor>& mask,  // attention mask
        const uint32_t q_heads, // num of query heads
        const uint32_t kv_heads, // num of key/value heads
        const float dropout_probability = 0.8F,          // default value?
        const bool return_intermediates = false);
};

}  // namespace ttml::metal::ops::sdpa_fw
