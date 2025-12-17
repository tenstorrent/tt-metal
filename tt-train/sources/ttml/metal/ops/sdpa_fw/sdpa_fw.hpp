// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_fw {

struct SDPAForwardOperation {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const ttnn::Tensor& query,
        const ttnn::Tensor& key,
        const ttnn::Tensor& value,
        const std::optional<ttnn::Tensor>& mask,  // attention mask
        const float dropout_probability = 0.0F,   // default value
        const bool return_intermediates = false,
        const bool fp32_dest_acc_en = true);
};

}  // namespace ttml::metal::ops::sdpa_fw
