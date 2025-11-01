// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_fw {

struct LayerNormForwardOperation {
    // Returns [output, mean, rstd] where mean and rstd are optional depending on return_mean_rstd
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const ttnn::Tensor& input_tensor,  // [B, 1, S, C] - input
        const ttnn::Tensor& gamma_tensor,  // [1, 1, 1, C] - scale parameter
        const ttnn::Tensor& beta_tensor,   // [1, 1, 1, C] - shift parameter
        float epsilon = 1e-5F,             // epsilon for numerical stability
        bool return_mean_rstd = false);    // whether to return mean and rstd
};

}  // namespace ttml::metal::ops::layernorm_fw
