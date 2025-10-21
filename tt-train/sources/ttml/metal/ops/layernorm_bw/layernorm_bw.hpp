// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_bw {

struct LayerNormBackwardOperation {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const ttnn::Tensor& input_tensor,     // [B, S, 1, C] - original input from forward
        const ttnn::Tensor& gamma_tensor,     // [1, 1, 1, C] - scale parameter
        const ttnn::Tensor& mean_tensor,      // [B, S, 1, 1] - mean from forward
        const ttnn::Tensor& rstd_tensor,      // [B, S, 1, 1] - reciprocal std from forward
        const ttnn::Tensor& dL_dout_tensor);  // [B, S, 1, C] - upstream gradient
};

}  // namespace ttml::metal::ops::layernorm_bw
