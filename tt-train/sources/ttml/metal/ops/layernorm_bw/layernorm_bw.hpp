// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Which grads layernorm_bw computes. A field set to false drops that grad's kernel work; at least one must be set.
struct LayerNormBackwardGrads {
    bool input = true;
    bool gamma = true;
    bool beta = true;
};

// Returns {dx, dgamma, dbeta}; a grad that is not requested is nullopt.
std::vector<std::optional<ttnn::Tensor>> layernorm_bw(
    const ttnn::Tensor& input_tensor,    // [B, 1, S, C] - original input from forward
    const ttnn::Tensor& gamma_tensor,    // [1, 1, 1, C] - scale parameter
    const ttnn::Tensor& mean_tensor,     // [B, 1, S, 1] - mean from forward
    const ttnn::Tensor& rstd_tensor,     // [B, 1, S, 1] - reciprocal std from forward
    const ttnn::Tensor& dL_dout_tensor,  // [B, 1, S, C] - upstream gradient
    const LayerNormBackwardGrads& grads = {});

}  // namespace ttml::metal
