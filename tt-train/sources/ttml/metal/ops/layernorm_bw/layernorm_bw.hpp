// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Returns {dx, dgamma, dbeta}. A grad that is not requested is nullopt and its kernel work is skipped; at least one
// must be requested.
std::vector<std::optional<ttnn::Tensor>> layernorm_bw(
    const ttnn::Tensor& input_tensor,    // [B, 1, S, C] - original input from forward
    const ttnn::Tensor& gamma_tensor,    // [1, 1, 1, C] - scale parameter
    const ttnn::Tensor& mean_tensor,     // [B, 1, S, 1] - mean from forward
    const ttnn::Tensor& rstd_tensor,     // [B, 1, S, 1] - reciprocal std from forward
    const ttnn::Tensor& dL_dout_tensor,  // [B, 1, S, C] - upstream gradient
    const bool compute_grad_input = true,
    const bool compute_grad_gamma = true,
    const bool compute_grad_beta = true);

}  // namespace ttml::metal
