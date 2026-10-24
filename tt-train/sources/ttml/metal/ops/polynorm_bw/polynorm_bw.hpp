// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Fused PolyNorm3 backward kernel entry point.
// Returns dL_dx, dL_dw ([1,1,1,3]), and dL_db ([1,1,1,1]). A grad that is not requested is nullopt and its kernel
// work is skipped; at least one must be requested.
std::tuple<std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> polynorm3_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& weight_tensor,
    float epsilon = 1e-5F,
    bool compute_grad_input = true,
    bool compute_grad_weight = true,
    bool compute_grad_bias = true);

}  // namespace ttml::metal
