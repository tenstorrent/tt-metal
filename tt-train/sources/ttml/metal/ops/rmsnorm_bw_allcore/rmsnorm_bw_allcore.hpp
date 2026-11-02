// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// RMSNorm backward. Returns {dL/dinput, dL/dgamma}; dL/dgamma is nullopt when compute_dgamma is false
// (frozen gamma), which also skips the dgamma-components pass and its reduction. `residual_grad` (same shape as
// the input) is added to dL/dinput inside the kernel: the gradient of a residual stream that is both the norm's
// input and the next sublayer's residual (fused add + RMSNorm).
std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw_allcore(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,  // intermediate from fw
    const ttnn::Tensor& dL_dout_tensor,
    bool compute_dgamma = true,
    const std::optional<ttnn::Tensor>& residual_grad = std::nullopt,
    bool transposed_grad = false);

}  // namespace ttml::metal
