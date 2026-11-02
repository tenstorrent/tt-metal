// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Residual add fused with RMSNorm, every tile-row split over several cores:
//   sum = residual + delta                        (bf16, the next residual stream)
//   rms = sqrt(mean(sum^2) + epsilon)             (bf16 [B, N, S, 1], column 0 of each tile)
//   out = sum / rms * gamma                       (bf16)
// Returns {sum, out, rms}. Two phases: per-(tile-row, slice) sums and partial sums of squares, then the row scale
// and the normalized output. All inputs bf16 TILE, C a multiple of 32.
std::vector<ttnn::Tensor> add_rmsnorm_fw_allcore(
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& delta_tensor,
    const ttnn::Tensor& gamma_tensor,
    float epsilon = 1e-6F,
    bool delta_transposed = false);

}  // namespace ttml::metal
