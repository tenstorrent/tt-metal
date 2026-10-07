// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Grouped, gated RMSNorm (Qwen3-Next / Gated DeltaNet output norm) on flat token-major activations.
//
//   input, gate : [B, 1, T, W] TILE bf16, W = num_groups * group  (group == gamma.shape[-1])
//   gamma       : [1, 1, 1, group] TILE bf16
//
//   out[..., g*group + c] = x[..., g*group + c] * rsqrt(mean_c(x[..., g*group + :]^2) + eps) * gamma[c] *
//   silu(gate[..., g*group + c])
//
// i.e. per-head RMSNorm over each contiguous `group`-wide slice of the last dim, followed by the
// `* gamma * silu(gate)` epilogue, without ever reshaping to [B, T*H, group].
ttnn::Tensor gated_rmsnorm_fw(
    const ttnn::Tensor& input, const ttnn::Tensor& gate, const ttnn::Tensor& gamma, float epsilon = 1e-6F);

// Backward of gated_rmsnorm_fw. Returns {dL/dinput, dL/dgate, dL/dgamma (only when compute_dgamma)}.
// dL/dgamma is reduced to [1, 1, 1, group].
std::vector<std::optional<ttnn::Tensor>> gated_rmsnorm_bw(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    float epsilon = 1e-6F,
    bool compute_dgamma = true);

}  // namespace ttml::metal
