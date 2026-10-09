// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Grouped gated RMSNorm on head-merged activations. `input`, `gate` [B, 1, T, H*V], `gamma`
// [1, 1, 1, V] shared by all heads, all TILE bf16 with T % 32 == 0 and V % 32 == 0:
//   out[.., h*V + c] = rmsnorm_V(input[.., h*V : (h+1)*V])[c] * gamma[c] * silu(gate[.., h*V + c])
// Each head is normalized in place on its own tiles, so no reshape to [.., V] is needed.
ttnn::Tensor gated_rmsnorm_fw(
    const ttnn::Tensor& input, const ttnn::Tensor& gate, const ttnn::Tensor& gamma, float epsilon = 1e-6F);

// Returns (dL/dinput, dL/dgate, dL/dgamma [1, 1, 1, V]); dL/dgamma is nullopt unless compute_dgamma.
std::tuple<ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>> gated_rmsnorm_bw(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    float epsilon = 1e-6F,
    bool compute_dgamma = true);

}  // namespace ttml::metal
