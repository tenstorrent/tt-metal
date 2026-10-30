// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ops/common/rmsnorm_common.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Returns {output, rms}; rms is nullopt unless return_intermediates. `input` may be BFLOAT16 or FLOAT32, everything
// else is BFLOAT16.
//
// With `gate`, output = rmsnorm(input) * gamma * silu(gate) (gated mode). `input` is head-major [B*H, 1, T, V],
// normalized over V, and `gate` head-merged [B, 1, T, H*V]; the output and its gradient take the gate's layout,
// while rms and the input gradient stay head-major. H = 1 is a plain gated RMSNorm.
//
// When whole tile-rows would leave cores idle, an ungated row may be split into up to `max_row_slices` (at most
// ops::rmsnorm::kMaxRowSlices) column slices on as many cores; 1 keeps whole rows.
std::vector<std::optional<ttnn::Tensor>> rmsnorm_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    bool return_intermediates = true,
    float epsilon = 1e-6F,
    const std::optional<ttnn::Tensor>& gate_tensor = std::nullopt,
    uint32_t max_row_slices = ops::rmsnorm::kMaxRowSlices);

}  // namespace ttml::metal
