// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr rmsnorm(const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, float epsilon);
autograd::TensorPtr rmsnorm_composite(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, float epsilon);

// rmsnorm(input) * gamma * silu(gate), normalized over each V-wide head (the Gated DeltaNet output norm). `input` is
// head-major [B*H, 1, T, V] in bf16 or fp32, `gate` head-merged [B, 1, T, H*V] and `gamma` [1, 1, 1, V]. The output
// has the gate's layout. Backward skips dL/dgamma when gamma is frozen.
autograd::TensorPtr gated_rmsnorm(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& gate,
    const autograd::TensorPtr& gamma,
    float epsilon = 1e-6F);

}  // namespace ttml::ops
